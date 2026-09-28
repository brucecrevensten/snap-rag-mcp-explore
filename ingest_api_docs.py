"""
ingest_api_docs.py - Index the Data API's own documentation pages.

Each service at https://earthmaps.io/<service>/ documents what it serves:
variable definitions (what `bui` or `dmc` mean), units, which models and
scenarios, the year span, caveats, the endpoints, and -- usefully -- links to
the catalog records that service is built from. None of that is in the
GeoNetwork records, so we harvest the pages, chunk them by heading, and embed
them into the "api_docs" collection, linked to those datasets.

The result: a question can match an explanation on a documentation page and
lead to the dataset(s) it describes, the same way ingest_papers.py does with
papers.

Usage:
    python ingest_api_docs.py                  # every service page
    python ingest_api_docs.py --limit 3 --show # try a few, print the chunks
    python ingest_api_docs.py --dry-run        # fetch + chunk, don't embed

Writes doc_chunks.jsonl (exactly what was embedded).
"""

import argparse
import json
import re
import textwrap
import urllib.request

from lxml import html as lxml_html

import lane1

API = "https://earthmaps.io"
USER_AGENT = "snap-rag-mcp-explore/0.1 (+https://github.com/brucecrevensten/snap-rag-mcp-explore)"
CATALOG_RECORD = "https://catalog.snap.uaf.edu/geonetwork/srv/eng/catalog.search#/metadata/{uuid}"
# Page furniture that is on every page and says nothing about the data.
SKIP_SECTIONS = {"", "Built by the Scenarios Network for Alaska + Arctic Planning"}


def http_get(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def services():
    """Service names from the API home page, home page first."""
    home = http_get(API + "/")
    found = sorted(set(re.findall(r'href="/([a-z0-9_]+)"', home)) - {"static"})
    return ["", *found]          # "" is the home page: provider, license, index


# ------------------------------------------------------------ page -> text

def text_of(el):
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


def page_sections(page_html):
    """[(heading, text)] for one documentation page, with the banner and
    footer dropped. Tables become one line per row ('bui | Numeric rating
    of...'), which is how the variable definitions read best."""
    tree = lxml_html.fromstring(page_html)
    for junk in tree.xpath("//footer | //nav | //script | //style | //*[@class='headerbanner']"):
        junk.getparent().remove(junk)

    sections, heading, lines = [], "", []
    for el in tree.xpath("//div[contains(@class,'content')]/*"):
        tag = el.tag.lower()
        if tag in ("h1", "h2", "h3", "h4"):
            if lines:
                sections.append((heading, "\n".join(lines)))
            heading, lines = text_of(el), []
        elif tag == "table":
            for row in el.xpath(".//tr"):
                cells = [text_of(c) for c in row.xpath("./th | ./td")]
                if any(cells):
                    lines.append(" | ".join(cells))
        elif tag in ("ul", "ol"):
            lines += [f"- {text_of(li)}" for li in el.xpath("./li") if text_of(li)]
        else:
            # Keep link targets: the references at the foot of a page are DOIs.
            for a in el.xpath(".//a[@href]"):
                href = a.get("href")
                if href.startswith("http") and href not in text_of(el):
                    a.tail = f" <{href}>" + (a.tail or "")
            if text_of(el):
                lines.append(text_of(el))
    if lines:
        sections.append((heading, "\n".join(lines)))
    return [(h, t) for h, t in sections if h not in SKIP_SECTIONS and t.strip()]


# ---------------------------------------------------------------- chunks

def build_doc_chunks(service, url, page_html, titles_by_uuid):
    sections = page_sections(page_html)
    if not sections:
        return []
    page_title = sections[0][0] or (service or "Alaska + Arctic Geospatial Data API")
    uuids = list(dict.fromkeys(re.findall(r"catalog\.search#/metadata/([0-9a-f-]{36})", page_html)))
    chunks = []
    for n, (heading, text) in enumerate(sections):
        header = f"API docs: {page_title}\nSection: {heading}\n\n"
        budget = max(lane1.MAX_TOKENS - 2 - lane1.count_tokens(header), 64)
        pieces = lane1.split_by_tokens(text, budget)
        for i, piece in enumerate(pieces):
            chunks.append({
                # Numbered, not named: headings like "Point query" repeat on a page.
                "id": f"doc:{service or 'home'}::s{n}::{i}",
                "document": header + piece,
                "metadata": {
                    "doc_key": f"doc:{service or 'home'}", "service": service or "home",
                    "url": url, "page_title": page_title, "heading": heading,
                    "section": "api_docs", "chunk": i, "n_chunks": len(pieces),
                    "section_text": text[:8000], "page": 0,
                    "citation": f"SNAP Data API documentation: {page_title} ({url})",
                    "doi": "", "full_text": True,
                    "dataset_uuids": ",".join(uuids),
                    "dataset_titles": " | ".join(titles_by_uuid.get(u, u) for u in uuids),
                }})
    return chunks


def show_chunk(c):
    m = c["metadata"]
    print(f"\n  ┌─ {c['id']}   {lane1.count_tokens(c['document']) + 2}/{lane1.MAX_TOKENS} tokens")
    print(f"  │  describes: {m['dataset_titles'][:110] or '(no catalog records linked)'}")
    print("  ├" + "─" * 60)
    for line in c["document"].splitlines():
        for piece in textwrap.wrap(line, 90) or [""]:
            print("  │ " + piece)
    print("  └" + "─" * 60)


# ------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="only the first N service pages")
    p.add_argument("--show", action="store_true", help="print every chunk")
    p.add_argument("--dry-run", action="store_true", help="fetch + chunk only; embed nothing")
    args = p.parse_args()

    names = services()[:args.limit]
    print(f"{len(names)} documentation pages\n")
    # Dataset titles, so a chunk can say which datasets its page describes.
    titles = {}
    try:
        for m in lane1.open_collection().get(include=["metadatas"])["metadatas"]:
            titles.setdefault(m["uuid"], m["title"])
    except Exception:
        pass                      # no catalog index yet; uuids alone still link

    all_chunks = []
    for n, service in enumerate(names, 1):
        url = f"{API}/{service}/" if service else f"{API}/"
        try:
            chunks = build_doc_chunks(service, url, http_get(url), titles)
        except Exception as e:
            print(f"  [{n}] {service or 'home':22} failed: {e.__class__.__name__}")
            continue
        all_chunks += chunks
        n_sections = len({c["metadata"]["heading"] for c in chunks})
        linked = len([u for u in chunks[0]["metadata"]["dataset_uuids"].split(",") if u]) if chunks else 0
        print(f"  [{n}] {service or 'home':22} {n_sections:>3} sections  {len(chunks):>3} chunks  "
              f"{linked} datasets linked")
        if args.show:
            for c in chunks:
                show_chunk(c)

    if args.dry_run:
        print(f"\nDry run: {len(all_chunks)} chunks; nothing embedded or written.")
        return

    collection = lane1.open_collection(create=True, name=lane1.DOCS_COLLECTION)
    for key in sorted({c["metadata"]["doc_key"] for c in all_chunks}):
        collection.delete(where={"doc_key": key})      # replace, don't pile up
    for i in range(0, len(all_chunks), 100):
        batch = all_chunks[i:i + 100]
        collection.add(ids=[c["id"] for c in batch], documents=[c["document"] for c in batch],
                       metadatas=[c["metadata"] for c in batch])
    with open("doc_chunks.jsonl", "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nEmbedded {len(all_chunks)} chunks from {len(names)} pages into "
          f"'{lane1.DOCS_COLLECTION}'")


if __name__ == "__main__":
    main()
