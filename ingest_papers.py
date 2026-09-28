"""
ingest_papers.py - Fetch the papers our datasets cite, chunk and embed them,
each chunk linked back to the dataset(s) that cite the paper.

For each publication in references.json (made by list_references.py):
  1. Ask OpenAlex (free, no key) about it: title, authors, year, abstract, and
     any LEGAL open-access copy (publisher or repository PDF). Paywalled
     papers are never fetched, and a server that refuses scripted downloads
     is taken at its word; for those we keep the abstract, or failing that
     the citation itself.
  2. Download the PDF to ./papers/ (cached), extract text page by page, and
     cut off the reference list at the end (pages of citations make noisy
     search matches).
  3. Split each page into token-sized chunks with the same splitter as
     lane1.py, and embed them into the "papers" collection of ./lane1_index.
     A chunk's metadata names the paper (DOI, citation) and the datasets
     that cite it; its section_text is the whole page (small-to-big).

Usage:
    python ingest_papers.py                    # every publication
    python ingest_papers.py --limit 3 --show   # try a few, print the chunks
    python ingest_papers.py --dry-run          # fetch + chunk, don't embed

Writes ./papers/*.pdf (cache) and paper_chunks.jsonl (exactly what was embedded).
"""

import argparse
import io
import json
import logging
import re
import textwrap
import urllib.parse
import urllib.request
from pathlib import Path

from pypdf import PdfReader

import lane1

logging.getLogger("pypdf").setLevel(logging.ERROR)   # quiet "wrong pointing object" noise

REFERENCES_FILE = "references.json"
PAPERS_DIR = Path("papers")
USER_AGENT = "snap-rag-mcp-explore/0.1 (+https://github.com/brucecrevensten/snap-rag-mcp-explore)"
# A heading that starts the reference list; everything after it is dropped.
REFERENCE_HEADING = re.compile(r"^\s*(references( and notes| cited)?|literature cited|bibliography"
                               r"|works cited)\s*:?\s*$", re.I | re.M)


# ------------------------------------------------------------- fetching

def http_get(url, accept=None):
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as r:
        return r.read()


def openalex(doi):
    """OpenAlex's record for a DOI, or None."""
    try:
        return json.loads(http_get("https://api.openalex.org/works/https://doi.org/"
                                   + urllib.parse.quote(doi)))
    except Exception:
        return None


def abstract_of(work):
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    index = (work or {}).get("abstract_inverted_index") or {}
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return " ".join(word for _, word in words)


def pdf_candidates(ref, work):
    """Open-access PDF URLs to try, best first."""
    urls = []
    if work:
        best = work.get("best_oa_location") or {}
        urls.append(best.get("pdf_url"))
        urls += [loc.get("pdf_url") for loc in work.get("locations", []) if loc.get("is_oa")]
    if ref["url"].lower().split("?")[0].endswith(".pdf"):
        # A GitHub "blob" page is HTML; the "raw" URL is the file itself.
        urls.append(ref["url"].replace("github.com", "raw.githubusercontent.com")
                              .replace("/blob/", "/") if "github.com" in ref["url"] else ref["url"])
    return [u for u in dict.fromkeys(urls) if u]


def fetch_pdf(ref, work):
    """(pdf bytes, url) for the first candidate that is really a PDF; cached."""
    PAPERS_DIR.mkdir(exist_ok=True)
    cache = PAPERS_DIR / (re.sub(r"[^a-z0-9]+", "_", ref["key"].lower())[:80] + ".pdf")
    if cache.exists():
        return cache.read_bytes(), str(cache)
    for url in pdf_candidates(ref, work):
        try:
            data = http_get(url, accept="application/pdf")
        except Exception:
            continue
        if data[:5] == b"%PDF-":          # some "PDF" links serve an HTML login page
            cache.write_bytes(data)
            return data, url
    return None, None


# ------------------------------------------------------------ text

def pdf_pages(data):
    """Clean text of each page, with the reference list cut off."""
    pages = [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]
    # Find the reference-list heading, only in the back half so a "References"
    # in a table of contents doesn't cut the whole paper.
    for i in range(len(pages) // 2, len(pages)):
        m = REFERENCE_HEADING.search(pages[i])
        if m:
            pages = pages[:i] + [pages[i][:m.start()]]
            break
    cleaned = []
    for text in pages:
        text = re.sub(r"-\n(?=[a-z])", "", text)     # re-join words hyphenated at line ends
        text = re.sub(r"\s*\n\s*", " ", text)         # PDF lines are layout, not meaning
        cleaned.append(re.sub(r"\s{2,}", " ", text).strip())
    return cleaned


def short_authors(work, ref):
    names = [a["author"]["display_name"] for a in (work or {}).get("authorships", [])]
    if not names:
        return ref["citation"].split("(")[0].strip(" ,")[:60]
    return names[0].split()[-1] + (" et al." if len(names) > 2 else
                                   f" & {names[1].split()[-1]}" if len(names) == 2 else "")


# ------------------------------------------------------------ chunks

def build_paper_chunks(ref, work, pages, full_text, section="abstract", min_chars=200):
    """Chunks for one paper; pages is a list of page texts (or [abstract])."""
    title = (work or {}).get("title") or ref.get("title") or ref["label"] or ref["url"]
    year = (work or {}).get("publication_year") or ref.get("year") or ""
    who = short_authors(work, ref)
    header = f"Paper: {title} ({who}, {year})\n\n"
    budget = max(lane1.MAX_TOKENS - 2 - lane1.count_tokens(header), 64)
    cited_by = ref["cited_by"]
    chunks = []
    for page_no, text in enumerate(pages, 1):
        if len(text) < min_chars:         # blank, figure-only, or title-block pages
            continue
        pieces = lane1.split_by_tokens(text, budget)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{ref['key']}::p{page_no}::{i}",
                "document": header + piece,
                "metadata": {
                    "paper_key": ref["key"], "doi": ref.get("doi", ""),
                    "url": ref["url"], "paper_title": title, "year": str(year),
                    "authors": who, "citation": ref["citation"] or title,
                    "full_text": full_text, "page": page_no, "n_pages": len(pages),
                    "chunk": i, "n_chunks": len(pieces),
                    "section": "paper" if full_text else section,
                    "section_text": text[:8000],
                    # Chroma metadata holds strings, not lists: comma / " | " joined.
                    "dataset_uuids": ",".join(c["uuid"] for c in cited_by),
                    "dataset_titles": " | ".join(dict.fromkeys(c["title"] for c in cited_by)),
                }})
    return chunks


def show_paper_chunk(c):
    m = c["metadata"]
    print(f"\n  ┌─ {c['id']}   {lane1.count_tokens(c['document']) + 2}/{lane1.MAX_TOKENS} tokens")
    print(f"  │  cited by: {m['dataset_titles'][:110]}")
    print("  ├" + "─" * 60)
    for line in c["document"].splitlines():
        for piece in textwrap.wrap(line, 90) or [""]:
            print("  │ " + piece)
    print("  └" + "─" * 60)


# --------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="only the first N publications")
    p.add_argument("--show", action="store_true", help="print every chunk")
    p.add_argument("--dry-run", action="store_true", help="fetch + chunk only; embed nothing")
    args = p.parse_args()

    try:
        refs = json.load(open(REFERENCES_FILE))["references"]
    except FileNotFoundError:
        raise SystemExit(f"{REFERENCES_FILE} not found. Run list_references.py first.")
    pubs = [r for r in refs if r["kind"] == "publication"][:args.limit]
    print(f"{len(pubs)} publications to fetch (of {len(refs)} references)\n")

    all_chunks, report = [], []
    for n, ref in enumerate(pubs, 1):
        work = openalex(ref["doi"]) if ref.get("doi") else None
        data, source = fetch_pdf(ref, work)
        pages = None
        if data:
            try:
                pages = pdf_pages(data)
            except Exception as e:
                source = f"PDF unreadable ({e.__class__.__name__})"
        # Why no PDF? "none" = no open-access copy known; "refused" = a copy
        # exists but the server wouldn't hand it to a script (we don't push).
        no_pdf = "PDF refused" if pdf_candidates(ref, work) else "no open-access PDF"
        if pages and sum(len(t) for t in pages) > 1000:
            how = f"full text, {len(pages)} pages"
            chunks = build_paper_chunks(ref, work, pages, full_text=True)
        elif abstract_of(work):
            how = f"abstract only ({no_pdf})"
            chunks = build_paper_chunks(ref, work, [abstract_of(work)], full_text=False)
        else:
            # Last resort: the citation itself (title, authors, venue), so the
            # paper -- and the datasets citing it -- can still match by topic.
            how = f"citation only ({no_pdf}, no abstract)"
            chunks = build_paper_chunks(ref, work, [ref["citation"] or ref["label"]],
                                        full_text=False, section="citation", min_chars=20)
        all_chunks += chunks
        report.append((ref["citation"] or ref["url"], how, len(chunks)))
        print(f"  [{n}] {how:38} {len(chunks):4} chunks  {(ref['citation'] or ref['url'])[:70]}")
        if args.show:
            for c in chunks:
                show_paper_chunk(c)

    if args.dry_run:
        print(f"\nDry run: {len(all_chunks)} chunks; nothing embedded or written.")
        return

    collection = lane1.open_collection(create=True, name=lane1.PAPERS_COLLECTION)
    keys = sorted({c["metadata"]["paper_key"] for c in all_chunks})
    for key in keys:                       # replace, don't pile up, on re-runs
        collection.delete(where={"paper_key": key})
    for i in range(0, len(all_chunks), 100):
        batch = all_chunks[i:i + 100]
        collection.add(ids=[c["id"] for c in batch], documents=[c["document"] for c in batch],
                       metadatas=[c["metadata"] for c in batch])
    with open("paper_chunks.jsonl", "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    counts = {kind: sum(1 for _, how, _ in report if how.startswith(kind))
              for kind in ("full text", "abstract only", "citation only")}
    print(f"\nEmbedded {len(all_chunks)} chunks from {len(pubs)} publications into "
          f"'{lane1.PAPERS_COLLECTION}': " + ", ".join(f"{n} {k}" for k, n in counts.items()))


if __name__ == "__main__":
    main()
