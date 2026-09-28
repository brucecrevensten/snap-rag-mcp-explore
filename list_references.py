"""
list_references.py - Every academic reference that each GeoNetwork dataset cites.

References hide in three places, so we look in all three:
  1. Online-resource links in the record (usually in distribution info),
     labelled "Publication", "Suggested citation", "Source dataset", ...
  2. DOIs written into free text (supplemental information, abstract, ...).
  3. The Data API's documentation pages (earthmaps.io/<service>/), which list
     academic references next to the catalog records that service serves.

Each DOI is then looked up at doi.org, which tells us what it is (journal
article, report, dataset, ...) and gives a formatted (APA) citation.

Writes references.json (read by ingest_papers.py) and prints a listing.

Usage:
    python list_references.py --csw https://catalog.snap.uaf.edu/geonetwork/srv/eng/csw
    python list_references.py --csw ... --limit 10     # quick look
"""

import argparse
import datetime
import html
import json
import re
import urllib.request

from lxml import etree

import lane1

API = "https://earthmaps.io"
CATALOG_RECORD = "https://catalog.snap.uaf.edu/geonetwork/srv/eng/catalog.search#/metadata/{uuid}"
OUT_FILE = "references.json"
USER_AGENT = "snap-rag-mcp-explore/0.1 (+https://github.com/brucecrevensten/snap-rag-mcp-explore)"

DOI = re.compile(r"10\.\d{4,9}/[^\s\"'<>;,)\]]+", re.I)
# A link counts as a reference if its label says so...
REFERENCE_WORDS = re.compile(r"publication|citation|cite|paper|article|journal|report|reference"
                             r"|doi|source data|parent data", re.I)
# ...or it points somewhere papers live.
PAPER_LINK = re.compile(r"doi\.org/|\.pdf($|\?)|journals\.|/article|sciencedirect|springer"
                        r"|wiley|agupubs|mdpi\.com|tandfonline|nature\.com|espis\.boem", re.I)
# Types doi.org reports that are publications rather than datasets. It passes
# through each registry's own vocabulary: Crossref ("journal-article") and
# CSL ("article-journal") names both appear.
PUBLICATION_TYPES = {"journal-article", "article-journal", "article", "proceedings-article",
                     "paper-conference", "book-chapter", "chapter", "book", "monograph",
                     "report", "report-component", "dissertation", "thesis",
                     "posted-content", "review", "entry"}


# ---------------------------------------------------------------- helpers

def http_get(url, accept=None):
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def clean_doi(doi):
    """'10.1002/hyp.9934/abstract.' -> '10.1002/hyp.9934' (DOIs are case-insensitive)."""
    doi = doi.rstrip(".")
    doi = re.sub(r"/(abstract|full|pdf|epdf)$", "", doi, flags=re.I)
    return doi.lower()


def text_of(el):
    """Text of an ISO element: its CharacterString / Anchor child, or itself."""
    if el is None:
        return ""
    return " ".join(t.strip() for t in el.itertext() if t.strip())


# ------------------------------------------------ 1 + 2: the catalog records

def references_in_record(rec):
    """[(key, found_in, label, url)] for one ISO record."""
    found = []
    root = etree.fromstring(rec.xml)

    # 1. Labelled online-resource links.
    for res in root.iter("{*}CI_OnlineResource"):
        url = text_of(res.find("{*}linkage"))
        label = " ".join(filter(None, [text_of(res.find("{*}name")),
                                       text_of(res.find("{*}description"))]))
        if not url or not (PAPER_LINK.search(url) or REFERENCE_WORDS.search(label)):
            continue
        if "data.snap.uaf.edu" in url and not url.lower().endswith(".pdf"):
            continue                       # a download folder, even if labelled "source data"
        doi = DOI.search(url)
        key = f"doi:{clean_doi(doi.group(0))}" if doi else f"url:{url}"
        found.append((key, "record link", label[:300], url))

    # 2. DOIs written anywhere in the text (supplemental information, abstract...).
    _, sections = lane1.record_to_sections(rec)
    for section, text in sections.items():
        for line in text.splitlines():
            for doi in DOI.findall(line):
                found.append((f"doi:{clean_doi(doi)}", f"record text ({section})",
                              line[:300], f"https://doi.org/{clean_doi(doi)}"))
    return found


# ------------------------------------------------ 3: Data API documentation

def references_on_api_pages():
    """{uuid: [(key, found_in, label, url)]} from earthmaps.io documentation pages."""
    home = http_get(API + "/")
    services = sorted(set(re.findall(r'href="/([a-z0-9_]+)"', home)) - {"static"})
    by_uuid = {}
    for service in services:
        try:
            page = http_get(f"{API}/{service}/")
        except Exception:
            continue
        uuids = set(re.findall(r"catalog\.search#/metadata/([0-9a-f-]{36})", page))
        refs = []
        for url in dict.fromkeys(re.findall(r'href="(https?://[^"]+)"', page)):
            doi = DOI.search(url)
            if doi and "zenodo.1163021" not in url:     # skip the rasdaman software DOI
                refs.append((f"doi:{clean_doi(doi.group(0))}", url))
            elif url.lower().endswith(".pdf"):
                refs.append((f"url:{url}", url))
        for uuid in uuids:
            for key, url in refs:
                by_uuid.setdefault(uuid, []).append(
                    (key, f"{API}/{service}/ documentation", "", url))
    return by_uuid


# --------------------------------------------------- what is each reference?

def describe_doi(doi):
    """Type, title, year and APA citation for a DOI, from doi.org."""
    info = {"doi": doi, "url": f"https://doi.org/{doi}"}
    try:
        csl = json.loads(http_get(info["url"], accept="application/vnd.citationstyles.csl+json"))
        info["type"] = csl.get("type", "")
        info["title"] = csl.get("title", "")
        info["year"] = ((csl.get("issued") or {}).get("date-parts") or [[None]])[0][0]
        apa = http_get(info["url"], accept="text/x-bibliography; style=apa")
        # DataCite returns HTML-flavoured text ("&amp;", "<i>Title</i>"): make it plain.
        info["citation"] = html.unescape(re.sub(r"<[^>]+>", "", apa)).strip()
    except Exception as e:
        info.update(type="", citation="", error=f"doi.org lookup failed: {e}")
    info["kind"] = ("dataset" if info.get("type") == "dataset"
                    else "publication" if info.get("type") in PUBLICATION_TYPES else "other")
    return info


def describe_url(url, label):
    return {"doi": "", "url": url, "type": "", "title": label, "year": None,
            "citation": label or url,
            "kind": "publication" if PAPER_LINK.search(url) else "other"}


# -------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csw", required=True, help="e.g. https://host/geonetwork/srv/eng/csw")
    p.add_argument("--limit", type=int, help="stop after N records (handy for testing)")
    args = p.parse_args()

    print("Reading the Data API documentation pages ...")
    api_refs = references_on_api_pages()

    print("Reading catalog records ...")
    datasets, cited = [], {}   # cited: {reference key: [citing dataset + where found]}
    for n, rec in enumerate(lane1.harvest_records(args.csw, limit=args.limit), 1):
        title, _ = lane1.record_to_sections(rec)
        found = references_in_record(rec) + api_refs.get(rec.identifier, [])
        keys = list(dict.fromkeys(key for key, *_ in found))
        datasets.append({"uuid": rec.identifier, "title": title,
                         "catalog_record": CATALOG_RECORD.format(uuid=rec.identifier),
                         "references": keys})
        for key, where, label, url in found:
            cited.setdefault(key, {"label": label, "url": url, "cited_by": []})
            entry = {"uuid": rec.identifier, "title": title, "found_in": where}
            if entry not in cited[key]["cited_by"]:
                cited[key]["cited_by"].append(entry)
            if label and not cited[key]["label"]:
                cited[key]["label"] = label
        print(f"  [{n}] {title[:70]}  ({len(keys)} references)")

    print(f"\nLooking up {len(cited)} references at doi.org ...")
    references = []
    for key, c in cited.items():
        info = describe_doi(key[4:]) if key.startswith("doi:") else describe_url(c["url"], c["label"])
        references.append({"key": key, **info, "label": c["label"], "cited_by": c["cited_by"]})

    with open(OUT_FILE, "w") as f:
        json.dump({"generated": datetime.date.today().isoformat(), "catalog": args.csw,
                   "datasets": datasets, "references": references}, f, indent=1, ensure_ascii=False)

    # ---- listing: per dataset, its references
    by_key = {r["key"]: r for r in references}
    print()
    for d in datasets:
        if not d["references"]:
            continue
        print(d["title"])
        print(f"  {d['catalog_record']}")
        for key in d["references"]:
            r = by_key[key]
            where = "; ".join(sorted({c["found_in"] for c in r["cited_by"] if c["uuid"] == d["uuid"]}))
            print(f"  - [{r['kind']}] {r['citation'] or r['url']}")
            print(f"      found in: {where}")
        print()
    kinds = {}
    for r in references:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    with_refs = sum(1 for d in datasets if d["references"])
    print(f"{with_refs} of {len(datasets)} datasets cite something; {len(references)} unique "
          f"references: {kinds}. Written to {OUT_FILE}")


if __name__ == "__main__":
    main()
