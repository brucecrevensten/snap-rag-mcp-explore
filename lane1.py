"""
lane1.py - A minimal, local "Lane 1" retrieval index over a GeoNetwork catalog.

  harvest : page through GeoNetwork via CSW, read EVERY value in each full
            ISO 19115/19139 record, group the values into sections (overview,
            supplemental, lineage, content, spatial, distribution, usage,
            contacts, record), split those into chunks, embed, store in Chroma.
  search  : embed a question, find the nearest chunks, and return the FULL
            section each one came from, with attribution.

Chunking is "small-to-big": every chunk is sized in tokens to fit inside what
the embedding model reads (256 tokens for MiniLM), so every sentence counts
toward matching. Each chunk also carries its complete section text, and that
is what search hands back, so the reader always gets the whole description.

Setup:
    mamba env create -f environment.yml
    mamba activate lane1

Usage:
    python lane1.py harvest --csw https://YOUR_HOST/geonetwork/srv/eng/csw
    python lane1.py harvest --csw ... --limit 5 --show --dry-run   # just look
    python lane1.py search "How will snowfall change in interior Alaska?"

Everything is stored in ./lane1_index (vectors) and ./chunks.jsonl (the raw
chunks, so you can read exactly what was embedded).
"""

import argparse
import functools
import json
import re
import textwrap

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from owslib.csw import CatalogueServiceWeb
from lxml import etree
from owslib.fes import Or, PropertyIsEqualTo, SortBy, SortProperty
from tokenizers import Tokenizer

import places

ISO_GMD = "http://www.isotc211.org/2005/gmd"
INDEX_DIR = "./lane1_index"
COLLECTION = "geonetwork_metadata"
PLACES_COLLECTION = "places"   # gazetteer, kept apart so places never crowd out datasets
PAPERS_COLLECTION = "papers"   # papers datasets cite (ingest_papers.py), also kept apart
DOCS_COLLECTION = "api_docs"   # the Data API's own documentation (ingest_api_docs.py)
# Added to a paper passage's distance before it competes with catalog records.
# Papers are evidence FOR a dataset, not the dataset itself; long reports have
# so many chunks that one of them lands near almost any question. Tuned by
# comparing hit rates with ask_all.py --papers (see README).
PAPER_PENALTY = 0.1
DOC_PENALTY = 0.1      # same idea for the API's documentation pages
OVERLAP_TOKENS = 40   # tokens shared by neighbouring chunks, so a sentence cut
                      # at a chunk boundary still appears whole in one of them


# ------------------------------------------------------------------ embedder
#
# ONE embedder object does both jobs: counting tokens while we chunk, and
# turning chunks into vectors. So chunk size always matches what the model can
# actually read. To try a different model, change EMBEDDER; MAX_TOKENS and all
# the chunk sizes follow from it automatically.

EMBEDDER = ONNXMiniLM_L6_V2()       # all-MiniLM-L6-v2: small, local, CPU-friendly
MAX_TOKENS = EMBEDDER.max_tokens()  # 256 for MiniLM; it ignores everything after


@functools.cache
def tokenizer():
    """A copy of the embedder's tokenizer with the 256-token cutoff switched off,
    so we can measure long texts instead of silently losing their tail."""
    tok = Tokenizer.from_str(EMBEDDER.tokenizer.to_str())
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(text):
    return len(tokenizer().encode(text, add_special_tokens=False))


# ---------------------------------------------------------------- harvesting

def harvest_records(csw_url, page_size=50, limit=None):
    """Yield parsed ISO records (OWSLib MD_Metadata objects) from a CSW endpoint."""
    csw = CatalogueServiceWeb(csw_url, timeout=120)
    # Only ask for datasets/series. Catalogs often also hold ISO 19110 feature
    # catalogues or service records; GeoNetwork can't convert 19110 to gmd and
    # fails the WHOLE page if one lands in it.
    only_data = [Or([PropertyIsEqualTo("dc:type", "dataset"),
                     PropertyIsEqualTo("dc:type", "series")])]
    # Without an explicit sort, paging order isn't stable and the same record
    # can show up on two pages. Sort by identifier, and de-dupe just in case.
    by_id = SortBy([SortProperty("dc:identifier", "ASC")])
    start, seen = 1, set()
    while True:
        csw.getrecords2(constraints=only_data, sortby=by_id, outputschema=ISO_GMD,
                        esn="full", startposition=start, maxrecords=page_size)
        for rec in csw.records.values():
            if rec.identifier in seen:
                continue
            seen.add(rec.identifier)
            yield rec
            if limit and len(seen) >= limit:
                return
        nxt = csw.results.get("nextrecord", 0)
        if not nxt or nxt > csw.results.get("matches", 0) or not csw.records:
            return
        start = nxt


# ------------------------------------------------ reading the full ISO record
#
# Rather than picking out a few fields, we walk EVERY element of the ISO 19139
# XML and keep every value in it. ISO XML alternates two kinds of element:
#   property elements, lowerCamelCase: what a value MEANS   (abstract, keyword)
#   type elements,     UpperCamelCase: wrappers             (CI_Citation,
#                                      MD_DataIdentification, gco:CharacterString)
# So a value's list of property ancestors, e.g.
#   identificationInfo > descriptiveKeywords > keyword
# tells us both what to call it ("Keyword") and which section it belongs in.

SECTIONS = ["overview",      # title, abstract, purpose, keywords, extents, resolution
            "supplemental",  # supplementalInformation: often the real methods notes
            "lineage",       # dataQualityInfo: lineage statement, process steps, sources
            "content",       # contentInfo: variables / bands, units
            "spatial",       # grid dimensions, coordinate reference system
            "distribution",  # formats, download and service links
            "usage",         # licences and use/access constraints
            "contacts",      # people and organisations
            "record"]        # about the metadata itself: ids, dates, standard
SKIP_SECTIONS = set()        # e.g. {"record", "contacts"} to keep them out of the index

# Property names that say little on their own ("Name: GeoTIFF"), so the label
# gets the parent property in front ("Distribution format name: GeoTIFF").
GENERIC = {"name", "description", "code", "date", "title", "type", "linkage",
           "protocol", "function", "version", "role", "statement", "level",
           "distance", "value"}
# Nicer wording for a few ISO names.
LABELS = {"onLine": "online resource", "electronicMailAddress": "email",
          "supplementalInformation": "supplemental information (methods / notes)"}


def humanize(prop):
    """'westBoundLongitude' -> 'west bound longitude'."""
    return LABELS.get(prop) or re.sub(r"(?<!^)(?=[A-Z])", " ", prop).lower()


def label_for(props):
    if props == ["identificationInfo", "citation", "title"]:
        return "Title"                                    # the dataset's own title
    last = props[-1]
    if last in GENERIC:
        parents = [p for p in props[:-1]
                   if p not in ("identificationInfo", "citation", last)]
        if parents:
            return f"{humanize(parents[-1])} {humanize(last)}".capitalize()
    return humanize(last).capitalize()


def section_for(props):
    top, parts = props[0], set(props)
    if "resourceConstraints" in parts or top == "metadataConstraints":
        return "usage"
    if top == "contact" or parts & {"pointOfContact", "citedResponsibleParty",
                                    "distributorContact"}:
        return "contacts"
    if top == "identificationInfo":
        if "supplementalInformation" in parts:
            return "supplemental"
        if "graphicOverview" in parts:
            return "distribution"
        return "overview"
    return {"dataQualityInfo": "lineage",
            "contentInfo": "content",
            "spatialRepresentationInfo": "spatial",
            "referenceSystemInfo": "spatial",
            "distributionInfo": "distribution"}.get(top, "record")


def value_of(el):
    """The value an element holds, or None if it's just a wrapper."""
    if el.get("codeListValue"):                 # e.g. <gmd:CI_RoleCode codeListValue="author">
        return el.get("codeListValue")
    text = (el.text or "").strip()
    if not text or len(el):                     # empty, or has child elements
        return None
    if el.get("uom"):                           # measurements: "12" + uom "km"
        text += f" {el.get('uom')}"
    href = el.get("{http://www.w3.org/1999/xlink}href")
    if href and href != text:                   # gmx:Anchor: text plus a link
        text += f" <{href}>"
    return text


def record_to_sections(rec):
    """Turn one full ISO record into (title, {section_name: text})."""
    root = etree.fromstring(rec.xml)
    title, entries = None, {s: [] for s in SECTIONS}
    for el in root.iter():
        if not isinstance(el.tag, str):         # skip XML comments
            continue
        value = value_of(el)
        if value is None:
            continue
        chain = [el] + list(el.iterancestors())[:-1]   # up to, not incl., the root
        props = [etree.QName(e).localname for e in reversed(chain)]
        props = [p for p in props if p[0].islower()]    # property elements only
        if not props:
            continue
        label = label_for(props)
        if label == "Title" and title is None:
            title = value
        lines = entries[section_for(props)]
        # Runs of short values with the same label share one line: "Keyword: a; b; c"
        if lines and lines[-1][0] == label and "\n" not in value and len(lines[-1][1]) < 400:
            lines[-1][1] += "; " + value
        else:
            lines.append([label, value])

    sections = {s: "\n".join(f"{label}: {value}" for label, value in lines)
                for s, lines in entries.items() if lines and s not in SKIP_SECTIONS}
    return title or "(untitled)", sections


# ------------------------------------------------------------------ chunking

def split_by_tokens(text, budget, overlap=OVERLAP_TOKENS):
    """Split text into pieces of at most `budget` tokens, as the embedder counts them.

    Cuts prefer to land at the end of a line or sentence, and each piece starts
    with a little of the previous one (`overlap` tokens) so no sentence is only
    ever seen cut in half.
    """
    enc = tokenizer().encode(text, add_special_tokens=False)
    toks, offs = enc.tokens, enc.offsets   # offs[k] = (first char, last char) of token k
    if len(toks) <= budget:
        return [text]

    def sentence_end(k):
        # Token k is the last on its line (lists and labelled fields have no
        # full stops), or a '.', '?', '!' followed by a space or the end.
        # Not the '.' inside "SSP5-8.5", nor ';' (it separates keywords).
        after = text[offs[k][1]:offs[k][1] + 1]
        return after == "\n" or (toks[k] in {".", "?", "!"} and after in {"", " "})

    pieces, start = [], 0
    while True:
        end = min(start + budget, len(toks))
        if end < len(toks):
            # Walk back to the last sentence end in the second half of the window.
            for k in range(end - 1, start + budget // 2, -1):
                if sentence_end(k):
                    end = k + 1
                    break
        pieces.append(text[offs[start][0]:offs[end - 1][1]])
        if end >= len(toks):
            return pieces
        # Next piece begins `overlap` tokens back, at a sentence start if one
        # falls in that window, and never halfway through a word ("##" marks
        # a word continuation, e.g. "tasmax" -> "ta", "##sma", "##x").
        start = max(end - overlap, start + 1)
        for k in range(start, end - 1):
            if sentence_end(k):
                start = k + 1
                break
        while start < end - 1 and toks[start].startswith("##"):
            start += 1


def bbox_of(rec):
    """The record's bounding box as numbers {west, east, south, north}, or {}.

    Stored on every chunk so search can ask Chroma for "datasets whose box
    contains this point". Uses the first box if a record has several.
    """
    box = etree.fromstring(rec.xml).find(".//{*}EX_GeographicBoundingBox")
    if box is None:
        return {}
    try:
        b = {side: float(box.findtext(f"{{*}}{tag}/{{*}}Decimal"))
             for side, tag in [("west", "westBoundLongitude"), ("east", "eastBoundLongitude"),
                               ("south", "southBoundLatitude"), ("north", "northBoundLatitude")]}
    except (TypeError, ValueError):
        return {}
    if b["west"] > b["east"]:
        # Crosses the 180° line (e.g. west 172, east -130). The simple
        # west <= lon <= east test would call every point outside it.
        print(f"  WARNING: {rec.identifier} bounding box crosses 180°; "
              f"spatial filtering will treat it as covering nothing")
    return b


def build_chunks(rec, catalog_base):
    """Small-to-big: small chunks get embedded (for matching); each one also
    carries its full section text in metadata (for what search returns)."""
    uuid = rec.identifier
    title, sections = record_to_sections(rec)
    bbox = bbox_of(rec)
    landing = f"{catalog_base}/srv/api/records/{uuid}"
    chunks = []
    for section, text in sections.items():
        # Contextual header: every chunk knows which dataset it came from.
        header = f"Dataset: {title}\nSection: {section}\n\n"
        # Token budget for the text: the model's limit, minus 2 special tokens
        # it always adds ([CLS] and [SEP]), minus the header on every chunk.
        # (The max() only matters for absurdly long titles.)
        budget = max(MAX_TOKENS - 2 - count_tokens(header), 64)
        pieces = split_by_tokens(text, budget)
        for i, piece in enumerate(pieces):
            chunks.append({
                "id": f"{uuid}::{section}::{i}",
                "document": header + piece,               # small: gets embedded
                "metadata": {"uuid": uuid, "title": title,
                             "section": section, "url": landing,
                             "chunk": i, "n_chunks": len(pieces),
                             "section_text": text,        # big: gets returned
                             **bbox},                     # west/east/south/north
            })
    return chunks


# --------------------------------------------------------- inspecting chunks

def embedded_chars(text):
    """How many characters of `text` the embedding model actually reads.

    Uses the embedder's real tokenizer (cutoff switched on), so it's an
    independent check that the chunking above did its job.
    """
    # Each token remembers which characters it came from; the last one tells
    # us where the model stopped reading.
    return max(end for _, end in EMBEDDER.tokenizer.encode(text).offsets)


def show_chunk(chunk, n, total):
    """Print one chunk exactly as it will be handed to Chroma."""
    doc, meta = chunk["document"], chunk["metadata"]
    seen = embedded_chars(doc)
    used = count_tokens(doc) + 2   # + [CLS] and [SEP]
    print(f"\n  ┌─ chunk {n}/{total}   id = {chunk['id']}")
    print(f"  │  metadata: section={meta['section']}  uuid={meta['uuid']}")
    print(f"  │            url={meta['url']}")
    print(f"  │  parent:   full {meta['section']} section is "
          f"{len(meta['section_text'])} chars (returned by search)")
    if seen < len(doc.rstrip()):
        print(f"  │  size:     {used}/{MAX_TOKENS} tokens, TOO LONG: embedder reads "
              f"only the first {seen} of {len(doc)} chars")
        doc = doc[:seen] + "  ⟪── embedder stops here ──⟫  " + doc[seen:]
    else:
        print(f"  │  size:     {used}/{MAX_TOKENS} tokens, {len(doc)} chars, all embedded")
    print("  ├" + "─" * 60)
    for line in doc.splitlines():
        # Wrap long lines for the screen only; the real text isn't changed.
        for piece in textwrap.wrap(line, 90) or [""]:
            print("  │ " + piece)
    print("  └" + "─" * 60)


# ---------------------------------------------------------------- commands

def cmd_harvest(args):
    catalog_base = args.csw.split("/srv/")[0]

    n_recs, all_chunks = 0, []
    for rec in harvest_records(args.csw, limit=args.limit):
        chunks = build_chunks(rec, catalog_base)
        if not chunks:
            continue
        all_chunks.extend(chunks)
        n_recs += 1
        print(f"  [{n_recs}] {chunks[0]['metadata']['title'][:70]}  ({len(chunks)} chunks)")
        if args.show:
            for n, c in enumerate(chunks, 1):
                show_chunk(c, n, len(chunks))
            print()

    if args.dry_run:
        print(f"\nDry run: built {len(all_chunks)} chunks from {n_recs} records; "
              f"nothing embedded or written.")
        return

    collection = open_collection(create=True)
    # Drop any older chunks of these records first. If a record now splits into
    # fewer chunks than last time, its leftover old chunks would otherwise stay
    # in the index and keep turning up in searches.
    uuids = sorted({c["metadata"]["uuid"] for c in all_chunks})
    for i in range(0, len(uuids), 100):
        collection.delete(where={"uuid": {"$in": uuids[i:i + 100]}})

    # Embedding happens here: Chroma runs EMBEDDER on each chunk's document.
    for i in range(0, len(all_chunks), 100):
        batch = all_chunks[i:i + 100]
        collection.upsert(ids=[c["id"] for c in batch],
                          documents=[c["document"] for c in batch],
                          metadatas=[c["metadata"] for c in batch])

    with open("chunks.jsonl", "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"\nIndexed {len(all_chunks)} chunks from {n_recs} records into {INDEX_DIR}")


def open_collection(create=False, name=COLLECTION):
    # Pass EMBEDDER explicitly so harvest and search are guaranteed to use the
    # same model. Vectors from two different models can't be compared.
    client = chromadb.PersistentClient(path=INDEX_DIR)
    try:
        if create:
            return client.get_or_create_collection(name, embedding_function=EMBEDDER)
        return client.get_collection(name, embedding_function=EMBEDDER)
    except ValueError as e:
        if "Embedding function conflict" not in str(e):
            raise
        raise SystemExit(f"{INDEX_DIR} was built with a different embedder "
                         f"(or an older lane1.py). Delete it and harvest again:\n"
                         f"    rm -rf {INDEX_DIR}")


def linked_hits(collection_name, questions, n, penalty, point=None):
    """Search a collection of material ABOUT datasets -- the papers they cite
    (ingest_papers.py) or the Data API's documentation (ingest_api_docs.py) --
    and turn each matching passage into a hit for every dataset it links to.

    Both collections store `dataset_uuids`, so one routine serves both.
    Returns one list of (document, metadata, distance) per question, shaped
    like dataset hits: the dataset's uuid/title/url/extent, plus the passage
    and its citation. `penalty` is added to the distance, because a passage is
    evidence FOR a dataset, not the dataset's own description. Empty lists if
    nothing has been ingested into that collection yet.
    """
    try:
        collection = open_collection(name=collection_name)
    except (chromadb.errors.NotFoundError, SystemExit):
        return [[] for _ in questions]
    res = collection.query(query_texts=list(questions), n_results=min(n, collection.count()))
    # What each linked dataset looks like (title, landing URL, extent).
    linked = sorted({u for metas in res["metadatas"] for m in metas
                     for u in m["dataset_uuids"].split(",") if u})
    datasets = {}
    if linked:
        for m in open_collection().get(where={"uuid": {"$in": linked}},
                                       include=["metadatas"])["metadatas"]:
            datasets.setdefault(m["uuid"], m)
    results = []
    for docs, metas, dists in zip(res["documents"], res["metadatas"], res["distances"]):
        hits = []
        for doc, m, dist in zip(docs, metas, dists):
            for uuid in filter(None, m["dataset_uuids"].split(",")):
                d = datasets.get(uuid)
                if d is None:
                    continue                  # links to a dataset that isn't in the index
                if point and places.covers(d, *point) is not True:
                    continue                  # same rule as the dataset filter
                meta = {k: d[k] for k in ("uuid", "title", "url", "west", "east", "south", "north")
                        if k in d}
                meta.update({k: m[k] for k in ("section", "chunk", "n_chunks", "section_text",
                                               "citation", "doi", "page", "full_text")
                             if k in m})
                meta["source_distance"] = dist         # before the penalty, for display
                hits.append((doc, meta, dist + penalty))
        results.append(hits)
    return results


def search_docs(questions, k=3):
    """Documentation passages in their own right, for "what does bui mean?"
    questions. Needed because only about half the API's pages link to a
    catalog record, so the rest can never surface via a dataset.
    Returns one list of (document, metadata, distance) per question."""
    try:
        collection = open_collection(name=DOCS_COLLECTION)
    except (chromadb.errors.NotFoundError, SystemExit):
        return [[] for _ in questions]
    res = collection.query(query_texts=list(questions), n_results=min(k, collection.count()))
    return [list(zip(docs, metas, dists)) for docs, metas, dists
            in zip(res["documents"], res["metadatas"], res["distances"])]


def search_many(questions, k=5, section=None, per="section", point=None, papers=False,
                docs=False):
    """Search for several questions at once; return one result list per question.

    Each result is (document, metadata, distance), nearest first.
    Small-to-big: we match against small chunks, but several chunks can come
    from the same place, so we fetch extra and keep only the best chunk per
    group. per="section" groups by (dataset, section); per="dataset" by dataset
    alone, which is what you want for "which dataset fits this question?".
    point=(lat, lon): only datasets whose bounding box contains that point.
    papers=True: also search the papers datasets cite; a matching passage
    counts as a hit for the citing dataset(s), marked section "paper".
    docs=True: the same for the Data API's documentation pages, marked
    section "api_docs".
    """
    collection = open_collection()
    conditions = [{"section": section}] if section else []
    if point:
        # Chroma compares numbers in metadata, so "box contains point" is
        # four comparisons against the west/east/south/north stored at harvest.
        lat, lon = point
        conditions += [{"south": {"$lte": lat}}, {"north": {"$gte": lat}},
                       {"west": {"$lte": lon}}, {"east": {"$gte": lon}}]
    where = ({"$and": conditions} if len(conditions) > 1
             else conditions[0] if conditions else None)
    n = min(k * 10, collection.count())
    # Chroma embeds all the questions in one batch and searches for each.
    res = collection.query(query_texts=list(questions), n_results=n, where=where)
    # Neither is a record section, so a --section filter leaves them out.
    extra = [[] for _ in questions]
    for use, name, penalty in [(papers, PAPERS_COLLECTION, PAPER_PENALTY),
                               (docs, DOCS_COLLECTION, DOC_PENALTY)]:
        if use and not section:
            for hits, more in zip(extra, linked_hits(name, questions, n, penalty, point)):
                hits += more
    results = []
    for docs, metas, dists, more in zip(res["documents"], res["metadatas"],
                                        res["distances"], extra):
        hits = sorted(list(zip(docs, metas, dists)) + more, key=lambda h: h[2])
        best = {}
        for doc, meta, dist in hits:
            key = meta["uuid"] if per == "dataset" else (meta["uuid"], meta["section"])
            best.setdefault(key, (doc, meta, dist))   # first seen = nearest
        top = list(best.values())[:k]
        # Papers as supporting evidence: a dataset found through its own record
        # still gets the closest passage from a paper it cites, if one matched.
        closest_paper = {}
        for doc, meta, dist in more:                  # nearest first
            closest_paper.setdefault(meta["uuid"], (doc, meta))
        for _, meta, _ in top:
            if "citation" not in meta and meta["uuid"] in closest_paper:
                doc, paper = closest_paper[meta["uuid"]]
                meta["supporting_paper"] = {
                    "citation": paper["citation"], "doi": paper["doi"], "page": paper["page"],
                    "passage": doc.split("\n\n", 1)[-1],
                    "distance": round(paper["source_distance"], 3)}
        results.append(top)
    return results


def place_in(question, verbose=True):
    """The first place named in the question that has coordinates, or None.
    With verbose, also prints every place found, so you can see what happened."""
    found = places.find_places(question)
    point_place = next((p for _, p, _ in found if places.has_point(p)), None)
    if verbose:
        for text, p, n in found:
            also = f"  ({n - 1} other places share this name)" if n > 1 else ""
            note = "" if places.has_point(p) else "  (an area: no coordinates in the places list)"
            print(f"Place: \"{text}\" -> {places.label(p)}{also}{note}")
    return point_place


def coverage_mark(meta, place):
    """✓ the dataset's box contains the place, ✗ it doesn't, ? no box / no place."""
    if place is None:
        return ""
    inside = places.covers(meta, place["latitude"], place["longitude"])
    return {True: "  ✓ covers", False: "  ✗ outside", None: "  ? no extent"}[inside]


def cmd_search(args):
    place = place_in(args.question)
    point = (place["latitude"], place["longitude"]) if (args.spatial and place) else None
    if args.spatial and not place:
        print("--spatial: no place with coordinates found in the question; searching everywhere")
    elif point:
        print(f"--spatial: only datasets whose extent contains {point[0]:.3f}, {point[1]:.3f}")
    hits = search_many([args.question], k=args.k, section=args.section, point=point,
                       papers=args.papers, docs=args.docs)[0]
    for rank, (doc, meta, dist) in enumerate(hits, 1):
        print(f"\n#{rank}  distance={dist:.3f}  [{meta['section']}]  {meta['title']}"
              f"{coverage_mark(meta, place)}")
        print(f"    source: {meta['url']}")
        if "citation" in meta:        # matched a paper or a documentation page
            page = f" (page {meta['page']})" if meta.get("page") else ""
            print(f"    via {meta['section']}{page}: {meta['citation'][:160]}")
        print(f"    matched chunk {meta['chunk'] + 1} of {meta['n_chunks']}")
        # --brief: just the chunk that matched. Default: the whole section.
        text = doc.split("\n\n", 1)[-1] if args.brief else meta["section_text"]
        for line in text.splitlines():
            for piece in textwrap.wrap(line, 90) or [""]:
                print("    " + piece)

    if args.docs:      # documentation pages that explain variables, units, methods
        found = search_docs([args.question], k=3)[0]
        if found:
            print("\n--- Data API documentation")
        for doc, meta, dist in found:
            print(f"\n  distance={dist:.3f}  {meta['page_title']} > {meta['heading']}")
            print(f"    {meta['url']}")
            print(textwrap.indent(textwrap.shorten(doc.split("\n\n", 1)[-1], 300), "    "))


def cmd_harvest_places(args):
    print(f"Fetching {places.PLACES_URL} ...")
    plist = places.fetch_places()
    print(f"Saved {len(plist)} places to {places.PLACES_FILE} (used for exact name matching)")
    if args.no_embed:
        return
    # Rebuild the places collection from scratch each time; simpler than
    # working out which places changed.
    client = chromadb.PersistentClient(path=INDEX_DIR)
    if PLACES_COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(PLACES_COLLECTION)
    collection = open_collection(create=True, name=PLACES_COLLECTION)
    print(f"Embedding {len(plist)} places (a few minutes on a laptop CPU) ...")
    for i in range(0, len(plist), 1000):
        batch = plist[i:i + 1000]
        collection.add(
            ids=[p["id"] for p in batch],
            documents=[places.describe(p) for p in batch],
            # Chroma metadata can't hold None, so leave missing values out.
            metadatas=[{k: p[k] for k in ("name", "alt_name", "type", "country", "region",
                                          "latitude", "longitude") if p.get(k) is not None}
                       for p in batch])
        print(f"  {min(i + 1000, len(plist))}/{len(plist)}")
    print(f"Embedded {len(plist)} places into collection '{PLACES_COLLECTION}' in {INDEX_DIR}")


def cmd_places(args):
    """Side by side, for one question: exact name matching vs embedding search."""
    print("Exact name matching (what search uses):")
    found = places.find_places(args.question)
    for text, p, n in found:
        also = f"  (+{n - 1} other places with this name)" if n > 1 else ""
        print(f"  \"{text}\" -> {places.label(p)}{also}")
    if not found:
        print("  (no place names found)")

    print("\nEmbedding search over the places collection (for comparison):")
    try:
        collection = open_collection(name=PLACES_COLLECTION)
    except chromadb.errors.NotFoundError:
        raise SystemExit("  no places collection yet: run  python lane1.py harvest-places")
    res = collection.query(query_texts=[args.question], n_results=args.k)
    for doc, dist in zip(res["documents"][0], res["distances"][0]):
        print(f"  {dist:.3f}  {doc}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--csw", required=True, help="e.g. https://host/geonetwork/srv/eng/csw")
    h.add_argument("--limit", type=int, help="stop after N records (handy for testing)")
    h.add_argument("--show", action="store_true",
                   help="print every chunk in full, exactly as it goes into Chroma")
    h.add_argument("--dry-run", action="store_true",
                   help="fetch and chunk only; don't embed or write anything")
    s = sub.add_parser("search")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--section", choices=SECTIONS)
    s.add_argument("--brief", action="store_true",
                   help="print only the chunk that matched, not its whole section")
    s.add_argument("--spatial", action="store_true",
                   help="only datasets whose extent contains the place named in the question")
    s.add_argument("--papers", action="store_true",
                   help="also match the papers datasets cite (after ingest_papers.py)")
    s.add_argument("--docs", action="store_true",
                   help="also match the Data API's documentation (after ingest_api_docs.py)")
    hp = sub.add_parser("harvest-places",
                        help="download the place-name gazetteer and embed it")
    hp.add_argument("--no-embed", action="store_true",
                    help="only download places.json (seconds); skip embedding (minutes)")
    pl = sub.add_parser("places", help="find places in a question: exact names vs embeddings")
    pl.add_argument("question")
    pl.add_argument("-k", type=int, default=5)
    args = p.parse_args()
    {"harvest": cmd_harvest, "search": cmd_search,
     "harvest-places": cmd_harvest_places, "places": cmd_places}[args.cmd](args)


if __name__ == "__main__":
    main()
