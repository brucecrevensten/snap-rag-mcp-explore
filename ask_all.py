"""
ask_all.py - Run every question in a text file against the Lane 1 index and
print a table: each question, then the datasets it retrieved and their distances.
If you say which dataset SHOULD come back, it also scores the retrieval.

Usage:
    python ask_all.py                      # reads questions.txt, top 3 per question
    python ask_all.py my_questions.txt -k 5
    python ask_all.py --csv results.csv    # also save the table for a spreadsheet
    python ask_all.py --list-datasets      # every dataset title in the index
    python ask_all.py --papers             # also match papers the datasets cite

Question file: one question per line; blank lines and lines starting with #
are skipped. To score a question, add " | " and the expected dataset:

    How is winter changing in Interior Alaska? | degree day totals
    When will the growing season get longer? | DOF/DOT/LOGS; growing season

The expected part is a piece of the dataset title (case-insensitive) or its
exact uuid. Separate alternatives with ";" -- any of them counts as a hit.
Questions without " | " are still searched, just not scored.

Scoring: hit@1 = the expected dataset came back first; hit@k = it came back
in the top k. For misses we look deeper (top 10) to report how far down it was.

Uses the same search as `lane1.py search` (lane1.search_many), grouped per
dataset: each dataset appears at most once per question, with its best chunk.
Distance: lower = closer match. Compare distances within one question
rather than across questions.
"""

import argparse
import csv
import textwrap

import lane1
import places

LOOK_DEEPER = 10   # how far down the ranking we look for a missed expected dataset


def read_questions(path):
    """Return a list of (question, [expected, ...]); the list is empty if unscored."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            question, _, expected = line.partition("|")
            alternatives = [e.strip() for e in expected.split(";") if e.strip()]
            items.append((question.strip(), alternatives))
    return items


def all_datasets():
    """{uuid: title} for every dataset in the index."""
    metas = lane1.open_collection().get(include=["metadatas"])["metadatas"]
    return {m["uuid"]: m["title"] for m in metas}


def is_expected(meta, alternatives):
    return any(e == meta["uuid"] or e.lower() in meta["title"].lower()
               for e in alternatives)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("questions", nargs="?", default="questions.txt")
    p.add_argument("-k", type=int, default=3, help="datasets per question (default 3)")
    p.add_argument("--section", choices=lane1.SECTIONS,
                   help="only match chunks from this section")
    p.add_argument("--width", type=int, default=60, help="max width of dataset names")
    p.add_argument("--csv", metavar="FILE", help="also write the rows to a CSV file")
    p.add_argument("--list-datasets", action="store_true",
                   help="print every dataset title in the index, then stop")
    p.add_argument("--spatial", action="store_true",
                   help="for questions naming a place, only datasets whose extent contains it")
    p.add_argument("--papers", action="store_true",
                   help="also match the papers datasets cite (after ingest_papers.py)")
    args = p.parse_args()

    datasets = all_datasets()
    if args.list_datasets:
        for uuid, title in sorted(datasets.items(), key=lambda kv: kv[1].lower()):
            print(f"{uuid}  {title}")
        print(f"\n{len(datasets)} datasets")
        return

    items = read_questions(args.questions)

    # Catch typos: an expected value that matches nothing can never be a hit.
    for qn, (question, alternatives) in enumerate(items, 1):
        for e in alternatives:
            if not any(is_expected({"uuid": u, "title": t}, [e]) for u, t in datasets.items()):
                print(f"WARNING: Q{qn} expects '{e}', which matches no dataset in the index")

    depth = max(args.k, LOOK_DEEPER)
    questions = [q for q, _ in items]
    # Places named in each question (exact name matching, see places.py), and
    # the first one with coordinates, which is what coverage is checked against.
    found = [places.find_places(q) for q in questions]
    place_for = [lane1.place_in(q, verbose=False) for q in questions]
    points = [(p["latitude"], p["longitude"]) if (args.spatial and p) else None
              for p in place_for]
    if any(points):
        # A different spatial filter per question, so search them one at a time.
        results = [lane1.search_many([q], k=depth, section=args.section,
                                     per="dataset", point=pt, papers=args.papers)[0]
                   for q, pt in zip(questions, points)]
    else:
        results = lane1.search_many(questions, k=depth, section=args.section, per="dataset",
                                    papers=args.papers)

    # One row per (question, retrieved dataset), top k only.
    rows, scores = [], []   # scores: rank where the expected dataset was found, or None
    for qn, ((question, alternatives), hits) in enumerate(zip(items, results), 1):
        place = place_for[qn - 1]
        for rank, (doc, meta, dist) in enumerate(hits[:args.k], 1):
            rows.append({"q": qn, "question": question, "rank": rank,
                         "dataset": meta["title"], "section": meta["section"],
                         "distance": round(dist, 3), "uuid": meta["uuid"],
                         "expected": "; ".join(alternatives),
                         "is_expected": is_expected(meta, alternatives),
                         "place": places.label(place) if place else "",
                         "covers_place": lane1.coverage_mark(meta, place).strip(),
                         "via_paper": meta.get("citation", "")})
        if alternatives:
            ranks = [rank for rank, (_, meta, _) in enumerate(hits, 1)
                     if is_expected(meta, alternatives)]
            scores.append(ranks[0] if ranks else None)

    # ---- print the table: a full-width line per question, then its datasets
    name_w = min(args.width, max((len(r["dataset"]) for r in rows), default=10))
    line = f"{'':4}{'#':>2}  {'dataset':<{name_w}}  {'section':<8}  {'distance':>8}"
    print(line)
    print("─" * len(line))
    score_iter = iter(scores)
    for qn, (question, alternatives) in enumerate(items, 1):
        print(f"Q{qn:<3}{question}")
        for text, p, _ in found[qn - 1]:
            note = "" if places.has_point(p) else "  (area: no coordinates, no coverage check)"
            print(f"{'':8}place: \"{text}\" -> {places.label(p)}{note}")
        if points[qn - 1]:
            print(f"{'':8}(--spatial: only datasets whose extent contains that point)")
        for r in (r for r in rows if r["q"] == qn):
            name = textwrap.shorten(r["dataset"], name_w, placeholder="…")
            mark = f"  {r['covers_place']}" if r["covers_place"] else ""
            mark += "  ✓ expected" if r["is_expected"] else ""
            print(f"{'':4}{r['rank']:>2}  {name:<{name_w}}  {r['section']:<8}  "
                  f"{r['distance']:>8.3f}{mark}")
            if r["via_paper"]:
                print(f"{'':8}via paper: {textwrap.shorten(r['via_paper'], name_w + 20)}")
        if alternatives:
            rank = next(score_iter)
            if rank is None:
                verdict = f"MISS, not in top {depth}"
            elif rank <= args.k:
                verdict = f"hit at rank {rank}"
            else:
                verdict = f"MISS, found lower down at rank {rank}"
            print(f"{'':8}expected: {'; '.join(alternatives)}  →  {verdict}")
        print()

    # ---- summary
    if scores:
        n = len(scores)
        hit1 = sum(1 for s in scores if s == 1)
        hitk = sum(1 for s in scores if s is not None and s <= args.k)
        print(f"Scored {n} of {len(items)} questions:  "
              f"hit@1 {hit1}/{n} ({100 * hit1 // n}%)   "
              f"hit@{args.k} {hitk}/{n} ({100 * hitk // n}%)")
    else:
        print("No questions have an expected dataset yet (add ' | title' to score).")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
