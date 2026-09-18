# Project: AI-ready climate data (Lane 1 retrieval prototype)

## Goal
Make our climate datacubes usable by AI agents so they can produce accurate,
well-attributed visualizations. Our stack: multidimensional raster datacubes
(dims: time, space, climate scenario, variable) served via Rasdaman and
GeoServer; metadata in GeoNetwork (ISO 19115); a custom Flask Data API that
shapes responses for web apps.

## Architecture we settled on (two lanes)
- **Lane 1 - discovery/context (RAG):** harvest GeoNetwork records + methods
  docs, glossaries, known caveats -> chunk -> embed -> vector store. Answers
  "which dataset fits, what do scenarios/baselines/methods mean?" Every chunk
  carries uuid, title, section, and landing URL for attribution.
- **Lane 2 - data access (tools):** an MCP server wrapping the Flask Data API
  so agents get exact numbers, never "retrieved" values. Responses will carry a
  provenance block (dataset_id, version, DOI, citation, license, metadata_url,
  source_query, units, baseline_period, caveats). Prototype: `mcp_server.py`
  (see Current state); talks to the live Data API at https://earthmaps.io.

## Current state
We are prototyping Lane 1 locally with open-source tools only.
- `lane1.py`: `harvest` pages GeoNetwork via CSW (OWSLib, ISO gmd schema),
  reads EVERY value in each full ISO 19139 record (generic XML walk, labels
  from ISO property names) into sections: overview, supplemental, lineage,
  content, spatial, distribution, usage, contacts, record (`SKIP_SECTIONS`
  to drop some). SNAP records store their real methods notes in
  `supplementalInformation`; lineage is mostly empty or boilerplate. Then
  "small-to-big" chunking: each section is split by TOKEN count (the embedder's
  own tokenizer) so every chunk fits MiniLM's 256-token window, header
  ("Dataset: X / Section: Y") included; each chunk carries the full
  section text in metadata. Embeds with MiniLM (`EMBEDDER`, passed explicitly),
  stores in `./lane1_index`, writes raw chunks to `chunks.jsonl`.
  `harvest --show` prints each chunk + token usage; `--dry-run` writes nothing.
  `search` matches chunks, returns the full section (best chunk per
  dataset/section); `--section` filter, `--brief` for matched chunk only.
  Each chunk also stores the record's bbox (west/east/south/north) so search
  can filter by "extent contains point" (`search --spatial`).
- `places.py`: Alaska-only gazetteer from https://earthmaps.io/places/all
  (Alaska communities + Alaskan area types; `keep()` / `CANADIAN_AREA_TYPES`).
  Places are found in questions by EXACT name matching incl. Indigenous /
  former names -- tested: embeddings match meaning, not names ("Snowfall near
  Bethel" -> Snow Lake, MB). `lane1.py harvest-places` downloads + embeds them
  anyway; `lane1.py places "..."` compares the two.
- `ask_all.py`: runs questions.txt ("question | expected title part"),
  prints table + hit@1/hit@k, place detection, coverage marks, `--spatial`.
- `mcp_server.py` (MCP SDK 2.x `MCPServer`; registered in `.mcp.json`):
  tools find_place, search_datasets, get_dataset, get_climate_data (curated
  earthmaps.io topics, yearly series -> decade means to fit agent context).
  Alaska places/polygons only. Every result has an attribution block (query
  URL, catalog records + reference DOIs scraped from the API doc pages,
  provider, license); server instructions require a Sources section.
- Repo: github.com/brucecrevensten/snap-rag-mcp-explore (public, MIT).
  `scratch/` is gitignored scratch space for test runs and outputs; `docs/`
  holds the metadata-recommendations write-up. Generated files (lane1_index/,
  chunks.jsonl, places.json) and machine-specific config (.mcp.json,
  .claude/settings.local.json) are gitignored; `.mcp.json.example` is the
  portable template.
- Setup: `mamba env create -f environment.yml && mamba activate lane1`
  (micromamba, env lives in ~/mamba/envs/lane1; all packages from conda-forge)

## Next steps to explore
1. Re-harvest the catalog so chunks carry bboxes (needed for coverage checks).
2. Grow questions.txt with expected answers per audience; measure hit rate;
   then run the same questions through Claude with the MCP server connected.
3. Add hybrid search (BM25 keyword + vector) for exact terms like variable
   codes (`tas`, `pr`), SSP names, and GCM names.
4. Ingest extra documents: methods notes, variable glossary, known pitfalls.
5. Consider a better embedding model (e.g. nomic-embed-text via Ollama) and
   pgvector for production since we already run PostgreSQL.

## Working preferences
- Keep things plain Python and readable; I'm learning the moving parts, so
  explain what new code does and why. Avoid heavy frameworks for now.
