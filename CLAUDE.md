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
- `list_references.py`: every reference each dataset cites, from record
  online-resource links (labelled Publication / Suggested citation / Source
  dataset...), DOIs in record text, and the earthmaps.io doc pages; typed and
  APA-formatted via doi.org -> references.json. 2026-09-18: 31/62 datasets
  cite something; 37 refs (29 publications, 5 datasets, 3 other).
- `ingest_api_docs.py`: the 25 earthmaps.io service pages -> heading-based
  sections (tables become "bui | Numeric rating ..." lines) -> "api_docs"
  collection, 295 chunks; only ~12 pages link catalog records, so
  `lane1.search_docs()` returns doc passages on their own (CLI `--docs`, MCP
  search_datasets `documentation`), while linked ones also count for their
  datasets after `DOC_PENALTY`. `linked_hits()` serves both papers and docs.
- `ingest_papers.py`: publications -> OpenAlex -> legal OA PDF (pypdf, reference
  list cut) or abstract or citation -> token chunks -> "papers" collection,
  metadata links back to citing datasets (dataset_uuids). 2026-09-18: 6 full
  text (mostly reports), 17 abstract, 6 citation only; many publishers refuse
  scripted PDF downloads (we don't work around that). `search --papers` /
  `ask_all.py --papers`: paper matches count for citing datasets after
  `PAPER_PENALTY` (0.1); unpenalized, two long BOEM reports (~1,750 of ~2,100
  chunks) swamped results (hit@3 3/6 -> 2/6 on a guessed test key). Datasets
  also get their closest `supporting_paper` passage (MCP search_datasets too).
- `mcp_server.py` (MCP SDK 2.x `MCPServer`; registered in `.mcp.json`):
  tools find_place, search_datasets, get_dataset, get_climate_data (curated
  earthmaps.io topics; yearly series -> decade means, day-of-year -> monthly
  means, active-fire GeoJSON -> counts + key fields, to fit agent context).
  Topics now include fire_weather (CMIP6 FWI; needs start_year/end_year, takes
  variables=bui,dc,dmc,ffmc,fwi,isi and operation=summer_fire_danger_rating_days
  (default) | {3,5,7}_day_rolling_average; area queries are HUC-only, boroughs
  and GMUs 404), wildfire_flammability + vegetation_type (ALFRESCO, point query
  returns the intersecting HUC-12) and wildfire_now (/fire/point, near-real-time
  danger, AQI, active fires -- not projections). Rolling averages are ~20s and
  300KB raw; monthly means bring them to ~1.5KB. Small values (flammability
  ~0.004) are rounded to 4 significant digits, not 2 decimals.
  All 24 Alaska-relevant API services are now topics (25 topics; conus_hydrology
  left out as non-Alaska). Options: variables / scenario / start_year+end_year /
  operation / stream_id, validated per topic. Sizes handled by compaction
  (monthly->annual over 24 keys, daily->monthly over 60, yearly->decades over 40)
  plus client-side filtering where the endpoint has no parameter (hydrology vars,
  cmip6 scenario, era5wrf + cmip6_downscaled year windows). Gotchas found:
  cmip6_downscaled needs models=7ModelAvg (docs say 6ModelAvg -> 422) or it is
  4.5MB/271s; sea ice and landfast ice must use a coastal community's
  ocean_lat1/lon1 (own coords are land cells) and inland places carry a useless
  far-away ocean point, so gate on is_coastal; arctic_hydrology is keyed by
  MERIT stream segment and the API has no lookup yet (docs say TBD).
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
