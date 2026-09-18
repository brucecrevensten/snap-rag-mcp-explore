# snap-rag-mcp-explore

> **Heads up: this is an AI-generated fever dream + demo.** It was built in a
> single exploratory afternoon, pair-programming with Claude (Claude Code),
> to learn how to make climate data usable by AI agents. Expect sharp edges.
> It is a personal experiment, not a supported product, and nothing here has
> been reviewed for production use.

The question it explores: **how can an AI agent answer questions about climate
change in Alaska accurately, and show where every number came from?**

It uses the catalog and Data API of the
[Scenarios Network for Alaska + Arctic Planning (SNAP)](https://www.snap.uaf.edu/)
at the University of Alaska Fairbanks, in two "lanes":

- **Lane 1: discovery and context (RAG).** Harvest every ISO 19115 record from
  SNAP's [GeoNetwork catalog](https://catalog.snap.uaf.edu/geonetwork), chunk it,
  embed it locally, and search it: *which dataset fits this question, and how
  was it made?*
- **Lane 2: exact numbers (tools).** An [MCP](https://modelcontextprotocol.io)
  server that lets Claude (or any MCP client) look up Alaska places and query the
  live [Alaska + Arctic Geospatial Data API](https://earthmaps.io), with an
  attribution block on every result.

## What's in here

| File | What it does |
|---|---|
| `lane1.py` | Harvests the catalog over CSW, reads every value in each full ISO record, splits it into token-sized chunks, embeds them with a local MiniLM model into Chroma, and searches them. Also downloads the place-name gazetteer. |
| `places.py` | Alaska-only gazetteer from `earthmaps.io/places/all`: finds place names in a question (including Indigenous and former names, e.g. *Mamterilleq* = Bethel) and checks whether a dataset's extent covers a place. |
| `ask_all.py` | Runs every question in `questions.txt` against the index and prints a table; if you add expected answers, it scores hit@1 / hit@k. |
| `mcp_server.py` | The MCP server: `find_place`, `search_datasets`, `get_dataset`, `get_climate_data`. |
| `questions.txt` | Example questions to test retrieval with. |
| `docs/` | A write-up of metadata improvements that would help retrieval (in Comic Sans, by request). |
| `CLAUDE.md` | Running project notes, also read by Claude Code. |

## Setup

Needs [mamba / micromamba](https://mamba.readthedocs.io) (or conda). Everything
comes from conda-forge.

```bash
git clone https://github.com/brucecrevensten/snap-rag-mcp-explore.git
cd snap-rag-mcp-explore
mamba env create -f environment.yml
mamba activate lane1
```

The first run downloads the MiniLM embedding model (~80 MB) to `~/.cache/chroma`.

## Lane 1: build and search the index

Harvest SNAP's catalog (about 60 dataset records; takes a minute or two). This
writes `./lane1_index` (vectors) and `./chunks.jsonl` (exactly what was
embedded, for reading):

```bash
python lane1.py harvest --csw https://catalog.snap.uaf.edu/geonetwork/srv/eng/csw
```

Useful flags: `--limit 5` for a quick test, `--show` to print every chunk as it
is embedded (with token counts), `--dry-run` to build chunks without writing
anything.

Search it:

```bash
python lane1.py search "How is winter changing in Interior Alaska?" -k 3 --brief
```

Download the Alaska place names (writes `./places.json`; `--no-embed` skips
embedding them, which the MCP server doesn't need):

```bash
python lane1.py harvest-places
```

Then place-aware search: `--spatial` keeps only datasets whose extent contains
the place named in the question:

```bash
python lane1.py search "How much warmer is it going to get in Two Rivers?" -k 3 --spatial --brief
```

Compare exact name matching with embedding search for finding places:

```bash
python lane1.py places "Snowfall trends near Bethel"
```

### Testing retrieval with a question set

```bash
python ask_all.py
```

To score it, add ` | ` and part of the expected dataset's title to a line in
`questions.txt` (separate alternatives with `;`):

```
When will average temperatures extend the growing season for potatoes? | DOF/DOT/LOGS
```

`python ask_all.py --list-datasets` prints every title in the index to copy from.
`--spatial` and `--csv results.csv` also work.

## Lane 2: the MCP server

| Tool | Does |
|---|---|
| `find_place` | Alaska communities (points) and areas (polygons: boroughs, watersheds, game management units, protected areas, ethnolinguistic regions...), including Indigenous and former names. |
| `search_datasets` | Lane 1 search, optionally keeping only datasets that cover a place. |
| `get_dataset` | A dataset's full catalog record: methods, limitations, license, DOIs. |
| `get_climate_data` | Live values from earthmaps.io for an Alaska place: temperature and precipitation, climate indicators, heating degree days, freezing and thawing indices, permafrost, snowfall, wet days. Yearly series are averaged by decade to fit an agent's context. |

Every result includes an **attribution** block: the exact query URL, the catalog
records and reference DOIs listed on the Data API's own documentation pages,
the provider credit, and the license. The server's instructions tell the agent
to take numbers only from `get_climate_data` and to end each answer with a
Sources section. Places are restricted to Alaska.

`find_place` and `get_climate_data` work straight away (`places.json` is
downloaded on first use); `search_datasets` and `get_dataset` need the Lane 1
index from `harvest`.

### Connect it to Claude Code

The server is launched by the client, so it needs absolute paths. Find the
environment's Python:

```bash
mamba run -n lane1 which python
```

Copy `.mcp.json.example` to `.mcp.json` and fill in that Python path and the
full path to `mcp_server.py`. Start Claude Code in this folder (`claude`, or a
new Code session on this folder in the Claude desktop app) and approve the
`alaska-climate` server when asked.

### Connect it to the Claude desktop app (Chat)

Add the same entry under `"mcpServers"` in
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS),
keeping any settings already in the file, then fully quit (⌘Q) and reopen the app:

```json
"mcpServers": {
  "alaska-climate": {
    "command": "/ABSOLUTE/PATH/TO/envs/lane1/bin/python",
    "args": ["/ABSOLUTE/PATH/TO/snap-rag-mcp-explore/mcp_server.py"]
  }
}
```

Then ask things like:

- *How much warmer will winters get in Two Rivers by late century?*
- *Is permafrost near Bethel projected to thaw? Explain the model.*
- *What do climate indicators say about summer days in the Denali Borough?*

## Things we learned along the way

- **Read the whole metadata record.** SNAP records keep their real methods
  notes in `supplementalInformation`; only 14 of 62 had a lineage statement.
- **Size chunks in tokens, not characters.** MiniLM reads 256 tokens and
  silently ignores the rest; technical text (`MPI-ESM1-2-HR`, `SSP5-8.5`) uses
  tokens fast. Here chunks are split with the embedder's own tokenizer, and
  search returns the whole section a chunk came from ("small-to-big").
- **Embeddings match meaning, not names.** "Snowfall trends near Bethel" found
  *Snow Lake, Manitoba*. Place names need exact string matching; spatial
  questions need a coverage check, not better embeddings.
- **CSW quirks:** filter to datasets/series (one ISO 19110 record can fail a
  whole page), and sort results, or paging returns duplicates.
- **Keep tool results small.** Some Data API responses are megabytes; decade
  means keep an answer's evidence readable.

## License and credits

Code: [MIT](LICENSE).

Data come from SNAP's catalog and Data API, whose data are available under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless a source
provider specifies otherwise. Credit: Scenarios Network for Alaska + Arctic Planning,
International Arctic Research Center, University of Alaska Fairbanks. Cite the
individual datasets listed in each tool result.
