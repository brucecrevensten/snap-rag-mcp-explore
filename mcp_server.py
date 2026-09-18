"""
mcp_server.py - Alaska climate knowledge for Claude (or any MCP client).

Four tools, all restricted to Alaska places and polygons:
  find_place        Alaska place names -> id, type, point or polygon extent
  search_datasets   Lane 1: which SNAP datasets fit a question, optionally at a place
  get_dataset       Lane 1: one dataset's full catalog record (methods, license, DOIs)
  get_climate_data  Lane 2: live numbers from https://earthmaps.io for an Alaska place

Every result carries an "attribution" block, and the server instructions tell the
agent to finish each answer with a Sources section built from them.

Claude Code starts this automatically via .mcp.json. To poke at it by hand:
    mamba activate lane1
    python mcp_server.py        # speaks MCP on stdin/stdout, so it just waits
"""

import functools
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Literal

# Clients start servers from anywhere; ./lane1_index and ./places.json live here.
os.chdir(Path(__file__).resolve().parent)

from mcp.server.mcpserver import MCPServer  # noqa: E402

import lane1  # noqa: E402
import places  # noqa: E402

API = "https://earthmaps.io"
CATALOG_RECORD = "https://catalog.snap.uaf.edu/geonetwork/srv/eng/catalog.search#/metadata/{uuid}"
PROVIDER = ("Scenarios Network for Alaska + Arctic Planning (SNAP), International Arctic "
            "Research Center, University of Alaska Fairbanks")
LICENSE = ("CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), unless otherwise "
           "specified by source providers")
MAX_RESULT_CHARS = 20_000   # keep a single tool result from flooding the agent's context


# ------------------------------------------------------------------ topics
#
# Curated Data API endpoints. Each is a question people ask, a URL template,
# and the API documentation page that lists its source datasets (we read
# attribution from that page, so it stays current).

TOPICS = {
    "temperature_precipitation": {
        "about": "Seasonal mean air temperature and precipitation at 2 km. Historical: "
                 "CRU TS 4.0. Projected: CMIP5 MRI-CGCM3, NCAR-CCSM4 and a 5-model average "
                 "under RCP 4.5, 6.0 and 8.5. Grouped by era, then season (DJF, MAM, JJA, SON), "
                 "model and scenario.",
        "units": "tas: °C; pr: mm",
        "point": "/taspr/point/{lat}/{lon}", "area": "/taspr/area/{id}", "doc": "/taspr/"},
    "climate_indicators": {
        "about": "Climate indicators from NCAR 12 km downscaled data (hot/cold day "
                 "thresholds, summer days, deep-winter days, heavy-precipitation days, "
                 "dry/wet spells...). Min/mean/max over historical 1980-2009 (Daymet), "
                 "mid-century 2040-2069 and late-century 2070-2099 (MRI-CGCM3, NCAR-CCSM4; "
                 "RCP 4.5 and 8.5).",
        "units": "see definitions",
        "point": "/indicators/cmip5/point/{lat}/{lon}", "area": "/indicators/cmip5/area/{id}",
        "doc": "/indicators/",
        "definitions_from": "1c1de476-cc9d-4c7b-b8ab-25e8f68a317e"},  # the 12 km indicators record
    "heating_degree_days": {
        "about": "Annual heating degree days (below 65 °F): building heating demand. "
                 "Decade means per CMIP5 model under RCP 4.5 and 8.5, 12 km.",
        "units": "°F·days", "point": "/degree_days/heating/{lat}/{lon}/1980/2099",
        "doc": "/degree_days/"},
    "freezing_index": {
        "about": "Annual freezing index (degree days below 32 °F): ground freezing, ice "
                 "roads, frost depth. Decade means per CMIP5 model, RCP 4.5 and 8.5, 12 km.",
        "units": "°F·days", "point": "/degree_days/freezing_index/{lat}/{lon}/1980/2099",
        "doc": "/degree_days/"},
    "thawing_index": {
        "about": "Annual thawing index (degree days above 32 °F): permafrost thaw, "
                 "foundation and road design, growing season. Decade means per CMIP5 "
                 "model, RCP 4.5 and 8.5, 12 km.",
        "units": "°F·days", "point": "/degree_days/thawing_index/{lat}/{lon}/1980/2099",
        "doc": "/degree_days/"},
    "degree_days_below_zero": {
        "about": "Annual degree days below 0 °F: severity of extreme cold. Decade means "
                 "per CMIP5 model, RCP 4.5 and 8.5, 12 km.",
        "units": "°F·days", "point": "/degree_days/below_zero/{lat}/{lon}/1980/2099",
        "doc": "/degree_days/"},
    "permafrost": {
        "about": "GIPL 2.0 model at 1 km, 2021-2120 (decade means): mean annual ground "
                 "temperature (magt) at several depths, permafrost top/base depth, talik "
                 "thickness; GFDL-CM3, NCAR-CCSM4, 5-model average; RCP 4.5 and 8.5. Also "
                 "permafrost extent and ground ice (Jorgenson et al. 2008) and 2000-2016 "
                 "ground temperature (Obu et al. 2018).",
        "units": "temperatures °C, depths and thicknesses m",
        "point": "/permafrost/point/{lat}/{lon}", "doc": "/permafrost/"},
    "snowfall": {
        "about": "Decadal mean annual snowfall water equivalent (SFE), historical and "
                 "projected.",
        "units": "SFE: mm", "point": "/snow/snowfallequivalent/{lat}/{lon}", "doc": "/snow/"},
    "wet_days": {
        "about": "Annual number of wet days (decade means), historical reanalysis and "
                 "projected models.",
        "units": "wdpy: days per year",
        "point": "/wet_days_per_year/all/point/{lat}/{lon}", "doc": "/wet_days_per_year/"},
}
Topic = Literal[tuple(TOPICS)]


# -------------------------------------------------------------- web helpers

def fetch(path):
    """GET a Data API path; returns (full URL, decoded JSON or text)."""
    url = API + path
    with urllib.request.urlopen(url, timeout=90) as r:   # follows the API's 308 redirects
        body = r.read().decode("utf-8")
    try:
        return url, json.loads(body)
    except json.JSONDecodeError:
        return url, body


@functools.cache
def doc_sources(doc_path):
    """Catalog records and references listed on the API's own documentation page."""
    _, page = fetch(doc_path)
    uuids = re.findall(r"catalog\.search#/metadata/([0-9a-f-]{36})", page)
    dois = re.findall(r'href="(https://doi\.org/[^"]+)"', page)
    return list(dict.fromkeys(uuids)), list(dict.fromkeys(dois))


def lane1_records(where):
    """Metadatas from the Lane 1 index matching `where`, or [] if there is no
    index yet (so live data still works before anyone has run a harvest)."""
    try:
        return lane1.open_collection().get(where=where, include=["metadatas"])["metadatas"]
    except (Exception, SystemExit):
        return []


def titles_for(uuids):
    """{uuid: title} for records that are in our Lane 1 index."""
    if not uuids:
        return {}
    return {m["uuid"]: m["title"] for m in lane1_records({"uuid": {"$in": list(uuids)}})}


def dataset_ref(uuid, title=None):
    return {"title": title or "(record not in local index; see link)",
            "catalog_record": CATALOG_RECORD.format(uuid=uuid)}


# -------------------------------------------------------- Alaska places

@functools.cache
def alaska_places():
    """The Alaska-only gazetteer; fetched on first use if missing. Filtered again
    here (places.keep) so an older, wider places.json can't widen the scope."""
    if not Path(places.PLACES_FILE).exists():
        places.fetch_places()
    return [p for p in places.load_places() if places.keep(p)]


@functools.cache
def places_by_id():
    return {p["id"]: p for p in alaska_places()}


@functools.cache
def polygon_extent(area_id):
    """Bounding box of an area's polygon from /boundary/area/{id}, or None."""
    try:
        _, geojson = fetch(f"/boundary/area/{area_id}")
    except Exception:
        return None
    lons, lats = [], []

    def walk(x):   # GeoJSON coordinates are nested lists ending in [lon, lat]
        if isinstance(x, list) and len(x) >= 2 and all(isinstance(v, (int, float)) for v in x[:2]):
            lons.append(x[0]); lats.append(x[1])
        elif isinstance(x, list):
            for item in x:
                walk(item)
        elif isinstance(x, dict):
            for key in ("features", "geometry", "geometries", "coordinates"):
                if key in x:
                    walk(x[key])
    walk(geojson)
    if not lons:
        return None
    return {"west": min(lons), "east": max(lons), "south": min(lats), "north": max(lats)}


def place_info(p, with_polygon=True):
    info = {"id": p["id"], "name": p["name"], "description": places.describe(p)}
    if places.has_point(p):
        info.update(kind="point", latitude=p["latitude"], longitude=p["longitude"])
    else:
        info.update(kind="polygon", boundary_geojson=f"{API}/boundary/area/{p['id']}")
        if with_polygon:
            info["extent"] = polygon_extent(p["id"])
    return info


def resolve(place):
    """An Alaska place from an id ('AK436', 'GMU20') or a name ('Two Rivers')."""
    by_id = places_by_id()
    if place in by_id:
        return by_id[place]
    found = places.find_places(place)
    if found:
        return found[0][1]
    raise ValueError(f"'{place}' is not an Alaska place id or name. Use find_place first.")


def covers(meta, p):
    """Does a dataset's extent contain the place? True / False / None (unknown)."""
    if p is None:
        return None
    if places.has_point(p):
        return places.covers(meta, p["latitude"], p["longitude"])
    box = polygon_extent(p["id"])
    if box is None or not all(k in meta for k in ("west", "east", "south", "north")):
        return None
    return (meta["west"] <= box["east"] and box["west"] <= meta["east"]
            and meta["south"] <= box["north"] and box["south"] <= meta["north"])


# ------------------------------------------------- shrinking API responses

YEAR = re.compile(r"^\d{4}$")
SPREAD_STATS = {"median", "q1", "q3", "hi_std", "lo_std"}


def mean_of(values):
    """Average numbers, or same-shaped dicts of numbers key by key."""
    nums = [v for v in values if isinstance(v, (int, float))]
    if nums and len(nums) == len([v for v in values if v is not None]):
        return sum(nums) / len(nums)
    dicts = [v for v in values if isinstance(v, dict)]
    if dicts:
        keys = dict.fromkeys(k for d in dicts for k in d)
        return {k: mean_of([d[k] for d in dicts if k in d]) for k in keys}
    return values[0] if values else None


def compact(x):
    """Make a response fit an agent's context without changing what it says:
    long yearly series become decade means, spread statistics are dropped
    (min/mean/max kept), numbers are rounded."""
    if isinstance(x, dict):
        if len(x) > 20 and all(YEAR.match(str(k)) for k in x):
            decades = {}
            for year, value in x.items():
                decades.setdefault(f"{str(year)[:3]}0s", []).append(value)
            return {d: compact(mean_of(v)) for d, v in decades.items()}
        return {k: compact(v) for k, v in x.items() if k not in SPREAD_STATS}
    if isinstance(x, list):
        return [compact(v) for v in x]
    if isinstance(x, float):
        return round(x, 2)
    return x


def definitions_from(uuid, codes):
    """Pull 'code: definition' lines for the given codes out of a catalog record
    in the Lane 1 index (the indicators record defines hd, su, dw, ...)."""
    text = "\n".join(dict.fromkeys(m["section_text"] for m in lane1_records({"uuid": uuid})))
    defs = {}
    for line in text.splitlines():
        m = re.match(r"\s*([a-z0-9]+):\s*(.+)", line)
        if m and m.group(1) in codes:
            defs.setdefault(m.group(1), m.group(2).strip())
    return defs


# -------------------------------------------------------------------- server

INSTRUCTIONS = """\
Alaska climate data and dataset knowledge from SNAP (University of Alaska Fairbanks).
Coverage is Alaska only.

Workflow:
1. Named place in the question -> find_place first. Use the returned id in other tools.
   If several places match, say which one you used.
2. To choose and explain datasets (what they contain, methods, baselines, caveats)
   -> search_datasets (pass the place id to keep only datasets covering it);
   get_dataset for a full record.
3. For numbers -> get_climate_data. Never state a climate value that did not come
   from get_climate_data in this conversation.

Every answer that uses these tools ends with a "Sources" section listing, from the
tools' attribution blocks: each dataset title with its catalog link, the exact Data
API query URL(s), reference DOIs, the provider credit, and the license. When giving
numbers, state units, model, scenario (RCP/SSP) and era or baseline period, and
mention relevant caveats or limitations from the dataset records.
"""

mcp = MCPServer("alaska-climate", instructions=INSTRUCTIONS)


def failure(e):
    message = str(e) or type(e).__name__
    if "does not exist" in message:   # no Lane 1 index built yet
        message += (" -- the dataset index hasn't been built. Run: python lane1.py harvest "
                    "--csw https://catalog.snap.uaf.edu/geonetwork/srv/eng/csw")
    return {"error": message}


@mcp.tool()
def find_place(name: str) -> dict[str, Any]:
    """Look up Alaska places by name: communities (with coordinates) and areas with
    polygons (boroughs, census areas, watersheds/HUCs, game management units,
    protected areas, ecoregions, ethnolinguistic regions, Native corporations,
    fire zones, climate divisions). Understands Indigenous and former names
    (e.g. Mamterilleq = Bethel, Barrow = Utqiaġvik). Use this before any other
    tool whenever a question names a place in Alaska. Pass a place name or a
    whole question."""
    try:
        exact = [p for _, p, _ in places.find_places(name)]
        if exact:
            matches, how = exact, "exact name"
        else:   # nothing exact: case-insensitive substring, for "Denali" -> Denali Borough etc.
            q = places.fold(name).lower().strip()
            matches = [p for p in alaska_places()
                       if q and q in places.fold(p["name"] + " " + (p.get("alt_name") or "")).lower()][:10]
            how = "partial name"
        return {"query": name, "matched_by": how,
                "places": [place_info(p, with_polygon=i < 3) for i, p in enumerate(matches)],
                "note": "No Alaska place matched." if not matches else
                        "Pass a place's id to search_datasets or get_climate_data.",
                "attribution": {"source": f"{API}/places/all", "provider": PROVIDER,
                                "license": LICENSE}}
    except Exception as e:
        return failure(e)


@mcp.tool()
def search_datasets(question: str, place: str | None = None, k: int = 5) -> dict[str, Any]:
    """Find SNAP/UAF climate datasets that fit a question about Alaska: temperature,
    precipitation, snow, permafrost, sea ice, wildfire, vegetation, hydrology,
    wind, degree days, climate indicators and more. Returns each dataset's title,
    catalog link and the part of its metadata that matched (description, methods,
    usage limits). Give `place` (an id or name from find_place) to keep only
    datasets whose extent covers that place. Use to choose datasets and to explain
    methods, scenarios, baselines and caveats; use get_climate_data for numbers."""
    try:
        p = resolve(place) if place else None
        hits = lane1.search_many([question], k=k * 3, per="dataset")[0]
        results = []
        for doc, meta, dist in hits:
            coverage = covers(meta, p)
            if coverage is False:
                continue
            text = meta["section_text"]
            results.append({
                **dataset_ref(meta["uuid"], meta["title"]), "uuid": meta["uuid"],
                "distance": round(dist, 3), "matched_section": meta["section"],
                "covers_place": coverage,
                "text": text if len(text) <= 1500 else text[:1500] + " [...] (get_dataset for all)"})
            if len(results) == k:
                break
        return {"question": question,
                "place": place_info(p, with_polygon=False) if p else None,
                "datasets": results,
                "note": "distance: lower = closer match. covers_place null = extent unknown.",
                "attribution": {"provider": PROVIDER, "license": LICENSE,
                                "catalog": "https://catalog.snap.uaf.edu/geonetwork"}}
    except Exception as e:
        return failure(e)


@mcp.tool()
def get_dataset(uuid: str) -> dict[str, Any]:
    """Full catalog record for one SNAP dataset (uuid from search_datasets):
    abstract, methods / supplemental information, lineage, variables, spatial
    extent and resolution, download links, license and use limitations, contacts.
    Use to explain how a dataset was made, what it can and cannot be used for,
    and to cite it (DOIs are listed)."""
    try:
        got = lane1.open_collection().get(where={"uuid": uuid}, include=["metadatas"])
        if not got["metadatas"]:
            return {"error": f"No dataset {uuid} in the index. Use search_datasets."}
        sections = {}
        for m in got["metadatas"]:
            sections.setdefault(m["section"], m["section_text"])
        title = got["metadatas"][0]["title"]
        all_text = "\n".join(sections.values())
        return {**dataset_ref(uuid, title), "uuid": uuid, "sections": sections,
                "attribution": {"dataset": title, "catalog_record": CATALOG_RECORD.format(uuid=uuid),
                                "dois": sorted(set(re.findall(r"https?://doi\.org/[^\s<>;]+", all_text))),
                                "license_and_constraints": sections.get("usage", LICENSE),
                                "provider": PROVIDER}}
    except Exception as e:
        return failure(e)


@mcp.tool()
def get_climate_data(place: str, topic: Topic) -> dict[str, Any]:
    """Live numbers from SNAP's Alaska + Arctic Data API (earthmaps.io) for one
    Alaska place: historical and projected temperature and precipitation,
    climate indicators, heating degree days, freezing and thawing indices,
    permafrost ground temperature and thaw depth, snowfall, wet days.
    `place` is an id or name from find_place. Communities give point values;
    areas (boroughs, watersheds, GMUs...) give area means where the topic
    supports it (temperature_precipitation, climate_indicators). Results
    include units, the exact query URL and full attribution."""
    try:
        p = resolve(place)
        spec = TOPICS[topic]
        if places.has_point(p):
            path = spec["point"].format(lat=p["latitude"], lon=p["longitude"])
        elif "area" in spec:
            path = spec["area"].format(id=p["id"])
        else:
            area_topics = [t for t, s in TOPICS.items() if "area" in s]
            return {"error": f"'{topic}' is only available for points, and {p['name']} is an "
                             f"area. Use a community inside it, or one of {area_topics}."}
        url, data = fetch(path)
        data = compact(data)
        text = json.dumps(data)
        if len(text) > MAX_RESULT_CHARS:
            data = {"too_large": f"{len(text):,} characters after compaction; open the "
                                 f"source query URL to see it all.",
                    "keys": list(data) if isinstance(data, dict) else None}
        uuids, dois = doc_sources(spec["doc"])
        titles = titles_for(uuids)
        result = {
            "place": place_info(p, with_polygon=False), "topic": topic,
            "about": spec["about"], "units": spec["units"],
            "processing": "Yearly series averaged by decade and spread statistics "
                          "(quartiles, std) dropped; values otherwise as returned by the API.",
            "data": data,
            "attribution": {
                "source_query": url,
                "api_documentation": API + spec["doc"],
                "datasets": [dataset_ref(u, titles.get(u)) for u in uuids],
                "references": dois,
                "provider": PROVIDER, "license": LICENSE}}
        if "definitions_from" in spec and isinstance(data, dict):
            result["definitions"] = definitions_from(spec["definitions_from"], set(data))
        return result
    except Exception as e:
        return failure(e)


if __name__ == "__main__":
    mcp.run()   # stdio: the client launches us and talks over stdin/stdout
