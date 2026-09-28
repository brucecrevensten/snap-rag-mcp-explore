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
import urllib.error
import urllib.parse
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
    "fire_weather": {
        "about": "Canadian Forest Fire Weather Index (FWI) System indices from "
                 "bias-corrected CMIP6 daily data and ERA5 reanalysis, 1980-2100, "
                 "April 1 - October 31, 0.25 degrees, North American boreal ecoregion "
                 "only. Needs a year range; pick the indices with `variables` and the "
                 "summary with `operation`.",
        "units": "index values (unitless ratings); fire danger days: days per year",
        "point": "/fire_weather/point/{lat}/{lon}/{start}/{end}",
        "area": "/fire_weather/area/{id}/{start}/{end}",
        "doc": "/fire_weather/",
        "years": (1980, 2100), "default_years": (2030, 2050),
        "variables": ["bui", "dc", "dmc", "ffmc", "fwi", "isi"],
        "operations": ["summer_fire_danger_rating_days", "3_day_rolling_average",
                       "5_day_rolling_average", "7_day_rolling_average"],
        "area_note": "fire_weather area queries take watershed (HUC) ids; boroughs and "
                     "game management units return 404. Use a community inside the area.",
        "slow_note": "The rolling averages return every day of the fire season for every "
                     "model; they take ~20s and are summarised here to monthly means. "
                     "Pass `variables` to keep them small."},
    "wildfire_flammability": {
        "about": "ALFRESCO modelled relative flammability at 1 km, as 30-year era means: "
                 "historical (CRU TS 4.0, 1950-1979 and 1980-2008) and projections "
                 "2010-2099 from GFDL-CM3, GISS-E2-R, IPSL-CM5A-LR, MRI-CGCM3, "
                 "NCAR-CCSM4 and a 5-model average under RCP 4.5, 6.0 and 8.5. A point "
                 "query returns the intersecting HUC-12 watershed: single ALFRESCO "
                 "pixels are not meaningful on their own.",
        "units": "average number of times a pixel burned per year (0-1)",
        "point": "/alfresco/flammability/local/{lat}/{lon}",
        "area": "/alfresco/flammability/area/{id}", "doc": "/alfresco/"},
    "vegetation_type": {
        "about": "ALFRESCO modelled vegetation composition at 1 km, as 30-year era means "
                 "(same models, scenarios and eras as wildfire_flammability). A point "
                 "query returns the intersecting HUC-12 watershed.",
        "units": "percent of area per vegetation type",
        "point": "/alfresco/veg_type/local/{lat}/{lon}",
        "area": "/alfresco/veg_type/area/{id}", "doc": "/alfresco/"},
    "wildfire_now": {
        "about": "Near-real-time wildfire conditions at a point (NOT climate "
                 "projections): today's fire danger rating and snow cover, PM2.5 air "
                 "quality over the last 6-48 hours, currently active fires nearby with "
                 "cause and size, land cover, and projected relative flammability.",
        "units": "aqi: US AQI; pm25_conc: ug/m3; fire size: acres",
        "point": "/fire/point/{lat}/{lon}", "doc": "/fire/", "trim": "fires"},
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
DAY_OF_YEAR = re.compile(r"^\d{2}-\d{2}$")
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
SPREAD_STATS = {"median", "q1", "q3", "hi_std", "lo_std"}


def summarize_fires(data):
    """The wildfire_now response lists every nearby fire as GeoJSON with a prose
    summary: ~18 KB of mostly geometry. Keep what a person would ask about."""
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for key in ("fire_points", "fire_polygons"):
        features = out.get(key)
        if not isinstance(features, list):
            continue
        fires = []
        for f in features[:15]:
            p = f.get("properties", {})
            fires.append({"name": p.get("NAME"), "cause": p.get("CAUSE"),
                          "acres": p.get("acres"), "active": p.get("active") == "1",
                          "summary": (p.get("SUMMARY") or "")[:300]})
        out[key] = {"count": len(features), "showing": len(fires), "fires": fires}
    return out


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
    long yearly series become decade means, day-of-year series become monthly
    means, spread statistics are dropped (min/mean/max kept), numbers rounded."""
    if isinstance(x, dict):
        if len(x) > 20 and all(YEAR.match(str(k)) for k in x):
            decades = {}
            for year, value in x.items():
                decades.setdefault(f"{str(year)[:3]}0s", []).append(value)
            return {d: compact(mean_of(v)) for d, v in decades.items()}
        if len(x) > 20 and all(DAY_OF_YEAR.match(str(k)) for k in x):
            # Fire weather gives every day from 04-01 to 10-31, per model and
            # index: hundreds of kilobytes. Monthly means say the same thing.
            months = {}
            for day, value in x.items():
                months.setdefault(MONTHS[int(str(day)[:2]) - 1], []).append(value)
            return {m: compact(mean_of(v)) for m, v in months.items()}
        return {k: compact(v) for k, v in x.items() if k not in SPREAD_STATS}
    if isinstance(x, list):
        return [compact(v) for v in x]
    if isinstance(x, float):
        # 2 decimals for ordinary values, but flammability rates are ~0.004:
        # rounding those to 2 decimals would turn every one of them into 0.0.
        return round(x, 2) if abs(x) >= 1 else float(f"{x:.4g}")
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
   get_dataset for a full record. `via_paper` means the dataset was found through
   a passage of a paper it cites; `supporting_paper` is the closest passage from a
   paper it cites. Use those passages for findings and methods, and cite the paper.
3. For numbers -> get_climate_data. Never state a climate value that did not come
   from get_climate_data in this conversation.

Every answer that uses these tools ends with a "Sources" section listing, from the
tools' attribution blocks: each dataset title with its catalog link, the exact Data
API query URL(s), reference DOIs, any papers you drew on (citation + DOI),
the provider credit, and the license. When giving
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
def search_datasets(question: str, place: str | None = None, k: int = 5,
                    include_papers: bool = True) -> dict[str, Any]:
    """Find SNAP/UAF climate datasets that fit a question about Alaska: temperature,
    precipitation, snow, permafrost, sea ice, wildfire, vegetation, hydrology,
    wind, degree days, climate indicators and more. Returns each dataset's title,
    catalog link and the part of its metadata that matched (description, methods,
    usage limits). Give `place` (an id or name from find_place) to keep only
    datasets whose extent covers that place. Also searches the papers the
    datasets cite (include_papers): a dataset can be found through a passage of
    its paper, shown in `via_paper` with the paper's citation and DOI. Use to
    choose datasets and to explain methods, scenarios, baselines, findings and
    caveats; use get_climate_data for numbers."""
    try:
        p = resolve(place) if place else None
        hits = lane1.search_many([question], k=k * 3, per="dataset", papers=include_papers)[0]
        results = []
        for doc, meta, dist in hits:
            coverage = covers(meta, p)
            if coverage is False:
                continue
            text = meta["section_text"]
            result = {
                **dataset_ref(meta["uuid"], meta["title"]), "uuid": meta["uuid"],
                "distance": round(dist, 3), "matched_section": meta["section"],
                "covers_place": coverage,
                "text": text if len(text) <= 1500 else text[:1500] + " [...] (get_dataset for all)"}
            if "citation" in meta:     # found through a paper this dataset cites
                result["via_paper"] = {
                    "citation": meta["citation"], "page": meta["page"],
                    "doi": f"https://doi.org/{meta['doi']}" if meta["doi"] else None,
                    "text_is": "full-text page" if meta["full_text"] else "abstract"}
            elif "supporting_paper" in meta:   # found via its record; a cited paper also matched
                sp = meta["supporting_paper"]
                result["supporting_paper"] = {
                    **sp, "doi": f"https://doi.org/{sp['doi']}" if sp["doi"] else None}
            results.append(result)
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
def get_climate_data(place: str, topic: Topic, variables: str = "", operation: str = "",
                     start_year: int = 0, end_year: int = 0) -> dict[str, Any]:
    """Live numbers from SNAP's Alaska + Arctic Data API (earthmaps.io) for one
    Alaska place: historical and projected temperature and precipitation,
    climate indicators, heating degree days, freezing and thawing indices,
    permafrost ground temperature and thaw depth, snowfall, wet days, fire
    weather indices, modelled flammability and vegetation, and current
    wildfire conditions.

    `place` is an id or name from find_place. Communities give point values;
    areas (boroughs, watersheds, GMUs...) give area means where the topic
    supports it. Three options apply to some topics only; the error says so
    if they don't fit:
      variables  comma-separated ids to return, e.g. "fwi,bui" for fire_weather
                 (bui, dc, dmc, ffmc, fwi, isi). Empty = all of them.
      operation  how to summarise, for fire_weather:
                 "summer_fire_danger_rating_days" (default; mean June-August
                 days per year in each fire danger class) or
                 "3_day_rolling_average" / "5_day_..." / "7_day_..." (min, mean
                 and max of the rolling average through the fire season; slower
                 and larger, summarised here to monthly means).
      start_year, end_year  the year range a topic needs (fire_weather, 1980-2100).

    Results include units, the exact query URL and full attribution."""
    try:
        p = resolve(place)
        spec = TOPICS[topic]
        # Options only make sense for some topics; say which, rather than
        # silently ignoring what the caller asked for.
        for name, value, allowed in [("variables", variables, spec.get("variables")),
                                     ("operation", operation, spec.get("operations"))]:
            if value and not allowed:
                topics = [t for t, s in TOPICS.items() if s.get(name if name == "variables"
                                                                else "operations")]
                return {"error": f"'{topic}' takes no {name}. Topics that do: {topics}"}
            for item in ([operation] if name == "operation" and value else
                         [v.strip() for v in value.split(",")] if value else []):
                if item not in allowed:
                    return {"error": f"{name} '{item}' is not one of {allowed} for '{topic}'."}
        years = {}
        if "years" in spec:
            first, last = spec["years"]
            start, end = start_year or spec["default_years"][0], end_year or spec["default_years"][1]
            if not (first <= start < end <= last):
                return {"error": f"'{topic}' needs start_year < end_year within "
                                 f"{first}-{last} (got {start}-{end})."}
            years = {"start": start, "end": end}
        elif start_year or end_year:
            return {"error": f"'{topic}' takes no year range; it returns fixed eras."}

        if places.has_point(p):
            path = spec["point"].format(lat=p["latitude"], lon=p["longitude"], **years)
        elif "area" in spec:
            path = spec["area"].format(id=p["id"], **years)
        else:
            area_topics = [t for t, s in TOPICS.items() if "area" in s]
            return {"error": f"'{topic}' is only available for points, and {p['name']} is an "
                             f"area. Use a community inside it, or one of {area_topics}."}
        query = {k: v for k, v in [("vars", variables),
                                   ("op", operation or spec.get("operations", [""])[0])] if v}
        if query:
            path += "?" + urllib.parse.urlencode(query)
        try:
            url, data = fetch(path)
        except urllib.error.HTTPError as e:
            if e.code == 404 and not places.has_point(p) and spec.get("area_note"):
                return {"error": f"{API}{path} returned 404. {spec['area_note']}"}
            raise
        data = summarize_fires(data) if spec.get("trim") == "fires" else data
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
            "processing": "Yearly series averaged by decade, day-of-year series by month, "
                          "spread statistics (quartiles, std) dropped, long feature lists "
                          "summarised; values otherwise as returned by the API.",
            "data": data,
            "attribution": {
                "source_query": url,
                "api_documentation": API + spec["doc"],
                "datasets": [dataset_ref(u, titles.get(u)) for u in uuids],
                "references": dois,
                "provider": PROVIDER, "license": LICENSE}}
        for note in ("slow_note", "area_note"):
            if spec.get(note):
                result.setdefault("notes", []).append(spec[note])
        if query:
            result["query_options"] = query
        if "definitions_from" in spec and isinstance(data, dict):
            result["definitions"] = definitions_from(spec["definitions_from"], set(data))
        return result
    except Exception as e:
        return failure(e)


if __name__ == "__main__":
    mcp.run()   # stdio: the client launches us and talks over stdin/stdout
