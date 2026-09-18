"""
places.py - The place-name gazetteer for Lane 1, from our Data API.

  fetch_places()  download https://earthmaps.io/places/all to ./places.json
  find_places()   which places does a question name? EXACT name matching,
                  including Indigenous / alternate names ("Mamterilleq" -> Bethel)
  covers()        does a dataset's bounding box contain a place's point?

Why exact matching and not embeddings? We tested it: an embedding model
matches MEANING, so "How much warmer will it get in Two Rivers?" found
"Headwaters Foraker River", "Snowfall near Bethel" found "Snow Lake, Manitoba",
and "Mamterilleq" found nothing. Names need string matching. (The places are
still embedded, see `lane1.py places`, so you can compare the two yourself.)
"""

import functools
import json
import unicodedata
import urllib.request

PLACES_URL = "https://earthmaps.io/places/all"
PLACES_FILE = "./places.json"
# When one name exists in several countries ("Bethel": Alaska and Ontario),
# prefer these, in this order. Areas (watersheds, boroughs...) have no country
# and come after.
PREFERRED_COUNTRIES = ["US", "CA"]
COUNTRY_NAMES = {"US": "United States", "CA": "Canada", "SE": "Sweden", "RU": "Russia",
                 "NO": "Norway", "FI": "Finland", "FO": "Faroe Islands", "GL": "Greenland",
                 "IS": "Iceland"}
MAX_NAME_WORDS = 8   # longest place name we look for, in words


# Area types in the list that are Canadian (Yukon / NWT). Areas carry no
# country field, so we go by type. All other area types are Alaskan.
CANADIAN_AREA_TYPES = {"yt_game_management_subzone", "yt_watershed",
                       "yt_fire_district", "first_nation"}


def keep(p):
    """Only Alaska: Alaska communities, plus the Alaskan area types
    (watersheds, GMUs, boroughs, census areas, protected areas, ...)."""
    if p["type"] == "community":
        return p.get("region") == "Alaska"
    return p["type"] not in CANADIAN_AREA_TYPES


def fetch_places(url=PLACES_URL, path=None):
    with urllib.request.urlopen(url, timeout=120) as response:
        data = [p for p in json.load(response) if keep(p)]
    with open(path or PLACES_FILE, "w") as f:
        json.dump(data, f)
    return data


@functools.cache
def load_places(path=None):
    path = path or PLACES_FILE
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"{path} not found. Fetch it first:\n    python lane1.py harvest-places")


def describe(p):
    """One readable line per place; also the text we embed."""
    if p["type"] == "community":
        text = f"{p['name']}: community in {p['region']}, {COUNTRY_NAMES.get(p['country'], p['country'])}."
    else:
        kind = p.get("area_type") or p["type"].replace("_", " ")
        text = f"{p['name']}: {kind}."
    if p.get("alt_name"):
        text += f" Also known as {p['alt_name']}."
    return text


def has_point(p):
    return p.get("latitude") is not None and p.get("longitude") is not None


def label(p):
    """Short form for printing: 'Two Rivers (community, Alaska, US) at 64.877, -147.038'."""
    if p["type"] == "community":
        where = f"community, {p['region']}, {p['country']}"
    else:
        where = p.get("area_type") or p["type"].replace("_", " ")
    text = f"{p['name']} ({where})"
    if has_point(p):
        text += f" at {p['latitude']:.3f}, {p['longitude']:.3f}"
    return text


# ------------------------------------------------------- exact name matching

def fold(text):
    """Make spelling variants compare equal: 'Utqiaġvik' = 'Utqiagvik',
    curly and straight apostrophes the same. Case is kept on purpose."""
    text = text.replace("’", "'").replace("‘", "'")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


@functools.cache
def name_index():
    """Two lookups over every name and alternate name:
    exact  {folded name: [places]}               -- case-sensitive
    loose  {folded name in lower case: [places]} -- multi-word names only

    The loose one lets "Denali National Park and Preserve" find "... Park And
    Preserve". One-word names stay case-sensitive: that's what keeps "moose"
    and "hope" from being places.
    """
    exact, loose = {}, {}
    for p in load_places():
        if not keep(p):          # Alaska only, even if places.json is an older, wider download
            continue
        names = [p["name"]] + [a.strip() for a in (p.get("alt_name") or "").split("/")]
        for name in names:
            if name:
                exact.setdefault(fold(name), []).append(p)
                if " " in name:
                    loose.setdefault(fold(name).lower(), []).append(p)
    return exact, loose


def lookup(text):
    exact, loose = name_index()
    if fold(text) in exact:
        return exact[fold(text)]
    # Case-insensitive only for several words, and only if written with a capital.
    if " " in text and text[0].isupper():
        return loose.get(fold(text).lower())
    return None


def preferred_first(candidates):
    def rank(p):
        country = p.get("country")
        return (PREFERRED_COUNTRIES.index(country) if country in PREFERRED_COUNTRIES
                else len(PREFERRED_COUNTRIES))
    return sorted(candidates, key=rank)


def find_places(question):
    """Places named in the question, in the order they appear.

    Returns a list of (text as written, best place, number of places with that name).
    Longest names win ("Fort Yukon" over "Yukon"). Case-sensitive on purpose:
    "Moose Pass" is a place, "moose" is an animal.
    """
    words = question.split()
    found, used = [], set()
    for n in range(min(MAX_NAME_WORDS, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            if used & set(range(i, i + n)):
                continue                              # part of a longer match already
            text = " ".join(words[i:i + n]).strip("?!.,;:\"()[]")
            if text.endswith(("'s", "’s")):          # possessive: "Bethel's"
                text = text[:-2]
            text = text.rstrip("'’")                  # possessive: "Fairbanks'"
            if not text:
                continue
            candidates = lookup(text)
            if candidates:
                found.append((i, text, preferred_first(candidates)[0], len(candidates)))
                used |= set(range(i, i + n))
    return [(text, place, n) for _, text, place, n in sorted(found, key=lambda f: f[0])]


# ------------------------------------------------------------------- spatial

def covers(meta, lat, lon):
    """Does a dataset's bounding box contain the point? None if it has no box."""
    if not all(k in meta for k in ("west", "east", "south", "north")):
        return None
    return meta["south"] <= lat <= meta["north"] and meta["west"] <= lon <= meta["east"]
