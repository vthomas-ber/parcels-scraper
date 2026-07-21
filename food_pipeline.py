"""
food_pipeline.py — deterministic-first food-data extraction (verify-then-read).

This module is INTENTIONALLY SELF-CONTAINED. It does not import from app.py so
it can be unit-tested in isolation (feed it a session + an EAN and it returns
structured food data). A handful of small pure helpers and the GOLDMINE /
exclusion constants are duplicated from app.py on purpose — that duplication is
the price of isolation. If you change the goldmine list or exclusion tokens in
app.py, mirror the change here.

Public entry point
------------------
    await extract_food_data(session, ean, ground_truth, market,
                            registry_name, keys,
                            go_upc_data=None, candidate_pages=None,
                            ean_verified_hint=False)
      -> {
           "fields":        {gemini-style field keys -> value|""},
           "source_links":  [urls actually read, trust-ordered],
           "source_routes": {url: "A"|"B"},
           "provenance":    str,     # for the "Data Provenance" column
           "reliability":   "H"|"M"|"L",
           "status":        "success"|"page_not_readable"|"no_source",
           "name":          str,
           "ean_verified":  bool,
           "retailer_urls": [all candidate urls seen],
           "diagnostics":   [log lines],
         }

Design rules (per the rebuild brief)
------------------------------------
* Verify-then-read is the PRIMARY path. Candidate pages come from the discovery
  ladder (Tier 1A Go-UPC -> 1B EAN-Search -> Attempt 2 brand -> 2.5 name+EAN ->
  3 goldmine -> 4 bare-GTIN -> 5 name fallback). In production app.py passes the
  already-discovered pages via `candidate_pages`; standalone/testing lets the
  module run the ladder itself.
* Each candidate is VERIFIED: Route A (EAN in a labelled GTIN context on the
  page) or Route B (product name >=80% token overlap vs user input OR the
  registry name, with no conflicting weight). Only verified pages are read.
* Extraction is DETERMINISTIC-FIRST, in this fallback order:
      JSON-LD (application/ld+json Product / NutritionInformation / gtin)
        -> microdata / OpenGraph
        -> labelled HTML nutrition table.
  Numbers (nutrition, GTIN, weight) ONLY ever come from the deterministic
  parser — never from an LLM.
* Gemini is a LAST-RESORT fallback for messy free-text fields only
  (ingredients / allergens / may-contain / manufacturer / origin), fed ONLY the
  fetched page text, temperature 0, "extract only what's present, return null
  otherwise, do not search."
* Fields are filled ACROSS the top verified pages: richest / highest-trust
  source wins per field; genuinely absent fields stay blank.
* Grading: any readable Route-A page -> H; only Route-B -> M; no readable
  verified page -> page_not_readable / L (first-pass registry data is kept, not
  blanked).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from html import unescape

try:
    from bs4 import BeautifulSoup
    _BS4_OK = True
except ImportError:  # pragma: no cover - defensive; requirements pins bs4
    _BS4_OK = False

# google-genai is only needed for the free-text fallback. Import lazily so the
# deterministic parsers (and their tests) work without the SDK present.
try:
    from google import genai
    from google.genai import types
    _GENAI_OK = True
except ImportError:  # pragma: no cover
    _GENAI_OK = False

# playwright is only needed for the JS-render fallback (pages that verify but
# come back data-thin via a plain HTTP GET — client-side-rendered content a
# aiohttp fetch can never see). Import lazily so the deterministic parsers
# (and everything else) keep working without it installed; see
# fetch_html_rendered's docstring for the deployment step this needs.
try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_OK = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_OK = False


# ============================================================================
# CONFIG (kept in sync with app.py — see module docstring)
# ============================================================================

EXTRACTION_MODEL = "gemini-2.5-flash"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

# Max verified pages to read and fill across (richest/highest-trust wins).
MAX_READ_PAGES = 4
# Max candidate pages to fetch+verify before giving up (keeps HTTP bounded).
MAX_CANDIDATES = 12
_NAME_MATCH_THRESHOLD = 0.80

# ---- JS-render fallback (headless Chromium via Playwright) -----------------
# A plain aiohttp GET can't see content a page builds with client-side JS
# (common on modern "drive"/click-and-collect storefronts) — such a page
# verifies (EAN/name matches) but extract_page_fields finds nothing, which
# grades as "page_not_readable". This fallback re-fetches ONLY those specific
# thin/unverified pages with a real browser and re-runs the SAME deterministic
# parsers on the rendered DOM. It is an order of magnitude slower/heavier than
# a plain GET, so it is bounded on every axis: a small number of renders per
# product, a small global concurrency cap (many EANs run in parallel), and an
# env-var kill switch so it can be turned off instantly without a redeploy if
# it strains a memory-constrained host.
JS_RENDER_ENABLED = os.environ.get("ENABLE_JS_RENDER", "1").strip().lower() not in (
    "0", "false", "no", "off")
RENDER_MAX_CONCURRENT = int(os.environ.get("JS_RENDER_MAX_CONCURRENT", "2"))
RENDER_TIMEOUT_MS = int(os.environ.get("JS_RENDER_TIMEOUT_MS", "15000"))
# Per extract_food_data() call — caps worst-case added latency/cost per EAN
# even if every candidate page happens to be JS-rendered.
MAX_RENDERS_PER_PRODUCT = 3

GOLDMINE = {
    "FR": ("site:carrefour.fr OR site:auchan.fr OR site:coursesu.com "
           "OR site:leclerc.fr OR site:intermarche.fr OR site:monoprix.fr "
           "OR site:cora.fr OR site:casino.fr OR site:magasins-u.com "
           "OR site:grandfrais.com"),
    "UK": ("site:ocado.com OR site:waitrose.com OR site:asda.com "
           "OR site:tesco.com OR site:sainsburys.co.uk OR site:morrisons.com "
           "OR site:iceland.co.uk OR site:boots.com OR site:coop.co.uk "
           "OR site:aldi.co.uk OR site:marksandspencer.com"),
    "NL": ("site:ah.nl OR site:jumbo.com OR site:plus.nl OR site:dirk.nl "
           "OR site:coop.nl OR site:vomar.nl OR site:hoogvliet.com"),
    "BE": ("site:delhaize.be OR site:colruyt.be OR site:carrefour.be "
           "OR site:spar.be OR site:lidl.be OR site:okay.be OR site:aldi.be"),
    "DE": ("site:rewe.de OR site:edeka.de OR site:kaufland.de "
           "OR site:dm.de OR site:rossmann.de OR site:metro.de "
           "OR site:budni.de OR site:ecoinform.de OR site:netto-online.de "
           "OR site:netto-marken-discount.de OR site:globus.de "
           "OR site:denns-biomarkt.de"),
    "AT": ("site:billa.at OR site:spar.at OR site:gurkerl.at "
           "OR site:hofer.at OR site:mpreis.at OR site:interspar.at "
           "OR site:unimarkt.at OR site:adeg.at"),
    "DK": ("site:nemlig.com OR site:matsmart.dk OR site:rema1000.dk "
           "OR site:coop.dk OR site:salling.dk OR site:netto.dk "
           "OR site:foetex.dk OR site:bilkatogo.dk OR site:meny.dk"),
    "IT": ("site:carrefour.it OR site:conad.it OR site:coop.it "
           "OR site:esselunga.it OR site:eurospin.it OR site:lidl.it "
           "OR site:pampanorama.it OR site:iper.it OR site:naturasi.it "
           "OR site:tigros.it"),
    "ES": ("site:carrefour.es OR site:mercadona.es OR site:dia.es "
           "OR site:alcampo.es OR site:eroski.es OR site:lidl.es "
           "OR site:consum.es OR site:caprabo.es OR site:ahorramas.com "
           "OR site:condis.es"),
    "PL": ("site:carrefour.pl OR site:auchan.pl OR site:frisco.pl "
           "OR site:lidl.pl OR site:kaufland.pl OR site:netto.pl "
           "OR site:dino-polska.pl"),
}
GLOBAL_SITES = "site:billigkaffee.eu OR site:fivestartrading-holland.eu"

# Approved / goldmine domains are the "prime read targets" — pages on these
# rank higher when filling fields across multiple verified pages, and are the
# TIER-1 domains for reliability grading (see _is_trusted_domain below).
# Kept in sync with the GOLDMINE query dict above — every domain referenced
# in a GOLDMINE site: clause must also appear here.
_GOLDMINE_DOMAINS = frozenset({
    "ocado.com", "waitrose.com", "asda.com", "tesco.com", "sainsburys.co.uk",
    "morrisons.com", "iceland.co.uk", "boots.com", "coop.co.uk", "aldi.co.uk",
    "marksandspencer.com",
    "carrefour.fr", "auchan.fr", "coursesu.com", "leclerc.fr",
    "intermarche.fr", "monoprix.fr", "cora.fr", "casino.fr", "magasins-u.com",
    "grandfrais.com",
    "carrefour.it", "conad.it", "coop.it", "esselunga.it", "eurospin.it",
    "lidl.it", "pampanorama.it", "iper.it", "naturasi.it", "tigros.it",
    "rewe.de", "edeka.de", "kaufland.de", "dm.de", "rossmann.de", "metro.de",
    "budni.de", "ecoinform.de", "netto-online.de", "netto-marken-discount.de",
    "globus.de", "denns-biomarkt.de",
    "ah.nl", "jumbo.com", "plus.nl", "dirk.nl", "coop.nl", "vomar.nl",
    "hoogvliet.com",
    "delhaize.be", "colruyt.be", "carrefour.be", "spar.be", "lidl.be",
    "okay.be", "aldi.be",
    "billa.at", "spar.at", "gurkerl.at", "hofer.at", "mpreis.at",
    "interspar.at", "unimarkt.at", "adeg.at",
    "nemlig.com", "rema1000.dk", "coop.dk", "salling.dk", "netto.dk",
    "foetex.dk", "bilkatogo.dk", "meny.dk",
    "carrefour.es", "mercadona.es", "dia.es", "alcampo.es", "eroski.es",
    "lidl.es", "consum.es", "caprabo.es", "ahorramas.com", "condis.es",
    "carrefour.pl", "auchan.pl", "frisco.pl", "lidl.pl", "kaufland.pl",
    "netto.pl", "dino-polska.pl",
})

_EXCLUDED_DOMAIN_TOKENS = (
    "amazon.", "ebay.", "aliexpress.", "alibaba.",
    "barcodelookup.", "go-upc.",
    # Open Food Facts must never be queried, fetched, or shown as a source.
    "openfoodfacts.org", "openfoodfacts.net", "off.",
)

# Data aggregators carry clean, structured nutrition/ingredient tables and are
# often the FIRST hit on a bare-EAN Google search. They are market-agnostic
# (most cover the whole EU) and far more reliable to parse than JS-heavy retail
# SPAs, so we search them explicitly and treat them as prime read targets.
_DATA_AGGREGATOR_QUERY = (
    "site:codecheck.info OR site:das-ist-drin.de "
    "OR site:fddb.info OR site:supermarktcheck.de OR site:digit-eyes.com "
    "OR site:piccantino.com OR site:questionmark.com OR site:yuka.io"
)
_AGGREGATOR_DOMAINS = frozenset({
    "codecheck.info",
    "das-ist-drin.de", "fddb.info", "fddb.mobi", "supermarktcheck.de",
    "digit-eyes.com", "piccantino.com", "questionmark.com", "yuka.io",
})

_BRAND_STOPWORDS = {
    "mandorle", "anacardi", "arachidi", "nocciole", "pistacchi", "noci",
    "mix", "assortiti", "ricoperte", "tostate", "salate", "bio", "organic",
    "cioccolato", "fondente", "bianco", "latte", "cocco", "yogurt", "limone",
    "arancia", "fragola", "lampone", "vaniglia", "caramello", "miele",
    "preparazione", "confezione", "formato", "gusto", "sapore",
    "preparation", "aromatisation", "saveur", "pour", "avec",
    "product", "item", "food", "snack", "pack", "bag", "box",
    "gr", "kg", "ml", "cl", "da", "al", "di", "con", "per",
}

_GARBAGE_NAME_PATTERNS = [
    r"upc lookup", r"ninguno", r"^lookup ", r"barcode\s+\d",
    r"unknown product", r"^no\s+(name|product)", r"^[\d\s#]+$", r"#####",
]


# ============================================================================
# PURE HELPERS (self-contained copies of app.py utilities)
# ============================================================================

def _is_excluded_domain(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(tok in u for tok in _EXCLUDED_DOMAIN_TOKENS)


def barcode_matches(returned: str, queried: str) -> bool:
    if not returned:
        return False
    return str(returned).strip().lstrip("0") == str(queried).strip().lstrip("0")


def is_garbage_name(name: str) -> bool:
    if not name or len(name.strip()) < 3:
        return True
    low = name.lower().strip()
    if low.startswith("http") or "://" in low or "www." in low:
        return True
    return any(re.search(p, low) for p in _GARBAGE_NAME_PATTERNS)


def strip_pack_notation(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'\(\s*\d+\s*[Pp]ack[^)]*\)', '', text)
    text = re.sub(r'\(\s*[Pp]ack\s+of\s+\d+[^)]*\)', '', text)
    text = re.sub(
        r'\b\d+\s*[xX×]\s*(\d+(?:[.,]\d+)?\s*(?:g|gr|kg|ml|cl|l|oz|lb))\b',
        r'\1', text, flags=re.IGNORECASE)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text).strip().strip(',').strip()
    return text


def extract_brand(ground_truth: str) -> str:
    if not ground_truth:
        return ""
    for token in ground_truth.split():
        clean = token.strip(".,;:()-/")
        if len(clean) < 3 or not clean[0].isupper():
            continue
        if clean.lower() in _BRAND_STOPWORDS or clean.replace(".", "").isdigit():
            continue
        return clean
    return ""


def brand_matches_domain(brand: str, domain: str) -> bool:
    if not brand or not domain:
        return False
    b = re.sub(r"[^a-z0-9]", "", brand.lower())
    d = re.sub(r"[^a-z0-9.]", "", domain.lower())
    return len(b) >= 4 and b in d


def _sig_tokens(s: str) -> set:
    if not s:
        return set()
    return {t for t in re.sub(r"[^\w\s]", " ", s.lower()).split() if len(t) > 3}


def _name_match_ratio(anchor: str, text: str) -> float:
    a = _sig_tokens(anchor)
    if not a:
        return 0.0
    hay = text.lower()
    return sum(1 for t in a if t in hay) / len(a)


def _best_name_ratio(anchors: list, text: str) -> float:
    return max((_name_match_ratio(a, text) for a in anchors if a), default=0.0)


def extract_weight_hint(text: str) -> str:
    if not text:
        return ""
    m = re.search(r'\b\d+\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(g|gr|kg|ml|cl|l|oz|lb)\b',
                  text, re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    m = re.search(r'\b(\d+(?:[.,]\d+)?)\s*(g|gr|kg|ml|cl|l|oz|lb)\b',
                  text, re.IGNORECASE)
    return f"{m.group(1)}{m.group(2).lower()}" if m else ""


def _weights_conflict(anchors: list, text: str) -> bool:
    tw = extract_weight_hint(text)
    if not tw:
        return False
    for a in anchors:
        aw = extract_weight_hint(a)
        if aw and aw != tw:
            return True
    return False


def _domain_of(url: str) -> str:
    return url.split("/")[2] if url.startswith("http") and "/" in url[8:] else \
        (url.split("/")[2] if url.startswith("http") else "")


def _is_goldmine(url: str) -> bool:
    dom = _domain_of(url)
    return any(dom == g or dom.endswith("." + g) for g in _GOLDMINE_DOMAINS)


def _is_aggregator(url: str) -> bool:
    dom = _domain_of(url)
    return any(dom == g or dom.endswith("." + g) for g in _AGGREGATOR_DOMAINS)


def is_trusted_page(url: str, brand: str = "") -> bool:
    """
    Tier-1 trust = a goldmine (approved retailer) domain OR the brand's own
    official site. This is the gate for reliability High and for which
    identity-match methods are acceptable (EAN-only / name+EAN / name+weight
    only are all fine from a trusted domain; an untrusted domain needs the
    EAN actually confirmed on the page — see verify_route).
    """
    if _is_goldmine(url):
        return True
    if brand:
        return brand_matches_domain(brand, _domain_of(url))
    return False


# Distinct non-tier-1 domains that must independently agree on a field's
# value before that field counts as Medium reliability; a single such
# source, uncorroborated, stays Low.
MEDIUM_CORROBORATION_MIN = 2


def _values_agree(key: str, a: str, b: str) -> bool:
    """Do two independently-extracted values for the same field agree closely
    enough to count as corroboration? Numeric fields compare parsed floats
    with a small tolerance (decimal/rounding noise); text fields compare
    normalised strings, falling back to significant-token overlap so minor
    formatting differences (punctuation, ordering) don't block a real match."""
    if not a or not b:
        return False
    if key in _NUTRI_KEYS or key in ("net_weight", "gross_weight"):
        fa, fb = _to_float(a), _to_float(b)
        if fa is None or fb is None:
            return False
        return abs(fa - fb) <= max(0.5, 0.03 * max(abs(fa), abs(fb)))
    na = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", a.lower())).strip()
    nb = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", b.lower())).strip()
    if na == nb:
        return True
    ta, tb = _sig_tokens(a), _sig_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / max(len(ta), len(tb)) >= 0.8


def _result_matches_gt(res: dict, ground_truth: str) -> bool:
    if not ground_truth:
        return True
    gt = _sig_tokens(ground_truth)
    if not gt:
        return True
    hay = (res.get("title", "") + " " + res.get("snippet", "")).lower()
    return sum(1 for t in gt if t in hay) / len(gt) >= 0.3


# ============================================================================
# NUMBER / UNIT PARSING
# ============================================================================

def _to_float(raw) -> float | None:
    """Parse a European or English decimal number out of a messy string."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = re.search(r'(\d[\d.\s]*[.,]?\d*)', s.replace("\xa0", " "))
    if not m:
        return None
    num = m.group(1).replace(" ", "")
    # If both separators present, the last one is the decimal separator.
    if "," in num and "." in num:
        if num.rfind(",") > num.rfind("."):
            num = num.replace(".", "").replace(",", ".")
        else:
            num = num.replace(",", "")
    elif "," in num:
        # Comma as decimal separator (EU) unless it's clearly a thousands group.
        if re.fullmatch(r'\d{1,3}(,\d{3})+', num):
            num = num.replace(",", "")
        else:
            num = num.replace(",", ".")
    elif "." in num:
        # A single dot is a DECIMAL POINT ("13.062" -> 13.062, "0.039" -> 0.039,
        # "18.5" -> 18.5). Only MULTIPLE dot groups ("1.234.567") are an
        # unambiguous European thousands grouping. (An earlier heuristic treated
        # any "X.XXX" as thousands, which corrupted clean per-100g decimals such
        # as Open Food Facts' "13.062 kJ" into "13062" and "0.039 g" into "39".)
        if num.count(".") > 1:
            num = num.replace(".", "")
    try:
        return float(num)
    except ValueError:
        return None


def _fmt_num(v: float | None) -> str:
    if v is None:
        return ""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _energy_to_kj(raw) -> str:
    """Return energy as kJ. Accepts kJ directly, or kcal (converts *4.184)."""
    if raw is None:
        return ""
    s = str(raw).lower()
    val = _to_float(s)
    if val is None:
        return ""
    if "kj" in s:
        return _fmt_num(val)
    if "kcal" in s or "cal" in s:
        return _fmt_num(val * 4.184)
    # Ambiguous bare number: a value >400 is almost certainly already kJ for a
    # per-100 basis; smaller values are likely kcal. This only fires when the
    # source gave no unit at all.
    return _fmt_num(val if val > 400 else val * 4.184)


def _sodium_to_salt(raw) -> str:
    """schema.org sodiumContent -> salt (g). salt = sodium * 2.5."""
    if raw is None:
        return ""
    s = str(raw).lower()
    val = _to_float(s)
    if val is None:
        return ""
    if "mg" in s:            # mg -> g
        val = val / 1000.0
    return _fmt_num(val * 2.5)


# Plausible per-100g/ml ranges. Values outside these are almost always a
# mis-parse (a price, an RDA %, a pack count) and are rejected rather than shown.
_NUTRI_BOUNDS = {
    "energy_kj": (10, 4000),      # water ~0, pure fat ~3700 kJ
    "fat_g": (0, 100),
    "saturates_g": (0, 100),
    "carbohydrates_g": (0, 100),
    "sugars_g": (0, 100),
    "protein_g": (0, 100),
    "fiber_g": (0, 100),
    "salt_g": (0, 100),           # realistically <30, but keep a safe ceiling
}


def _bounded(key: str, value: str) -> str:
    """Return value only if it falls inside the plausible per-100 range."""
    if not value:
        return ""
    lo, hi = _NUTRI_BOUNDS.get(key, (None, None))
    if lo is None:
        return value
    fv = _to_float(value)
    if fv is None or fv < lo or fv > hi:
        return ""
    return value


# Canonical nutrition field keys used across the module (== app.py row/data keys)
_NUTRI_KEYS = ("energy_kj", "fat_g", "saturates_g", "carbohydrates_g",
               "sugars_g", "protein_g", "fiber_g", "salt_g")
_TEXT_KEYS = ("ingredients", "allergens", "may_contain",
              "manufacturer_name", "manufacturer_address", "place_of_origin")
_OTHER_KEYS = ("brand", "net_weight", "gross_weight", "nutritional_info",
               "organic_certification_id")
ALL_FIELD_KEYS = _NUTRI_KEYS + _TEXT_KEYS + _OTHER_KEYS


def _blank_fields() -> dict:
    return {k: "" for k in ALL_FIELD_KEYS}


def _num_or_none(v):
    """Parse a formatted field string ('12', '0,5', '1.3') to float, else None."""
    if v in (None, "", "None"):
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def _implausible_nutrition(fields: dict) -> str:
    """
    Last-line physics backstop against parser / decimal corruption. Returns a
    reason string when the per-100 g nutrition is physically impossible, else "".

    The key invariant: fat + carbohydrate + protein cannot exceed 100 g per
    100 g. Decimal corruption (0.039 -> 39, 0.062 -> 62) inflates individual
    macros so their sum blows past 100, which is how a tea ends up "reading"
    39 g fat + 62 g protein. Real foods — even oils (100 g fat) or sugar
    (100 g carb) or protein isolate (~90 g protein) — always satisfy this.
    """
    def val(k):
        v = fields.get(k)
        if v in (None, "", "None"):
            return None
        try:
            return float(str(v).replace(",", "."))
        except ValueError:
            return None

    fat, carb, prot = val("fat_g"), val("carbohydrates_g"), val("protein_g")
    sat, sug, fib, salt = (val("saturates_g"), val("sugars_g"),
                           val("fiber_g"), val("salt_g"))

    macro_sum = sum(x for x in (fat, carb, prot) if x is not None)
    if macro_sum > 100.5:                     # 0.5 g tolerance for rounding
        return f"fat+carb+protein = {macro_sum:g} g/100 g (>100)"
    if fat is not None and sat is not None and sat > fat + 0.5:
        return "saturated fat exceeds total fat"
    if carb is not None and sug is not None and sug > carb + 0.5:
        return "sugars exceed total carbohydrate"
    # Energy vs macros: 9·fat + 4·carb + 4·protein + 2·fibre should roughly match
    # stated energy. A macro corrupted ×1000 implies far more energy than stated.
    energy = val("energy_kj")
    if energy is not None and energy > 0:
        kcal = 9 * (fat or 0) + 4 * (carb or 0) + 4 * (prot or 0) + 2 * (fib or 0)
        kj_from_macros = kcal * 4.184
        if kj_from_macros > max(energy * 3, energy + 800):
            return (f"macros imply ~{kj_from_macros:.0f} kJ but energy states "
                    f"{energy:g} kJ")
    for k, cap in (("fat_g", 100), ("carbohydrates_g", 100), ("protein_g", 92),
                   ("sugars_g", 100), ("saturates_g", 100), ("fiber_g", 90),
                   ("salt_g", 100), ("energy_kj", 4000)):
        v = val(k)
        if v is not None and v > cap:
            return f"{k} = {v:g} exceeds {cap}/100 g"
    return ""


# ============================================================================
# DETERMINISTIC PARSER 1 — JSON-LD  (application/ld+json)
# ============================================================================

def _iter_jsonld_objects(html: str):
    """Yield every dict found in <script type=application/ld+json> blocks,
    flattening @graph and lists."""
    for m in re.finditer(
        r'<script[^>]*type=[\'"]application/ld\+json[\'"][^>]*>(.*?)</script>',
        html, re.S | re.I
    ):
        block = m.group(1).strip()
        if not block:
            continue
        try:
            data = json.loads(block)
        except Exception:
            # Some sites emit multiple JSON objects concatenated or with stray
            # trailing commas; try a lenient salvage of the first {...}.
            try:
                data = json.loads(re.search(r"\{.*\}", block, re.S).group(0))
            except Exception:
                continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            item = stack.pop()
            if isinstance(item, list):
                stack.extend(item)
                continue
            if not isinstance(item, dict):
                continue
            if "@graph" in item and isinstance(item["@graph"], list):
                stack.extend(item["@graph"])
            yield item


def _type_matches(item: dict, wanted: str) -> bool:
    t = item.get("@type", "")
    if isinstance(t, list):
        return any(wanted.lower() == str(x).lower() for x in t)
    return str(t).lower() == wanted.lower()


def parse_jsonld(html: str, ean: str = "") -> dict:
    """Extract product/nutrition fields from JSON-LD. Numbers only from here."""
    out = _blank_fields()
    gtins = set()
    nutrition_obj = None
    product = None

    for item in _iter_jsonld_objects(html):
        # Collect GTINs from any object.
        for gk in ("gtin", "gtin8", "gtin12", "gtin13", "gtin14", "sku", "mpn"):
            if item.get(gk):
                gtins.add(re.sub(r"\D", "", str(item[gk])))
        if _type_matches(item, "Product") and product is None:
            product = item
        if _type_matches(item, "NutritionInformation"):
            nutrition_obj = item

    if product is not None:
        # brand
        b = product.get("brand")
        if isinstance(b, dict):
            b = b.get("name", "")
        if b and not is_garbage_name(str(b)):
            out["brand"] = str(b).strip()
        # weight / size
        for wk in ("weight", "size"):
            wv = product.get(wk)
            if isinstance(wv, dict):
                wv = wv.get("value") or wv.get("name")
            if wv:
                nw = _to_float(wv)
                if nw is not None:
                    out["net_weight"] = _fmt_num(nw)
                    break
        # country of origin
        origin = product.get("countryOfOrigin")
        if isinstance(origin, dict):
            origin = origin.get("name", "")
        if origin:
            out["place_of_origin"] = str(origin).strip()
        # manufacturer
        man = product.get("manufacturer")
        if isinstance(man, dict):
            if man.get("name"):
                out["manufacturer_name"] = str(man["name"]).strip()
            addr = man.get("address")
            if isinstance(addr, dict):
                parts = [addr.get(k, "") for k in
                         ("streetAddress", "postalCode", "addressLocality",
                          "addressCountry")]
                addr = ", ".join(p for p in parts if p)
            if addr:
                out["manufacturer_address"] = str(addr).strip()
        elif isinstance(man, str) and man:
            out["manufacturer_name"] = man.strip()
        # ingredients may live under a few nonstandard keys
        for ik in ("ingredients", "recipeIngredient"):
            iv = product.get(ik)
            if isinstance(iv, list):
                iv = ", ".join(str(x) for x in iv)
            if iv and len(str(iv)) > 5:
                out["ingredients"] = _clean_text(str(iv))
                break
        # nutrition attached to the product
        if nutrition_obj is None and isinstance(product.get("nutrition"), dict):
            nutrition_obj = product["nutrition"]

    if isinstance(nutrition_obj, dict):
        _apply_schema_nutrition(nutrition_obj, out)

    if gtins:
        out["_gtins"] = gtins  # internal; used for Route-A confirmation
    return out


def _apply_schema_nutrition(n: dict, out: dict) -> None:
    """Map schema.org NutritionInformation onto canonical nutrition keys."""
    energy = n.get("calories") or n.get("energy")
    if energy:
        out["energy_kj"] = _energy_to_kj(energy)
    mapping = {
        "fatContent": "fat_g",
        "saturatedFatContent": "saturates_g",
        "carbohydrateContent": "carbohydrates_g",
        "sugarContent": "sugars_g",
        "proteinContent": "protein_g",
        "fiberContent": "fiber_g",
        "fibreContent": "fiber_g",
    }
    for sk, ck in mapping.items():
        if n.get(sk) not in (None, ""):
            val = _to_float(n[sk])
            if val is not None:
                out[ck] = _fmt_num(val)
    # salt: prefer an explicit salt field, else derive from sodium.
    if n.get("saltContent") not in (None, ""):
        v = _to_float(n["saltContent"])
        if v is not None:
            out["salt_g"] = _fmt_num(v)
    elif n.get("sodiumContent") not in (None, ""):
        out["salt_g"] = _sodium_to_salt(n["sodiumContent"])
    serving = n.get("servingSize")
    if serving:
        out["nutritional_info"] = f"per {str(serving).strip()}"


# ============================================================================
# DETERMINISTIC PARSER 2 — microdata / OpenGraph
# ============================================================================

def parse_microdata_og(html: str) -> dict:
    """Cheap identity/brand/GTIN signals from itemprop + OpenGraph meta tags."""
    out = _blank_fields()
    gtins = set()

    for m in re.finditer(
        r'itemprop=[\'"](gtin\d*|sku|mpn)[\'"][^>]*content=[\'"]([^\'"]+)',
        html, re.I):
        gtins.add(re.sub(r"\D", "", m.group(2)))
    # itemprop on visible elements (value in text is harder; grab content attr)
    m = re.search(r'itemprop=[\'"]brand[\'"][^>]*content=[\'"]([^\'"]+)', html, re.I)
    if m and not is_garbage_name(m.group(1)):
        out["brand"] = m.group(1).strip()

    og_title = _meta(html, "og:title")
    og_desc = _meta(html, "og:description")
    if og_desc and len(og_desc) > 20:
        out["nutritional_info"] = out["nutritional_info"] or ""
    # OpenGraph rarely carries nutrition; it mostly helps Route-B name matching.
    if gtins:
        out["_gtins"] = gtins
    out["_og_title"] = og_title or ""
    out["_og_desc"] = og_desc or ""
    return out


def _meta(html: str, prop: str) -> str:
    m = re.search(
        rf'<meta[^>]+(?:property|name)=[\'"]{re.escape(prop)}[\'"][^>]+content=[\'"]([^\'"]+)',
        html, re.I)
    if not m:
        m = re.search(
            rf'<meta[^>]+content=[\'"]([^\'"]+)[\'"][^>]+(?:property|name)=[\'"]{re.escape(prop)}[\'"]',
            html, re.I)
    return unescape(m.group(1)).strip() if m else ""


# ============================================================================
# DETERMINISTIC PARSER 3 — labelled HTML nutrition table + text paragraphs
# ============================================================================

# Multilingual nutrition labels -> canonical key. Order matters: check
# "saturated" style before plain "fat", "sugars" before "carbohydrate", etc.
_NUTRI_LABELS = [
    ("saturates_g", ["of which saturates", "saturated fat", "gesättigte",
                     "davon gesättigte", "acides gras saturés", "dont acides gras saturés",
                     "di cui acidi grassi saturi", "acidi grassi saturi",
                     "waarvan verzadigd", "verzadigde vetzuren", "grasas saturadas",
                     "de las cuales saturadas"]),
    ("sugars_g", ["of which sugars", "davon zucker", "dont sucres", "di cui zuccheri",
                  "waarvan suikers", "de los cuales azúcares", "azúcares", "sugars",
                  "zucker", "sucres", "zuccheri", "suikers"]),
    ("fat_g", ["fat", "fett", "matières grasses", "grassi", "vetten", "vet",
               "grasas", "lipides"]),
    ("carbohydrates_g", ["carbohydrate", "kohlenhydrate", "glucides", "carboidrati",
                         "koolhydraten", "hidratos de carbono", "carbohidratos"]),
    ("protein_g", ["protein", "eiweiß", "eiweiss", "protéines", "proteine",
                   "eiwitten", "eiwit", "proteínas", "proteinas"]),
    ("fiber_g", ["fibre", "fiber", "ballaststoffe", "fibres", "fibra", "fibre alimentari",
                 "vezels", "vezel", "fibra alimentaria"]),
    ("salt_g", ["salt", "salz", "sel", "sale", "zout", "sal"]),
    ("energy_kj", ["energy", "energie", "énergie", "energia", "valore energetico",
                   "brennwert", "valor energético"]),
]

_INGREDIENT_LABELS = ["ingredients", "zutaten", "ingrédients", "ingredienti",
                      "ingrediënten", "ingredienten", "ingredientes"]
_ALLERGEN_LABELS = ["allergens", "allergene", "allergènes", "allergeni",
                    "allergenen", "alérgenos", "allergy advice", "allergie"]
_MAYCONTAIN_LABELS = ["may contain", "kann spuren", "kann enthalten",
                      "peut contenir", "può contenere", "kan sporen",
                      "puede contener", "traces of", "spuren von"]


def _clean_text(s: str) -> str:
    s = unescape(s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_html_nutrition(html: str) -> dict:
    """Labelled-table + labelled-paragraph fallback parser (uses bs4 if present)."""
    out = _blank_fields()
    if not _BS4_OK:
        return _parse_html_nutrition_regex(html, out)
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return _parse_html_nutrition_regex(html, out)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # ---- Nutrition from <table> rows ----------------------------------------
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for tr in rows:
            cells = [_clean_text(c.get_text(" ")) for c in tr.find_all(["th", "td"])]
            if len(cells) < 2:
                continue
            label = cells[0].lower()
            # Prefer a value column whose header/text references "100"; else 2nd cell.
            value_cell = _pick_per100_cell(cells)
            _assign_nutri_label(label, value_cell, out)

    # ---- Nutrition from definition lists / label:value pairs ----------------
    text_blob = _clean_text(soup.get_text(" "))
    _scan_inline_nutrition(text_blob, out)

    # ---- Ingredients / allergens / may-contain paragraphs -------------------
    _extract_labelled_paragraph(soup, text_blob, out)
    return out


def _pick_per100_cell(cells: list) -> str:
    """Given a row's cells, return the value most likely to be the per-100 figure."""
    # If any cell text mentions 100 alongside a number, that's not it (that's a
    # header). We just take the first cell after the label that contains a digit.
    for c in cells[1:]:
        if re.search(r"\d", c):
            return c
    return cells[1] if len(cells) > 1 else ""


def _assign_nutri_label(label: str, value: str, out: dict) -> None:
    for key, variants in _NUTRI_LABELS:
        if out[key]:
            continue
        if any(v in label for v in variants):
            if key == "energy_kj":
                v = _energy_to_kj(value)
            else:
                fv = _to_float(value)
                v = _fmt_num(fv) if fv is not None else ""
            v = _bounded(key, v)
            if v:
                out[key] = v
            return


def _scan_inline_nutrition(text: str, out: dict) -> None:
    """Catch 'Label 12,3 g' patterns in flat text when there was no table.
    Requires a unit next to the number so stray figures (prices, RDA %, pack
    counts) aren't captured, and clamps to plausible per-100 ranges."""
    low = text.lower()
    for key, variants in _NUTRI_LABELS:
        if out[key]:
            continue
        # Energy may legitimately be unit-suffixed kj/kcal; macros must end in g.
        unit = r'(?:kj|kcal)' if key == "energy_kj" else r'(?:g|mg)'
        for v in variants:
            m = re.search(re.escape(v) + r'[^0-9<]{0,25}(\d[\d.,]*)\s*' + unit, low)
            if m:
                raw = m.group(1)
                if key == "energy_kj":
                    # re-attach the matched unit so kcal converts correctly
                    val = _energy_to_kj(m.group(0))
                else:
                    fv = _to_float(raw)
                    val = _fmt_num(fv) if fv is not None else ""
                val = _bounded(key, val)
                if val:
                    out[key] = val
                break


def _extract_labelled_paragraph(soup, text_blob: str, out: dict) -> None:
    low = text_blob.lower()
    # Generic stops: a new section heading ends the current paragraph.
    common_stops = (r'Nutrition|N\u00e4hrwert|Valeurs nutrit|Valori nutriz|'
                    r'Voedingswaarde|Informaci\u00f3n nutricional|Angaben pro|'
                    r'Durchschnittliche|Nettof\u00fcllmenge|Netto|Aufbewahrung|'
                    r'Verwendung|Hersteller|Kontakt zum|Alle Preise|Preise und|'
                    r'Produkthinweise|Brennwert|Kalorien|Energie|Energy|'
                    r'Valore energetico')
    # ingredients need a real list; allergen / may-contain segments can be as
    # short as a single word ("nuts"), so they get a lower length floor. Each
    # field also stops when the NEXT field's label begins, so an allergen line
    # doesn't swallow the ingredients list that follows it (and vice versa).
    for key, labels, min_len, extra_stop in (
            ("ingredients", _INGREDIENT_LABELS, 8, r'Allerg'),
            ("may_contain", _MAYCONTAIN_LABELS, 3, r'Zutaten|Ingredient|Allerg'),
            ("allergens", _ALLERGEN_LABELS, 3, r'Zutaten|Ingredient|Ingr\u00e9dient|Ingredienti')):
        if out[key]:
            continue
        for lab in labels:
            idx = low.find(lab)
            if idx == -1:
                continue
            # Grab the sentence/segment after the label up to a sensible stop.
            after = text_blob[idx + len(lab): idx + len(lab) + 600]
            after = re.sub(r'^[\s:：\-–—.]+', '', after)
            stop = common_stops + "|" + extra_stop
            seg = re.split(r'(?:\.\s|\n|' + stop + r')', after, 1)[0]
            seg = _clean_text(seg)
            # Strip a repeated inline label prefix ("Zutaten: ...", "Ingredients: ...").
            seg = re.sub(r'^(?:zutaten|ingredients?|ingr\u00e9dients?|ingredienti|'
                         r'ingredi\u00ebnten|ingredienten|ingredientes|enthaltene\s+allergene|'
                         r'allergene?|allergens?|info[s]?)\s*[:：]?\s*', '', seg, flags=re.I)
            # For allergen fields, reject "see the ingredient list" style pointers
            # (they carry no allergen data, e.g. "Siehe Hervorhebungen im ...").
            if key in ("allergens", "may_contain") and re.search(
                    r'\b(siehe|see|voir|zie|vedi|ver|hervorhebung|zutatenverzeichnis|'
                    r'ingredient list|ingredient statement|markering|as highlighted|'
                    r'in bold|fett gedruckt|in grassetto)\b', seg, flags=re.I):
                continue
            if len(seg) >= min_len:
                out[key] = seg[:500]
                break


def _parse_html_nutrition_regex(html: str, out: dict) -> dict:
    """bs4-free fallback: strip tags and scan flat text only."""
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = _clean_text(re.sub(r"<[^>]+>", " ", text))
    _scan_inline_nutrition(text, out)
    _extract_labelled_paragraph(None, text, out)
    return out


# ============================================================================
# PAGE FETCH + ROUTE A/B VERIFICATION
# ============================================================================

def _page_has_ean(html: str, ean: str) -> bool:
    """Route-A test: EAN present in a labelled GTIN context (strong evidence)."""
    stripped = ean.lstrip("0")
    if re.search(rf'(gtin\d*|"sku"|itemprop=["\']gtin)[\s"\':=]+["\']?0*{re.escape(stripped)}',
                 html, re.IGNORECASE):
        return True
    if re.search(rf'(ean|gtin|barcode|strichcode|streepjescode|'
                 rf'codice\s*a\s*barre|c[oó]digo\s*de\s*barras|upc)'
                 rf'[^0-9]{{0,30}}0*{re.escape(stripped)}', html, re.IGNORECASE):
        return True
    return False


def _page_title_bits(html: str) -> str:
    bits = []
    for pat in (r'<meta[^>]+og:title[^>]+content=["\']([^"\']+)',
                r'<title[^>]*>([^<]+)</title>',
                r'<h1[^>]*>(.*?)</h1>'):
        m = re.search(pat, html, re.I | re.S)
        if m:
            bits.append(re.sub(r"<[^>]+>", " ", m.group(1)))
    return _clean_text(" ".join(bits))


def verify_route(html: str, ean: str, anchors: list, gtins: set | None = None,
                  trusted: bool = False) -> str | None:
    """
    Return 'A' (EAN confirmed), 'B' (name match >=80%, no weight conflict), or None.

    Policy: EAN-only and name+EAN matches (Route A) are strong enough to accept
    from ANY domain. A name(+weight)-only match with NO EAN on the page
    (Route B) is comparatively weak evidence, so it is only accepted from a
    `trusted` domain (goldmine retailer or the brand's own site) — an
    untrusted/general-web page that merely shares ~80% of its title tokens
    with the product name is too easily a different SKU/variant and must not
    be treated as verified.
    """
    if _page_has_ean(html, ean):
        return "A"
    if gtins and (ean in gtins or ean.lstrip("0") in {g.lstrip("0") for g in gtins}):
        return "A"
    if not trusted:
        return None
    identity = _page_title_bits(html)
    if identity and anchors:
        ratio = _best_name_ratio(anchors, identity)
        if ratio >= _NAME_MATCH_THRESHOLD and not _weights_conflict(anchors, identity):
            return "B"
    return None


_FETCH_HEADERS = {
    "User-Agent": _UA,
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    # Cover every market we operate in so retailers serve the localised page.
    "Accept-Language": "de,en-US;q=0.9,en;q=0.8,fr;q=0.7,it;q=0.6,es;q=0.5,nl;q=0.4",
    "Accept-Encoding": "gzip, deflate",
}

# Retail product pages (globus, rewe, etc.) inline the ENTIRE site navigation
# mega-menu before the product content, so the nutrition table lives near the
# very end of the HTML. A small read cap silently truncates it away — read
# generously (up to ~4 MB) so the product block always survives.
_MAX_HTML_BYTES = 4_000_000


async def _fetch_html(session, url: str, timeout: int = 15) -> str:
    if not url or not url.startswith("http") or _is_excluded_domain(url):
        return ""
    try:
        async with session.get(url, headers=_FETCH_HEADERS, timeout=timeout,
                               allow_redirects=True) as r:
            if r.status != 200:
                return ""
            raw = await r.content.read(_MAX_HTML_BYTES)
            # Honour the declared charset when it isn't UTF-8 (e.g. digit-eyes
            # and other legacy pages use latin-1), else fall back to utf-8.
            charset = (r.charset or "").lower()
            if charset and charset not in ("utf-8", "utf8"):
                try:
                    return raw.decode(charset, errors="ignore")
                except (LookupError, TypeError):
                    pass
            return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ============================================================================
# JS-RENDER FALLBACK (headless Chromium, via Playwright)
# ============================================================================

# A single browser process is launched lazily and reused for the lifetime of
# the app (launching Chromium per-page would dwarf the cost of everything
# else this pipeline does). _RENDER_LOCK serialises the one-time launch;
# _RENDER_SEM bounds how many pages may be rendering AT ONCE across every
# concurrent extract_food_data() call, which is the number that actually
# controls memory/CPU pressure on the host.
_RENDER_LOCK: asyncio.Lock | None = None
_RENDER_SEM: asyncio.Semaphore | None = None
_PLAYWRIGHT_CTX = None
_RENDER_BROWSER = None
# Set once, permanently, the first time a launch attempt fails or times out.
# Without this, a broken/missing browser on the host gets RE-attempted (and
# re-blocks the shared _RENDER_LOCK) on every single page that wants the
# fallback — under Streamlit's synchronous per-session execution, a single
# hung launch can stall the whole app's response long enough to look like a
# 502 to Render's proxy, however much RAM/CPU the instance has.
_RENDER_UNAVAILABLE = False
_RENDER_LAUNCH_TIMEOUT_S = int(os.environ.get("JS_RENDER_LAUNCH_TIMEOUT_S", "25"))

# Best-effort cookie/consent-banner dismissal. Many EU retailer sites gate
# the underlying page content behind a consent click even after JS has
# otherwise finished rendering; this is a small, non-exhaustive set of the
# most common patterns (OneTrust's id is extremely widely used).
_CONSENT_SELECTORS = (
    "#onetrust-accept-btn-handler",
    "button:has-text('Accept all')", "button:has-text('Accept All')",
    "button:has-text('Accept')", "button:has-text('Alle akzeptieren')",
    "button:has-text('Akzeptieren')", "button:has-text('Tout accepter')",
    "button:has-text('Accetta tutto')", "button:has-text('Accetta')",
    "button:has-text('Alles accepteren')", "button:has-text('Akkoord')",
    "button:has-text('Aceptar todo')", "button:has-text('Aceptar')",
)



# Playwright/Chromium's default automation fingerprint (navigator.webdriver
# = true, --enable-automation banner, etc.) is trivially detected by any
# common WAF/anti-bot layer (Cloudflare, Akamai, DataDome, ...), which then
# serves a challenge/block page INSTEAD of the real content — indistinguishable
# from a genuine JS-rendering failure unless you know to look for it. These
# are the standard, well-established mitigations (not a full stealth suite,
# just the highest-value/lowest-risk ones): a launch flag that removes the
# most obvious automation flag Blink itself exposes, and an init script that
# patches the handful of properties most detection scripts actually check.
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
window.chrome = window.chrome || { runtime: {} };
"""


async def _launch_render_browser():
    """The actual launch sequence, run under a wait_for() timeout by the caller."""
    global _PLAYWRIGHT_CTX
    _PLAYWRIGHT_CTX = await async_playwright().start()
    launch_kwargs = {
        "headless": True,
        "args": ["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                  "--disable-blink-features=AutomationControlled"],
    }
    # Optional override for environments (like sandboxed dev containers) that
    # ship a pre-installed browser at a nonstandard path instead of the one
    # `playwright install chromium` puts under ~/.cache/ms-playwright. Unset
    # in normal deployment — Playwright finds its own installed browser
    # automatically.
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM_PATH")
    if exe:
        launch_kwargs["executable_path"] = exe
    return await _PLAYWRIGHT_CTX.chromium.launch(**launch_kwargs)


async def _get_render_browser():
    """
    Lazily launch (once) and return the shared headless Chromium instance, or
    None if it's unavailable on this host. Bounded by _RENDER_LAUNCH_TIMEOUT_S
    so a broken install can never hang instead of failing; a failure is
    cached in _RENDER_UNAVAILABLE so it's detected ONCE per process, not
    re-attempted (and re-blocking the shared lock) on every subsequent page.
    """
    global _RENDER_LOCK, _RENDER_BROWSER, _RENDER_UNAVAILABLE
    if _RENDER_UNAVAILABLE:
        return None
    if _RENDER_LOCK is None:
        _RENDER_LOCK = asyncio.Lock()
    async with _RENDER_LOCK:
        if _RENDER_UNAVAILABLE:
            return None
        if _RENDER_BROWSER is None:
            try:
                _RENDER_BROWSER = await asyncio.wait_for(
                    _launch_render_browser(), timeout=_RENDER_LAUNCH_TIMEOUT_S)
            except Exception as e:
                _RENDER_UNAVAILABLE = True
                # Visible in Render's runtime logs even though callers only
                # ever see "" — this is the one place worth a hard print,
                # since fetch_html_rendered's fail-soft contract would
                # otherwise swallow this completely silently.
                print(f"[js_render] Chromium unavailable — disabling the "
                      f"JS-render fallback for this process ("
                      f"{type(e).__name__}: {str(e)[:200]}).")
                return None
        return _RENDER_BROWSER


async def fetch_html_rendered(url: str, timeout_ms: int = RENDER_TIMEOUT_MS) -> str:
    """
    Headless-browser fallback fetch for pages whose content is built by
    client-side JavaScript — the "JS/cookie wall" case a plain aiohttp GET
    (_fetch_html) cannot read. This is a FALLBACK ONLY: callers should use it
    exclusively for candidates that already came back unverified or
    data-thin from the fast path, never as the primary fetch, since it is
    roughly an order of magnitude slower and heavier per page.

    Bounded by RENDER_MAX_CONCURRENT (global, across every EAN being
    processed concurrently) so this can't exhaust memory on a small host.
    Returns "" on any failure — same fail-closed contract as _fetch_html.

    DEPLOYMENT NOTE: requires `playwright install chromium` (or
    `--with-deps chromium`) to have been run wherever this app is deployed —
    `pip install playwright` alone only installs the Python client, not the
    browser binary. See requirements.txt / README for the build-step change
    this needs on Render. Fails soft (JS_RENDER_ENABLED / _PLAYWRIGHT_OK
    checks) if that step hasn't been done, so its absence degrades this one
    fallback rather than breaking the app.
    """
    global _RENDER_SEM
    if not (_PLAYWRIGHT_OK and JS_RENDER_ENABLED):
        return ""
    if not url or not url.startswith("http") or _is_excluded_domain(url):
        return ""
    if _RENDER_SEM is None:
        _RENDER_SEM = asyncio.Semaphore(RENDER_MAX_CONCURRENT)

    async with _RENDER_SEM:
        context = None
        try:
            browser = await _get_render_browser()
            if browser is None:
                return ""   # unavailable on this host — _RENDER_UNAVAILABLE is now cached
            context = await browser.new_context(
                user_agent=_UA,
                extra_http_headers={"Accept-Language": _FETCH_HEADERS["Accept-Language"]},
                viewport={"width": 1366, "height": 900},
            )
            context.set_default_timeout(timeout_ms)
            await context.add_init_script(_STEALTH_INIT_SCRIPT)
            page = await context.new_page()
            try:
                await page.goto(url, timeout=timeout_ms, wait_until="networkidle")
            except Exception:
                # networkidle can time out on pages with long-poll/analytics
                # traffic even though the actual product content already
                # rendered — a partially-settled DOM is still far better than
                # giving up, so fall through rather than aborting the fetch.
                pass
            for sel in _CONSENT_SELECTORS:
                try:
                    btn = page.locator(sel).first
                    if await btn.is_visible(timeout=800):
                        await btn.click(timeout=800)
                        await page.wait_for_timeout(400)
                        break
                except Exception:
                    continue
            html = await page.content()
            return (html or "")[:_MAX_HTML_BYTES]
        except Exception:
            return ""
        finally:
            if context is not None:
                try:
                    await context.close()
                except Exception:
                    pass


# ============================================================================
# DETERMINISTIC EXTRACTION FOR ONE PAGE (all three parsers, one pass)
# ============================================================================

_EMBEDDED_JSON_RE = re.compile(
    r'<script[^>]+(?:id=["\']__NEXT_DATA__["\']|type=["\']application/json["\'])'
    r'[^>]*>(.*?)</script>', re.S | re.I)
_STATE_JSON_RE = re.compile(
    r'window\.__(?:NUXT|INITIAL_STATE|APOLLO_STATE|PRELOADED_STATE)__\s*=\s*'
    r'(\{.*?\})\s*(?:;|</script>)', re.S)

# nutriment-ish JSON keys -> canonical field, with a converter.
_JSON_NUTRI_KEYS = {
    "energy-kj_100g": ("energy_kj", lambda x: _bounded("energy_kj", _fmt_num(_to_float(x)))),
    "energy-kj": ("energy_kj", lambda x: _bounded("energy_kj", _fmt_num(_to_float(x)))),
    "energy-kcal_100g": ("energy_kj", lambda x: _bounded("energy_kj", _energy_to_kj(str(x) + " kcal"))),
    "fat_100g": ("fat_g", lambda x: _bounded("fat_g", _fmt_num(_to_float(x)))),
    "saturated-fat_100g": ("saturates_g", lambda x: _bounded("saturates_g", _fmt_num(_to_float(x)))),
    "carbohydrates_100g": ("carbohydrates_g", lambda x: _bounded("carbohydrates_g", _fmt_num(_to_float(x)))),
    "sugars_100g": ("sugars_g", lambda x: _bounded("sugars_g", _fmt_num(_to_float(x)))),
    "proteins_100g": ("protein_g", lambda x: _bounded("protein_g", _fmt_num(_to_float(x)))),
    "fiber_100g": ("fiber_g", lambda x: _bounded("fiber_g", _fmt_num(_to_float(x)))),
    "salt_100g": ("salt_g", lambda x: _bounded("salt_g", _fmt_num(_to_float(x)))),
    "sodium_100g": ("salt_g", lambda x: _bounded("salt_g", _sodium_to_salt(str(x)))),
}


def _walk_json_for_nutrition(node, out: dict, gtins: set) -> None:
    """Depth-first walk of an arbitrary JSON tree pulling nutriment-style keys.
    Handles the state blobs that JS retail SPAs ship in the initial HTML."""
    if isinstance(node, dict):
        for k, v in node.items():
            kl = str(k).lower()
            if kl in _JSON_NUTRI_KEYS and isinstance(v, (str, int, float)):
                key, conv = _JSON_NUTRI_KEYS[kl]
                if not out.get(key):
                    val = conv(v)
                    if val:
                        out[key] = val
            elif kl in ("gtin", "gtin13", "gtin12", "gtin8", "ean", "barcode") and v:
                gtins.add(re.sub(r"\D", "", str(v)))
            elif kl in ("ingredients_text", "ingredientstext", "ingredients") and \
                    isinstance(v, str) and len(v) > 8 and not out.get("ingredients"):
                out["ingredients"] = v.strip()[:500]
            else:
                _walk_json_for_nutrition(v, out, gtins)
    elif isinstance(node, list):
        for item in node:
            _walk_json_for_nutrition(item, out, gtins)


def parse_embedded_json(html: str) -> dict:
    """Extract nutrition/ingredients from __NEXT_DATA__ / state JSON blobs that
    JS-rendered retail pages embed in their initial HTML (site-agnostic)."""
    out = _blank_fields()
    gtins: set = set()
    blobs = _EMBEDDED_JSON_RE.findall(html) + _STATE_JSON_RE.findall(html)
    for blob in blobs[:6]:                       # cap work
        blob = blob.strip()
        if not blob.startswith(("{", "[")):
            continue
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError):
            continue
        _walk_json_for_nutrition(data, out, gtins)
    out["_gtins"] = gtins
    return out


def extract_page_fields(html: str, ean: str) -> dict:
    """Run JSON-LD -> microdata/OG -> embedded-JSON -> HTML-table parsers and
    merge (earlier parser wins per field). Returns canonical field dict."""
    jsonld = parse_jsonld(html, ean)
    micro = parse_microdata_og(html)
    embedded = parse_embedded_json(html)
    table = parse_html_nutrition(html)

    merged = _blank_fields()
    gtins = set()
    for src in (jsonld, micro, embedded, table):     # priority order
        gtins |= src.get("_gtins", set())
        for k in ALL_FIELD_KEYS:
            if not merged[k] and src.get(k):
                merged[k] = src[k]
    merged["_gtins"] = gtins
    merged["_og_title"] = micro.get("_og_title", "")
    merged["_og_desc"] = micro.get("_og_desc", "")
    return merged


def page_visible_text(html: str, limit: int = 20000) -> str:
    """Plain visible text of a page (for the Gemini free-text fallback)."""
    if _BS4_OK:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            return _clean_text(soup.get_text(" "))[:limit]
        except Exception:
            pass
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    return _clean_text(re.sub(r"<[^>]+>", " ", text))[:limit]


# ============================================================================
# GEMINI FREE-TEXT FALLBACK  (text fields only — NEVER numbers, NEVER search)
# ============================================================================

# Only these messy free-text fields may be filled by the LLM fallback.
_GEMINI_FALLBACK_FIELDS = ("ingredients", "allergens", "may_contain",
                           "manufacturer_name", "manufacturer_address",
                           "place_of_origin")


def gemini_freetext_from_page(page_text: str, ean: str, product_name: str,
                              market_code: str, gemini_key: str,
                              want_fields: list) -> dict:
    """
    Last-resort extraction of unstructured TEXT fields from a single verified
    page's text. Fed ONLY the page text, temperature 0, no search. Returns
    {field_key: value|None}. Numbers are never requested here.
    """
    if not (_GENAI_OK and gemini_key and page_text and len(page_text) >= 200):
        return {}
    want = [f for f in want_fields if f in _GEMINI_FALLBACK_FIELDS]
    if not want:
        return {}
    field_lines = "\n".join(f'    "{k}": "value or null",' for k in want)
    prompt = f"""You are reading ONE product page's text. Extract ONLY the fields
below and ONLY from the PAGE TEXT provided. Do NOT search. Do NOT use outside
knowledge. If a field is not clearly present in this text, return null for it —
never guess or infer. Translate text into the {market_code} market language,
verbatim. Product for reference: {product_name} (EAN {ean}).

Return ONLY a JSON object, no prose:
{{
{field_lines}
}}

PAGE TEXT:
\"\"\"{page_text[:16000]}\"\"\"
"""
    try:
        client = genai.Client(api_key=gemini_key)
        resp = client.models.generate_content(
            model=EXTRACTION_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0,
                                                max_output_tokens=1536),
        )
        raw = ""
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if getattr(part, "text", None):
                    raw += part.text
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return {}
        parsed = json.loads(m.group(0))
        out = {}
        for k in want:
            v = parsed.get(k)
            if v in ("", "null", "None", None):
                out[k] = None
            else:
                out[k] = _clean_text(str(v))[:500]
        return out
    except Exception:
        return {}


# ============================================================================
# MARKET-LANGUAGE LOCALISATION FOR ALLERGENS / MAY-CONTAIN
# The selected market is NOT a source filter (discovery is deliberately
# market-agnostic — see _serp_discover). Its only remaining job is choosing
# the OUTPUT LANGUAGE for the allergens / may-contain fields, since those are
# the fields a market's own compliance/labelling process actually reads.
# Ingredients and every other text field are left exactly as extracted —
# translating those risks altering a manufacturer's exact wording.
# ============================================================================

_MARKET_LANGUAGE_NAME = {
    "de": "German", "nl": "Dutch", "fr": "French", "it": "Italian",
    "es": "Spanish", "da": "Danish", "en": "English", "pl": "Polish",
}
_ALLERGEN_LOCALIZE_FIELDS = ("allergens", "may_contain")


def translate_allergen_fields(fields: dict, market: str, gemini_key: str) -> dict:
    """
    Translate ONLY allergens/may_contain into the selected market's language,
    verbatim (temperature 0, no content added/removed/inferred) — a pure
    localisation pass over whatever language the winning source happened to
    be in. Returns `fields` unchanged on any failure (fail-safe, not
    fail-blank: an untranslated value is still correct data).
    """
    if not (_GENAI_OK and gemini_key):
        return fields
    want = {k: fields[k] for k in _ALLERGEN_LOCALIZE_FIELDS if fields.get(k)}
    if not want:
        return fields
    _, hl, _ = _serp_locale(market)
    lang_name = _MARKET_LANGUAGE_NAME.get(hl, "English")
    prompt = f"""Translate ONLY the JSON values below into {lang_name}, verbatim.
Do not add, remove, infer, or otherwise change any information — translate the
language only. If a value is already in {lang_name}, return it unchanged.
Return ONLY a JSON object with the same keys, no prose:
{json.dumps(want, ensure_ascii=False)}
"""
    try:
        client = genai.Client(api_key=gemini_key)
        resp = client.models.generate_content(
            model=EXTRACTION_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=512),
        )
        raw = ""
        if resp.candidates and resp.candidates[0].content and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if getattr(part, "text", None):
                    raw += part.text
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return fields
        translated = json.loads(m.group(0))
        out = dict(fields)
        for k in want:
            v = translated.get(k)
            if v and isinstance(v, str):
                out[k] = _clean_text(v)[:500]
        return out
    except Exception:
        return fields


# ============================================================================
# DISCOVERY LADDER  (used only when candidate_pages is not supplied — e.g. tests)
# Production app.py passes already-discovered pages, so this stays dormant there.
# ============================================================================

_GO_UPC_LOCK: asyncio.Lock | None = None
_GO_UPC_LAST = [0.0]
_SERP_SEM: asyncio.Semaphore | None = None
_SERP_LOOP = None


async def fetch_go_upc(session, ean: str, go_upc_key: str) -> dict | None:
    """Tier 1A — Go-UPC barcode lookup (2 req/s). Registry data only."""
    global _GO_UPC_LOCK
    if not go_upc_key:
        return None
    if _GO_UPC_LOCK is None:
        _GO_UPC_LOCK = asyncio.Lock()
    async with _GO_UPC_LOCK:
        now = asyncio.get_event_loop().time()
        wait = 0.5 - (now - _GO_UPC_LAST[0])
        if wait > 0:
            await asyncio.sleep(wait)
        _GO_UPC_LAST[0] = asyncio.get_event_loop().time()
        try:
            async with session.get(
                f"https://go-upc.com/api/v1/code/{ean}",
                headers={"Authorization": f"Bearer {go_upc_key}"},
                timeout=10) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                p = data.get("product") or {}
                if not p or not p.get("name"):
                    return None
                if not barcode_matches(str(data.get("code", "")), ean):
                    return None
                return {
                    "name": p.get("name", ""),
                    "brand": p.get("brand", ""),
                    "ingredients": (p.get("ingredients") or {}).get("text", "")
                    if isinstance(p.get("ingredients"), dict) else "",
                }
        except Exception:
            return None


# Per-market Google localisation so SERP results match what a human in that
# market sees (gl=country, hl=interface language, google_domain=local Google).
_SERP_LOCALE = {
    "DE": ("de", "de", "google.de"),   "AT": ("at", "de", "google.at"),
    "NL": ("nl", "nl", "google.nl"),   "BE": ("be", "nl", "google.be"),
    "FR": ("fr", "fr", "google.fr"),   "IT": ("it", "it", "google.it"),
    "ES": ("es", "es", "google.es"),   "DK": ("dk", "da", "google.dk"),
    "UK": ("uk", "en", "google.co.uk"), "PL": ("pl", "pl", "google.pl"),
}


def _serp_locale(market: str):
    return _SERP_LOCALE.get((market or "").upper(),
                            ((market or "world").lower(), "en", "google.com"))


async def _serp_discover(session, ean, market, serp_key,
                         ground_truth="", name="", limit=10) -> list:
    """
    SERP-FIRST discovery ladder. Runs live web search BEFORE any reliance on the
    EAN databases, in priority order:
        1. Goldmine-targeted   ({market goldmine site} {EAN})      [Search 1, LOCALISED]
        2. Bare-EAN organic    ({EAN})                             [google-the-barcode, WORLDWIDE]
        3. Name + EAN          ({name} {EAN})                      [WORLDWIDE]
        3b. Goldmine + name    ({goldmine} "{name}")                [LOCALISED]
        4. Name only, exact    ("{name}")                          [WORLDWIDE]
        5. Name only, natural  ({name})                            [WORLDWIDE]

    The selected market is NOT a source filter. It only localises the two
    goldmine-site: queries (1, 3b) so the user's own market's trusted retailers
    get a fair, language-appropriate shot. Queries 2-5 carry no site:
    restriction, so they deliberately run WITHOUT gl/hl/google_domain bias —
    a trusted (goldmine/brand) page in ANY country must be discoverable for a
    product sold anywhere (e.g. a BE-market EAN whose only clean data page is
    on an Italian retailer). Geo-biasing those queries silently suppressed
    exactly that case. Market only ever controls output LANGUAGE for
    allergens/may-contain (see translate_allergen_fields), never eligibility.

    Query 5 exists because query 4's exact-phrase quoting is often too
    strict: `ground_truth` is frequently a translated/localised product name
    (e.g. a Dutch description of an Italian product) that won't appear
    verbatim on ANY retailer's page, so the exact-phrase query returns
    nothing even when the product is trivially findable — exactly what a
    human typing the same words into Google *without* quotes gets right,
    because Google's normal relevance ranking (word order/synonym flexible)
    is what actually finds it, not a literal substring match.

    Collects organic result URLs in order (deduped, excluded domains removed).
    The Go-UPC registry contributes ONLY the `name` seed here — it never gates
    or replaces this.
    """
    if not serp_key or not ean:
        return []
    gl, hl, gd = _serp_locale(market)
    gm = (GOLDMINE.get((market or "").upper(), "") or "").strip()
    nm = (strip_pack_notation(name or ground_truth) or name or ground_truth or "").strip()

    # (query, localize) — localize=True adds the market's gl/hl/google_domain;
    # localize=False runs worldwide, unbiased.
    queries = []
    if gm:
        queries.append((f"{gm} {ean}", True))              # 1. goldmine + EAN
    queries.append((str(ean), False))                       # 2. bare EAN
    if nm:
        queries.append((f"{nm} {ean}", False))              # 3. name + EAN
        if gm:
            queries.append((f'{gm} "{nm}"', True))          # 3b. goldmine + name
        queries.append((f'"{nm}"', False))                  # 4. name only, exact phrase
        queries.append((nm, False))                         # 5. name only, natural (no quotes)

    urls: list = []

    def _add(link):
        if (link and link.startswith("http")
                and not _is_excluded_domain(link) and link not in urls):
            urls.append(link)

    for q, localize in queries:
        params = {"engine": "google", "q": q, "num": 10, "api_key": serp_key}
        if localize:
            params.update({"gl": gl, "hl": hl, "google_domain": gd})
        data = await _serp_get(session, params)
        for res in (data or {}).get("organic_results", [])[:6]:
            _add(res.get("link", ""))
        if len(urls) >= limit:
            break
    return urls[:limit]


async def _serp_ean_organic(session, ean, market, serp_key,
                            ground_truth="", limit=8) -> list:
    """
    One (or two) localised Google *organic* searches for the bare EAN — this is
    the "search the barcode on Google" step a human does, and the step that
    surfaces the retailer pages (onfos, delitea, swedishness, ...) that actually
    carry the nutrition table. App-side discovery is name-resolution-gated and
    skips this once a registry names the product, which is the main coverage
    hole; running it here guarantees data-bearing pages are always considered.

    Returns a de-duplicated list of candidate URLs (excluded domains removed).
    """
    if not serp_key or not ean:
        return []
    gl, hl, gd = _serp_locale(market)
    queries = [str(ean)]
    if ground_truth:
        gt = strip_pack_notation(ground_truth) or ground_truth
        queries.append(f"{gt} {ean}")
    urls: list = []
    for q in queries:
        data = await _serp_get(session, {
            "engine": "google", "q": q, "gl": gl, "hl": hl,
            "google_domain": gd, "num": 10, "api_key": serp_key})
        for res in (data or {}).get("organic_results", [])[:limit]:
            link = res.get("link", "")
            if (link and link.startswith("http")
                    and not _is_excluded_domain(link) and link not in urls):
                urls.append(link)
        if len(urls) >= limit:
            break
    return urls[:limit]


async def _serp_get(session, params: dict, timeout: int = 15) -> dict | None:
    global _SERP_SEM, _SERP_LOOP
    loop = asyncio.get_running_loop()
    if _SERP_SEM is None or _SERP_LOOP is not loop:
        _SERP_SEM = asyncio.Semaphore(8)
        _SERP_LOOP = loop
    async with _SERP_SEM:
        for attempt in range(3):
            try:
                async with session.get("https://serpapi.com/search",
                                       params=params, timeout=timeout) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status == 429 or r.status >= 500:
                        if attempt < 2:
                            await asyncio.sleep(3 * (attempt + 1))
                            continue
                    return None
            except asyncio.TimeoutError:
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                return None
            except Exception:
                return None
    return None


async def discover_candidate_pages(session, ean, ground_truth, market,
                                   registry_name, keys, go_upc_data) -> tuple:
    """
    Run the ladder (Attempts 2-5) and return (candidate_urls, resolved_name,
    ean_verified). Tier 1A/1B name resolution is folded in via go_upc_data /
    EAN-Search. SerpAPI is used for Attempts 2-5. Order is UNCHANGED from the
    original fetch_basic_info.
    """
    serp_key = keys.get("serp", "")
    ean_token = keys.get("ean_search", "")
    gl = market.lower()
    market_upper = market.upper()
    urls: list = []
    name = registry_name or ""
    ean_verified = bool(go_upc_data)  # Tier 1A registry hit == barcode-verified

    clean_gt = strip_pack_notation(ground_truth) if ground_truth else ground_truth

    # Tier 1B: EAN-Search (only if Go-UPC gave nothing)
    if not go_upc_data and ean_token:
        try:
            u = (f"https://api.ean-search.org/api?token={ean_token}"
                 f"&op=barcode-lookup&ean={ean}&format=json")
            async with session.get(u, timeout=6) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and isinstance(data, list) and "error" not in data[0]:
                        rb = str(data[0].get("barcode", data[0].get("ean", "")))
                        if barcode_matches(rb, ean):
                            ean_verified = True
                            cn = data[0].get("name", "")
                            if cn and not is_garbage_name(cn):
                                name = name or cn
        except Exception:
            pass

    def _add(link):
        if link and link.startswith("http") and not _is_excluded_domain(link) \
                and link not in urls:
            urls.append(link)

    if not serp_key:
        return urls, name, ean_verified

    # Attempt 2: brand-site
    if clean_gt:
        brand = extract_brand(clean_gt)
        if brand:
            data = await _serp_get(session, {"q": f"{brand} {ean}", "gl": gl,
                                             "api_key": serp_key})
            for res in (data or {}).get("organic_results", [])[:5]:
                link = res.get("link", "")
                dom = _domain_of(link)
                if _is_excluded_domain(link):
                    continue
                if brand_matches_domain(brand, dom):
                    _add(link)
                    if not name:
                        name = res.get("title", "").split("-")[0].split("|")[0].strip()
                    break

    # Attempt 2.5: combined name + EAN
    if clean_gt:
        data = await _serp_get(session, {"q": f"{clean_gt} {ean}", "gl": gl,
                                         "api_key": serp_key})
        for res in (data or {}).get("organic_results", [])[:5]:
            link = res.get("link", "")
            if _is_excluded_domain(link):
                continue
            snip = res.get("snippet", "") or ""
            if ean in snip or ean.lstrip("0") in snip:
                _add(link)
                if not name:
                    t = res.get("title", "").split("-")[0].split("|")[0].strip()
                    if not is_garbage_name(t):
                        name = t
                ean_verified = True

    # Attempt 2.6: data aggregators (openfoodfacts/codecheck/das-ist-drin/...)
    # These carry clean structured tables and are the usual bare-EAN top hits.
    data = await _serp_get(session, {"q": f"{_DATA_AGGREGATOR_QUERY} {ean}",
                                     "gl": gl, "api_key": serp_key})
    for res in (data or {}).get("organic_results", [])[:5]:
        _add(res.get("link", ""))

    # Attempt 3: goldmine
    goldmine = f"{GOLDMINE.get(market_upper, '')} OR {GLOBAL_SITES}".strip(" OR")
    data = await _serp_get(session, {"q": f"{goldmine} {ean}", "gl": gl,
                                     "api_key": serp_key})
    org = (data or {}).get("organic_results", [])
    if org and not name:
        cand = org[0].get("title", "").split("-")[0].split("|")[0].strip()
        if not is_garbage_name(cand):
            name = cand
    for res in org[:4]:
        _add(res.get("link", ""))

    # Attempt 4: bare GTIN
    if len(urls) < 4:
        data = await _serp_get(session, {"q": str(ean), "gl": gl, "api_key": serp_key})
        for res in (data or {}).get("organic_results", [])[:4]:
            if _result_matches_gt(res, clean_gt):
                _add(res.get("link", ""))

    # Attempt 5: name-based fallback
    if clean_gt and len(urls) < 4:
        gm = GOLDMINE.get(market_upper, "").strip()
        for q in ([f'{gm} "{clean_gt}"'] if gm else []) + [f'"{clean_gt}"']:
            data = await _serp_get(session, {"q": q, "gl": gl, "api_key": serp_key})
            for res in (data or {}).get("organic_results", [])[:4]:
                link = res.get("link", "")
                if _is_excluded_domain(link):
                    continue
                title = res.get("title", "").split("-")[0].split("|")[0].strip()
                if is_garbage_name(title):
                    continue
                _add(link)
                if not name:
                    name = title
            if len(urls) >= 4:
                break

    return urls, name, ean_verified


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def extract_food_data(session, ean, ground_truth, market, registry_name,
                            keys, *, go_upc_data=None, candidate_pages=None,
                            ean_verified_hint=False):
    """
    Verify-then-read food extraction. See module docstring for the return shape.

    session         : aiohttp.ClientSession
    ean             : queried GTIN string
    ground_truth    : user-supplied product description (may be "")
    market          : market code, e.g. "DE"
    registry_name   : Go-UPC / EAN-Search product name ("" if none)
    keys            : {"serp","gemini","ean_search","go_upc"}
    go_upc_data     : pre-fetched Go-UPC dict (optional; else module fetches it)
    candidate_pages : pages already discovered by app.py (optional; else the
                      module runs the ladder itself)
    ean_verified_hint : whether the barcode was already registry-verified
    """
    diag: list = []
    gemini_key = keys.get("gemini", "")
    clean_gt = strip_pack_notation(ground_truth) if ground_truth else ground_truth

    # Tier 1A registry (shared from app.py if provided).
    if go_upc_data is None and keys.get("go_upc"):
        go_upc_data = await fetch_go_upc(session, ean, keys["go_upc"])
    reg_name = registry_name or (go_upc_data or {}).get("name", "") or ""
    ean_verified = bool(ean_verified_hint or go_upc_data)
    market_upper = (market or "").upper()

    # ---- Candidate discovery (SERP-FIRST) -----------------------------------
    # Live web search leads. Open Food Facts is NOT queried at all. The product
    # name used to sharpen the goldmine/name queries comes from the Go-UPC
    # registry (if available) or the user's own input — never a food database.
    resolved_name = reg_name
    seed_name = reg_name or ""

    if candidate_pages:
        app_candidates = [u for u in candidate_pages
                          if u and u.startswith("http") and not _is_excluded_domain(u)]
    else:
        app_candidates, resolved_name2, ev2 = await discover_candidate_pages(
            session, ean, ground_truth, market, reg_name, keys, go_upc_data)
        resolved_name = resolved_name or resolved_name2
        ean_verified = ean_verified or ev2

    serp_candidates = []
    if keys.get("serp"):
        serp_candidates = await _serp_discover(
            session, ean, market, keys["serp"], ground_truth, name=seed_name)
        diag.append(f"SERP-first discovery: {len(serp_candidates)} page(s) "
                    f"({', '.join(_domain_of(u) for u in serp_candidates[:5])}).")

    # SERP-discovered pages FIRST, then app/image-pipeline pages as backup.
    candidates = serp_candidates + [u for u in app_candidates
                                    if u not in serp_candidates]
    diag.append(f"{len(candidates)} candidate page(s) "
                f"({len(serp_candidates)} from SERP, "
                f"{len(app_candidates)} from app/images).")

    # De-dupe by domain, keep order, cap.
    anchors = [a for a in (clean_gt, reg_name) if a]
    brand_for_trust = extract_brand(clean_gt or "") or extract_brand(reg_name or "")
    seen_dom, ordered = set(), []
    for u in candidates:
        dom = _domain_of(u)
        if dom and dom not in seen_dom:
            seen_dom.add(dom)
            ordered.append(u)
    ordered = ordered[:MAX_CANDIDATES]

    # ---- Verify + read each candidate ---------------------------------------
    read_pages: list = []       # dicts: {url, route, fields, text}
    source_routes: dict = {}
    verified_any = False
    readable_any = False
    renders_done = 0   # bounds total JS-render fallback calls for this product

    for url in ordered:
        if len([p for p in read_pages if p["fields_nonempty"]]) >= MAX_READ_PAGES:
            break
        html = await _fetch_html(session, url)
        trusted = is_trusted_page(url, brand_for_trust)
        page_fields = extract_page_fields(html, ean) if html else _blank_fields()
        route = (verify_route(html, ean, anchors, page_fields.get("_gtins"),
                              trusted=trusted) if html else None)
        nonempty = any(page_fields.get(k) for k in ALL_FIELD_KEYS)

        # JS-render fallback: the plain fetch either couldn't verify this
        # candidate at all (route is None — could be a bot-blocked or
        # cookie-gated response) or verified it but came back data-thin (a
        # classic client-side-rendered page). Retry with a real browser and
        # re-run the SAME deterministic parsers on the rendered DOM — bounded
        # per-product (MAX_RENDERS_PER_PRODUCT) so a page-full of thin
        # candidates can't blow up per-EAN latency.
        rendered_used = False
        if (route is None or not nonempty) and renders_done < MAX_RENDERS_PER_PRODUCT:
            rendered_html = await fetch_html_rendered(url)
            if rendered_html and rendered_html != html:
                renders_done += 1
                r_fields = extract_page_fields(rendered_html, ean)
                r_route = verify_route(rendered_html, ean, anchors,
                                       r_fields.get("_gtins"), trusted=trusted)
                r_nonempty = any(r_fields.get(k) for k in ALL_FIELD_KEYS)
                if r_route is not None and (route is None or r_nonempty):
                    page_fields, route, nonempty, html = \
                        r_fields, r_route, r_nonempty, rendered_html
                    rendered_used = True

        if route is None:
            reason = "fetch failed" if not html else "neither EAN nor trusted-name match"
            diag.append(f"skip ({reason}): {_domain_of(url)}")
            continue
        verified_any = True
        source_routes[url] = route
        # A page is "readable" for grading purposes when the deterministic
        # parsers pulled ANY real field out of it. A JS/cookie-walled shell
        # returns markup with no product data -> nonempty is False -> the page
        # counts as verified-but-not-readable (handled below).
        if nonempty:
            readable_any = True
        read_pages.append({
            "url": url, "route": route, "fields": page_fields,
            "text": page_visible_text(html), "trusted": trusted,
            "readable": nonempty, "fields_nonempty": nonempty,
        })
        diag.append(f"verified Route {route} ({'trusted' if trusted else 'untrusted'}, "
                    f"{'read' if nonempty else 'thin'}"
                    f"{', JS-rendered' if rendered_used else ''}): {_domain_of(url)}")

    # ---- Rescue pass: if the pages so far gave thin nutrition, run ONE -------
    # aggregator-targeted bare-EAN search (codecheck / fddb / digit-eyes / ...)
    # and read those clean-data pages too. Costs a SERP call only when the
    # common path came up short, so bulk cost stays low.
    def _nutri_hits(pages):
        got = set()
        for p in pages:
            for k in _NUTRI_KEYS:
                if p["fields"].get(k):
                    got.add(k)
        return len(got)

    # Rescue search when the REAL pages read so far are data-thin. (OFF is not
    # counted — it can't be displayed, so it can't rescue the row.)
    if keys.get("serp") and _nutri_hits(read_pages) < 4:
        diag.append("thin data -> aggregator rescue search")
        try:
            # Aggregators (codecheck/fddb/etc.) are pan-market — no gl bias.
            rescue = await _serp_get(session, {
                "q": f"{_DATA_AGGREGATOR_QUERY} {ean}", "api_key": keys["serp"]})
            rescue_urls = [r.get("link", "") for r in
                           (rescue or {}).get("organic_results", [])[:6]]
            for url in rescue_urls:
                if len([p for p in read_pages if p["fields_nonempty"]]) >= MAX_READ_PAGES:
                    break
                dom = _domain_of(url)
                if not url or _is_excluded_domain(url) or dom in seen_dom:
                    continue
                seen_dom.add(dom)
                html = await _fetch_html(session, url)
                rescue_trusted = is_trusted_page(url, brand_for_trust)
                pf = extract_page_fields(html, ean) if html else _blank_fields()
                route = (verify_route(html, ean, anchors, pf.get("_gtins"),
                                      trusted=rescue_trusted) if html else None)
                ne = any(pf.get(k) for k in ALL_FIELD_KEYS)

                rescue_rendered = False
                if (route is None or not ne) and renders_done < MAX_RENDERS_PER_PRODUCT:
                    rendered_html = await fetch_html_rendered(url)
                    if rendered_html and rendered_html != html:
                        renders_done += 1
                        r_pf = extract_page_fields(rendered_html, ean)
                        r_route = verify_route(rendered_html, ean, anchors,
                                               r_pf.get("_gtins"), trusted=rescue_trusted)
                        r_ne = any(r_pf.get(k) for k in ALL_FIELD_KEYS)
                        if r_route is not None and (route is None or r_ne):
                            pf, route, ne, html = r_pf, r_route, r_ne, rendered_html
                            rescue_rendered = True

                if route is None:
                    continue
                verified_any = True
                source_routes[url] = route
                if ne:
                    readable_any = True
                read_pages.append({"url": url, "route": route, "fields": pf,
                                   "text": page_visible_text(html),
                                   "trusted": rescue_trusted,
                                   "readable": ne, "fields_nonempty": ne})
                diag.append(f"rescue Route {route} ({'trusted' if rescue_trusted else 'untrusted'}, "
                            f"{'JS-rendered, ' if rescue_rendered else ''}"
                            f"{'read' if ne else 'thin'}): {dom}")
        except Exception:
            pass

    # ---- Open Food Facts is NOT used ----------------------------------------
    # OFF is never queried, fetched, or displayed. Every value in the results
    # table comes from a real retailer/manufacturer page discovered via SERP.
    # (Its domains are also in _EXCLUDED_DOMAIN_TOKENS as a safety net.)

    # ---- Rank pages: trusted (goldmine/brand) domains first — they set the ----
    # row's grade and should win field-value contests; then EAN-verified
    # (Route A) before name-matched (Route B, only possible on a trusted page);
    # aggregators (clean structured tables) break remaining ties.
    def _rank(p):
        return (0 if p["trusted"] else 1,
                {"A": 0, "B": 1}.get(p["route"], 2),
                0 if _is_aggregator(p["url"]) else 1)
    read_pages.sort(key=_rank)

    # ---- Fill fields ACROSS verified pages (highest-trust wins per field) ----
    fields = _blank_fields()
    used_urls: list = []
    for p in read_pages:
        contributed = False
        for k in ALL_FIELD_KEYS:
            if not fields[k] and p["fields"].get(k):
                fields[k] = p["fields"][k]
                contributed = True
        if contributed or p["fields_nonempty"]:
            if p["url"] not in used_urls:
                used_urls.append(p["url"])

    # ---- Go-UPC / EAN-Search registry: SEED ONLY, never a field source -------
    # Policy: every value shown in the table must trace back to a verified,
    # SerpAPI-discovered page. The barcode registries are used upstream only
    # to resolve an identity (name/brand token, ean_verified flag) that seeds
    # the SerpAPI queries — they must NOT write ingredients/brand/etc. directly
    # into `fields`, since that data would have no source link behind it.
    registry_used = False

    # ---- Gemini last-resort for messy TEXT fields still missing --------------
    gemini_used = False
    if gemini_key:
        missing_text = [k for k in _GEMINI_FALLBACK_FIELDS if not fields[k]]
        if missing_text:
            # Feed the richest readable verified page's text only.
            best = next((p for p in read_pages if p["readable"] and p["text"]), None)
            if best:
                got = await asyncio.to_thread(
                    gemini_freetext_from_page, best["text"], ean,
                    resolved_name or reg_name, market, gemini_key, missing_text)
                for k, v in (got or {}).items():
                    if v and not fields[k]:
                        fields[k] = v
                        gemini_used = True

    # ---- Market-language localisation for allergens / may-contain -----------
    # Market selection controls ONLY this — never which sources are eligible.
    if gemini_key:
        fields = translate_allergen_fields(fields, market, gemini_key)

    # ---- Last-line plausibility backstop (decimal / parse corruption) --------
    # The physics check is now the sole corruption guard (OFF is not queried).
    # If the nutrition is physically impossible, blank it rather than shipping
    # garbage as "verified". Text fields (ingredients, brand) are kept.
    nutrition_flag = _implausible_nutrition(fields)
    if nutrition_flag:
        for k in _NUTRI_KEYS:
            fields[k] = ""
        diag.append(f"Nutrition blanked — implausible ({nutrition_flag}).")

    # ---- Grade + provenance: trust-tier + cross-source corroboration --------
    # High   — at least one shown field was read from a TRUSTED page (the
    #          brand's own site, or a tier-1/goldmine retailer). EAN-only,
    #          name+EAN, or name+weight-only matches are all acceptable
    #          evidence there (see verify_route / is_trusted_page).
    # Medium — no trusted page contributed, but a field's value is
    #          independently corroborated by >= MEDIUM_CORROBORATION_MIN
    #          distinct non-tier-1 domains (each necessarily EAN-verified —
    #          untrusted domains can only reach Route A; see verify_route).
    # Low    — only a single non-tier-1 source backs the data, uncorroborated.
    live = [p for p in read_pages if p["fields_nonempty"]]
    has_trusted_contributor = any(p["trusted"] for p in live)

    def _field_corroborated(k: str) -> bool:
        val = fields.get(k)
        if not val:
            return False
        agreeing_domains = {
            _domain_of(p["url"]) for p in live
            if not p["trusted"] and _values_agree(k, val, p["fields"].get(k, ""))
        }
        return len(agreeing_domains) >= MEDIUM_CORROBORATION_MIN

    has_corroborated_field = any(_field_corroborated(k) for k in ALL_FIELD_KEYS)

    if readable_any:
        status = "success"
        if has_trusted_contributor:
            reliability = "H"
        elif has_corroborated_field:
            reliability = "M"
        else:
            reliability = "L"
        if gemini_used:
            provenance = "Verified page + fallback ⚠️"
        else:
            provenance = "Verified page"
    elif verified_any:
        # Pages verified (Route A/B) but none readable — JS/cookie wall etc.
        status = "page_not_readable"
        reliability = "L"
        provenance = "Page not readable"
    else:
        status = "no_source"
        reliability = "L"
        provenance = "No verified source"

    # Corrupt/implausible nutrition was blanked above — never present such a row
    # as trustworthy, whatever the source was.
    if nutrition_flag:
        reliability = "L"
        provenance = f"Nutrition failed plausibility check ⚠️ ({nutrition_flag})"

    diag.append(f"status={status} reliability={reliability} "
                f"pages_read={sum(1 for p in read_pages if p['fields_nonempty'])}")

    # Trust-ordered source links = the pages that actually contributed data.
    source_links = used_urls or [p["url"] for p in read_pages]

    return {
        "fields": {k: fields[k] for k in ALL_FIELD_KEYS},
        "source_links": source_links,
        "source_routes": {u: source_routes.get(u, "") for u in source_links},
        "provenance": provenance,
        "reliability": reliability,
        "status": status,
        "name": resolved_name or reg_name or "",
        "ean_verified": ean_verified,
        "retailer_urls": ordered,
        "diagnostics": diag,
    }
