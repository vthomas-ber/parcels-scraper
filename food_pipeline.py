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


# ============================================================================
# CONFIG (kept in sync with app.py — see module docstring)
# ============================================================================

EXTRACTION_MODEL = "gemini-2.5-flash"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")

# Max verified pages to read and fill across (richest/highest-trust wins).
MAX_READ_PAGES = 4
# Max candidate pages to fetch+verify before giving up (keeps HTTP bounded).
MAX_CANDIDATES = 8
_NAME_MATCH_THRESHOLD = 0.80

GOLDMINE = {
    "FR": ("site:carrefour.fr OR site:auchan.fr OR site:coursesu.com "
           "OR site:leclerc.fr OR site:intermarche.fr OR site:monoprix.fr"),
    "UK": ("site:ocado.com OR site:waitrose.com OR site:asda.com "
           "OR site:tesco.com OR site:sainsburys.co.uk OR site:morrisons.com"),
    "NL": "site:ah.nl OR site:jumbo.com OR site:plus.nl OR site:dirk.nl",
    "BE": ("site:delhaize.be OR site:colruyt.be OR site:carrefour.be "
           "OR site:spar.be OR site:lidl.be"),
    "DE": ("site:rewe.de OR site:edeka.de OR site:kaufland.de "
           "OR site:dm.de OR site:rossmann.de OR site:metro.de "
           "OR site:budni.de OR site:ecoinform.de"),
    "AT": ("site:billa.at OR site:spar.at OR site:gurkerl.at "
           "OR site:hofer.at OR site:mpreis.at"),
    "DK": ("site:nemlig.com OR site:matsmart.dk OR site:rema1000.dk "
           "OR site:coop.dk OR site:salling.dk"),
    "IT": ("site:carrefour.it OR site:conad.it OR site:coop.it "
           "OR site:esselunga.it OR site:eurospin.it OR site:lidl.it"),
    "ES": ("site:carrefour.es OR site:mercadona.es OR site:dia.es "
           "OR site:alcampo.es OR site:eroski.es OR site:lidl.es"),
    "PL": ("site:carrefour.pl OR site:auchan.pl OR site:frisco.pl "
           "OR site:lidl.pl OR site:kaufland.pl"),
}
GLOBAL_SITES = "site:billigkaffee.eu OR site:fivestartrading-holland.eu"

# Approved / goldmine domains are the "prime read targets" — pages on these
# rank higher when filling fields across multiple verified pages.
_GOLDMINE_DOMAINS = frozenset({
    "ocado.com", "waitrose.com", "asda.com", "tesco.com", "sainsburys.co.uk",
    "morrisons.com", "carrefour.fr", "auchan.fr", "coursesu.com", "leclerc.fr",
    "intermarche.fr", "monoprix.fr", "carrefour.it", "conad.it", "coop.it",
    "esselunga.it", "eurospin.it", "lidl.it", "rewe.de", "edeka.de",
    "kaufland.de", "dm.de", "rossmann.de", "metro.de", "budni.de",
    "ecoinform.de", "ah.nl", "jumbo.com", "plus.nl", "dirk.nl", "delhaize.be",
    "colruyt.be", "carrefour.be", "spar.be", "lidl.be", "billa.at", "spar.at",
    "gurkerl.at", "hofer.at", "mpreis.at", "nemlig.com", "rema1000.dk",
    "coop.dk", "salling.dk", "carrefour.es", "mercadona.es", "dia.es",
    "alcampo.es", "eroski.es", "lidl.es", "carrefour.pl", "auchan.pl",
    "frisco.pl", "lidl.pl", "kaufland.pl",
})

_EXCLUDED_DOMAIN_TOKENS = (
    "amazon.", "ebay.", "openfoodfacts.", "aliexpress.", "alibaba.",
    "barcodelookup.", "go-upc.",
)

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
        # Dot may be an EU thousands separator ("1.980" -> 1980) rather than a
        # decimal point. Treat as thousands ONLY when it matches the strict
        # grouped pattern; "18.5" / "1.5" keep the dot as a decimal point.
        if re.fullmatch(r'\d{1,3}(\.\d{3})+', num):
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
    return f"{v:.2f}".rstrip("0").rstrip(".")


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
            if v:
                out[key] = v
            return


def _scan_inline_nutrition(text: str, out: dict) -> None:
    """Catch 'Label 12,3 g' patterns in flat text when there was no table."""
    low = text.lower()
    for key, variants in _NUTRI_LABELS:
        if out[key]:
            continue
        for v in variants:
            m = re.search(re.escape(v) + r'[^0-9<]{0,25}(<?\s*\d[\d.,]*\s*(?:kj|kcal|g|mg|mcg)?)',
                          low)
            if m:
                raw = m.group(1)
                if key == "energy_kj":
                    val = _energy_to_kj(raw)
                else:
                    fv = _to_float(raw)
                    val = _fmt_num(fv) if fv is not None else ""
                if val:
                    out[key] = val
                break


def _extract_labelled_paragraph(soup, text_blob: str, out: dict) -> None:
    low = text_blob.lower()
    # ingredients need a real list; allergen / may-contain segments can be as
    # short as a single word ("nuts"), so they get a lower length floor.
    for key, labels, min_len in (
            ("ingredients", _INGREDIENT_LABELS, 8),
            ("may_contain", _MAYCONTAIN_LABELS, 3),
            ("allergens", _ALLERGEN_LABELS, 3)):
        if out[key]:
            continue
        for lab in labels:
            idx = low.find(lab)
            if idx == -1:
                continue
            # Grab the sentence/segment after the label up to a sensible stop.
            after = text_blob[idx + len(lab): idx + len(lab) + 600]
            after = re.sub(r'^[\s:：\-–—.]+', '', after)
            seg = re.split(r'(?:\.\s|\n|Nutrition|Nährwerte|Valeurs|Valori|'
                           r'Voedingswaarde|Información nutricional)', after, 1)[0]
            seg = _clean_text(seg)
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


def verify_route(html: str, ean: str, anchors: list, gtins: set | None = None) -> str | None:
    """Return 'A' (EAN confirmed), 'B' (name match >=80%, no weight conflict), or None."""
    if _page_has_ean(html, ean):
        return "A"
    if gtins and (ean in gtins or ean.lstrip("0") in {g.lstrip("0") for g in gtins}):
        return "A"
    identity = _page_title_bits(html)
    if identity and anchors:
        ratio = _best_name_ratio(anchors, identity)
        if ratio >= _NAME_MATCH_THRESHOLD and not _weights_conflict(anchors, identity):
            return "B"
    return None


async def _fetch_html(session, url: str, timeout: int = 10) -> str:
    if not url or not url.startswith("http") or _is_excluded_domain(url):
        return ""
    try:
        async with session.get(url, headers={"User-Agent": _UA}, timeout=timeout) as r:
            if r.status != 200:
                return ""
            raw = await r.content.read(700_000)
            return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


# ============================================================================
# DETERMINISTIC EXTRACTION FOR ONE PAGE (all three parsers, one pass)
# ============================================================================

def extract_page_fields(html: str, ean: str) -> dict:
    """Run JSON-LD -> microdata/OG -> HTML-table parsers and merge (earlier
    parser wins per field). Returns canonical field dict (+ _gtins/_og_*)."""
    jsonld = parse_jsonld(html, ean)
    micro = parse_microdata_og(html)
    table = parse_html_nutrition(html)

    merged = _blank_fields()
    gtins = set()
    for src in (jsonld, micro, table):           # priority order
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

    # ---- Candidate discovery -------------------------------------------------
    resolved_name = reg_name
    if candidate_pages:
        candidates = [u for u in candidate_pages
                      if u and u.startswith("http") and not _is_excluded_domain(u)]
        diag.append(f"Using {len(candidates)} candidate page(s) from app.py.")
    else:
        candidates, resolved_name2, ev2 = await discover_candidate_pages(
            session, ean, ground_truth, market, reg_name, keys, go_upc_data)
        resolved_name = resolved_name or resolved_name2
        ean_verified = ean_verified or ev2
        diag.append(f"Ladder discovered {len(candidates)} candidate page(s).")

    # De-dupe by domain, keep order, cap.
    anchors = [a for a in (clean_gt, reg_name) if a]
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

    for url in ordered:
        if len([p for p in read_pages if p["fields_nonempty"]]) >= MAX_READ_PAGES:
            break
        html = await _fetch_html(session, url)
        if not html:
            diag.append(f"unreadable (fetch failed): {_domain_of(url)}")
            continue
        page_fields = extract_page_fields(html, ean)
        route = verify_route(html, ean, anchors, page_fields.get("_gtins"))
        if route is None:
            diag.append(f"skip (neither EAN nor name match): {_domain_of(url)}")
            continue
        verified_any = True
        source_routes[url] = route
        # A page is "readable" for grading purposes when the deterministic
        # parsers pulled ANY real field out of it. A JS/cookie-walled shell
        # returns markup with no product data -> nonempty is False -> the page
        # counts as verified-but-not-readable (handled below).
        nonempty = any(page_fields.get(k) for k in ALL_FIELD_KEYS)
        if nonempty:
            readable_any = True
        read_pages.append({
            "url": url, "route": route, "fields": page_fields,
            "text": page_visible_text(html),
            "readable": nonempty, "fields_nonempty": nonempty,
        })
        diag.append(f"verified Route {route} ({'read' if nonempty else 'thin'}): "
                    f"{_domain_of(url)}")

    # ---- Rank pages: Route A before B; goldmine before other; then order ----
    def _rank(p):
        return (0 if p["route"] == "A" else 1,
                0 if _is_goldmine(p["url"]) else 1)
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

    # ---- Registry (Go-UPC) first-pass fill for genuinely-absent text fields --
    registry_used = False
    if go_upc_data:
        if not fields["ingredients"] and go_upc_data.get("ingredients"):
            fields["ingredients"] = _clean_text(go_upc_data["ingredients"])[:500]
            registry_used = True
        if not fields["brand"] and go_upc_data.get("brand"):
            fields["brand"] = _clean_text(go_upc_data["brand"])
            registry_used = True

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

    # ---- Grade + provenance --------------------------------------------------
    has_route_a = any(p["route"] == "A" for p in read_pages if p["fields_nonempty"])
    has_route_b = any(p["route"] == "B" for p in read_pages if p["fields_nonempty"])

    if readable_any:
        status = "success"
        reliability = "H" if has_route_a else "M"
        if registry_used or gemini_used:
            provenance = "Verified page + fallback ⚠️"
        else:
            provenance = "Verified page"
    elif verified_any:
        # Pages verified (Route A/B) but none readable — JS/cookie wall etc.
        status = "page_not_readable"
        reliability = "L"
        provenance = ("Registry only — page not readable ⚠️" if registry_used
                      else "Page not readable")
    else:
        status = "no_source"
        reliability = "L"
        provenance = ("Registry only — no verified page ⚠️" if registry_used
                      else "No verified source")

    # A name-only (Route-B) match can never be H even if flagged as such above.
    if reliability == "H" and not has_route_a:
        reliability = "M"

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
