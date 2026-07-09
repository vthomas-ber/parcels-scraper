"""Deterministic verified-page food-data pipeline.

This module deliberately does not import Streamlit and does not use Gemini / Google
Search grounding for food-data discovery. The workflow is:

1. Resolve registry identity from Go-UPC, then EAN-Search only if Go-UPC is empty.
2. Discover candidate product pages with SerpAPI using the existing ladder.
3. Verify pages by Route A (EAN on page) or Route B (>=80% product-name token match
   against user input / registry name, with no clear weight mismatch).
4. Read food fields from the verified page HTML only, deterministic first:
   JSON-LD -> microdata/OpenGraph -> labelled HTML nutrition/attribute tables.
5. Use Gemini only as an optional no-search fallback for messy free-text fields,
   fed only the fetched page text.
"""

from __future__ import annotations

import asyncio
import html as html_lib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except Exception:  # pragma: no cover - production dependency guard
    BeautifulSoup = None  # type: ignore
    BS4_OK = False

try:
    from google import genai
    from google.genai import types
    GENAI_OK = True
except Exception:  # pragma: no cover - optional fallback only
    genai = None  # type: ignore
    types = None  # type: ignore
    GENAI_OK = False

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

MAX_FETCH_BYTES = 900_000
MAX_CANDIDATES = 18
MAX_VERIFIED_READS = 6
NAME_MATCH_THRESHOLD = 0.80

EXCLUDED_DOMAIN_TOKENS: Tuple[str, ...] = (
    "amazon.",
    "ebay.",
    "openfoodfacts.",
    "aliexpress.",
    "alibaba.",
)

GOLDMINE: Dict[str, str] = {
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
           "OR site:rema1000.dk OR site:coop.dk OR site:salling.dk"),
    "IT": ("site:carrefour.it OR site:conad.it OR site:coop.it "
           "OR site:esselunga.it OR site:eurospin.it OR site:lidl.it"),
    "ES": ("site:carrefour.es OR site:mercadona.es OR site:dia.es "
           "OR site:alcampo.es OR site:eroski.es OR site:lidl.es"),
    "PL": ("site:carrefour.pl OR site:auchan.pl OR site:frisco.pl "
           "OR site:lidl.pl OR site:kaufland.pl"),
}
GLOBAL_SITES = "site:billigkaffee.eu OR site:fivestartrading-holland.eu"

GOLDMINE_DOMAINS: Dict[str, Tuple[str, ...]] = {
    "UK": ("ocado.com", "waitrose.com", "asda.com", "tesco.com", "sainsburys.co.uk", "morrisons.com"),
    "FR": ("carrefour.fr", "auchan.fr", "coursesu.com", "leclerc.fr", "intermarche.fr", "monoprix.fr"),
    "IT": ("carrefour.it", "conad.it", "coop.it", "esselunga.it", "eurospin.it", "lidl.it"),
    "DE": ("rewe.de", "edeka.de", "kaufland.de", "dm.de", "rossmann.de", "metro.de", "budni.de", "ecoinform.de"),
    "NL": ("ah.nl", "jumbo.com", "plus.nl", "dirk.nl"),
    "BE": ("delhaize.be", "colruyt.be", "carrefour.be", "spar.be", "lidl.be"),
    "AT": ("billa.at", "spar.at", "gurkerl.at", "hofer.at", "mpreis.at"),
    "DK": ("nemlig.com", "matsmart.dk", "rema1000.dk", "coop.dk", "salling.dk"),
    "ES": ("carrefour.es", "mercadona.es", "dia.es", "alcampo.es", "eroski.es", "lidl.es"),
    "PL": ("carrefour.pl", "auchan.pl", "frisco.pl", "lidl.pl", "kaufland.pl"),
}

# Output field names mirror the previous app.py food schema to keep wiring stable.
FOOD_FIELDS: Tuple[str, ...] = (
    "product_name", "brand", "uom", "packaging", "fragile_item",
    "net_weight", "gross_weight", "organic_product", "net_weight_customer_facing",
    "ingredients", "allergens", "may_contain", "nutritional_info",
    "manufacturer_name", "manufacturer_address", "place_of_origin",
    "organic_certification_id", "energy_kj", "fat_g", "saturates_g",
    "carbohydrates_g", "sugars_g", "protein_g", "fiber_g", "salt_g",
)

NUMERIC_FIELDS = {
    "energy_kj", "fat_g", "saturates_g", "carbohydrates_g", "sugars_g",
    "protein_g", "fiber_g", "salt_g", "net_weight", "gross_weight",
}

LLM_FALLBACK_FIELDS: Tuple[str, ...] = (
    "ingredients", "allergens", "may_contain", "manufacturer_name",
    "manufacturer_address", "place_of_origin", "organic_certification_id",
)

NULLISH = {"", "null", "none", "n/a", "na", "-", "–", "—", "tbd", "k.a.", "keine angabe"}

NUTRIENT_LABELS = {
    "energy_kj": (
        "energy", "energie", "énergie", "energia", "energía", "energi", "wartość energetyczna",
    ),
    "fat_g": (
        "fat", "fett", "matières grasses", "materias grasas", "grasas", "grassi", "vet", "fedt", "tłuszcz",
    ),
    "saturates_g": (
        "saturates", "saturated fat", "of which saturates", "davon gesättigte", "gesättigte fettsäuren",
        "dont acides gras saturés", "ácidos grasos saturados", "grassi saturi", "verzadigde vetzuren",
        "heraf mættede", "kwasy tłuszczowe nasycone",
    ),
    "carbohydrates_g": (
        "carbohydrate", "carbohydrates", "kohlenhydrate", "glucides", "hidratos de carbono", "carboidrati",
        "koolhydraten", "kulhydrat", "węglowodany",
    ),
    "sugars_g": (
        "sugars", "sugar", "davon zucker", "zucker", "dont sucres", "azúcares", "zuccheri",
        "suikers", "sukkerarter", "cukry",
    ),
    "protein_g": (
        "protein", "proteins", "eiweiß", "eiweiss", "protéines", "proteínas", "proteine", "eiwitten",
        "protein", "białko",
    ),
    "fiber_g": (
        "fibre", "fiber", "ballaststoffe", "fibres", "fibra", "vezels", "kostfibre", "błonnik",
    ),
    "salt_g": (
        "salt", "salz", "sel", "sale", "sal", "zout", "sól",
    ),
}

LABEL_ALIASES = {
    "ingredients": (
        "ingredients", "ingredient", "zutaten", "ingrédients", "ingredientes", "ingredienti", "ingrediënten",
        "składniki", "ingredienser", "sammansättning",
    ),
    "allergens": (
        "allergens", "allergen", "allergene", "allergènes", "allergeni", "alérgenos", "allergenen",
        "contains", "enthält", "contient", "contiene", "zawiera",
    ),
    "may_contain": (
        "may contain", "may also contain", "kann enthalten", "kann spuren", "spuren von",
        "peut contenir", "puede contener", "può contenere", "kan sporen bevatten", "może zawierać",
    ),
    "manufacturer_name": (
        "manufacturer", "producer", "hersteller", "hergestellt von", "fabriquant", "fabricant",
        "producteur", "produttore", "fabricante", "producent",
    ),
    "manufacturer_address": (
        "manufacturer address", "address", "adresse", "anschrift", "kontakt", "contact", "indirizzo", "dirección", "adres",
    ),
    "place_of_origin": (
        "origin", "country of origin", "place of origin", "herkunft", "ursprung", "origine", "origen", "provenienza", "pochodzenie",
    ),
    "net_weight_customer_facing": (
        "net weight", "net content", "net quantity", "nettofüllmenge", "füllmenge", "inhalt", "poids net",
        "contenu net", "peso neto", "peso netto", "netto inhoud", "nettogewicht", "massa netto",
    ),
    "organic_certification_id": (
        "organic certification", "bio", "öko", "oekologisch", "eko", "organic", "certification",
    ),
}

PACKAGING_PATTERNS = [
    ("Glass jar", r"\b(glass jar|jar|glas|glasflasche|weckglas)\b"),
    ("Bottle", r"\b(bottle|flasche|bouteille|botella|bottiglia|fles)\b"),
    ("Can", r"\b(can|tin|dose|boîte|lata|lattina|blik)\b"),
    ("Box", r"\b(box|carton|karton|boîte|scatola|doos)\b"),
    ("Bag", r"\b(bag|pouch|beutel|sachet|sac|bolsa|busta|zak)\b"),
    ("Wrapper", r"\b(wrapper|wrap|flowpack|barquette|verpackung)\b"),
    ("Tub", r"\b(tub|cup|becher|pot|vasetto)\b"),
]

ORG_CERT_RE = re.compile(r"\b(?:[A-Z]{2}-)?(?:BIO|ÖKO|OKO|EKO|ECO|ORG)-\d{2,4}\b", re.I)


@dataclass
class CandidatePage:
    url: str
    title: str = ""
    snippet: str = ""
    attempt: str = ""
    rank: int = 0


@dataclass
class VerifiedPage:
    url: str
    final_url: str
    route: str  # A or B
    html: str
    text: str
    title: str
    attempt: str
    domain: str
    fields: Dict[str, Any] = field(default_factory=dict)
    score: int = 0
    readable: bool = False
    reason: str = ""


def _is_excluded_domain(url: str) -> bool:
    if not url:
        return False
    u = url.lower()
    return any(tok in u for tok in EXCLUDED_DOMAIN_TOKENS)


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _is_goldmine_domain(url: str, market: str) -> bool:
    dom = _domain(url)
    if not dom:
        return False
    return any(dom == d or dom.endswith("." + d) for d in GOLDMINE_DOMAINS.get(market.upper(), ()))


def _dedupe_append(candidates: List[CandidatePage], url: str, title: str = "", snippet: str = "", attempt: str = "") -> None:
    if not url or not url.startswith("http") or _is_excluded_domain(url):
        return
    if any(c.url == url for c in candidates):
        return
    candidates.append(CandidatePage(url=url, title=title or "", snippet=snippet or "", attempt=attempt, rank=len(candidates)))


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = html_lib.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _soup_text(soup) -> str:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return _clean_text(soup.get_text(" "))


def _sig_tokens(s: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", (s or "").lower())
    return {t for t in cleaned.split() if len(t) > 3 and not t.isdigit()}


def _name_match_ratio(anchor: str, text: str) -> float:
    tokens = _sig_tokens(anchor)
    if not tokens:
        return 0.0
    hay = (text or "").lower()
    return sum(1 for t in tokens if t in hay) / len(tokens)


def _best_name_ratio(anchors: Sequence[str], text: str) -> float:
    return max((_name_match_ratio(a, text) for a in anchors if a), default=0.0)


def _strip_pack_notation(text: str) -> str:
    if not text:
        return ""
    # 4x200ml -> keep 200ml, remove pack count noise.
    text = re.sub(r"\b\d+\s*[xX×]\s*(\d+(?:[.,]\d+)?\s*(?:g|gr|kg|ml|cl|l|oz|lb))\b", r"\1", text)
    text = re.sub(r"\((?:\d+\s*)?(?:pack|pcs?|pieces|st(?:ück)?|x)\)", " ", text, flags=re.I)
    return _clean_text(text)


def _extract_brand(ground_truth: str) -> str:
    if not ground_truth:
        return ""
    cleaned = _strip_pack_notation(ground_truth)
    tokens = cleaned.split()
    stop = {"the", "and", "with", "for", "bio", "organic", "vegan", "gluten", "free"}
    usable = []
    for t in tokens[:3]:
        bare = re.sub(r"[^A-Za-zÀ-ž0-9&'-]", "", t)
        if len(bare) < 2 or bare.lower() in stop or re.search(r"\d", bare):
            continue
        usable.append(bare)
    return usable[0] if usable else ""


def _brand_matches_domain(brand: str, domain: str) -> bool:
    if not brand or not domain:
        return False
    b = re.sub(r"[^a-z0-9]", "", brand.lower())
    d = re.sub(r"[^a-z0-9]", "", domain.lower())
    return len(b) >= 3 and b in d


def _ean_variants(ean: str) -> set[str]:
    e = str(ean or "").strip()
    stripped = e.lstrip("0") or e
    return {v for v in {e, stripped, e.zfill(12), e.zfill(13), e.zfill(14)} if v}


def barcode_matches(returned_barcode: str, queried_ean: str) -> bool:
    if not returned_barcode or not queried_ean:
        return False
    return str(returned_barcode).strip().lstrip("0") == str(queried_ean).strip().lstrip("0")


def _extract_weight_hint(text: str) -> Optional[Tuple[float, str, str]]:
    """Return normalized (amount, unit, display) where unit is g or ml."""
    if not text:
        return None
    m = re.search(r"\b\d+\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(kg|g|gr|ml|cl|l|oz|lb)\b", text, re.I)
    if not m:
        m = re.search(r"\b(\d+(?:[.,]\d+)?)\s*(kg|g|gr|ml|cl|l|oz|lb)\b", text, re.I)
    if not m:
        return None
    amount = float(m.group(1).replace(",", "."))
    unit_raw = m.group(2).lower()
    unit = "g"
    if unit_raw == "kg":
        amount *= 1000
        unit = "g"
    elif unit_raw in ("g", "gr"):
        unit = "g"
    elif unit_raw == "l":
        amount *= 1000
        unit = "ml"
    elif unit_raw == "cl":
        amount *= 10
        unit = "ml"
    elif unit_raw == "ml":
        unit = "ml"
    elif unit_raw == "oz":
        amount *= 28.3495
        unit = "g"
    elif unit_raw == "lb":
        amount *= 453.592
        unit = "g"
    amount = round(amount, 2)
    display = f"{int(amount) if amount.is_integer() else amount:g}{unit}"
    return amount, unit, display


def _weights_conflict(anchors: Sequence[str], identity_text: str, body_text: str = "") -> bool:
    page_w = _extract_weight_hint(identity_text) or _extract_weight_hint(body_text[:3000])
    if not page_w:
        return False
    for anchor in anchors:
        aw = _extract_weight_hint(anchor)
        if not aw:
            continue
        if aw[1] != page_w[1]:
            return True
        # Allow 15% tolerance for rounding / customer-facing formatting.
        base = max(abs(aw[0]), 1.0)
        if abs(aw[0] - page_w[0]) / base > 0.15:
            return True
    return False


def _real_value(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and math.isnan(v))
    s = str(v).strip()
    return bool(s) and s.lower() not in NULLISH


def _to_str(v: Any) -> str:
    if isinstance(v, list):
        return ", ".join(_to_str(x) for x in v if _real_value(x))
    if isinstance(v, dict):
        if _real_value(v.get("name")):
            return _to_str(v.get("name"))
        return _clean_text(" ".join(f"{k}: {_to_str(val)}" for k, val in v.items() if _real_value(val)))
    return _clean_text(str(v)) if _real_value(v) else ""


def _parse_number(value: Any, prefer_kj: bool = False) -> str:
    if value is None:
        return ""
    s = _to_str(value)
    if not s:
        return ""
    if prefer_kj:
        m_kj = re.search(r"([<>~≈]?\s*\d+(?:[.,]\d+)?)\s*k\s*j\b", s, re.I)
        if m_kj:
            return m_kj.group(1).replace(" ", "").replace(",", ".")
        # Deliberately do not convert kcal to kJ; the requested field is kJ and
        # food numbers should come from the source, not an inferred conversion.
        return ""
    m = re.search(r"([<>~≈]?\s*\d+(?:[.,]\d+)?)", s)
    return m.group(1).replace(" ", "").replace(",", ".") if m else ""


def _parse_weight_to_fields(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    w = _extract_weight_hint(text)
    if w:
        out["net_weight"] = f"{int(w[0]) if float(w[0]).is_integer() else w[0]:g}"
        out["uom"] = w[1]
        out["net_weight_customer_facing"] = w[2]
    return out


async def _serp_get(session, serp_key: str, query: str, market: str, timeout: int = 15) -> Dict[str, Any]:
    if not serp_key or not query:
        return {}
    try:
        async with session.get(
            "https://serpapi.com/search",
            params={"q": query, "gl": market.lower(), "api_key": serp_key},
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        ) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"_error": resp.status}
    except Exception as exc:
        return {"_error": str(exc)}


async def _fetch_go_upc(session, ean: str, go_upc_key: str) -> Optional[Dict[str, Any]]:
    if not go_upc_key:
        return None
    try:
        async with session.get(
            f"https://go-upc.com/api/v1/code/{ean}",
            headers={"Authorization": f"Bearer {go_upc_key}", "User-Agent": USER_AGENT},
            timeout=10,
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if not barcode_matches(str(data.get("code", "")), ean):
                return None
            product = data.get("product") or {}
            if not product:
                return None
            return {
                "name": product.get("name") or "",
                "brand": product.get("brand") or "",
                "description": product.get("description") or "",
                "ingredients": ((product.get("ingredients") or {}).get("text", "")
                                if isinstance(product.get("ingredients"), dict) else ""),
                "category": product.get("category") or "",
                "category_path": product.get("categoryPath") or [],
                "specs": dict(product.get("specs") or []),
            }
    except Exception:
        return None


async def _fetch_ean_search(session, ean: str, token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    try:
        url = f"https://api.ean-search.org/api?token={token}&op=barcode-lookup&ean={ean}&format=json"
        async with session.get(url, timeout=7, headers={"User-Agent": USER_AGENT}) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            if not data or not isinstance(data, list) or "error" in data[0]:
                return None
            item = data[0]
            returned = str(item.get("barcode", item.get("ean", "")))
            if not barcode_matches(returned, ean):
                return None
            return {"name": item.get("name", ""), "brand": item.get("categoryName", ""), "image": item.get("image", "")}
    except Exception:
        return None


def _organic_results(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not data or not isinstance(data, dict):
        return []
    return [r for r in data.get("organic_results", []) if isinstance(r, dict)]


async def _discover_candidates(
    session,
    ean: str,
    ground_truth: str,
    market: str,
    registry_name: str,
    serp_key: str,
    diag: List[str],
) -> List[CandidatePage]:
    candidates: List[CandidatePage] = []
    clean_gt = _strip_pack_notation(ground_truth) if ground_truth else ""
    anchor_name = registry_name or clean_gt
    brand = _extract_brand(clean_gt or registry_name)

    # Attempt 2: brand-site lookup.
    if serp_key and brand:
        diag.append(f"🔍 Attempt 2 brand-site query: {brand} + EAN")
        data = await _serp_get(session, serp_key, f"{brand} {ean}", market)
        if data.get("_error"):
            diag.append(f"❌ SERP_FAILED Attempt 2: {data.get('_error')}")
        for i, res in enumerate(_organic_results(data)[:6]):
            link = res.get("link", "")
            if _brand_matches_domain(brand, _domain(link)):
                _dedupe_append(candidates, link, res.get("title", ""), res.get("snippet", ""), "Attempt 2 brand-site")
        # Keep non-marketplace top brand query results as candidates too; some brand domains
        # do not contain the exact brand token in the host (e.g. parent-company sites).
        for i, res in enumerate(_organic_results(data)[:4]):
            _dedupe_append(candidates, res.get("link", ""), res.get("title", ""), res.get("snippet", ""), "Attempt 2 brand-site")

    # Attempt 2.5: combined name + EAN query.
    if serp_key and (clean_gt or registry_name):
        q_name = clean_gt or registry_name
        diag.append("🔍 Attempt 2.5 combined name + EAN query")
        data = await _serp_get(session, serp_key, f"{q_name} {ean}", market)
        if data.get("_error"):
            diag.append(f"❌ SERP_FAILED Attempt 2.5: {data.get('_error')}")
        for res in _organic_results(data)[:8]:
            _dedupe_append(candidates, res.get("link", ""), res.get("title", ""), res.get("snippet", ""), "Attempt 2.5 name+EAN")

    # Attempt 3: goldmine tier-1 retailers, prioritized as read targets.
    if serp_key:
        goldmine = f"{GOLDMINE.get(market.upper(), '')} OR {GLOBAL_SITES}".strip(" OR")
        if goldmine:
            diag.append("🔍 Attempt 3 goldmine/tier-1 retailers")
            data = await _serp_get(session, serp_key, f"{goldmine} {ean}", market)
            if data.get("_error"):
                diag.append(f"❌ SERP_FAILED Attempt 3: {data.get('_error')}")
            for res in _organic_results(data)[:10]:
                _dedupe_append(candidates, res.get("link", ""), res.get("title", ""), res.get("snippet", ""), "Attempt 3 goldmine")

    # Attempt 4: bare GTIN.
    if serp_key:
        diag.append("🔍 Attempt 4 bare-GTIN fallback")
        data = await _serp_get(session, serp_key, str(ean), market)
        if data.get("_error"):
            diag.append(f"❌ SERP_FAILED Attempt 4: {data.get('_error')}")
        for res in _organic_results(data)[:8]:
            # Keep relevance guard when user supplied a name; verification still happens on page.
            hay = f"{res.get('title', '')} {res.get('snippet', '')}"
            if clean_gt and _best_name_ratio([clean_gt], hay) < 0.30 and str(ean) not in hay:
                continue
            _dedupe_append(candidates, res.get("link", ""), res.get("title", ""), res.get("snippet", ""), "Attempt 4 bare-GTIN")

    # Attempt 5: name-based fallback.
    if serp_key and (clean_gt or registry_name):
        q_name = clean_gt or registry_name
        diag.append("🔍 Attempt 5 name-based fallback")
        queries = []
        goldmine = GOLDMINE.get(market.upper(), "").strip()
        if goldmine:
            queries.append(f'{goldmine} "{q_name}"')
        queries.append(f'"{q_name}"')
        for q in queries:
            data = await _serp_get(session, serp_key, q, market)
            if data.get("_error"):
                diag.append(f"❌ SERP_FAILED Attempt 5: {data.get('_error')}")
            for res in _organic_results(data)[:8]:
                _dedupe_append(candidates, res.get("link", ""), res.get("title", ""), res.get("snippet", ""), "Attempt 5 name fallback")
            if len(candidates) >= MAX_CANDIDATES:
                break

    # Prime goldmine read targets inside the same discovery order by sorting only within
    # comparable rank bands: market tier-1 pages beat generic pages from the same ladder depth.
    candidates.sort(key=lambda c: (c.rank // 10, 0 if _is_goldmine_domain(c.url, market) else 1, c.rank))
    return candidates[:MAX_CANDIDATES]


async def _fetch_html(session, url: str) -> Tuple[str, str]:
    try:
        async with session.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"},
            timeout=12,
            allow_redirects=True,
        ) as resp:
            ctype = (resp.headers.get("content-type") or "").lower()
            if resp.status >= 400:
                return "", str(resp.url)
            raw = await resp.content.read(MAX_FETCH_BYTES)
            if raw and ("text" in ctype or "html" in ctype or not ctype):
                return raw.decode("utf-8", errors="ignore"), str(resp.url)
            return raw.decode("utf-8", errors="ignore"), str(resp.url)
    except Exception:
        return "", url


def _iter_jsonld_objects(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if "@graph" in obj:
            yield from _iter_jsonld_objects(obj.get("@graph"))
        yield obj
        for v in obj.values():
            if isinstance(v, (dict, list)):
                yield from _iter_jsonld_objects(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_jsonld_objects(item)


def _jsonld_blocks(soup) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text(" ")
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            # Some sites stuff multiple JSON objects or HTML entities into scripts.
            cleaned = html_lib.unescape(raw)
            try:
                parsed = json.loads(cleaned)
            except Exception:
                continue
        out.extend([o for o in _iter_jsonld_objects(parsed) if isinstance(o, dict)])
    return out


def _object_type(obj: Dict[str, Any]) -> str:
    t = obj.get("@type", "")
    if isinstance(t, list):
        return " ".join(str(x) for x in t).lower()
    return str(t).lower()


def _extract_identity_text(soup, jsonld: List[Dict[str, Any]], fallback_title: str = "") -> str:
    bits: List[str] = []
    for obj in jsonld:
        if "product" in _object_type(obj):
            bits.extend([_to_str(obj.get("name")), _to_str(obj.get("brand")), _to_str(obj.get("description"))[:200]])
    for sel in [
        ("meta", {"property": "og:title"}),
        ("meta", {"name": "twitter:title"}),
        ("meta", {"property": "og:description"}),
    ]:
        tag = soup.find(sel[0], attrs=sel[1])
        if tag and tag.get("content"):
            bits.append(tag.get("content"))
    if soup.title and soup.title.string:
        bits.append(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        bits.append(h1.get_text(" "))
    if fallback_title:
        bits.append(fallback_title)
    return _clean_text(" ".join(x for x in bits if x))


def _html_contains_ean(html: str, jsonld: List[Dict[str, Any]], ean: str) -> bool:
    variants = _ean_variants(ean)
    # Structured GTIN keys.
    for obj in jsonld:
        for key, val in obj.items():
            if str(key).lower().startswith("gtin") or str(key).lower() in {"barcode", "ean", "upc", "sku"}:
                if any(barcode_matches(_to_str(val), v) or v in _to_str(val) for v in variants):
                    return True
    # Labeled or bare occurrence in fetched page. Route A is defined as EAN present on page.
    for v in variants:
        if not v:
            continue
        if re.search(rf"(ean|gtin|barcode|strichcode|streepjescode|upc|codice\s*a\s*barre|c[oó]digo\s*de\s*barras)[^0-9]{{0,50}}0*{re.escape(v.lstrip('0'))}", html, re.I):
            return True
        if v in html:
            return True
    return False


def _page_title(soup, fallback: str = "") -> str:
    for attrs in ({"property": "og:title"}, {"name": "twitter:title"}):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return _clean_text(tag.get("content"))
    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)
    h1 = soup.find("h1")
    if h1:
        return _clean_text(h1.get_text(" "))
    return fallback or ""


async def _verify_candidate(session, cand: CandidatePage, ean: str, anchors: Sequence[str]) -> Optional[VerifiedPage]:
    if not BS4_OK:
        return None
    html, final_url = await _fetch_html(session, cand.url)
    if not html or len(html) < 200:
        return None
    soup = BeautifulSoup(html, "html.parser")
    jsonld = _jsonld_blocks(soup)
    title = _page_title(soup, cand.title)
    identity = _extract_identity_text(soup, jsonld, cand.title)
    text = _soup_text(BeautifulSoup(html, "html.parser"))

    route = None
    reason = ""
    if _html_contains_ean(html, jsonld, ean):
        route = "A"
        reason = "EAN present on page"
    else:
        ratio = _best_name_ratio(anchors, identity)
        if ratio >= NAME_MATCH_THRESHOLD and not _weights_conflict(anchors, identity, text):
            route = "B"
            reason = f"name token match {ratio:.0%}, no weight conflict"

    if not route:
        return None
    readable = len(text) >= 500 or bool(jsonld)
    return VerifiedPage(
        url=cand.url,
        final_url=final_url,
        route=route,
        html=html,
        text=text[:50000],
        title=title,
        attempt=cand.attempt,
        domain=_domain(final_url or cand.url),
        readable=readable,
        reason=reason,
    )


def _set_field(fields: Dict[str, Any], key: str, value: Any) -> None:
    if key not in FOOD_FIELDS:
        return
    if not _real_value(value):
        return
    if key in NUMERIC_FIELDS:
        val = _parse_number(value, prefer_kj=(key == "energy_kj")) if key != "net_weight" and key != "gross_weight" else _parse_number(value)
        if not val:
            return
        fields[key] = val
    else:
        fields[key] = _to_str(value)


def _extract_from_jsonld(soup, html: str) -> Dict[str, Any]:
    jsonld = _jsonld_blocks(soup)
    fields: Dict[str, Any] = {}
    for obj in jsonld:
        typ = _object_type(obj)
        if "product" not in typ and "nutrition" not in typ:
            continue
        if "product" in typ:
            _set_field(fields, "product_name", obj.get("name"))
            _set_field(fields, "brand", obj.get("brand"))
            _set_field(fields, "ingredients", obj.get("ingredients") or obj.get("ingredient"))
            _set_field(fields, "manufacturer_name", obj.get("manufacturer"))
            _set_field(fields, "place_of_origin", obj.get("countryOfOrigin") or obj.get("areaServed"))
            weight_candidate = obj.get("weight") or obj.get("size") or obj.get("netWeight")
            if _real_value(weight_candidate):
                fields.update(_parse_weight_to_fields(_to_str(weight_candidate)))
            for prop in obj.get("additionalProperty", []) if isinstance(obj.get("additionalProperty"), list) else []:
                pname = _to_str(prop.get("name") if isinstance(prop, dict) else "").lower()
                pval = prop.get("value") if isinstance(prop, dict) else None
                if any(x in pname for x in ("ingredient", "zutaten", "ingrédients")):
                    _set_field(fields, "ingredients", pval)
                elif any(x in pname for x in ("allergen", "contains", "enthält")):
                    _set_field(fields, "allergens", pval)
                elif any(x in pname for x in ("net", "weight", "inhalt", "füllmenge", "content")):
                    fields.update(_parse_weight_to_fields(_to_str(pval)))
                elif "origin" in pname or "herkunft" in pname or "origine" in pname:
                    _set_field(fields, "place_of_origin", pval)
            nutrition = obj.get("nutrition")
            if isinstance(nutrition, dict):
                _extract_nutrition_object(nutrition, fields)
        if "nutrition" in typ:
            _extract_nutrition_object(obj, fields)
    return fields


def _extract_nutrition_object(n: Dict[str, Any], fields: Dict[str, Any]) -> None:
    mapping = {
        "energy_kj": ("energyContent", "calories"),
        "fat_g": ("fatContent",),
        "saturates_g": ("saturatedFatContent",),
        "carbohydrates_g": ("carbohydrateContent",),
        "sugars_g": ("sugarContent",),
        "protein_g": ("proteinContent",),
        "fiber_g": ("fiberContent",),
        "salt_g": ("saltContent",),
    }
    for out_key, keys in mapping.items():
        for k in keys:
            if k in n:
                _set_field(fields, out_key, n.get(k))
                break
    serving = n.get("servingSize") or n.get("servingSizeDescription")
    if _real_value(serving):
        _set_field(fields, "nutritional_info", f"per {serving}")


def _extract_microdata_opengraph(soup) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    # OpenGraph / Twitter / product metas.
    meta_map = {
        "og:title": "product_name",
        "twitter:title": "product_name",
        "product:brand": "brand",
        "og:description": "ingredients",  # only accepted later if label parser cannot do better; weak but deterministic text.
    }
    for prop, key in meta_map.items():
        tag = soup.find("meta", attrs={"property": prop}) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            val = tag.get("content")
            if key == "ingredients" and not re.search(r"ingredients|zutaten|ingrédients|ingredientes|ingredienti", val, re.I):
                continue
            _set_field(fields, key, val)

    itemprop_map = {
        "name": "product_name",
        "brand": "brand",
        "manufacturer": "manufacturer_name",
        "ingredients": "ingredients",
        "ingredient": "ingredients",
        "weight": "net_weight_customer_facing",
        "netWeight": "net_weight_customer_facing",
        "fatContent": "fat_g",
        "saturatedFatContent": "saturates_g",
        "carbohydrateContent": "carbohydrates_g",
        "sugarContent": "sugars_g",
        "proteinContent": "protein_g",
        "fiberContent": "fiber_g",
        "saltContent": "salt_g",
        "energyContent": "energy_kj",
        "calories": "energy_kj",
    }
    for tag in soup.find_all(attrs={"itemprop": True}):
        prop = tag.get("itemprop")
        key = itemprop_map.get(prop)
        if not key:
            continue
        val = tag.get("content") or tag.get("value") or tag.get_text(" ")
        if key == "net_weight_customer_facing":
            fields.update(_parse_weight_to_fields(_to_str(val)))
        else:
            _set_field(fields, key, val)
    return fields


def _label_to_field(label: str) -> Optional[str]:
    lab = re.sub(r"\s+", " ", (label or "").strip().lower())
    if not lab:
        return None
    # Nutrition rows first to prevent "of which sugars" from mapping to carbohydrates.
    for field, aliases in sorted(NUTRIENT_LABELS.items(), key=lambda kv: -max(len(a) for a in kv[1])):
        if any(a in lab for a in aliases):
            return field
    for field, aliases in LABEL_ALIASES.items():
        if any(a in lab for a in aliases):
            return field
    return None


def _extract_labelled_tables(soup, page_text: str) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    # Attribute / nutrition tables.
    for table in soup.find_all("table"):
        headers = [_clean_text(c.get_text(" ")).lower() for c in table.find_all(["th", "td"], recursive=True)[:12]]
        for row in table.find_all("tr"):
            cells = [_clean_text(c.get_text(" ")) for c in row.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if len(cells) < 2:
                continue
            label = cells[0]
            field = _label_to_field(label)
            if not field:
                continue
            value_cells = cells[1:]
            # Prefer per-100g / per-100ml value when headers expose columns.
            chosen = value_cells[0]
            row_header_text = " | ".join(headers)
            if len(value_cells) > 1 and re.search(r"100\s*(g|ml)", row_header_text, re.I):
                # Header cells often include first-column label, so align by offset best-effort.
                for idx, h in enumerate(headers[1:1 + len(value_cells)]):
                    if re.search(r"100\s*(g|ml)", h, re.I):
                        chosen = value_cells[idx]
                        break
            if field in NUMERIC_FIELDS and field not in {"net_weight", "gross_weight"}:
                _set_field(fields, field, chosen)
            elif field == "net_weight_customer_facing":
                fields.update(_parse_weight_to_fields(chosen))
            else:
                _set_field(fields, field, chosen)
            if field in NUTRIENT_LABELS and re.search(r"100\s*(g|ml)", table.get_text(" "), re.I):
                _set_field(fields, "nutritional_info", "per 100g/100ml")

    # Definition lists and common spec rows.
    for container in soup.find_all(["dl", "ul", "div"]):
        text = _clean_text(container.get_text(" "))
        if len(text) > 600:
            continue
        parts = re.split(r"\s{2,}|\n|\t", text)
        if len(parts) >= 2:
            field = _label_to_field(parts[0])
            if field:
                val = " ".join(parts[1:]).strip()
                if field == "net_weight_customer_facing":
                    fields.update(_parse_weight_to_fields(val))
                elif field in NUMERIC_FIELDS:
                    _set_field(fields, field, val)
                else:
                    _set_field(fields, field, val)

    # Regex sections in visible text for free-text fields.
    stop = r"(?=(?:\b(?:nutri|nutrition|nährwert|allergen|may contain|kann enthalten|manufacturer|hersteller|origin|herkunft|storage|aufbewahrung|preparation|zubereitung)\b\s*:)|$)"
    for field, aliases in LABEL_ALIASES.items():
        if field == "net_weight_customer_facing":
            continue
        for alias in aliases:
            pat = rf"\b{re.escape(alias)}\b\s*[:：]\s*(.{{3,700}}?){stop}"
            m = re.search(pat, page_text, re.I | re.S)
            if m:
                val = _clean_text(m.group(1))
                if field == "organic_certification_id":
                    cert = ORG_CERT_RE.search(val)
                    if cert:
                        _set_field(fields, field, cert.group(0).upper())
                else:
                    _set_field(fields, field, val)
                break

    # Certification and organic flags.
    cert = ORG_CERT_RE.search(page_text)
    if cert:
        _set_field(fields, "organic_certification_id", cert.group(0).upper())
        fields["organic_product"] = "Yes"
    elif re.search(r"\b(bio|organic|ökologisch|oeko|eco)\b", page_text[:8000], re.I):
        fields.setdefault("organic_product", "Yes")
    else:
        fields.setdefault("organic_product", "No")

    # Weight if not found in tables.
    if not fields.get("net_weight"):
        fields.update(_parse_weight_to_fields(page_text[:5000]))

    # Packaging / fragile.
    packaging_hay = page_text[:10000].lower()
    for packaging, pat in PACKAGING_PATTERNS:
        if re.search(pat, packaging_hay, re.I):
            fields.setdefault("packaging", packaging)
            break
    if fields.get("packaging") and re.search(r"glass|glas|jar", fields.get("packaging", ""), re.I):
        fields.setdefault("fragile_item", "Yes")
    elif fields.get("packaging"):
        fields.setdefault("fragile_item", "No")

    return fields


def _parse_page_fields(html: str, page_text: str) -> Dict[str, Any]:
    if not BS4_OK or not html:
        return {}
    soup = BeautifulSoup(html, "html.parser")
    fields: Dict[str, Any] = {}

    # Deterministic fallback order: JSON-LD -> microdata/OpenGraph -> labelled HTML.
    for source_fields in (
        _extract_from_jsonld(soup, html),
        _extract_microdata_opengraph(soup),
        _extract_labelled_tables(soup, page_text),
    ):
        for k, v in source_fields.items():
            if _real_value(v) and not _real_value(fields.get(k)):
                fields[k] = v

    # Title/name fallback and weight from title are deterministic and useful.
    if not fields.get("product_name"):
        title = _page_title(soup)
        if title:
            fields["product_name"] = title
    if fields.get("product_name") and not fields.get("net_weight"):
        fields.update(_parse_weight_to_fields(fields["product_name"]))

    return {k: v for k, v in fields.items() if k in FOOD_FIELDS and _real_value(v)}


def _field_richness(fields: Dict[str, Any]) -> int:
    food_core = [
        "ingredients", "allergens", "energy_kj", "fat_g", "saturates_g", "carbohydrates_g",
        "sugars_g", "protein_g", "fiber_g", "salt_g", "net_weight", "brand",
    ]
    return sum(2 if k in food_core else 1 for k, v in fields.items() if _real_value(v))


def _page_base_score(page: VerifiedPage, market: str) -> int:
    score = 100 if page.route == "A" else 65
    if _is_goldmine_domain(page.final_url or page.url, market):
        score += 20
    if page.attempt.startswith("Attempt 2 brand"):
        score += 15
    if page.attempt.startswith("Attempt 3 goldmine"):
        score += 10
    score += min(_field_richness(page.fields), 30)
    return score


def _merge_page_fields(pages: List[VerifiedPage], market: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    merged: Dict[str, Any] = {k: "" for k in FOOD_FIELDS}
    winners: Dict[str, Tuple[int, Any, str]] = {}
    for page in pages:
        base = _page_base_score(page, market)
        page.score = base
        for k, v in page.fields.items():
            if not _real_value(v):
                continue
            # Rich/high-trust source wins per field. Deterministic parsers only.
            current = winners.get(k)
            if current is None or base > current[0]:
                winners[k] = (base, v, page.url)
    provenance_by_field: Dict[str, str] = {}
    for k, (_, v, url) in winners.items():
        merged[k] = v
        provenance_by_field[k] = url
    return merged, provenance_by_field


def _extract_llm_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    clean = raw.strip().replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{.*\}", clean, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _llm_extract_free_text(page_text: str, ean: str, product_name: str, market: str, gemini_key: str, model: str, missing: Sequence[str]) -> Dict[str, Any]:
    if not GENAI_OK or not gemini_key or not page_text or not missing:
        return {}
    missing = [f for f in missing if f in LLM_FALLBACK_FIELDS]
    if not missing:
        return {}
    schema = "\n".join(f'  "{f}": "value or null",' for f in missing)
    page_excerpt = page_text[:22000]
    prompt = f"""
You are reading ONE fetched product page. Extract ONLY the requested fields from PAGE TEXT.
Do not search. Do not use outside knowledge. Do not infer values. Return null when absent.
Do not extract or invent numeric nutrition, GTIN, or weight values.
Product reference: {product_name} / EAN {ean}. Target market: {market}.
Return only a JSON object:
{{
{schema}
}}

PAGE TEXT:
{page_excerpt}
"""
    try:
        client = genai.Client(api_key=gemini_key)
        resp = client.models.generate_content(
            model=model,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=1400),
        )
        raw = getattr(resp, "text", "") or ""
        if not raw and getattr(resp, "candidates", None):
            for part in (resp.candidates[0].content.parts if resp.candidates[0].content and resp.candidates[0].content.parts else []):
                if getattr(part, "text", None):
                    raw += part.text
        parsed = _extract_llm_json(raw)
        return {k: v for k, v in parsed.items() if k in missing and _real_value(v)}
    except Exception:
        return {}


def _blank_fields() -> Dict[str, Any]:
    return {k: "" for k in FOOD_FIELDS}


def _status_reason_for_no_verified(unreadable_count: int, candidates_count: int) -> str:
    if unreadable_count:
        return "page not readable"
    if candidates_count:
        return "no candidate page passed Route A/B verification"
    return "no candidate page found"


async def extract_food_data(
    session,
    ean: str,
    ground_truth: str,
    market: str,
    registry_name: str = "",
    keys: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return deterministic food data from verified source pages only.

    Parameters
    ----------
    session:
        An aiohttp ClientSession.
    ean:
        Queried GTIN/EAN.
    ground_truth:
        User-provided product name/weight/brand context.
    market:
        Market code such as DE, FR, UK.
    registry_name:
        Optional pre-resolved name from the calling app. If omitted, the module
        tries Go-UPC and then EAN-Search according to the requested ladder.
    keys:
        {"serp_key", "ean_token", "go_upc_key", "gemini_key", "gemini_model",
         "go_upc_data"}. `go_upc_data` may be passed from app.py to avoid a
        duplicate Tier 1A request.
    """
    keys = keys or {}
    serp_key = keys.get("serp_key") or keys.get("serpapi_key") or ""
    ean_token = keys.get("ean_token") or keys.get("ean_search_token") or ""
    go_upc_key = keys.get("go_upc_key") or keys.get("go_upc_api_key") or ""
    gemini_key = keys.get("gemini_key") or keys.get("gemini_api_key") or ""
    gemini_model = keys.get("gemini_model") or "gemini-2.5-flash"
    go_upc_data = keys.get("go_upc_data") or None

    diag: List[str] = []
    fields = _blank_fields()
    source_links: List[str] = []
    source_routes: Dict[str, str] = {}
    provenance_by_field: Dict[str, str] = {}

    # Tier 1A: Go-UPC name. If app.py already fetched it, reuse it.
    registry_source = ""
    if go_upc_data:
        registry_source = "Go-UPC"
        registry_name = registry_name or (go_upc_data.get("name") or "")
        diag.append(f"✅ Tier 1A Go-UPC name: {registry_name or '(empty)'}")
    elif go_upc_key:
        go_upc_data = await _fetch_go_upc(session, ean, go_upc_key)
        if go_upc_data:
            registry_source = "Go-UPC"
            registry_name = registry_name or (go_upc_data.get("name") or "")
            diag.append(f"✅ Tier 1A Go-UPC name: {registry_name or '(empty)'}")

    # Go-UPC / EAN-Search are registry identity only. They seed the product name
    # for discovery and Route B comparison, but they do NOT populate food fields.
    # Ingredients, nutrition, weight, brand, manufacturer, and similar row values
    # must come from verified source pages.
    if go_upc_data and _real_value(go_upc_data.get("name")):
        fields["product_name"] = _to_str(go_upc_data.get("name"))

    # Tier 1B: EAN-Search only if Go-UPC empty.
    if not registry_name and ean_token:
        ean_data = await _fetch_ean_search(session, ean, ean_token)
        if ean_data:
            registry_source = "EAN-Search"
            registry_name = ean_data.get("name") or ""
            if registry_name:
                fields["product_name"] = registry_name
            diag.append(f"✅ Tier 1B EAN-Search name: {registry_name or '(empty)'}")
        else:
            diag.append("ℹ️ Tier 1B EAN-Search: empty")

    clean_gt = _strip_pack_notation(ground_truth) if ground_truth else ""
    anchors = [x for x in (clean_gt, registry_name) if _real_value(x)]
    if not fields.get("product_name"):
        fields["product_name"] = registry_name or clean_gt or f"Product with EAN {ean}"

    candidates = await _discover_candidates(session, ean, ground_truth, market, registry_name, serp_key, diag)
    diag.append(f"🔎 Candidate pages discovered: {len(candidates)}")

    verified: List[VerifiedPage] = []
    unreadable_count = 0
    for cand in candidates:
        if len(verified) >= MAX_VERIFIED_READS:
            break
        html, final_url = await _fetch_html(session, cand.url)
        if not html or len(html) < 200:
            unreadable_count += 1
            continue
        if not BS4_OK:
            continue
        soup = BeautifulSoup(html, "html.parser")
        jsonld = _jsonld_blocks(soup)
        title = _page_title(soup, cand.title)
        identity = _extract_identity_text(soup, jsonld, cand.title)
        text = _soup_text(BeautifulSoup(html, "html.parser"))
        route = None
        reason = ""
        if _html_contains_ean(html, jsonld, ean):
            route = "A"
            reason = "EAN present on page"
        else:
            ratio = _best_name_ratio(anchors, identity)
            if ratio >= NAME_MATCH_THRESHOLD and not _weights_conflict(anchors, identity, text):
                route = "B"
                reason = f"name match {ratio:.0%}, no weight conflict"
        if not route:
            continue
        page = VerifiedPage(
            url=cand.url,
            final_url=final_url,
            route=route,
            html=html,
            text=text[:50000],
            title=title,
            attempt=cand.attempt,
            domain=_domain(final_url or cand.url),
            readable=(len(text) >= 500 or bool(jsonld)),
            reason=reason,
        )
        page.fields = _parse_page_fields(html, text)
        if page.readable:
            # LLM fallback only for messy free-text fields, and only from this page text.
            missing = [f for f in LLM_FALLBACK_FIELDS if not _real_value(page.fields.get(f))]
            if missing and gemini_key:
                llm_fields = await asyncio.to_thread(
                    _llm_extract_free_text, page.text, ean, fields.get("product_name") or registry_name,
                    market, gemini_key, gemini_model, missing,
                )
                for k, v in llm_fields.items():
                    if _real_value(v) and k in LLM_FALLBACK_FIELDS:
                        page.fields[k] = v
        verified.append(page)
        diag.append(f"✅ Verified Route {route}: {page.domain} ({reason}); fields={_field_richness(page.fields)}")

    if not verified:
        reason = _status_reason_for_no_verified(unreadable_count, len(candidates))
        diag.append(f"❌ No readable verified page — {reason}")
        # Keep registry first-pass fields for visibility, but explicitly fail validation.
        return {
            "fields": fields,
            "source_links": [],
            "source_routes": {},
            "candidate_pages": [c.url for c in candidates[:10]],
            "provenance": "page not readable" if unreadable_count else "no verified page",
            "provenance_by_field": {},
            "reliability": "L",
            "status": "Failed Validation",
            "is_exact_match": False,
            "rejection_reason": "page_not_readable" if unreadable_count else "unconfirmed",
            "diagnostic_log": "\n".join(diag),
        }

    # Sort verified reads so the links and field-level winners prefer highest trust/richness.
    for page in verified:
        page.score = _page_base_score(page, market)
    verified.sort(key=lambda p: (p.route != "A", -p.score, p.domain))

    merged_page_fields, provenance_by_field = _merge_page_fields(verified, market)
    for k, v in merged_page_fields.items():
        if _real_value(v):
            fields[k] = v

    # If page did not expose a product name, preserve registry/user identity.
    if not fields.get("product_name"):
        fields["product_name"] = registry_name or clean_gt or f"Product with EAN {ean}"

    # Source links are verified pages only. These are the pages contributing data.
    for page in verified:
        link = page.final_url or page.url
        if link not in source_links and not _is_excluded_domain(link):
            source_links.append(link)
            source_routes[link] = page.route
    source_links = source_links[:5]

    has_route_a = any(p.route == "A" for p in verified)
    has_route_b = any(p.route == "B" for p in verified)
    if has_route_a:
        reliability = "H"
    elif has_route_b:
        reliability = "M"
    else:
        reliability = "L"

    if not any(p.readable for p in verified):
        # Should be rare because JSON-LD pages are treated as readable, but keep the
        # explicit v1 JS/cookie-wall failure mode requested.
        reliability = "L"
        status = "Failed Validation"
        provenance = "page not readable"
        exact = False
        rejection = "page_not_readable"
    else:
        status = "Success"
        provenance = "Verified page" if len(source_links) == 1 else "Verified pages"
        exact = True
        rejection = ""

    if registry_source:
        diag.append(f"ℹ️ Registry identity used as discovery context only: {registry_source}")
    diag.append(f"📄 Food data provenance: {provenance}; reliability={reliability}")

    return {
        "fields": fields,
        "source_links": source_links,
        "source_routes": source_routes,
        "candidate_pages": [c.url for c in candidates[:10]],
        "provenance": provenance,
        "provenance_by_field": provenance_by_field,
        "reliability": reliability,
        "status": status,
        "is_exact_match": exact,
        "rejection_reason": rejection,
        "diagnostic_log": "\n".join(diag),
    }
