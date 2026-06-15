import streamlit as st
import pandas as pd
import os
import json
import asyncio
import aiohttp
import re
import time
from io import BytesIO
from google import genai
from google.genai import types

# Image processing libraries (for display pipeline only - food extraction never touches these)
try:
    from PIL import Image
    import imagehash
    IMAGE_LIBS_OK = True
except ImportError:
    IMAGE_LIBS_OK = False

st.set_page_config(page_title="Food Data Researcher PRO", layout="wide")

# ============================================================================
# SQLITE CACHE
# ============================================================================
import sqlite3

DB_PATH = os.environ.get("CACHE_DB_PATH", "cache.db")


def init_cache():
    """
    Creates the cache table if it does not exist, then runs a non-destructive
    schema migration to add columns introduced in CACHE_VERSION 2.
    Existing rows keep their data; new columns default to NULL / 0.
    """
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ean_cache (
            ean         TEXT NOT NULL,
            market      TEXT NOT NULL,
            result_json TEXT NOT NULL,
            status      TEXT    DEFAULT 'unknown',
            confidence  REAL    DEFAULT 0.0,
            version     INTEGER DEFAULT 1,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ean, market)
        )
    """)
    # Non-destructive migration: add v2 columns to tables created under v1.
    # SQLite does not support IF NOT EXISTS on ALTER TABLE, so we catch the
    # OperationalError that fires when the column already exists.
    for col, defn in [
        ("status",     "TEXT DEFAULT 'unknown'"),
        ("confidence", "REAL DEFAULT 0.0"),
        ("version",    "INTEGER DEFAULT 1"),
    ]:
        try:
            con.execute(f"ALTER TABLE ean_cache ADD COLUMN {col} {defn}")
        except sqlite3.OperationalError:
            pass  # column already present — nothing to do
    con.commit()
    con.close()


def cache_get(ean: str, market: str):
    """
    Return a cached result only if it was stored under the current schema version
    AND its status was 'Success'.  Any other entry (error, failed-validation,
    needs-review, wrong schema version) is treated as a miss so the EAN is
    re-queried fresh.

    NOTE: On first deployment after upgrading to CACHE_VERSION 2, ALL existing
    entries will be invalidated because their version column is NULL / 1.
    This is intentional — the extraction prompt and source rules have changed
    significantly, so stale entries must not be served as authoritative data.
    """
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT result_json, status, version FROM ean_cache WHERE ean=? AND market=?",
            (ean, market)
        ).fetchone()
        con.close()
        if not row:
            return None
        result_json, status, version = row
        if version != CACHE_VERSION:
            return None           # stale schema — re-query
        if status != "Success":
            return None           # don't serve errors / reviews from cache
        return json.loads(result_json)
    except Exception:
        return None


def cache_set(ean: str, market: str, result_dict: dict,
              status: str = "unknown", confidence: float = 0.0):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            """INSERT OR REPLACE INTO ean_cache
               (ean, market, result_json, status, confidence, version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ean, market, json.dumps(result_dict), status, confidence, CACHE_VERSION)
        )
        con.commit()
        con.close()
    except Exception:
        pass


def cache_delete(ean: str, market: str):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "DELETE FROM ean_cache WHERE ean=? AND market=?",
            (ean, market)
        )
        con.commit()
        con.close()
    except Exception:
        pass


# ============================================================================
# MODEL & QUALITY CONSTANTS
# Update model strings here when migrating — never scatter them across the file.
# ============================================================================

# Food data extraction: uses Google Search grounding + full JSON output
EXTRACTION_MODEL = "gemini-2.5-flash"

# Image identity verification: vision-only, no grounding needed, must be cheap & fast
# gemini-2.0-flash-lite was SHUT DOWN on 2026-06-01. gemini-2.5-flash-lite is the
# direct replacement at the same price point ($0.10/$0.40 per 1M tokens) with
# better multimodal understanding and a stable 2.5 lifecycle.
VISION_MODEL = "gemini-2.5-flash-lite"

# Bump this whenever the extraction prompt or cache schema changes significantly.
# All rows stored under a lower version are treated as cache misses on first load.
CACHE_VERSION = 2

# Minimum vision confidence score (0.0–1.0) required to display an image.
# Candidates below this threshold are rejected; their URL is preserved in
# Image Source Link so users can visit the page manually.
IMAGE_MATCH_THRESHOLD = 0.55

# Lower threshold applied to images from pages that were EAN-verified
# (i.e. the retailer/brand page's HTML confirmed it contained the queried EAN).
# These images are more trustworthy, so a slightly more lenient gate is appropriate.
IMAGE_MATCH_THRESHOLD_TRUSTED = 0.35


# ============================================================================
# HARD-EXCLUDED DOMAINS
# These sources are NEVER used for food information or product images.
# Amazon and eBay: marketplaces with reseller-submitted, unverified content.
# OpenFoodFacts: open-source user-generated database.
# Add or remove tokens here to maintain the exclusion list centrally.
# ============================================================================

_EXCLUDED_DOMAIN_TOKENS: tuple = (
    "amazon.",       # covers amazon.com, amazon.co.uk, amazon.de, etc.
    "ebay.",         # covers ebay.com, ebay.co.uk, ebay.de, etc.
    "openfoodfacts.",
    "aliexpress.",
    "alibaba.",
)


def _is_excluded_domain(url: str) -> bool:
    """
    Returns True if the URL belongs to a hard-excluded marketplace or
    user-generated source.  Checks by substring so it catches all TLDs
    (amazon.co.uk, ebay.de, …) and CDN subdomains (m.media-amazon.com, etc.).
    """
    if not url:
        return False
    url_lower = url.lower()
    return any(token in url_lower for token in _EXCLUDED_DOMAIN_TOKENS)


# ============================================================================
# BARCODE & IDENTITY UTILITIES
# ============================================================================

def valid_gtin(ean: str) -> bool:
    """
    Luhn / mod-10 check-digit validation for EAN-8, EAN-13, EAN-14, UPC-12.
    Returns False for barcodes that fail the check digit, preventing wasted
    API calls on OCR-mangled or mistyped codes.
    """
    if not ean.isdigit() or len(ean) not in (8, 12, 13, 14):
        return False
    digits = [int(c) for c in ean]
    check  = digits[-1]
    body   = digits[:-1][::-1]
    total  = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(body))
    return (10 - total % 10) % 10 == check


def barcode_matches(returned_barcode: str, queried_ean: str) -> bool:
    """
    Compare a source's returned barcode against the queried EAN.
    Normalises leading zeros so '0012345678905' and '12345678905' match.
    """
    if not returned_barcode:
        return False
    return (str(returned_barcode).strip().lstrip("0")
            == str(queried_ean).strip().lstrip("0"))


async def page_contains_ean(session, url: str, ean: str) -> bool:
    """
    Fetch the first 200 KB of a page and check whether the EAN appears in
    the raw HTML.  Used to verify a brand/retailer page actually references
    the product before we trust OG/JSON-LD images from it.
    Fails closed: returns False on any network or parsing error.
    """
    _ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/122.0.0.0 Safari/537.36")
    try:
        async with session.get(url, headers={"User-Agent": _ua}, timeout=8) as r:
            if r.status != 200:
                return False
            raw  = await r.content.read(200_000)      # max 200 KB
            text = raw.decode("utf-8", errors="ignore")
            return ean in text or ean.lstrip("0") in text
    except Exception:
        return False


# ============================================================================
# CACHE POLICY
# ============================================================================

def should_cache(status: str) -> bool:
    """
    Only persist confident successes.  Errors, failed validation, and
    'Needs Review' results must be re-evaluated on next lookup.
    A cache miss is always safer than a stale wrong answer.
    """
    return status == "Success"


def _is_displayable_url(url: str) -> bool:
    """
    Change 4 — Returns True only for real, clickable source URLs.
    Filters out:
    - Vertex AI grounding redirect URLs (which 404 immediately)
    - google.com/search?q= URLs (Gemini internal search queries leaking out)
    - google.com/url? redirect wrappers
    - All hard-excluded marketplace/UGC domains
    """
    if not url or not url.startswith("http"):
        return False
    if "vertexaisearch.cloud.google.com" in url:
        return False
    if "google.com/search" in url:
        return False
    if "google.com/url?" in url:
        return False
    if _is_excluded_domain(url):
        return False
    return True


def _extract_weight_hint(ground_truth: str) -> str:
    """
    Change 7 — Extract unit weight/volume hint from ground truth for image
    verification. Handles pack-size notation by extracting the UNIT size:
      "4x200ml"      → "200ml"   (unit, not total)
      "6x330ml"      → "330ml"
      "(4 Pack) 80g" → "80g"
      "255g"         → "255g"
    """
    if not ground_truth:
        return ""
    # Pack notation: NxVolume or N x Volume — extract the unit, not pack total
    m = re.search(
        r'\b\d+\s*[xX×]\s*(\d+(?:[.,]\d+)?)\s*(g|gr|kg|ml|cl|l|oz|lb)\b',
        ground_truth, re.IGNORECASE
    )
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    # Standard single weight/volume
    m = re.search(
        r'\b(\d+(?:[.,]\d+)?)\s*(g|gr|kg|ml|cl|l|oz|lb)\b',
        ground_truth, re.IGNORECASE
    )
    return f"{m.group(1)}{m.group(2).lower()}" if m else ""


# ── Go-UPC rate limiter (2 req/s per plan documentation) ─────────────────────
_GO_UPC_SEM:  asyncio.Semaphore | None = None   # lazily initialised in event loop
_GO_UPC_LAST: list[float]              = [0.0]  # mutable so nested fn can update it


async def fetch_go_upc(session, ean: str, go_upc_key: str) -> dict | None:
    """
    Tier 1A — Go-UPC barcode lookup (https://go-upc.com/docs).

    Returns a structured dict on success, None on 404 / any error.
    Enforces the 2 requests/second rate limit documented by Go-UPC.

    Fields returned:
      name, brand, description, image_url, ingredients,
      category, category_path, specs (dict), is_food (bool)
    """
    global _GO_UPC_SEM, _GO_UPC_LAST
    if not go_upc_key:
        return None
    if _GO_UPC_SEM is None:
        _GO_UPC_SEM = asyncio.Semaphore(1)  # serialise → max 2 req/s with sleep below

    async with _GO_UPC_SEM:
        # Enforce 0.5 s minimum gap between consecutive calls (= max 2 req/s)
        now  = asyncio.get_event_loop().time()
        wait = 0.5 - (now - _GO_UPC_LAST[0])
        if wait > 0:
            await asyncio.sleep(wait)
        _GO_UPC_LAST[0] = asyncio.get_event_loop().time()

        url = f"https://go-upc.com/api/v1/code/{ean}"
        try:
            async with session.get(
                url,
                headers={"Authorization": f"Bearer {go_upc_key}"},
                timeout=10
            ) as resp:
                if resp.status == 404:
                    return None   # product not in Go-UPC database
                if resp.status == 429:
                    await asyncio.sleep(1.0)
                    return None   # rate-limited — skip rather than retry
                if resp.status != 200:
                    return None
                data = await resp.json()
                p    = data.get("product") or {}
                if not p or not p.get("name"):
                    return None
                # Verify the returned barcode matches the queried EAN
                returned = str(data.get("code", ""))
                if not barcode_matches(returned, ean):
                    return None
                # Determine food/non-food from Google Shopping category path
                cat_path = p.get("categoryPath") or []
                is_food  = bool(cat_path) and cat_path[0].lower().startswith("food")
                return {
                    "name":          p.get("name", ""),
                    "brand":         p.get("brand", ""),
                    "description":   p.get("description", ""),
                    "image_url":     p.get("imageUrl", "") or "",
                    "ingredients":   (p.get("ingredients") or {}).get("text", ""),
                    "category":      p.get("category", ""),
                    "category_path": cat_path,
                    "specs":         dict(p.get("specs") or []),
                    "is_food":       is_food,
                }
        except Exception:
            return None


# ============================================================================
# SHARED CONSTANTS (used by both pipelines)
# ============================================================================

GOLDMINE = {
    "FR": ("site:carrefour.fr OR site:auchan.fr OR site:coursesu.com "
           "OR site:leclerc.fr OR site:intermarche.fr OR site:monoprix.fr"),
    "UK": ("site:ocado.com OR site:waitrose.com OR site:asda.com "
           "OR site:tesco.com OR site:sainsburys.co.uk OR site:morrisons.com"),
    "NL": "site:ah.nl OR site:jumbo.com OR site:plus.nl OR site:dirk.nl",
    "BE": ("site:delhaize.be OR site:colruyt.be OR site:carrefour.be "
           "OR site:spar.be OR site:lidl.be"),
    "DE": ("site:rewe.de OR site:edeka.de OR site:kaufland.de "
           "OR site:dm.de OR site:rossmann.de OR site:metro.de"),
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

# ============================================================================
# PATH A: FOOD EXTRACTION PIPELINE
# ============================================================================

BAD_IMAGE_EXTENSIONS = {".svg", ".gif", ".ico", ".webmanifest", ".json", ".xml"}
BAD_IMAGE_PATTERNS = [
    "logo", "icon", "banner", "placeholder", "spinner", "loading",
    "payment", "paypal", "mastercard", "visa", "flag", "star",
    "cart", "account", "arrow", "check", "tick", "social",
    # Hard-excluded sources — images from these are never used
    "openfoodfacts", "pinterest", "ebay", "tiktok", "facebook",
    "instagram", "twitter", "youtube", "amazon-ads", "ad_",
    "s192", "width=250", "160x160", "200x200", "250x250", "300x300",
    "50x50", "75x30", "100x100", "128x128", "150x150", "_xs", "_xxs", "thumbnail"
]


def _is_valid_image_url(url: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    # Hard-exclude marketplace / UGC image URLs
    if _is_excluded_domain(url):
        return False
    url_lower = url.lower()
    path = url_lower.split("?")[0]

    if any(path.endswith(ext) for ext in BAD_IMAGE_EXTENSIONS): return False
    if any(p in url_lower for p in BAD_IMAGE_PATTERNS): return False

    is_bad_amazon = "media-amazon.com" in url_lower and ("," in url_lower or "_bo" in url_lower)
    if is_bad_amazon: return False

    return True


@st.cache_data
def load_taxonomy():
    """Loads the taxonomy CSV into memory once to prevent repeated disk I/O."""
    try:
        with open("taxonomy.csv", "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "Level 1,Level 2,Level 3,Level 4,Level 5,Level 6\nError: taxonomy.csv not found."


async def fetch_og_image(session, url):
    """Visits a retailer URL and extracts the high-quality Open Graph image."""
    if _is_excluded_domain(url):
        return None
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=5) as resp:
            if resp.status == 200:
                html = await resp.text()
                match = re.search(r'<meta[^>]*property=[\'"]og:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]', html, re.IGNORECASE)
                if match:
                    img_url = match.group(1)
                    if not _is_excluded_domain(img_url):
                        return img_url
    except Exception:
        pass
    return None


GEMINI_SUPPORTED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

MAGIC_BYTES_MAP = [
    (b"\xff\xd8\xff",       "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF",               "image/webp"),
    (b"GIF87a",             "image/gif"),
    (b"GIF89a",             "image/gif"),
]


def _sniff_mime(data: bytes) -> str | None:
    for magic, mime in MAGIC_BYTES_MAP:
        if data[:len(magic)] == magic:
            return mime
    return None


def _safe_mime(raw_mime: str, data: bytes) -> str | None:
    clean = raw_mime.split(";")[0].strip().lower() if raw_mime else ""
    if clean in GEMINI_SUPPORTED_MIMES:
        return clean
    sniffed = _sniff_mime(data)
    return sniffed


async def fetch_image_bytes_simple(session, url):
    """Image fetcher used by the food extraction pipeline."""
    if _is_excluded_domain(url):
        return None
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.read()
                if len(data) < 8000:
                    return None
                raw_mime = resp.headers.get("content-type", "")
                mime = _safe_mime(raw_mime, data)
                if mime is None:
                    return None
                return {"url": url, "mime": mime, "data": data}
    except Exception:
        pass
    return None



async def _serp_get(session, url: str, params: dict, timeout: int = 15,
                    max_retries: int = 2) -> dict | None:
    """
    SerpAPI GET with automatic 429 backoff-retry.
    Returns the parsed JSON dict, or None on persistent failure.
    Converts concurrent burst failures into safe sequential retries.

    Backoff schedule: 3 s after attempt 1, 6 s after attempt 2.
    """
    for attempt in range(max_retries + 1):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    if attempt < max_retries:
                        wait = 3 * (attempt + 1)   # 3 s, 6 s
                        await asyncio.sleep(wait)
                        continue
                    return None   # exhausted retries
                # Other HTTP error (403, 5xx…) — don't retry
                return None
        except asyncio.TimeoutError:
            if attempt < max_retries:
                await asyncio.sleep(1)
                continue
            return None
        except Exception:
            return None
    return None


async def fetch_basic_info(session, ean, serp_key, ean_token, market_code,
                          user_ground_truth="", go_upc_data=None):
    """
    Path A: Resolves the product name and gathers images for Gemini extraction.

    go_upc_data: pre-fetched Go-UPC result (from process_ean) or None.
    Waterfall: Go-UPC (Tier 1A) → EAN-Search (Tier 1B, only if Go-UPC empty)
               → SerpAPI goldmine → SerpAPI global fallback.

    Returns: (product_name, images, diagnostic_log, ean_verified)
    """
    gl           = market_code.lower()
    market_upper = market_code.upper()
    diagnostic_log = []

    product_name       = None
    retailer_urls      = []
    candidate_image_urls = []
    registry_image_url = None
    ean_verified       = False

    serp_url = "https://serpapi.com/search"

    # ── TIER 1A: Go-UPC (barcode-keyed, highest accuracy) ────────────────────
    if go_upc_data:
        diagnostic_log.append("✅ Tier 1A: Go-UPC data received.")
        ean_verified = True
        gname = go_upc_data.get("name", "")
        if gname and not is_garbage_name(gname):
            product_name = gname
            diagnostic_log.append(f"✅ Go-UPC verified name: {product_name}")
        img_url = go_upc_data.get("image_url", "")
        if img_url and _is_valid_image_url(img_url):
            candidate_image_urls.append(img_url)
            diagnostic_log.append("✅ Go-UPC image URL added for Gemini context.")
    else:
        diagnostic_log.append("ℹ️ Go-UPC: no data — falling through to EAN-Search.")

    # ── TIER 1B: EAN-Search (only if Go-UPC returned nothing) ────────────────
    if not go_upc_data and ean_token:
        diagnostic_log.append("🔍 Tier 1B: EAN-Search.org API...")
        ean_url = f"https://api.ean-search.org/api?token={ean_token}&op=barcode-lookup&ean={ean}&format=json"
        try:
            async with session.get(ean_url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) > 0 and "error" not in data[0]:
                        returned_barcode = str(data[0].get("barcode", data[0].get("ean", "")))
                        candidate_name   = data[0].get("name", "")
                        if barcode_matches(returned_barcode, ean):
                            ean_verified = True
                            if not is_garbage_name(candidate_name):
                                product_name       = candidate_name
                                registry_image_url = data[0].get("image")
                                diagnostic_log.append(f"✅ EAN-Search barcode verified: {product_name}")
                            else:
                                diagnostic_log.append("✅ EAN-Search barcode verified (name discarded — garbage pattern)")
                        else:
                            diagnostic_log.append(
                                f"⚠️ EAN-Search barcode mismatch "
                                f"(queried={ean}, returned={returned_barcode}) — name discarded"
                            )
        except Exception as e:
            diagnostic_log.append(f"⚠️ EAN-Search failed: {e}")

    # ── ATTEMPT 2: Brand-site lookup (if brand token in ground truth) ──
    # Strip pack notation before using ground truth in search queries so
    # "Eisberg Be Free Sparkling Rosé (4 Pack), 4x200ml" searches as
    # "Eisberg Be Free Sparkling Rosé 200ml" and finds the unit listing.
    _clean_ground_truth = _strip_pack_notation(user_ground_truth) if user_ground_truth else user_ground_truth
    brand_domain = None
    if serp_key and user_ground_truth:
        brand_candidate = _extract_brand(_clean_ground_truth)
        if brand_candidate:
            diagnostic_log.append(f"🔍 Brand-site lookup for '{brand_candidate}'...")
            try:
                data    = await _serp_get(session, serp_url,
                    params={"q": f"{brand_candidate} {ean}", "gl": gl, "api_key": serp_key})
                organic = data.get("organic_results", []) if data else []
                for res in organic[:5]:
                        link   = res.get("link", "")
                        domain = link.split("/")[2] if link.startswith("http") else ""
                        # Hard guard: never treat Amazon/eBay as a brand domain
                        if _is_excluded_domain(link):
                            continue
                        if _brand_matches_domain(brand_candidate, domain):
                            # Only trust this page if the EAN appears in its HTML
                            ean_on_page = await page_contains_ean(session, link, ean)
                            if ean_on_page:
                                if not product_name:
                                    product_name = res.get("title", "").split("-")[0].split("|")[0].strip()
                                    diagnostic_log.append(f"✅ Brand page EAN verified — name: {product_name} (from clean query: {_clean_ground_truth})")
                                if not ean_verified:
                                    ean_verified = True
                                if link not in retailer_urls:
                                    retailer_urls.insert(0, link)
                                brand_domain = domain
                            else:
                                diagnostic_log.append(
                                    f"⚠️ Brand page found ({domain}) but EAN not in page HTML — "
                                    f"adding as retailer URL but not using as identity source"
                                )
                                if link not in retailer_urls:
                                    retailer_urls.insert(0, link)
                                brand_domain = domain
                            break
            except Exception as e:
                diagnostic_log.append(f"⚠️ Brand-site lookup failed: {e}")

    # ── ATTEMPT 3: SerpAPI goldmine (tier-1 retailers for this market) ──
    if not product_name and serp_key:
        diagnostic_log.append("🔍 Goldmine Google Search for Name...")
        goldmine = f"{GOLDMINE.get(market_upper, '')} OR {GLOBAL_SITES}".strip(" OR")
        try:
            data = await _serp_get(session, serp_url,
                params={"q": f"{goldmine} {ean}", "gl": gl, "api_key": serp_key})
            organic = data.get("organic_results", []) if data else []
            if organic:
                product_name = organic[0].get("title", "").split("-")[0].split("|")[0].strip()
                if is_garbage_name(product_name):
                    product_name = None
                    diagnostic_log.append("⚠️ Goldmine name looks like a garbage/placeholder — discarded")
                else:
                    diagnostic_log.append(f"✅ Found Name via Goldmine: {product_name}")
                new_urls = [res.get("link") for res in organic[:4] if "link" in res
                            and not _is_excluded_domain(res.get("link", ""))]
                for u in new_urls:
                    if u not in retailer_urls:
                        retailer_urls.append(u)
            else:
                # ── ATTEMPT 4: Bare GTIN global fallback (low-confidence) ──
                diagnostic_log.append("⚠️ Goldmine returned nothing — bare GTIN global fallback (low confidence)...")
                data2   = await _serp_get(session, serp_url,
                    params={"q": str(ean), "gl": gl, "api_key": serp_key})
                organic2 = data2.get("organic_results", []) if data2 else []
                if organic2:
                    candidate = organic2[0].get("title", "").split("-")[0].split("|")[0].strip()
                    if not is_garbage_name(candidate):
                        product_name = candidate
                        diagnostic_log.append(f"✅ Found Name via bare GTIN fallback (unverified): {product_name}")
                    new_urls2 = [res.get("link") for res in organic2[:4] if "link" in res
                                 and not _is_excluded_domain(res.get("link", ""))]
                    for u in new_urls2:
                        if u not in retailer_urls:
                            retailer_urls.append(u)
        except Exception as e:
            diagnostic_log.append(f"⚠️ Google text search failed: {e}")

    if not product_name:
        diagnostic_log.append("⚠️ Name not found via any source — Gemini will attempt EAN-grounded search.")
        product_name = f"Product with EAN {ean}"
    elif not product_name.startswith("Product with EAN"):
        # Change 5: guard resolved name against obviously non-food titles
        if not _title_looks_like_food(product_name, ""):
            diagnostic_log.append(
                f"⚠️ Resolved name '{product_name}' appears non-food — treating as not found."
            )
            product_name = f"Product with EAN {ean}"
            ean_verified = False  # non-food name cannot count as a verified identity

    # ── GATHER CANDIDATE IMAGES FOR GEMINI ───────────────────────────────────
    # Stage A: Google Shopping — primary EAN-anchored image source.
    # Shopping matches on the barcode directly; images are retailer-submitted
    # product photos, not arbitrary search results.
    # Also yields retailer product-page URLs for Stage B OG scraping.
    if serp_key:
        diagnostic_log.append("🛒 Google Shopping images for Gemini (primary)...")
        shopping_data_a = await _serp_get(session, serp_url,
            params={"engine": "google_shopping", "q": ean,
                    "gl": gl, "api_key": serp_key}, timeout=15)
        if shopping_data_a:
            for item in shopping_data_a.get("shopping_results", [])[:6]:
                thumb = item.get("thumbnail", "")
                link  = item.get("link", "")
                if thumb and _is_valid_image_url(thumb) and thumb not in candidate_image_urls:
                    candidate_image_urls.append(thumb)
                # Enrich retailer_urls so Stage B can scrape product-page images
                if link and not _is_excluded_domain(link) and link not in retailer_urls:
                    retailer_urls.append(link)
            diagnostic_log.append(
                f"   Shopping: {len(shopping_data_a.get('shopping_results', []))} results.")
        else:
            diagnostic_log.append("   ⚠️ Google Shopping returned no data.")

    # Stage B: OG/JSON-LD images scraped from retailer pages
    # (includes pages discovered via Shopping above)
    if retailer_urls:
        diagnostic_log.append("🌐 Scraping OG images from approved retailer pages...")
        tasks     = [fetch_og_image(session, url) for url in retailer_urls]
        og_images = await asyncio.gather(*tasks)
        for img in og_images:
            if img and _is_valid_image_url(img) and img not in candidate_image_urls:
                candidate_image_urls.append(img)

    # Stage C: Google Images — secondary fallback only
    if serp_key:
        diagnostic_log.append("🖼️ Google Images (secondary fallback for Gemini)...")
        _mkt_excl_a = ("-site:amazon.com -site:amazon.co.uk -site:amazon.de "
                       "-site:amazon.fr -site:ebay.com -site:ebay.co.uk")
        for img_q in [f'"{ean}" {_mkt_excl_a}',
                      f'site:barcodelookup.com OR site:go-upc.com "{ean}"']:
            img_data = await _serp_get(session, serp_url,
                params={"q": img_q, "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10)
            if img_data:
                for item in img_data.get("images_results", [])[:6]:
                    url = item.get("original", "")
                    if _is_valid_image_url(url) and url not in candidate_image_urls:
                        candidate_image_urls.append(url)

    if registry_image_url and _is_valid_image_url(registry_image_url) and registry_image_url not in candidate_image_urls:
        candidate_image_urls.append(registry_image_url)

    # ── DOWNLOAD, SIZE, AND DEDUPLICATE IMAGES (for Gemini) ──
    final_downloaded_images = []
    seen_b64_prefixes = []

    for url in candidate_image_urls:
        if len(final_downloaded_images) >= 2:
            break
        img_payload = await fetch_image_bytes_simple(session, url)
        if not img_payload:
            continue
        prefix = img_payload["data"][:120]
        if prefix in seen_b64_prefixes:
            continue
        seen_b64_prefixes.append(prefix)
        final_downloaded_images.append(img_payload)

    diagnostic_log.append(f"✅ Secured {len(final_downloaded_images)} image(s) for Gemini extraction.")
    return product_name, final_downloaded_images, "\n".join(diagnostic_log), ean_verified


def run_gemini_sync(ean, product_name, market_code, gemini_key, taxonomy_text,
                    image_bytes_list, user_ground_truth, go_upc_data=None):
    """
    Path A: Sends the resolved product name + images to Gemini for structured
    food data extraction.  Uses Google Search grounding so Gemini can reach
    tier-1 retailer and brand pages directly.

    go_upc_data: optional dict from fetch_go_upc — used as search-anchoring
    context only. Never cited as an output source.

    Source exclusion rules are enforced both in this prompt and in the
    Python-side domain blocklist (_is_excluded_domain).
    """
    market_upper   = market_code.upper()
    goldmine_sites = GOLDMINE.get(market_upper, "Major Tier-1 Supermarkets")

    # Build barcode-database context string for search anchoring (never output source)
    _go_upc_ctx = ""
    if go_upc_data:
        _parts = []
        if go_upc_data.get("brand"):
            _parts.append(f"Brand: {go_upc_data['brand']}")
        if go_upc_data.get("category"):
            _parts.append(f"Category: {go_upc_data['category']}")
        if go_upc_data.get("ingredients"):
            _parts.append(f"Ingredients reference: {go_upc_data['ingredients'][:400]}")
        _go_upc_ctx = "\n".join(_parts)

    prompt = f"""
    You are the Lead Food Product Researcher.
    TARGET EAN: {ean}
    ONLINE PRODUCT NAME FOUND: {product_name}
    USER INPUT (GROUND TRUTH): {user_ground_truth if user_ground_truth else "None provided (Proceed normally)"}
    MARKET: {market_code}
    BARCODE DATABASE CONTEXT (search anchoring only — do NOT cite as output source):
    {_go_upc_ctx if _go_upc_ctx else "No additional context available."}

    CORE DIRECTIVES:
    0. VALIDATION GATE — TIERED CONFIRMATION:

       FOOD CHECK (mandatory first step): This tool covers food, drink, and pet food products. If the product you find is clearly NOT in any of these categories — for example household cleaning products, air fresheners, plug-in refills, cling wrap, aluminium foil, baking paper, personal care, cosmetics, electronics, furniture, tools, clothing, or stationery — set "is_exact_match" to false immediately and do not extract any data. NOTE: tinned fish, seafood, and all animal products for human consumption ARE food. Pet food, animal feed, and treats sold for pets are also acceptable — do not reject them.

       TIER A — Highest confidence: The exact EAN/barcode {ean} appears explicitly in the page text, URL, GTIN field, or structured data of an approved food product page. This is the strongest possible confirmation. Set food_info_reliability to "H" if Tier A is confirmed and sources are tier-1.

       TIER B — Good confidence (only when USER INPUT ground truth is non-empty): The EAN {ean} does NOT appear in page text, BUT all three of the following match the USER INPUT (GROUND TRUTH):
         - Brand name matches the product you found (minor spelling differences allowed)
         - Product name matches closely (key words present, language differences allowed)
         - Unit weight or volume matches within 15% of what USER INPUT states
       In this case, accept the product and note in chain_of_thought that EAN was confirmed via name/brand/weight match rather than page text. Set food_info_reliability to "M" at most for Tier B confirmations. If USER INPUT (GROUND TRUTH) is empty, Tier B is NOT available — only Tier A is accepted.

       TIER C — Reject immediately (set is_exact_match to false):
         - Product is clearly not food, drink, or pet food (see FOOD CHECK above)
         - Brand OR product name clearly do not match USER INPUT (for a clear, meaningful mismatch — not cosmetic differences)
         - EAN found on page but belongs to a different product
         - No food/pet-food product can be found for EAN {ean} on any approved source

       COSMETIC DIFFERENCES: Do NOT fail for minor formatting differences (e.g. 'da 100 gr.' vs '100g', abbreviations, capitalisation, language translation). Only reject for clear, meaningful mismatches. A null result is always better than wrong data.

       PACK SIZE LENIENCY (important): If USER INPUT states a multi-pack format like '4x200ml', '6x330ml', or '(4 Pack) 80g', compare only the UNIT size against what you find online. A product listed as '200ml' MATCHES a USER INPUT of '4x200ml'. A product listed as '80g' MATCHES '(4 Pack) 80g'. Do NOT fail validation because the online listing shows the individual unit while the USER INPUT describes the pack. The barcode on the pack is the same as the barcode on the unit in most retailer systems.
    1. ACCURACY: You have access to Google Search. You MUST prioritise official brand websites and approved tier-1 retailers only.
    2. SOURCE EXCLUSION (HARD CONSTRAINT): The following sources are STRICTLY FORBIDDEN — do NOT use them under any circumstances, even as a last resort:
       - Amazon (any amazon.* domain or subdomain including amazon.com, amazon.co.uk, amazon.de, etc.)
       - eBay (any ebay.* domain or subdomain including ebay.com, ebay.co.uk, ebay.de, etc.)
       - openfoodfacts.org or any Open Food Facts mirror or API
       - barcodelookup.com, go-upc.com, or similar barcode aggregator sites
       - Any wiki (including Wikipedia), forum, Reddit, social media, or user-generated database
       - Any marketplace, reseller, third-party seller, or affiliate page
       ONLY approved sources: official brand website and the Tier-1 retailers listed below.
    3. TARGET MARKET LANGUAGE: You MUST translate and output ALL product text (Ingredients, Allergens, May Contain, Dietary Info, Nutritional Context) into the native language of the TARGET MARKET ({market_code}). Write it verbatim. EXCEPTION: The 6 taxonomy categories AND the Tags (Dietary, Occasion, Seasonal) MUST remain exactly as they appear in the English lists below.
    4. MISSING DATA & SOURCE CASCADE: You MUST try to fill every field. Follow this cascade:
        STEP 1 — Official brand website (highest priority). Extract everything available.
        STEP 2 — If ANY field is still null after Step 1, search the Tier-1 retailers for {market_code}: {goldmine_sites}. Cross-reference and fill any remaining nulls.
        STEP 3 — If a field is STILL null after Steps 1 and 2, return "null" for that field. Do NOT fall back to forbidden sources (Amazon, eBay, Open Food Facts, barcodelookup, go-upc, wikis, forums, or any marketplace). Returning "null" is always better than data from a forbidden or unverified source.
        STEP 4 — Do NOT guess nutritional values. Do NOT invent ingredients. Do NOT copy from similar products.
        IMPORTANT: A brand page that has ingredients but no nutrition table is NOT a complete source. Continue to Step 2 for the missing fields.
        Do NOT attempt to deduce "May Contain" warnings from the ingredient list; only populate "May Contain" if you find an explicit warning on the source website or packaging.
    5. TAXONOMY MAPPING: Classify the product into the 6-level taxonomy provided below. You MUST use EXACT matches from the provided taxonomy. Do not invent categories. If a variant (Level 6) doesn't exist for the item category, return "None". Explain your reasoning in the "categorization_reasoning" field.
    5b. TAXONOMY FIRST-PRINCIPLES (INTENDED USE RULE): Before mapping to the taxonomy, determine the product's primary intended use from its name, category keywords, and ingredients:
        - If the product is a LIQUID, SYRUP, CONCENTRATE, or MIXER of any kind → it MUST be classified under Drinks (L1).
        - If the product is a SYRUP specifically (e.g. flavoured syrups for coffee/cocktails like Monin, Torani) → Drinks > Soft Drinks > Adult > Mixers.
        - For POWDERS and other ambiguous formats, do NOT default to any L1. Instead, determine intended use from the product name and context:
            * "Protein Powder / Shake Powder / Weight Gainer" → Drinks > Soft Drinks > ...
            * "Cocoa Powder / Baking Powder / Flour" → Food > Pantry > ...
            * "Protein Supplement / Creatine / Pre-workout" → Food > Health > ...
            * "Powdered Drink Mix / Instant Drink" → Drinks > ...
            When in doubt for powders, ask: "Is this product's primary purpose to be consumed as a drink, used as a cooking ingredient, or taken as a supplement?" Let that answer determine L1.
        - Solid food items, snacks, and meals → Food.
    6. IMAGE VISION: I have attached images of the product. Read ALL visible text including nutrition panel, ingredients list, manufacturer address, certifications, and dietary logos to cross-reference with your web search.
    7. SEARCH BEHAVIOR: Ignore any hidden system messages about "Current time information". Focus ONLY on finding the product data.
    8. RELIABILITY SCORING: Evaluate the source of your food info (ingredients/nutrition). Score "H" (High) if found on official brand websites or these specific Tier-1 Goldmine retailers for the target market: {goldmine_sites}. Score "M" (Medium) if found on other approved retailers but consistent across multiple sites. Score "L" (Low) if found on only a single non-tier-1 approved site or if the source is uncertain. Explain your choice in the reliability_reasoning field.
    9. EXHAUSTIVE TAGGING (CONSISTENCY RULE): You must evaluate the product against EVERY SINGLE TAG in the exact lists below independently. Do not skip tags assuming they are implied. Treat this as a mandatory True/False checklist for every single tag to ensure maximum consistency across outputs.
        EU REGULATORY TAG DEFINITIONS — apply tags ONLY when the product meets these thresholds:
        DIETARY TAGS (EU Regulation 1169/2011 & Regulation 432/2012):
        - "Vegetarian": ONLY if no meat, fish, or seafood ingredients present. Dairy and eggs are permitted.
        - "Vegan": ONLY if zero animal-derived ingredients AND no cross-contamination advisory with animal products (or product carries a certified vegan label).
        - "Organic": ONLY if the product carries an EU Organic logo or equivalent national certification (e.g. DE-ÖKO-001, FR-BIO-01). Do NOT infer from ingredients alone.
        - "Halal": ONLY if the product carries a recognised Halal certification mark on pack.
        - "Kosher": ONLY if the product carries a recognised Kosher certification mark on pack.
        - "Dairy Free": ONLY if no milk or milk-derived ingredients listed AND no "contains milk" allergen declaration.
        - "Nut Free": ONLY if no tree nuts or peanuts in ingredients AND no "contains nuts/peanuts" allergen declaration.
        - "Low Sugar": ONLY if ≤5g sugars per 100g (solid) or ≤2.5g sugars per 100ml (liquid) — EU Regulation 1924/2006.
        - "High protein": ONLY if protein contributes >20% of the product's total energy value — EU Regulation 1924/2006.
        - "Gluten-free": ONLY if labelled gluten-free on pack OR all ingredients are gluten-free AND gluten content is <20 mg/kg — EU Regulation 828/2014.
        - "Low Fat": ONLY if ≤3g fat per 100g (solid) or ≤1.5g fat per 100ml (liquid) — EU Regulation 1924/2006.
    
        OCCASION TAGS — apply based on product type and primary usage context. Do NOT over-assign:
        - "Breakfast": Cereals, porridge, breakfast biscuits, morning drinks, pastries.
        - "Lunchbox": Individually portioned snacks, sandwich accompaniments, small-format items.
        - "BBQ": Condiments, marinades, grillable meats, charcoal, disposable BBQ accessories.
        - "Party": Multi-serve sharing formats, party snack packs, celebration cakes, mixers/soft drinks in large format.
        - "Christmas": Only if the product is explicitly Christmas-themed or a recognised Christmas food/drink tradition (e.g. mince pies, mulled wine spice).
        - "Ramadan": Only if product is specifically marketed for Ramadan or is a traditional Ramadan food (e.g. dates, harira).
        - "Meal prep": Bulk staples, dry goods in large quantities, ingredient-focused products.
        - "Quick dinner": Ready meals, instant noodles, stir-in sauces, products with <15 min prep.
        - "Kids snack": Products explicitly marketed at children OR inherently child-targeted by format/size/packaging.
    
        SEASONAL TAGS — apply ONLY if the SKU is specifically marketed or packaged for that season. Default is empty:
        - "Christmas": Limited-edition Christmas packaging or an explicitly seasonal SKU.
        - "Easter": Limited-edition Easter packaging.
        - "Back to School": Explicitly back-to-school themed products.
        - "Valentines Day": Explicitly Valentine's Day themed.
        - "Mothers Day": Explicitly Mother's Day themed.
        - "Halloween": Limited-edition Halloween packaging.
        - "Other": A seasonal angle exists but fits none of the above.
        - If NONE of the above apply, return "" (empty string) for seasonal_tags. Do NOT default to "Other".
        
    --- START TAXONOMY REFERENCE (CSV FORMAT) ---
    {taxonomy_text}
    --- END TAXONOMY REFERENCE ---

    CRITICAL JSON RULES:
    - YOUR ENTIRE RESPONSE MUST BE A SINGLE VALID JSON OBJECT. NO EXCEPTIONS.
    - NEVER write conversational text outside the JSON object. All thoughts, summaries, and reasoning MUST go inside the "chain_of_thought" field.
    - EVEN IF YOU FIND ABSOLUTELY NO DATA, YOU MUST RETURN THE JSON WITH ALL FIELDS SET TO "null". NEVER ABORT OR SKIP THE JSON.
    - To avoid RECITATION errors (copyright filters), do NOT copy-paste long paragraphs of text verbatim. You MUST paraphrase and summarize descriptions in your own words.
    - JSON REQUIRES double quotes (") for keys and string values. You MUST use double quotes for the JSON structure.
    - If you need to use quotes INSIDE a string value, use single quotes ('). NEVER use unescaped double quotes inside a value.
    - Do not use literal newlines/tabs inside strings.

    SCHEMA:
    {{
        "is_exact_match": true or false,
        "chain_of_thought": "Step-by-step reasoning. If validation failed, explain why. If passed, briefly explain how you found the data, translated it, and read the images. Include which sources you used and confirm EAN match.",
        "food_info_reliability": "H, M, or L",
        "reliability_reasoning": "Explain why H, M, or L was assigned based on the specific URLs/sources used",
        "category_1": "Level 1 Category (English)",
        "category_2": "Level 2 Category (English)",
        "category_3": "Level 3 Category (English)",
        "category_4": "Level 4 Category (English)",
        "category_5": "Level 5 Category (English)",
        "category_6": "Level 6 Variant or None (English)",
        "categorization_reasoning": "Brief explanation of why these categories were chosen",
        "dietary_tags": "Comma-separated tags from the exact Dietary list (English)",
        "occasion_tags": "Comma-separated tags from the exact Occasion list (English)",
        "seasonal_tags": "Comma-separated tags from the exact Seasonal list (English)",
        "tagging_reasoning": "Brief explanation for the chosen tags. Be concise. Only explain assigned tags, do not explain rejected ones.",
        "brand": "Brand Name",
        "uom": "Strictly write 'g' (or 'ml' for liquids). Do not write 'gram', 'grams', 'gr'.",
        "packaging": "Packaging type (e.g., Box, Bottle, Wrapper)",
        "fragile_item": "Yes or No",
        "net_weight": "Weight/Volume number only",
        "gross_weight": "Gross weight if found, else null",
        "organic_product": "Yes or No",
        "net_weight_customer_facing": "How weight is displayed on pack",
        "ingredients": "Full list as a single string (Translated to {market_code} language)",
        "allergens": "List as a single string (Translated to {market_code} language)",
        "may_contain": "List as a single string (Translated to {market_code} language)",
        "nutritional_info": "Context (e.g., per 100g or per serving) (Translated to {market_code} language)",
        "manufacturer_name": "Legal name of the manufacturer company",
        "manufacturer_address": "Full address",
        "place_of_origin": "Country/Region of origin",
        "organic_certification_id": "e.g., DE-ÖKO-001 or null",
        "energy_kj": "Value in kJ",
        "fat_g": "Value",
        "saturates_g": "Value",
        "carbohydrates_g": "Value",
        "sugars_g": "Value",
        "protein_g": "Value",
        "fiber_g": "Value",
        "salt_g": "Value",
        "sources": ["Array of full URLs (starting with https://) from approved sources only — NO Amazon, eBay, OpenFoodFacts, or any forbidden domain"]
    }}
    """

    client = genai.Client(api_key=gemini_key)

    contents_payload = [prompt]
    for img in image_bytes_list:
        contents_payload.append(
            types.Part.from_bytes(data=img["data"], mime_type=img["mime"])
        )

    last_error = "Unknown error"

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=EXTRACTION_MODEL,
                contents=contents_payload,
                config=types.GenerateContentConfig(
                    temperature=0.25,
                    tools=[{"google_search": {}}],
                    max_output_tokens=8192,
                    # NOTE: To disable thinking tokens on gemini-2.5-flash and control costs,
                    # uncomment the line below (requires google-genai >= 0.8.0):
                    # thinking_config=types.ThinkingConfig(thinking_budget=0),
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        )
                    ]
                )
            )

            if not response.candidates:
                raw_resp_str = str(response)[:500].replace('\n', ' ')
                raise Exception(f"Request blocked entirely. Raw response: {raw_resp_str}")

            raw_text = ""
            if response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, 'text', None):
                        raw_text += part.text + "\n"

            if not raw_text.strip() and getattr(response, 'text', None):
                raw_text = response.text

            raw_text = raw_text.strip() if raw_text else ""

            if not raw_text:
                candidate    = response.candidates[0]
                finish_reason = candidate.finish_reason
                raise Exception(f"Empty text extracted. Reason: {finish_reason}")

            # ── Collect grounding redirect URLs as supplementary source references ──
            grounding_urls = []
            try:
                if response.candidates and response.candidates[0].grounding_metadata:
                    metadata = response.candidates[0].grounding_metadata
                    if metadata.grounding_chunks:
                        for chunk in metadata.grounding_chunks:
                            if chunk.web and chunk.web.uri:
                                grounding_urls.append(chunk.web.uri)
            except Exception:
                pass

            start_idx = raw_text.find('{')
            end_idx   = raw_text.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean_json = raw_text[start_idx:end_idx+1]
            else:
                rogue_preview = raw_text[:200].replace('\n', ' ')
                raise Exception(f"Could not find JSON object. AI wrote: {rogue_preview}...")

            data = json.loads(clean_json, strict=False)

            # ── Merge sources: model-claimed URLs first, grounding redirects appended ──
            # Model sources are the real claimed domains; grounding redirects are
            # supplementary references (Vertex redirect URLs, not real domains).
            # Filter out any forbidden domains from model-claimed sources.
            model_sources = data.get("sources", [])
            if isinstance(model_sources, str):
                model_sources = [s.strip() for s in model_sources.split(",") if s.strip()]
            elif not isinstance(model_sources, list):
                model_sources = []

            # Hard filter: remove any forbidden domains that Gemini may have included
            model_sources = [s for s in model_sources if not _is_excluded_domain(s)]

            # Merge: model sources take priority; grounding redirects appended as refs
            all_sources = list(dict.fromkeys([*model_sources, *grounding_urls]))
            data["sources"] = all_sources

            return data

        except json.JSONDecodeError as e:
            last_error = f"JSON Error: {str(e)}"
        except Exception as e:
            last_error = str(e)

        if attempt < 2:
            time.sleep(3)

    return {"error": f"API Error (Failed after 3 attempts). Last error: {last_error}"}


# ============================================================================
# PATH B: DISPLAY IMAGE PIPELINE
# ============================================================================

DISPLAY_MIN_LONG_EDGE_PX       = 120
DISPLAY_ASPECT_MIN             = 0.3
DISPLAY_ASPECT_MAX             = 3.0
DISPLAY_PHASH_DUPLICATE_THRESHOLD = 5
MAX_DISPLAY_IMAGES             = 2

DISPLAY_BAD_PATH_TOKENS = {
    "logo", "icon", "banner", "placeholder", "spinner", "loading",
    "payment", "paypal", "mastercard", "visa", "social",
    "pinterest", "tiktok", "facebook", "instagram", "twitter", "youtube",
    "thumbnail", "thumb", "avatar", "favicon", "sprite",
}

# Hard-excluded domain substrings for image URLs.
# This covers product image CDNs as well as storefront URLs.
DISPLAY_BAD_SUBSTRINGS = {
    # ── Forbidden marketplaces and UGC databases ──
    "openfoodfacts",
    # Amazon — storefront domains and product image CDNs
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr",
    "amazon.it", "amazon.es", "amazon.nl", "amazon.pl", "amazon.be",
    "amazon.se", "amazon.dk", "amazon.at",
    "m.media-amazon.com", "images-amazon.com", "media-amazon.com",
    # eBay — storefront domains and image CDN
    "ebay.com", "ebay.co.uk", "ebay.de", "ebay.fr",
    "ebay.it", "ebay.es", "ebay.nl", "ebay.pl", "ebay.be",
    "i.ebayimg.com", "ebayimg.com",
    # ── Ad networks ──
    "amazon-ads", "ad_servlet", "doubleclick",
    # ── Size indicators ──
    "_xs.", "_xxs.", "/icons/", "/logos/",
}

DISPLAY_SOURCE_PRIORITY = {
    "jsonld_retailer":     100,  # scraped from page confirmed to contain the EAN
    "go_upc_tier1":         95,  # Go-UPC API — barcode-keyed, S3-hosted image
    "og_retailer":          80,  # OG tag from retailer page
    "shopping_result":      75,  # Google Shopping — EAN-matched, retailer-submitted
    "twitter_retailer":     70,  # Twitter card from retailer page
    "serpapi_strict":       60,  # Google Images EAN search (marketplace-excluded)
    "serpapi_barcode":      55,  # Barcode-site image search
    "serpapi_name":         50,  # Google Images name+EAN search
    "serpapi_hailmary":     45,  # Google Images generic fallback
    "ean_search_registry":  40,  # EAN-Search.org registry image
}

DISPLAY_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

EAN_SEARCH_GARBAGE_PATTERNS = [
    r"upc lookup", r"ninguno", r"^lookup ", r"barcode\s+\d",
    r"unknown product", r"^no\s+(name|product)", r"^[\d\s#]+$", r"#####",
]


def is_garbage_name(name: str) -> bool:
    if not name or len(name.strip()) < 3:
        return True
    name_lower = name.lower().strip()
    return any(re.search(p, name_lower) for p in EAN_SEARCH_GARBAGE_PATTERNS)


def _check_display_url(url: str):
    """URL filter for the display pipeline. Returns (is_valid, reject_reason)."""
    if not url or not url.startswith("http"):
        return False, "Not a valid http(s) URL"
    # Hard-exclude marketplace / UGC domains before any other check
    if _is_excluded_domain(url):
        return False, f"Hard-excluded domain (marketplace/UGC): {url[:80]}"
    url_lower = url.lower()
    path      = url_lower.split("?")[0]

    for ext in BAD_IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return False, f"Bad extension: {ext}"
    for substr in DISPLAY_BAD_SUBSTRINGS:
        if substr in url_lower:
            return False, f"Domain/substring blocklist: '{substr}'"
    tokens = set(re.split(r"[/_\-.]+", path))
    for token in tokens:
        if token in DISPLAY_BAD_PATH_TOKENS:
            return False, f"Path token blocklist: '{token}'"
    if "media-amazon.com" in url_lower and ("," in url_lower or "_bo" in url_lower):
        return False, "Amazon UI composite (commas / _bo modifier)"
    return True, ""


class ImageDiagnostics:
    """Per-EAN diagnostics for the display image pipeline."""

    def __init__(self, ean):
        self.ean            = ean
        self.text_log       = []
        self.candidates     = []
        self.final_selected = []
        self.image_2_failure = ""

    def log(self, msg):
        self.text_log.append(msg)

    def log_candidate(self, source, url, status, reason="", width=None, height=None):
        self.candidates.append({
            "source":   source,
            "url":      url[:120] + "..." if len(url) > 120 else url,
            "full_url": url,
            "status":   status,
            "reason":   reason,
            "width":    width,
            "height":   height,
        })

    def status_counts(self):
        counts = {}
        for c in self.candidates:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        return counts

    def summary_string(self):
        counts = self.status_counts()
        parts  = [f"{k}={v}" for k, v in counts.items()]
        return (f"Selected {len(self.final_selected)}/{MAX_DISPLAY_IMAGES} images "
                f"from {len(self.candidates)} candidates. " + ", ".join(parts))

    def to_dict_list(self):
        return [{
            "Source": c["source"],
            "Status": c["status"],
            "Reason": c["reason"],
            "Width":  c["width"],
            "Height": c["height"],
            "URL":    c["url"],
        } for c in self.candidates]


async def display_extract_from_page(session, url):
    """Scrape JSON-LD + OG + Twitter images from a retailer page."""
    # Hard guard: never scrape excluded marketplace pages
    if _is_excluded_domain(url):
        return []
    images  = []
    headers = {"User-Agent": DISPLAY_USER_AGENTS[0]}
    try:
        async with session.get(url, headers=headers, timeout=8) as resp:
            if resp.status != 200:
                return images
            html = await resp.text()

            for match in re.finditer(
                r'<script[^>]*type=[\'"]application/ld\+json[\'"][^>]*>(.*?)</script>',
                html, re.S | re.I
            ):
                try:
                    data  = json.loads(match.group(1).strip())
                    items = data if isinstance(data, list) else [data]
                    expanded = []
                    for it in items:
                        if isinstance(it, dict) and "@graph" in it:
                            expanded.extend(it["@graph"])
                        else:
                            expanded.append(it)
                    for item in expanded:
                        if not isinstance(item, dict):
                            continue
                        item_type  = item.get("@type", "")
                        is_product = (("Product" in item_type) if isinstance(item_type, list)
                                      else (item_type == "Product"))
                        if is_product:
                            img = item.get("image")
                            if isinstance(img, list):
                                for i in img:
                                    if isinstance(i, str) and not _is_excluded_domain(i):
                                        images.append(("jsonld_retailer", i))
                                    elif isinstance(i, dict) and i.get("url") and not _is_excluded_domain(i["url"]):
                                        images.append(("jsonld_retailer", i["url"]))
                            elif isinstance(img, str) and not _is_excluded_domain(img):
                                images.append(("jsonld_retailer", img))
                            elif isinstance(img, dict) and img.get("url") and not _is_excluded_domain(img["url"]):
                                images.append(("jsonld_retailer", img["url"]))
                except Exception:
                    continue

            og = re.search(r'<meta[^>]*property=[\'"]og:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]', html, re.I)
            if og and not _is_excluded_domain(og.group(1)):
                images.append(("og_retailer", og.group(1)))
            tw = re.search(r'<meta[^>]*name=[\'"]twitter:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]', html, re.I)
            if tw and not _is_excluded_domain(tw.group(1)):
                images.append(("twitter_retailer", tw.group(1)))
    except Exception:
        pass

    seen   = set()
    unique = []
    for src, u in images:
        if u not in seen:
            seen.add(u)
            unique.append((src, u))
    return unique


async def display_fetch_image_bytes(session, url):
    """Display pipeline image fetcher with UA rotation on 403/429."""
    last_status = None
    last_error  = None
    for attempt, ua in enumerate(DISPLAY_USER_AGENTS):
        headers = {
            "User-Agent":      ua,
            "Accept":          "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":         "https://www.google.com/",
        }
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                last_status = resp.status
                if resp.status == 200:
                    data = await resp.read()
                    mime = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
                    return {"url": url, "mime": mime, "data": data}
                if resp.status in (403, 429) and attempt < len(DISPLAY_USER_AGENTS) - 1:
                    continue
                return {"error": f"HTTP {resp.status}"}
        except asyncio.TimeoutError:
            last_error = "timeout"
            if attempt < len(DISPLAY_USER_AGENTS) - 1:
                continue
        except Exception as e:
            last_error = str(e)[:80]
            if attempt < len(DISPLAY_USER_AGENTS) - 1:
                continue
    return {"error": f"HTTP {last_status} after retries" if last_status else f"Network error: {last_error}"}


def display_inspect_image(data: bytes):
    """PIL inspection for display pipeline."""
    if not IMAGE_LIBS_OK:
        return {"width": None, "height": None, "ok": len(data) >= 5000, "reason": "PIL unavailable"}
    try:
        img  = Image.open(BytesIO(data))
        w, h = img.size
        if h == 0:
            return {"width": w, "height": h, "ok": False, "reason": "Zero height"}
        aspect   = w / h
        long_edge = max(w, h)
        if long_edge < DISPLAY_MIN_LONG_EDGE_PX:
            return {"width": w, "height": h, "long_edge": long_edge, "aspect": aspect,
                    "ok": False, "reason": f"Too small ({long_edge}px, need {DISPLAY_MIN_LONG_EDGE_PX}+)"}
        if aspect < DISPLAY_ASPECT_MIN or aspect > DISPLAY_ASPECT_MAX:
            return {"width": w, "height": h, "long_edge": long_edge, "aspect": aspect,
                    "ok": False, "reason": f"Wrong aspect ({aspect:.2f}, outside {DISPLAY_ASPECT_MIN}-{DISPLAY_ASPECT_MAX})"}
        return {"width": w, "height": h, "long_edge": long_edge, "aspect": aspect, "ok": True, "reason": ""}
    except Exception as e:
        return {"width": None, "height": None, "ok": False, "reason": f"Unreadable: {str(e)[:60]}"}


def display_compute_phash(data: bytes):
    if not IMAGE_LIBS_OK:
        return None
    try:
        return imagehash.phash(Image.open(BytesIO(data)))
    except Exception:
        return None


def display_phash_distance(h1, h2):
    if h1 is None or h2 is None:
        return 999
    try:
        return h1 - h2
    except Exception:
        return 999


_BAD_TITLE_TOKENS = {
    "hardware", "tool", "screw", "bolt", "nut ", "drill", "plumb", "pipe",
    "furniture", "sofa", "chair", "table", "bed ", "bath", "shower", "toilet",
    "clothing", "shirt", "shoe", "dress", "jacket",
    "electronic", "laptop", "phone", "cable", "printer", "router",
    "paint", "roller", "brush", "wallpaper",
    "vehicle", "car ", "truck", "tyre", "tire",
    "garden", "plant", "flower", "seed ",
    "safari", "tour", "travel", "lodge", "camp ", "wildlife",
}


def _title_looks_like_food(title: str, product_name: str) -> bool:
    """Quick text check: does the image title/source seem food-related?"""
    if not title:
        return True
    title_lower = title.lower()
    for bad in _BAD_TITLE_TOKENS:
        if bad in title_lower:
            return False
    if product_name:
        pwords = [w.lower() for w in product_name.split() if len(w) > 3]
        if any(w in title_lower for w in pwords):
            return True
    return True


# ============================================================================
# IMAGE VERIFIER  (rewritten: VISION_MODEL, float return, fail-closed)
# ============================================================================

def verify_image_with_gemini(image_data: bytes, mime: str,
                              product_name: str, brand: str,
                              gemini_key: str,
                              weight_hint: str = "") -> float:
    """
    Identity verification: how confident are we that this image shows
    THIS specific product (correct brand, variant, AND weight/volume)?

    Returns:
      -1.0  : API unavailable / network error (cannot make a judgement)
      0.0–1.0 : Model's confidence score (0 = definitively wrong product)

    The sentinel -1.0 lets callers distinguish "model said no" from
    "model couldn't run" — trusted sources allow -1.0 through, untrusted
    sources reject it (fail-closed).

    weight_hint: optional weight/volume string extracted from ground truth
    (e.g. "330ml", "100g") appended to the label for tighter matching.

    Model: VISION_MODEL (gemini-2.5-flash-lite).
    """
    if not gemini_key or not image_data:
        return -1.0   # can't verify — caller decides what to do

    base_label = f"{brand} {product_name}".strip() if brand else (product_name or "")
    label      = f"{base_label} {weight_hint}".strip() if weight_hint else base_label
    prompt = (
        f"You are verifying a product photo for a food database.\n"
        f"Target product: '{label}'.\n\n"
        f"Reply with ONLY a number 0-100 representing your confidence that this image "
        f"shows the packaging of THAT SPECIFIC product (same brand AND same variant/flavour).\n\n"
        f"Scoring guide:\n"
        f"  0-15  : Non-food item — this includes: hardware, tools, furniture, "
        f"landscapes, people, logos, blank barcodes, safari/travel imagery, "
        f"household cleaning products, air fresheners, plug-in refills, "
        f"cling wrap, aluminium foil, baking paper, film rolls, "
        f"personal care products, cosmetics, pet products, electronics, "
        f"stationery, or ANY item that is not food or drink for human consumption.\n"
        f"  16-45 : Food or drink item but clearly a DIFFERENT brand or product variant.\n"
        f"  46-79 : Plausibly the right product but brand or variant is unclear.\n"
        f"  80-100: Brand name and product visibly match '{label}'.\n\n"
        f"Respond with a single integer only. No explanation."
    )
    try:
        client = genai.Client(api_key=gemini_key)
        resp   = client.models.generate_content(
            model=VISION_MODEL,
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=mime),
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=8,
            ),
        )
        txt = ""
        if resp.candidates and resp.candidates[0].content:
            for part in resp.candidates[0].content.parts:
                if getattr(part, "text", None):
                    txt += part.text
        m = re.search(r"\d{1,3}", txt)
        return (min(100, int(m.group())) / 100.0) if m else 0.0
    except Exception:
        return -1.0   # API error — caller decides; -1 != "model said wrong"


# Sources that come directly from a brand/retailer product page.
# These still undergo vision verification but with a lower threshold (0.35)
# because the page-scraping context gives us higher prior confidence.
# IMPORTANT: this trust is only applied after _is_excluded_domain() confirms
# the URL is not from a hard-excluded marketplace.
TRUSTED_SOURCES = {"jsonld_retailer", "og_retailer", "twitter_retailer", "ean_search_registry", "go_upc_tier1"}


async def display_evaluate_candidate(session, source, original_url, thumbnail_url, diag,
                                     gemini_key="", product_name="", brand="",
                                     title="", src_domain="", weight_hint=""):
    """
    Download + inspect a single candidate display image.
    weight_hint: optional weight/volume string (e.g. "330ml") passed to the
    vision verifier so it can distinguish weight variants of the same product.
    Strategy:
      1. Hard reject any URL from an excluded marketplace/UGC domain.
      2. Pre-filter on title/source text (instant, free).
      3. Try original URL; fall back to Google thumbnail on error.
      4. PIL dimension check.
      5. Gemini vision verification (mandatory for all sources).
         - Excluded-domain URLs NEVER reach this step.
         - 'Trusted' source images use a lower threshold (0.35).
         - All other images use IMAGE_MATCH_THRESHOLD (0.55).
         - FAILS CLOSED: returns 0.0 on error, never silently allows.
    """
    url_to_use = original_url or thumbnail_url
    if not url_to_use:
        return None

    # ── Hard reject: excluded marketplace / UGC image URL ──
    if _is_excluded_domain(url_to_use):
        diag.log_candidate(source, url_to_use, "rejected_excluded_domain",
                           f"Hard-excluded domain: {url_to_use[:80]}")
        return None

    # A source is 'effectively trusted' only when it is in TRUSTED_SOURCES
    # AND the URL itself is not from an excluded domain.
    # This guards against Amazon/eBay listing pages being tagged as jsonld_retailer.
    effective_trusted = (source in TRUSTED_SOURCES) and not _is_excluded_domain(url_to_use)

    # ── Pre-filter: reject obviously wrong content by title ──
    if not effective_trusted and not _title_looks_like_food(title, product_name):
        diag.log_candidate(source, url_to_use, "rejected_title",
                           f"Title suggests non-food content: '{title[:80]}'")
        return None

    # ── Try original URL first ──
    payload  = None
    used_url = None

    if original_url:
        is_valid, reject_reason = _check_display_url(original_url)
        if is_valid:
            payload = await display_fetch_image_bytes(session, original_url)
            if payload and "error" not in payload:
                used_url = original_url
            else:
                diag.log(f"   ⚠️ Original URL failed ({(payload or {}).get('error','no response')}), trying thumbnail...")
        else:
            diag.log(f"   ⚠️ Original URL pre-rejected: {reject_reason}")

    # ── Fall back to Google thumbnail ──
    if (not payload or "error" in payload) and thumbnail_url:
        thumb_valid, _ = _check_display_url(thumbnail_url)
        if thumb_valid:
            payload = await display_fetch_image_bytes(session, thumbnail_url)
            if payload and "error" not in payload:
                used_url = thumbnail_url
                diag.log(f"   ✅ Using Google thumbnail fallback: {thumbnail_url[:80]}")

    if not payload or "error" in payload:
        err = payload.get("error", "unknown") if payload else "no response"
        diag.log_candidate(source, url_to_use, "rejected_download", err)
        return None

    # ── Dimension check ──
    inspection = display_inspect_image(payload["data"])
    if not inspection["ok"] and used_url != thumbnail_url:
        diag.log_candidate(source, used_url, "rejected_dimensions", inspection["reason"],
                           width=inspection.get("width"), height=inspection.get("height"))
        return None

    safe_mime = _safe_mime(payload["mime"], payload["data"]) or "image/jpeg"

    # ── Vision verification ───────────────────────────────────────────────────
    # TRUSTED sources (brand/retailer pages that passed _is_excluded_domain):
    #   These are scraped directly from known-good product pages.
    #   Amazon/eBay are already blocked by effective_trusted above.
    #   No vision verification — the source page is the trust signal.
    # ── Vision verification ───────────────────────────────────────────────────
    # Both trusted and untrusted sources are verified.
    # The sentinel value from verify_image_with_gemini:
    #   -1.0  : API error / model unavailable
    #    0.0–1.0 : Model score (0 = definitively wrong product)
    #
    # TRUSTED sources (brand/retailer pages, not Amazon/eBay):
    #   -1.0 (API error)           → allow with neutral score (source trust applies)
    #   0.0 – threshold            → REJECT: model clearly says wrong product
    #   threshold – 1.0            → allow
    #   Threshold: IMAGE_MATCH_THRESHOLD_TRUSTED (0.35) — lenient because
    #   page-scraping context already gives prior confidence.
    #
    # UNTRUSTED sources (SerpAPI image results):
    #   -1.0 or < threshold        → REJECT (fail-closed, no benefit of the doubt)
    #   Threshold: IMAGE_MATCH_THRESHOLD (0.55)
    #
    _product_known = (product_name and
                      not product_name.startswith("Product with EAN"))
    match_score = 0.5   # default neutral

    if gemini_key and _product_known:
        raw = await asyncio.to_thread(
            verify_image_with_gemini,
            payload["data"], safe_mime, product_name, brand, gemini_key,
            weight_hint  # Change 7: include weight for tighter variant matching
        )

        if effective_trusted:
            if raw < 0:
                # API error — allow trusted source; log for transparency
                diag.log(f"   ⚠️ Trusted — vision API unavailable; showing: {used_url[:80]}")
                match_score = 0.5
            elif raw < IMAGE_MATCH_THRESHOLD_TRUSTED:
                # Model says clearly wrong product — reject even trusted source
                diag.log_candidate(
                    source, used_url, "rejected_vision",
                    f"Trusted source — vision score {raw:.2f} < {IMAGE_MATCH_THRESHOLD_TRUSTED:.2f} "
                    f"(model says wrong product: '{brand} {product_name}')",
                    width=inspection.get("width"), height=inspection.get("height")
                )
                return {
                    "url": used_url, "mime": safe_mime, "data": None,
                    "source": source, "inspection": inspection,
                    "phash": None, "display": False, "match": raw,
                }
            else:
                match_score = raw
                diag.log(f"   ✅ Trusted vision verified (score={raw:.2f}): {used_url[:80]}")
        else:
            # Untrusted: fail-closed — both API error and low score → reject
            if raw < 0 or raw < IMAGE_MATCH_THRESHOLD:
                diag.log_candidate(
                    source, used_url, "rejected_vision",
                    f"Untrusted — score {raw:.2f} (API error or wrong product for '{brand} {product_name}')",
                    width=inspection.get("width"), height=inspection.get("height")
                )
                return {
                    "url": used_url, "mime": safe_mime, "data": None,
                    "source": source, "inspection": inspection,
                    "phash": None, "display": False, "match": max(raw, 0),
                }
            match_score = raw
            diag.log(f"   ✅ Vision verified (score={raw:.2f}): {used_url[:80]}")

    elif not effective_trusted:
        # No Gemini key or unknown product name — reject untrusted source
        reason = "GEMINI_API_KEY not set" if not gemini_key else "Product name unknown"
        diag.log_candidate(source, used_url, "rejected_no_verify", reason)
        return None
    else:
        # Trusted source, no product name or no API key — allow with neutral score
        diag.log(f"   ⚠️ Trusted source — no verification possible: {used_url[:80]}")

    return {
        "url":        used_url,
        "mime":       safe_mime,
        "data":       payload["data"],
        "source":     source,
        "inspection": inspection,
        "phash":      display_compute_phash(payload["data"]),
        "display":    True,
        "match":      match_score,
    }


def display_select(candidates, diag):
    """
    Pick top MAX_DISPLAY_IMAGES for rendering.
    Scoring: source priority + resolution + vision match confidence.
    """
    if not candidates:
        diag.image_2_failure = "No candidate images survived quality checks."
        return []

    displayable    = [c for c in candidates if c.get("display", True) and c.get("data")]
    rejected_links = [c for c in candidates if not c.get("display", True)]

    if rejected_links:
        diag.log(f"   ℹ️ {len(rejected_links)} image(s) rejected by vision — URLs preserved for Image Source Link.")

    def score(c):
        base      = DISPLAY_SOURCE_PRIORITY.get(c["source"], 30)
        long_edge = c["inspection"].get("long_edge") or 0
        match     = c.get("match", 0.5)
        # Vision match is weighted heavily — a smaller but correctly identified
        # product image outranks a large but uncertain one.
        return base + min(20, long_edge / 75) + 40 * match

    sorted_cands = sorted(displayable, key=score, reverse=True)

    selected        = []
    selected_hashes = []
    for c in sorted_cands:
        if any(display_phash_distance(c["phash"], h) < DISPLAY_PHASH_DUPLICATE_THRESHOLD
               for h in selected_hashes):
            diag.log_candidate(
                c["source"], c["url"], "rejected_dedup",
                f"Near-identical to selected (phash < {DISPLAY_PHASH_DUPLICATE_THRESHOLD})",
                width=c["inspection"].get("width"), height=c["inspection"].get("height")
            )
            continue
        selected.append(c)
        if c["phash"] is not None:
            selected_hashes.append(c["phash"])
        diag.log_candidate(
            c["source"], c["url"], "selected",
            f"Display image #{len(selected)} "
            f"(score={score(c):.0f}, match={c.get('match',0):.2f}, "
            f"{c['inspection']['width']}x{c['inspection']['height']})",
            width=c["inspection"]["width"], height=c["inspection"]["height"]
        )
        if len(selected) >= MAX_DISPLAY_IMAGES:
            break

    diag.final_selected = [c["url"] for c in selected]

    if len(selected) < MAX_DISPLAY_IMAGES:
        if len(displayable) > 1:
            diag.image_2_failure = "Multiple candidates found but only 1 unique image (rest were near-duplicates)."
        elif len(displayable) == 0 and rejected_links:
            diag.image_2_failure = "No displayable image found — see Image Source Link to access manually."
        else:
            diag.image_2_failure = "Only 1 candidate image survived quality checks."

    return selected


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


def _strip_pack_notation(text: str) -> str:
    """
    Remove pack-size notation from ground truth before using it in search
    queries or weight comparisons.  Prevents "4x200ml" or "(4 Pack)" from
    confusing retailer lookups into returning the wrong pack variant.
    Examples:
      "Eisberg Be Free Sparkling Rosé (4 Pack), 4x200ml"
        → "Eisberg Be Free Sparkling Rosé 200ml"
      "McVities HobNobs 6x255g"
        → "McVities HobNobs 255g"
    """
    if not text:
        return text
    # Remove "(N Pack)" / "(Pack of N)" style annotations
    text = re.sub(r'\(\s*\d+\s*[Pp]ack[^)]*\)', '', text)
    text = re.sub(r'\(\s*[Pp]ack\s+of\s+\d+[^)]*\)', '', text)
    # Convert "NxVolume" → "Volume" (keep the unit size, drop the count)
    text = re.sub(
        r'\b\d+\s*[xX×]\s*(\d+(?:[.,]\d+)?\s*(?:g|gr|kg|ml|cl|l|oz|lb))\b',
        r'\1', text, flags=re.IGNORECASE
    )
    # Clean up extra punctuation left behind
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s{2,}', ' ', text).strip().strip(',').strip()
    return text


def _extract_brand(ground_truth: str) -> str:
    """
    Extract a brand token from ground_truth.
    Skips common food/descriptive words and short tokens.
    """
    if not ground_truth:
        return ""
    tokens = ground_truth.split()
    for token in tokens:
        clean = token.strip(".,;:()-/")
        if len(clean) < 3:
            continue
        if not clean[0].isupper():
            continue
        if clean.lower() in _BRAND_STOPWORDS:
            continue
        if clean.replace(".", "").isdigit():
            continue
        return clean
    return ""


def _brand_matches_domain(brand: str, domain: str) -> bool:
    """
    Fuzzy brand-to-domain match, stripping apostrophes, hyphens, and
    spaces before comparing.  Handles cases like:
      "Hellmann's" -> "hellmanns"  matches "uk.hellmanns.com"
      "Coca-Cola"  -> "cocacola"   matches "coca-cola.com"
    Requires at least 4 clean characters to avoid false positives.
    """
    if not brand or not domain:
        return False
    brand_clean  = re.sub(r"[^a-z0-9]", "", brand.lower())
    domain_clean = re.sub(r"[^a-z0-9.]", "", domain.lower())
    return len(brand_clean) >= 4 and brand_clean in domain_clean


async def fetch_display_images(session, ean, serp_key, ean_token, market_code,
                                ground_truth="", gemini_key="", go_upc_data=None):
    """
    Path B: Find and verify the 2 images shown in the results table.
    Completely separate from food extraction; runs in parallel with Path A.

    Hard constraints enforced here:
    - Amazon and eBay domains are skipped at every stage.
    - Brand pages are only trusted when page_contains_ean() confirms the EAN.
    - Images from unconfirmed pages use a lower-priority source tag (og_retailer).
    """
    gl           = market_code.lower()
    market_upper = market_code.upper()
    diag         = ImageDiagnostics(ean)
    serp_url     = "https://serpapi.com/search"

    product_name         = None
    retailer_urls        = []
    candidate_image_urls = []
    registry_image_url   = None
    brand_for_verify     = _extract_brand(ground_truth) if ground_truth else ""
    weight_for_verify    = _extract_weight_hint(ground_truth) if ground_truth else ""

    # ── Tier 1A: Go-UPC — inject verified name and image before any search ────
    if go_upc_data:
        diag.log("✅ Tier 1A: Go-UPC data received.")
        if not product_name:
            gname = go_upc_data.get("name", "")
            if gname and not is_garbage_name(gname):
                product_name = gname
                diag.log(f"✅ Go-UPC verified name: {product_name}")
        img_url = go_upc_data.get("image_url", "")
        if img_url and not _is_excluded_domain(img_url):
            candidate_image_urls.append(("go_upc_tier1", img_url, "", "", "go-upc"))
            diag.log(f"✅ Go-UPC Tier 1 image added: {img_url[:80]}")
    else:
        diag.log("ℹ️ Go-UPC: no data — falling through to EAN-Search.")

    # Marketplace exclusion string — used in every SerpAPI image query so
    # Google Images returns retailer/brand images instead of Amazon/eBay CDN URLs.
    _mkt_excl = ("-site:amazon.com -site:amazon.co.uk -site:amazon.de "
                 "-site:amazon.fr -site:amazon.it -site:amazon.es "
                 "-site:ebay.com -site:ebay.co.uk -site:ebay.de "
                 "-site:ebay.fr -site:ebay.it")

    # ── Stage 0: Brand-site image lookup ──────────────────────────────────────
    brand_domain_found  = None
    brand_ean_verified  = False   # True only if page_contains_ean() passed

    _clean_gt = _strip_pack_notation(ground_truth) if ground_truth else ground_truth
    if serp_key and ground_truth:
        brand_candidate = _extract_brand(_clean_gt)
        if brand_candidate:
            diag.log(f"🔍 Brand-site lookup for '{brand_candidate}' (cleaned: '{_clean_gt[:50]}')")
            try:
                data    = await _serp_get(session, serp_url,
                    params={"q": f"{brand_candidate} {ean}", "gl": gl, "api_key": serp_key})
                if data:
                    organic = data.get("organic_results", [])
                if data:
                        for res in organic[:5]:
                            link   = res.get("link", "")
                            domain = link.split("/")[2] if link.startswith("http") else ""
                            # Hard guard: never accept Amazon/eBay as a brand domain
                            if _is_excluded_domain(link):
                                diag.log(f"   ⚠️ Skipping excluded domain: {domain}")
                                continue
                            if _brand_matches_domain(brand_candidate, domain):
                                # Verify the page actually contains the EAN in its HTML
                                ean_on_page = await page_contains_ean(session, link, ean)
                                if ean_on_page:
                                    brand_ean_verified = True
                                    diag.log(f"✅ Brand page EAN verified: {domain}")
                                else:
                                    diag.log(f"⚠️ Brand page found ({domain}) but EAN not in HTML — lower trust")
                                if not product_name:
                                    product_name = res.get("title", "").split("-")[0].split("|")[0].strip()
                                    diag.log(f"✅ Brand-site name: {product_name}")
                                if link not in retailer_urls:
                                    retailer_urls.insert(0, link)
                                brand_domain_found = domain
                                break
            except Exception as e:
                diag.log(f"⚠️ Brand text search failed: {e}")

            # Google Images restricted to brand domain
            if brand_domain_found and serp_key:
                diag.log(f"🖼️ Image search on site:{brand_domain_found}...")
                name_hint       = " ".join((product_name or ground_truth).split()[:4])
                site_img_query  = f"site:{brand_domain_found} {name_hint}"
                # Use jsonld_retailer tag only for EAN-verified brand pages;
                # use og_retailer for unverified (still useful but lower priority)
                brand_img_tag = "jsonld_retailer" if brand_ean_verified else "og_retailer"
                try:
                    img_data = await _serp_get(session, serp_url,
                        params={"q": site_img_query, "tbm": "isch",
                                "gl": gl, "api_key": serp_key}, timeout=10)
                    if img_data:
                        if True:
                            hits     = 0
                            for item in img_data.get("images_results", [])[:8]:
                                original   = item.get("original", "")
                                thumbnail  = item.get("thumbnail", "")
                                title      = item.get("title", "").lower()
                                src_domain = item.get("source", "").lower()
                                # Skip any image from an excluded domain
                                if _is_excluded_domain(original) or _is_excluded_domain(thumbnail):
                                    continue
                                if original or thumbnail:
                                    candidate_image_urls.append(
                                        (brand_img_tag, original, thumbnail, title, src_domain)
                                    )
                                    hits += 1
                            diag.log(f"   Found {hits} brand-domain image candidates (tag={brand_img_tag}).")
                except Exception as e:
                    diag.log(f"⚠️ Brand domain image search failed: {e}")

    # ── Stage 1B: EAN-Search (only if Go-UPC returned nothing) ─────────────────
    if not go_upc_data and ean_token:
        diag.log("🔍 Tier 1B: EAN-Search.org API...")
        ean_url = f"https://api.ean-search.org/api?token={ean_token}&op=barcode-lookup&ean={ean}&format=json"
        try:
            async with session.get(ean_url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) > 0 and "error" not in data[0]:
                        returned_barcode = str(data[0].get("barcode", data[0].get("ean", "")))
                        candidate_name   = data[0].get("name", "")
                        registry_image_url = data[0].get("image")
                        if barcode_matches(returned_barcode, ean):
                            if is_garbage_name(candidate_name):
                                diag.log(f"⚠️ EAN-Search barcode verified but name discarded (garbage): '{candidate_name}'")
                            else:
                                product_name = candidate_name
                                diag.log(f"✅ EAN-Search name (barcode verified): {product_name}")
                        else:
                            diag.log(f"⚠️ EAN-Search barcode mismatch (got {returned_barcode}) — discarded")
                            registry_image_url = None  # don't trust registry image either
        except Exception as e:
            diag.log(f"⚠️ EAN-Search failed: {e}")

    if serp_key:
        diag.log("🔍 Goldmine search...")
        goldmine = f"{GOLDMINE.get(market_upper, '')} OR {GLOBAL_SITES}".strip(" OR")
        try:
            data    = await _serp_get(session, serp_url,
                params={"q": f"{goldmine} {ean}", "gl": gl, "api_key": serp_key})
            organic = data.get("organic_results", []) if data else []
            if organic:
                if not product_name:
                    candidate = organic[0].get("title", "").split("-")[0].split("|")[0].strip()
                    if not is_garbage_name(candidate):
                        product_name = candidate
                        diag.log(f"✅ Goldmine name: {product_name}")
                before = len(retailer_urls)
                retailer_urls.extend([
                    res.get("link") for res in organic[:4]
                    if "link" in res
                    and res.get("link")
                    and not _is_excluded_domain(res.get("link", ""))
                ])
                added = len(retailer_urls) - before
                diag.log(f"   Goldmine: {len(organic)} results → {added} retailer URLs added.")
            elif data is None:
                diag.log("   ⚠️ Goldmine: SerpAPI returned no data (rate-limited or quota).")
        except Exception as e:
            diag.log(f"⚠️ Goldmine failed: {e}")

    if serp_key and (not retailer_urls or not product_name):
        diag.log("🔍 Global GTIN search for retailer URLs...")
        try:
            data    = await _serp_get(session, serp_url,
                params={"q": str(ean), "gl": gl, "api_key": serp_key})
            organic = data.get("organic_results", []) if data else []
            if organic:
                if not product_name:
                    candidate = organic[0].get("title", "").split("-")[0].split("|")[0].strip()
                    if not is_garbage_name(candidate):
                        product_name = candidate
                        diag.log(f"✅ Global GTIN name (unverified): {product_name}")
                new_urls = [
                    res.get("link") for res in organic[:5]
                    if "link" in res and not _is_excluded_domain(res.get("link", ""))
                ]
                for u in new_urls:
                    if u not in retailer_urls:
                        retailer_urls.append(u)
            elif data is None:
                diag.log("   ⚠️ Global GTIN: SerpAPI returned no data (rate-limited or quota).")
        except Exception as e:
            diag.log(f"⚠️ Global search failed: {e}")

    # ── Stage 2a: Google Shopping (primary EAN-anchored image source) ───────────
    # Google Shopping matches on GTIN/barcode and returns retailer-submitted
    # product images.  These are directly tied to a product listing, not
    # ranked by image SEO, making them far more product-specific than Google
    # Images results.  Shopping results also enrich retailer_urls so Stage 2b
    # can scrape additional JSON-LD/OG images from the product pages.
    if serp_key:
        diag.log("🛒 Google Shopping by EAN (primary image source)...")
        shopping_data = await _serp_get(session, serp_url,
            params={"engine": "google_shopping", "q": ean,
                    "gl": gl, "api_key": serp_key}, timeout=15)
        if shopping_data:
            shopping_results = shopping_data.get("shopping_results", [])
            shopping_hits    = 0
            for item in shopping_results[:10]:
                thumbnail   = item.get("thumbnail", "")
                link        = item.get("link", "")
                title_txt   = item.get("title", "").lower()
                source_name = item.get("source", "").lower()
                # Hard guard: skip excluded domains
                if _is_excluded_domain(thumbnail) or _is_excluded_domain(link):
                    continue
                # Add retailer product-page URL for JSON-LD/OG scraping in Stage 2b
                if link and link not in retailer_urls:
                    retailer_urls.append(link)
                # Collect thumbnail as a Shopping candidate
                if thumbnail:
                    candidate_image_urls.append(
                        ("shopping_result", thumbnail, "", title_txt, source_name)
                    )
                    shopping_hits += 1
            diag.log(f"   Shopping: {shopping_hits}/{len(shopping_results)} product images found.")
            # Bootstrap product name from Shopping if still unknown
            if not product_name and shopping_results:
                candidate_nm = shopping_results[0].get("title", "")
                if candidate_nm and not is_garbage_name(candidate_nm):
                    product_name = candidate_nm
                    diag.log(f"✅ Product name from Shopping: {product_name}")
        else:
            diag.log("   ⚠️ Google Shopping: no results (rate-limited or quota).")

    # ── Stage 2b: Retailer page scraping (now enriched with Shopping URLs) ───
    # ── Stage 2: Gather candidate image URLs ─────────────────────────────────
    if retailer_urls:
        diag.log(f"🌐 Scraping JSON-LD/OG/Twitter from {min(len(retailer_urls), 5)} retailer pages...")
        scrape_tasks    = [display_extract_from_page(session, url) for url in retailer_urls[:5]]
        per_page_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        scraped_count   = 0
        for result in per_page_results:
            if isinstance(result, Exception):
                continue
            for source_tag, img_url in result:
                if not _is_excluded_domain(img_url):
                    candidate_image_urls.append((source_tag, img_url))
                    scraped_count += 1
        diag.log(f"   Scraped {scraped_count} URLs from retailer pages.")

    if serp_key:
        diag.log("🖼️ SerpAPI image search by EAN (marketplace-excluded)...")
        # Sequential (not concurrent) to avoid 429 burst.
        # _serp_get handles 429 with automatic backoff-retry.
        for q_str, tag in [
            (f'"{ean}" {_mkt_excl}', "serpapi_strict"),
            (f'site:barcodelookup.com OR site:go-upc.com "{ean}"', "serpapi_barcode"),
        ]:
            img_data = await _serp_get(session, serp_url,
                params={"q": q_str, "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10)
            if img_data:
                items    = img_data.get("images_results", [])
                accepted = 0
                for item in items[:12]:
                    original   = item.get("original", "")
                    thumbnail  = item.get("thumbnail", "")
                    title      = item.get("title", "").lower()
                    source     = item.get("source", "").lower()
                    if _is_excluded_domain(original) or _is_excluded_domain(thumbnail):
                        continue
                    if original or thumbnail:
                        candidate_image_urls.append((tag, original, thumbnail, title, source))
                        accepted += 1
                diag.log(f"   [{tag}] {accepted}/{len(items)} URLs kept after exclusion filter.")
            else:
                diag.log(f"   ⚠️ [{tag}] No results (rate-limited, quota, or empty).")

    if serp_key and product_name and not product_name.startswith("Product with EAN"):
        name_words = " ".join(product_name.split()[:5])

        # ── Supplementary Shopping search by name + EAN ──────────────────────
        # A second Google Shopping call using the product name alongside the EAN
        # surfaces variants and additional retailer listings that the EAN-only
        # call may have missed.
        diag.log("🛒 Google Shopping by name+EAN (supplementary)...")
        shopping_name_data = await _serp_get(session, serp_url,
            params={"engine": "google_shopping", "q": f"{name_words} {ean}",
                    "gl": gl, "api_key": serp_key}, timeout=15)
        if shopping_name_data:
            name_shop_hits = 0
            for item in shopping_name_data.get("shopping_results", [])[:6]:
                thumbnail   = item.get("thumbnail", "")
                link        = item.get("link", "")
                title_txt   = item.get("title", "").lower()
                source_name = item.get("source", "").lower()
                if _is_excluded_domain(thumbnail) or _is_excluded_domain(link):
                    continue
                if link and link not in retailer_urls:
                    retailer_urls.append(link)
                if thumbnail:
                    candidate_image_urls.append(
                        ("shopping_result", thumbnail, "", title_txt, source_name)
                    )
                    name_shop_hits += 1
            diag.log(f"   Name+EAN Shopping: {name_shop_hits} images found.")
        else:
            diag.log("   ⚠️ Name+EAN Shopping: no results.")

        # ── Google Images hailmary — last resort for name-based image search ──
        hailmary_query = (f'{name_words} "{ean}" {_mkt_excl}'
                          f' -site:aliexpress.com -site:alibaba.com')
        img_data = await _serp_get(session, serp_url,
            params={"q": hailmary_query, "tbm": "isch",
                    "gl": gl, "api_key": serp_key}, timeout=10)
        if img_data:
            name_hits = 0
            for item in img_data.get("images_results", [])[:6]:
                original   = item.get("original", "")
                thumbnail  = item.get("thumbnail", "")
                title      = item.get("title", "").lower()
                source     = item.get("source", "").lower()
                if _is_excluded_domain(original) or _is_excluded_domain(thumbnail):
                    continue
                if original or thumbnail:
                    candidate_image_urls.append(
                        ("serpapi_hailmary", original, thumbnail, title, source))
                    name_hits += 1
            diag.log(f"   [hailmary] Found {name_hits} Google Images candidates.")
        else:
            diag.log("   ⚠️ [hailmary] No results (rate-limited or empty).")

    if registry_image_url and not _is_excluded_domain(registry_image_url):
        candidate_image_urls.append(("ean_search_registry", registry_image_url, "", "", ""))

    # ── Stage 3: Normalise and deduplicate ────────────────────────────────────
    seen_urls = set()
    deduped   = []
    for entry in candidate_image_urls:
        if len(entry) == 2:
            src, original = entry
            thumbnail, title, src_domain = "", "", ""
        else:
            src, original, thumbnail, title, src_domain = entry
        key = original or thumbnail
        if key and key not in seen_urls and not _is_excluded_domain(key):
            seen_urls.add(key)
            deduped.append((src, original, thumbnail, title, src_domain))

    diag.log(f"📊 {len(deduped)} unique candidate URLs after deduplication. Evaluating...")

    # ── Stage 4: Evaluate candidates ─────────────────────────────────────────
    _pname_for_verify = product_name or ""
    eval_tasks = [
        display_evaluate_candidate(
            session, src, original, thumbnail, diag,
            gemini_key=gemini_key,
            product_name=_pname_for_verify,
            brand=brand_for_verify,
            title=title, src_domain=src_domain,
            weight_hint=weight_for_verify  # Change 7
        )
        for src, original, thumbnail, title, src_domain in deduped[:20]
    ]
    eval_results   = await asyncio.gather(*eval_tasks, return_exceptions=True)
    valid_candidates = [r for r in eval_results if r and not isinstance(r, Exception)]

    diag.log(f"✅ {len(valid_candidates)} candidates passed quality checks.")

    # ── Stage 5: Hail-mary fallback ───────────────────────────────────────────
    if len(valid_candidates) == 0 and serp_key:
        diag.log("🆘 HAIL MARY: zero viable — trying Shopping then Google Images...")
        hail = []

        # Hail mary attempt 1: Google Shopping by EAN (no results yet = likely quota)
        # This is a last-chance retry of the primary Shopping search with a fresh call.
        hm_shopping = await _serp_get(session, serp_url,
            params={"engine": "google_shopping", "q": ean,
                    "gl": gl, "api_key": serp_key}, timeout=15)
        if hm_shopping:
            for item in hm_shopping.get("shopping_results", [])[:6]:
                thumbnail   = item.get("thumbnail", "")
                title_txt   = item.get("title", "").lower()
                source_name = item.get("source", "").lower()
                key         = thumbnail
                if key and key not in seen_urls and not _is_excluded_domain(key):
                    hail.append(("shopping_result", thumbnail, "", title_txt, source_name))
                    seen_urls.add(key)
            diag.log(f"   Hail mary Shopping: {len(hm_shopping.get('shopping_results', []))} results.")

        # Hail mary attempt 2: Google Images generic fallback
        generic_base  = (product_name if product_name and not product_name.startswith("Product with EAN")
                         else str(ean))
        generic_query = f"{generic_base} {_mkt_excl}"
        img_data = await _serp_get(session, serp_url,
            params={"q": generic_query, "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10)
        if img_data:
            for item in img_data.get("images_results", [])[:10]:
                original   = item.get("original", "")
                thumbnail  = item.get("thumbnail", "")
                title      = item.get("title", "").lower()
                src_domain = item.get("source", "").lower()
                key        = original or thumbnail
                if key and key not in seen_urls and not _is_excluded_domain(key):
                    hail.append(("serpapi_hailmary", original, thumbnail, title, src_domain))
                    seen_urls.add(key)
            diag.log(f"   Hail mary Images: {len(img_data.get('images_results', []))} results.")

        diag.log(f"   Total hail mary candidates: {len(hail)}.")
        if hail:
            h_eval = await asyncio.gather(
                *[display_evaluate_candidate(
                    session, s, orig, thumb, diag,
                    gemini_key=gemini_key,
                    product_name=_pname_for_verify,
                    brand=brand_for_verify,
                    title=ttl, src_domain=sdomain,
                    weight_hint=weight_for_verify  # Change 7
                  ) for s, orig, thumb, ttl, sdomain in hail],
                return_exceptions=True
            )
            h_valid = [r for r in h_eval if r and not isinstance(r, Exception)]
            valid_candidates.extend(h_valid)
            diag.log(f"   Rescued {len(h_valid)} viable images.")

    # ── Stage 6: Select top images ────────────────────────────────────────────
    selected = display_select(valid_candidates, diag)

    # Change 8: Image Source Link — only include pages confirmed to contain the EAN
    # This ensures the link actually leads to the correct product page.
    source_links      = []
    seen_link_domains = set()
    for page_url in retailer_urls:
        if not page_url or _is_excluded_domain(page_url):
            continue
        dom = page_url.split("/")[2] if page_url.startswith("http") else ""
        if dom and dom not in seen_link_domains:
            ean_confirmed = await page_contains_ean(session, page_url, ean)
            if ean_confirmed:
                source_links.append(page_url)
                seen_link_domains.add(dom)
                diag.log(f"✅ Image Source Link confirmed (EAN in page): {dom}")
            else:
                diag.log(f"   ⚠️ Source link skipped — EAN not found on page: {dom}")
        if len(source_links) >= 2:
            break

    return selected, diag, source_links


# ============================================================================
# ORCHESTRATION: run Path A + Path B in parallel, merge results
# ============================================================================

async def process_ean(sem, session, item, serp_key, gemini_key, ean_token,
                      go_upc_key, market, taxonomy_text):
    ean           = item["ean"]
    ground_truth  = item.get("ground_truth", "")
    force_refresh = item.get("force_refresh", False)

    # ── Cache check ──────────────────────────────────────────────────────────
    if not force_refresh:
        cached = cache_get(ean, market)
        if cached:
            cached["Cached"] = "✅ Cached"
            empty_diag = ImageDiagnostics(ean)
            empty_diag.log("✅ Loaded from cache.")
            return {"row": cached, "image_diag": empty_diag, "food_diag": None}
    else:
        cache_delete(ean, market)

    async with sem:
        # ── Tier 1A: Go-UPC — single call, result shared by both paths ───────
        go_upc_data = await fetch_go_upc(session, ean, go_upc_key)
        if go_upc_data:
            print(f"[Go-UPC] {ean}: {go_upc_data.get('name','')} "
                  f"| food={go_upc_data.get('is_food')} "
                  f"| img={'yes' if go_upc_data.get('image_url') else 'no'}")

        # Go-UPC non-food early exit is intentionally disabled.
        # Go-UPC's category taxonomy is unreliable (e.g. tinned salmon can be
        # tagged as "Fish Food" for aquariums). The user also sells pet food
        # which would be incorrectly rejected by a category-based gate.
        # Gemini's tiered validation gate handles food classification with
        # much better context and accuracy.

        # ── Run both pipelines concurrently ──────────────────────────────────
        food_task  = fetch_basic_info(session, ean, serp_key, ean_token, market,
                                      user_ground_truth=ground_truth,
                                      go_upc_data=go_upc_data)
        image_task = fetch_display_images(session, ean, serp_key, ean_token, market,
                                           ground_truth=ground_truth, gemini_key=gemini_key,
                                           go_upc_data=go_upc_data)

        (name, gemini_images, food_diag, ean_verified), \
        (display_images, image_diag, source_links) = await asyncio.gather(food_task, image_task)

        # ── Change 1: Early exit when no approved source found anything ───────
        no_sources = (
            name.startswith("Product with EAN") and
            not ean_verified and
            len(gemini_images) == 0 and
            go_upc_data is None
        )
        if no_sources:
            empty_diag2 = ImageDiagnostics(ean)
            empty_diag2.log("❌ No approved source indexed this EAN.")
            row = {
                "Image 1": "", "Image 2": "", "Image Source Link": "",
                "GTIN / EAN": ean, "User Input": ground_truth,
                "Status": "⚠️ Not Found — no approved source indexed this EAN",
                "Cached": "🔄 Fresh",
            }
            return {"row": row, "image_diag": image_diag, "food_diag": food_diag}

        # ── Path A: Gemini food extraction ───────────────────────────────────
        data = await asyncio.to_thread(
            run_gemini_sync, ean, name, market, gemini_key,
            taxonomy_text, gemini_images, ground_truth, go_upc_data
        )

        # ── Path B output ─────────────────────────────────────────────────────
        display_urls  = [img["url"] for img in display_images]
        imgs          = (display_urls + ["", ""])[:2]
        img_src_links = (source_links + ["", ""])[:2]

        # ── Error branch — do NOT cache ───────────────────────────────────────
        if "error" in data:
            row = {
                "Image 1": imgs[0],
                "Image 2": imgs[1],
                "Image Source Link": img_src_links[0],
                "GTIN / EAN": ean,
                "User Input": ground_truth,
                "Status": f"{data['error']}",
                "Cached": "🔄 Fresh"
            }
            # Errors are never cached — they will be retried on next lookup
            return {"row": row, "image_diag": image_diag, "food_diag": food_diag}

        # ── Parse sources — merge model claims + grounding refs, strip forbidden ──
        sources = data.get("sources", [])
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",") if s.strip()]
        elif not isinstance(sources, list):
            sources = []
        # Change 4: keep only real, clickable URLs — filter Vertex grounding redirects
        # and any forbidden domains Gemini may have slipped in
        sources = [s for s in sources if s and _is_displayable_url(s)]
        srcs    = (sources + ["", "", "", "", ""])[:5]

        # ── Determine real approved sources (filter grounding redirect URLs) ──
        real_sources = [
            s for s in srcs
            if s and not s.startswith("https://vertexaisearch.cloud.google.com")
            and not _is_excluded_domain(s)
        ]

        # ── Validation gate — is_exact_match=False means wrong product ────────
        if data.get("is_exact_match") is False:
            # Images are suppressed: Path B searched under the wrong product
            # name, so any images found belong to the wrong product.
            row = {
                "Image 1": "",
                "Image 2": "",
                "Image Source Link": "",
                "Image 2 Failure Reason": "Images suppressed — EAN validation failed",
                "Status": "Failed Validation",
                "GTIN / EAN": ean,
                "User Input": ground_truth,
                "Product Name": name,
                "Categorization Diagnosis": "Error: EAN does not correspond to the product found.",
                "Info Reliability": "",
                "Reliability Reasoning": data.get("chain_of_thought", ""),
                "Chain of Thought": data.get("chain_of_thought", ""),
                "Category L1": "", "Category L2": "", "Category L3": "",
                "Category L4": "", "Category L5": "", "Category L6": "",
                "Dietary Tags": "", "Occasion Tags": "", "Seasonal Tags": "",
                "Tagging Reasoning": "",
                "Brand": "", "UoM": "", "Packaging": "", "Fragile Item": "",
                "Net Weight (g) / Volume": "", "Gross Weight (g)": "",
                "Organic Product": "", "Net Weight/ Volume (Customer Facing)": "",
                "Ingredients": "", "Allergens": "", "May Contain": "",
                "Nutritional Info": "", "Manufacturer Name": "",
                "Manufacturer Address": "", "Place of Origin": "",
                "Organic Certification ID": "",
                "Energy (kJ)": "", "Fat (g)": "",
                "Of Which Saturated Fatty Acids (g)": "", "Carbohydrates (g)": "",
                "Of Which Sugars (g)": "", "Protein (g)": "",
                "Fiber (g)": "", "Salt (g)": "",
                "Source 1": srcs[0], "Source 2": srcs[1], "Source 3": srcs[2],
                "Source 4": srcs[3], "Source 5": srcs[4],
                "Cached": "🔄 Fresh"
            }
            # Failed validation is never cached — the product may be retried with
            # corrected ground truth
            return {"row": row, "image_diag": image_diag, "food_diag": food_diag}

        # ── Determine final status ────────────────────────────────────────────
        # Priority: wrong answer < needs review < success.
        # Rules (most conservative first):
        #   L reliability         → always Needs Review
        #   No approved sources   → always Needs Review
        #   M reliability + no EAN barcode verification → Needs Review
        #   Otherwise             → Success
        reliability = data.get("food_info_reliability", "")

        if reliability == "L" or not real_sources:
            final_status = "⚠️ Needs Review"
        elif reliability == "M" and not ean_verified:
            final_status = "⚠️ Needs Review"
        else:
            final_status = "Success"

        row = {
            "Image 1": imgs[0],
            "Image 2": imgs[1],
            "Image Source Link": img_src_links[0],
            "Image 2 Failure Reason": image_diag.image_2_failure,
            "Status": final_status,
            "GTIN / EAN": ean,
            "User Input": ground_truth,
            "Product Name": name,
            "Info Reliability": reliability,
            "Reliability Reasoning": data.get("reliability_reasoning", ""),
            "Chain of Thought": data.get("chain_of_thought", ""),
            "Category L1": data.get("category_1", ""),
            "Category L2": data.get("category_2", ""),
            "Category L3": data.get("category_3", ""),
            "Category L4": data.get("category_4", ""),
            "Category L5": data.get("category_5", ""),
            "Category L6": data.get("category_6", ""),
            "Categorization Diagnosis": data.get("categorization_reasoning", ""),
            "Dietary Tags": data.get("dietary_tags", ""),
            "Occasion Tags": data.get("occasion_tags", ""),
            "Seasonal Tags": data.get("seasonal_tags", ""),
            "Tagging Reasoning": data.get("tagging_reasoning", ""),
            "Brand": data.get("brand", ""),
            "UoM": data.get("uom", ""),
            "Packaging": data.get("packaging", ""),
            "Fragile Item": data.get("fragile_item", ""),
            "Net Weight (g) / Volume": data.get("net_weight", ""),
            "Gross Weight (g)": data.get("gross_weight", ""),
            "Organic Product": data.get("organic_product", ""),
            "Net Weight/ Volume (Customer Facing)": data.get("net_weight_customer_facing", ""),
            "Ingredients": data.get("ingredients", ""),
            "Allergens": data.get("allergens", ""),
            "May Contain": data.get("may_contain", ""),
            "Nutritional Info": data.get("nutritional_info", ""),
            "Manufacturer Address": data.get("manufacturer_address", ""),
            "Manufacturer Name": data.get("manufacturer_name", ""),
            "Place of Origin": data.get("place_of_origin", ""),
            "Organic Certification ID": data.get("organic_certification_id", ""),
            "Energy (kJ)": data.get("energy_kj", ""),
            "Fat (g)": data.get("fat_g", ""),
            "Of Which Saturated Fatty Acids (g)": data.get("saturates_g", ""),
            "Carbohydrates (g)": data.get("carbohydrates_g", ""),
            "Of Which Sugars (g)": data.get("sugars_g", ""),
            "Protein (g)": data.get("protein_g", ""),
            "Fiber (g)": data.get("fiber_g", ""),
            "Salt (g)": data.get("salt_g", ""),
            "Source 1": srcs[0],
            "Source 2": srcs[1],
            "Source 3": srcs[2],
            "Source 4": srcs[3],
            "Source 5": srcs[4],
            "Cached": "🔄 Fresh"
        }

        # ── Cache only confirmed successes ────────────────────────────────────
        if should_cache(final_status):
            cache_set(ean, market, row, status=final_status, confidence=1.0)

        return {"row": row, "image_diag": image_diag, "food_diag": food_diag}


async def run_main(parsed_inputs, serp_key, gemini_key, ean_token, go_upc_key,
                   market, taxonomy_text, progress_bar, status_text):
    sem       = asyncio.Semaphore(5)
    total     = len(parsed_inputs)
    completed = 0

    async with aiohttp.ClientSession() as session:
        async def tracked(item):
            nonlocal completed
            res = await process_ean(sem, session, item, serp_key, gemini_key,
                                     ean_token, go_upc_key, market, taxonomy_text)
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Processed {completed}/{total} items...")
            return res

        all_results = await asyncio.gather(*[tracked(item) for item in parsed_inputs])

        results     = [r["row"]        for r in all_results]
        diagnostics = [r["image_diag"] for r in all_results]

        # ── Cross-EAN duplicate image detection ────────────────────────────
        # If two different EANs ended up with the same Image 1 URL, the image
        # pipeline found a generic brand photo rather than a product-specific
        # shot.  Flag those rows so users know to verify manually.
        seen_img_urls: dict = {}
        for row in results:
            for col in ("Image 1", "Image 2"):
                url = row.get(col, "")
                if not url:
                    continue
                ean_key = row.get("GTIN / EAN", "")
                if url in seen_img_urls and seen_img_urls[url] != ean_key:
                    # Same URL appeared for a different EAN — flag both rows
                    current_note = row.get("Image 2 Failure Reason", "")
                    flag = f"⚠️ Duplicate image shared with EAN {seen_img_urls[url]} — may be a generic brand photo, not product-specific"
                    row["Image 2 Failure Reason"] = flag if not current_note else f"{current_note}; {flag}"
                else:
                    seen_img_urls[url] = ean_key

        return results, diagnostics


# ============================================================================
# UI APP (STREAMLIT)
# ============================================================================

st.title("🔬 Food Data Researcher PRO")

SERP_KEY    = os.environ.get("SERPAPI_KEY", "")
GEMINI_KEY  = os.environ.get("GEMINI_API_KEY", "")
EAN_TOKEN   = os.environ.get("EAN_SEARCH_TOKEN", "")
GO_UPC_KEY  = os.environ.get("GO_UPC_API_KEY", "")

taxonomy_text = load_taxonomy()
init_cache()

with st.sidebar:
    st.header("⚙️ Settings")
    market_selection = st.selectbox(
        "Target Market",
        [
            "Belgium (BE)", "Denmark (DK)", "Germany (DE)", "Austria (AT)",
            "Netherlands (NL)", "France (FR)", "Italy (IT)", "Spain (ES)",
            "United Kingdom (UK)", "Poland (PL)"
        ]
    )
    market_code = market_selection.split("(")[1].replace(")", "")

    if not EAN_TOKEN:
        st.warning("⚠️ EAN_SEARCH_TOKEN not found in environment variables.")
    if not GO_UPC_KEY:
        st.info("ℹ️ GO_UPC_API_KEY not set — Tier 1A barcode lookup disabled. "
                "Add GO_UPC_API_KEY to Render environment to enable.")

    if "Error" in taxonomy_text:
        st.error("⚠️ taxonomy.csv missing from project root!")

    if not IMAGE_LIBS_OK:
        st.error("⚠️ Pillow & imagehash not installed. Run: pip install Pillow imagehash")

st.markdown("""
**Input Instructions:** Paste rows directly from Excel or type them manually.
The system will automatically find the EAN (8-14 digits) anywhere in the line. Any other text on that line (Brand, Name, Weight) will be used by the AI to strictly validate if it found the correct product online.
""")

ean_input = st.text_area("Insert Data (EANs + Optional Name/Weight/Brand):")

if st.button("🚀 Start Deep Research", type="primary"):
    if not SERP_KEY or not GEMINI_KEY:
        st.error("API Keys are missing! Set SERPAPI_KEY and GEMINI_API_KEY.")
        st.stop()

    parsed_inputs = []
    for line in ean_input.split("\n"):
        line = line.strip()
        if not line:
            continue

        force_refresh = False
        if line.upper().startswith("REFRESH"):
            force_refresh = True
            line = line[7:].strip()

        match = re.search(r'\b\d{8,14}\b', line)
        if match:
            ean_raw      = match.group(0)
            ground_truth = line.replace(ean_raw, "").strip()
            ground_truth = re.sub(r'\s+', ' ', ground_truth)

            # ── EAN check-digit validation ────────────────────────────────────
            if not valid_gtin(ean_raw):
                st.warning(
                    f"⚠️ '{ean_raw}' failed the barcode check digit (mod-10 validation). "
                    f"Please verify this EAN is correct before submitting. Skipping."
                )
                continue

            parsed_inputs.append({
                "ean":           ean_raw,
                "ground_truth":  ground_truth,
                "force_refresh": force_refresh
            })
        else:
            st.warning(f"⚠️ No valid 8-14 digit EAN in line: '{line}' - Skipping.")

    if parsed_inputs:
        progress_bar = st.progress(0.0)
        status_text  = st.empty()
        with st.spinner(f"Analyzing {len(parsed_inputs)} products concurrently..."):
            all_data, all_diags = asyncio.run(
                run_main(parsed_inputs, SERP_KEY, GEMINI_KEY, EAN_TOKEN, GO_UPC_KEY,
                         market_code, taxonomy_text, progress_bar, status_text)
            )

            st.session_state["results_df"]   = pd.DataFrame(all_data)
            st.session_state["image_diags"]  = all_diags

if "results_df" in st.session_state:
    df = st.session_state["results_df"].copy()
    df = df.drop(columns=["Image 2 Failure Reason", "Packaging"], errors="ignore")

    if "Cached" not in df.columns:
        df["Cached"] = "🔄 Fresh"

    df.insert(0, "Re-run?", False)

    column_order = [
        "Re-run?",
        "Cached",
        "Image 1",
        "Image 2",
        "Image Source Link",
        "GTIN / EAN",
        "User Input",
        "Product Name",
        "Brand",
        "Status",
        "Info Reliability",
        "Reliability Reasoning",
        "Chain of Thought",
        "Category L1",
        "Category L2",
        "Category L3",
        "Category L4",
        "Category L5",
        "Category L6",
        "Categorization Diagnosis",
        "Dietary Tags",
        "Occasion Tags",
        "Seasonal Tags",
        "Tagging Reasoning",
        "UoM",
        "Fragile Item",
        "Net Weight (g) / Volume",
        "Gross Weight (g)",
        "Organic Product",
        "Net Weight/ Volume (Customer Facing)",
        "Ingredients",
        "Allergens",
        "May Contain",
        "Nutritional Info",
        "Manufacturer Address",
        "Manufacturer Name",
        "Place of Origin",
        "Organic Certification ID",
        "Energy (kJ)",
        "Fat (g)",
        "Of Which Saturated Fatty Acids (g)",
        "Carbohydrates (g)",
        "Of Which Sugars (g)",
        "Protein (g)",
        "Fiber (g)",
        "Salt (g)",
        "Source 1",
        "Source 2",
        "Source 3",
        "Source 4",
        "Source 5",
    ]

    existing_ordered = [c for c in column_order if c in df.columns]
    remaining        = [c for c in df.columns if c not in column_order]
    df               = df[existing_ordered + remaining]

    st.subheader("📊 Results")
    edited_df = st.data_editor(
        df,
        column_config={
            "Re-run?": st.column_config.CheckboxColumn(
                "Re-run?",
                help="Tick to re-run this EAN and overwrite cached result",
                default=False,
            ),
            "Cached": st.column_config.TextColumn(
                "Source",
                help="Whether this result came from cache or was freshly scraped",
                disabled=True,
            ),
            "Image 1":          st.column_config.ImageColumn(),
            "Image 2":          st.column_config.ImageColumn(),
            "Image Source Link": st.column_config.LinkColumn(
                "Image Source Link",
                display_text="🔗 Image Source",
                help="Direct URL to the product image. Click to open if the image above could not be rendered.",
            ),
            "Source 1": st.column_config.LinkColumn(display_text="Link 1"),
            "Source 2": st.column_config.LinkColumn(display_text="Link 2"),
            "Source 3": st.column_config.LinkColumn(display_text="Link 3"),
            "Source 4": st.column_config.LinkColumn(display_text="Link 4"),
            "Source 5": st.column_config.LinkColumn(display_text="Link 5"),
        },
        width='stretch',
        hide_index=True
    )

    # Re-run selected rows
    rerun_rows = edited_df[edited_df["Re-run?"] == True]
    rerun_eans = rerun_rows["GTIN / EAN"].tolist()
    if rerun_eans:
        if st.button(f"🔄 Re-run {len(rerun_eans)} selected EAN(s)", type="primary"):
            rerun_inputs = [
                {
                    "ean":           row["GTIN / EAN"],
                    "ground_truth":  str(row.get("User Input", "") or "").strip(),
                    "force_refresh": True
                }
                for _, row in rerun_rows.iterrows()
            ]
            rerun_progress = st.progress(0.0)
            rerun_status   = st.empty()
            with st.spinner(f"Re-running {len(rerun_inputs)} EAN(s)..."):
                rerun_data, _ = asyncio.run(
                    run_main(rerun_inputs, SERP_KEY, GEMINI_KEY, EAN_TOKEN, GO_UPC_KEY,
                             market_code, taxonomy_text, rerun_progress, rerun_status)
                )
                rerun_df = pd.DataFrame(rerun_data)
                if "Cached" not in rerun_df.columns:
                    rerun_df["Cached"] = "🔄 Fresh"
                rerun_df.insert(0, "Re-run?", False)
                base_df = st.session_state["results_df"].copy()
                for _, fresh_row in rerun_df.iterrows():
                    mask = base_df["GTIN / EAN"] == fresh_row["GTIN / EAN"]
                    if mask.any():
                        idx = base_df.index[mask][0]
                        for col in fresh_row.index:
                            if col in base_df.columns:
                                base_df.at[idx, col] = fresh_row[col]
                st.session_state["results_df"] = base_df
                st.success(f"✅ Re-run complete for {len(rerun_eans)} EAN(s). Scroll up to see updated results.")
                st.rerun()

    # ── Image pipeline diagnostics ────────────────────────────────────────────
    # Expandable panel showing exactly what happened in the image pipeline
    # for each EAN: how many candidates were found, fetched, and why any
    # were rejected.  Use this to debug missing or wrong images.
    if "image_diags" in st.session_state:
        # Surface duplicate-image warnings as visible banners above the table
        if "results_df" in st.session_state:
            for _, warn_row in st.session_state["results_df"].iterrows():
                reason = warn_row.get("Image 2 Failure Reason", "")
                if reason and "Duplicate image" in str(reason):
                    st.warning(
                        f"**{warn_row.get('GTIN / EAN', '')}** — {reason}"
                    )
        with st.expander("🔬 Image pipeline diagnostics", expanded=False):
            all_diags = st.session_state["image_diags"]
            results_eans = (st.session_state["results_df"]["GTIN / EAN"].tolist()
                            if "results_df" in st.session_state else [])
            for i, diag_obj in enumerate(all_diags):
                ean_label = results_eans[i] if i < len(results_eans) else f"EAN #{i+1}"
                st.markdown(f"**{ean_label}** — {diag_obj.summary_string()}")
                if diag_obj.text_log:
                    st.text("\n".join(diag_obj.text_log))
                cand_data = diag_obj.to_dict_list()
                if cand_data:
                    st.dataframe(
                        pd.DataFrame(cand_data),
                        hide_index=True,
                        use_container_width=True,
                    )
                st.divider()
