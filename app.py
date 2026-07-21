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

# Bump this on every deploy. It's shown in the sidebar so you can confirm at a
# glance which build Render is actually running (deployment lag has bitten us).
BUILD_VERSION = "2026-07-10c · SERP-first · no OFF · plausibility guard"

# Styled DMF Excel export (color-coded clusters, legend sheet, hidden audit cols)
try:
    from xlsx_export import build_dmf_workbook
    XLSX_EXPORT_OK = True
except ImportError:
    XLSX_EXPORT_OK = False

# Deterministic-first food-data extraction (verify-then-read primary path).
# Replaces run_gemini_sync's grounded-search food extraction: food fields now
# come from the SAME verified pages that populate the Source columns.
try:
    from food_pipeline import extract_food_data, ALL_FIELD_KEYS
    FOOD_PIPELINE_OK = True
except ImportError:
    FOOD_PIPELINE_OK = False

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
    con.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()


def get_setting(key: str, default: str = "") -> str:
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        con.close()
        return row[0] if row else default
    except Exception:
        return default


def set_setting(key: str, value: str) -> None:
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
            (key, value)
        )
        con.commit()
        con.close()
    except Exception:
        pass


def _cache_key(ean: str, ground_truth: str = "") -> str:
    """
    Cache key = EAN + normalized ground truth. Two different inputs for the SAME
    barcode (e.g. "Fanta Exotic 500ml" vs "120x Fanta Exotic 500ml") must not
    collide — the ground truth drives Route B name-matching and validation, so
    their results genuinely differ. Normalization (lowercase, collapse spaces)
    keeps trivial variants from fragmenting the cache.
    """
    gt = re.sub(r"\s+", " ", (ground_truth or "").strip().lower())
    return f"{ean}|{gt}" if gt else ean


def cache_get(ean: str, market: str, ground_truth: str = ""):
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
    key = _cache_key(ean, ground_truth)
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT result_json, status, version FROM ean_cache WHERE ean=? AND market=?",
            (key, market)
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
              status: str = "unknown", confidence: float = 0.0,
              ground_truth: str = ""):
    key = _cache_key(ean, ground_truth)
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            """INSERT OR REPLACE INTO ean_cache
               (ean, market, result_json, status, confidence, version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, market, json.dumps(result_dict), status, confidence, CACHE_VERSION)
        )
        con.commit()
        con.close()
    except Exception:
        pass


def cache_delete(ean: str, market: str, ground_truth: str = ""):
    key = _cache_key(ean, ground_truth)
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "DELETE FROM ean_cache WHERE ean=? AND market=?",
            (key, market)
        )
        con.commit()
        con.close()
    except Exception:
        pass


def cache_count() -> int:
    """Number of cached rows (for the sidebar control)."""
    try:
        con = sqlite3.connect(DB_PATH)
        n = con.execute("SELECT COUNT(*) FROM ean_cache").fetchone()[0]
        con.close()
        return int(n)
    except Exception:
        return 0


def cache_clear(market: str | None = None) -> int:
    """
    Clear cached results. If `market` is given, clears only that market;
    otherwise clears everything. Returns the number of rows removed.
    Used after logic/schema changes so stale entries can't be served.
    """
    try:
        con = sqlite3.connect(DB_PATH)
        if market:
            cur = con.execute("DELETE FROM ean_cache WHERE market=?", (market,))
        else:
            cur = con.execute("DELETE FROM ean_cache")
        removed = cur.rowcount
        con.commit()
        con.close()
        return int(removed)
    except Exception:
        return 0


# ============================================================================
# MODEL & QUALITY CONSTANTS
# Update model strings here when migrating — never scatter them across the file.
# ============================================================================

# Categorization/tagging + free-text fallback extraction. NOT used for search
# or grounding anywhere — every call is fed pre-fetched page text or already-
# extracted fields only ("no web access" / "do not search" in the prompt).
# All page/source discovery goes through SerpAPI exclusively; see food_pipeline.py.
EXTRACTION_MODEL = "gemini-2.5-flash"

# Image identity verification: vision-only, no grounding needed, must be cheap & fast
# gemini-2.0-flash-lite was SHUT DOWN on 2026-06-01. gemini-2.5-flash-lite is the
# direct replacement at the same price point ($0.10/$0.40 per 1M tokens) with
# better multimodal understanding and a stable 2.5 lifecycle.
VISION_MODEL = "gemini-2.5-flash-lite"

# Bump this whenever the extraction prompt or cache schema changes significantly.
# All rows stored under a lower version are treated as cache misses on first load.
CACHE_VERSION = 3

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


async def page_contains_ean(session, url: str, ean: str,
                            ground_truth: str = "") -> bool:
    """
    Verify that a page is genuinely ABOUT the queried EAN — not merely a page
    where the digit string happens to appear (article numbers, order codes,
    and delivery lists on unrelated shops produce 13-digit false positives).

    STRONG evidence (always accepted):
      - EAN within a labeled GTIN context: JSON-LD/microdata gtin fields, or
        a nearby label like EAN / GTIN / Barcode / Strichcode / UPC.
    WEAK evidence (bare digit occurrence anywhere in HTML):
      - Accepted ONLY if the page <title>/og:title overlaps the ground truth
        (≥40% of significant tokens). Without ground truth, weak evidence is
        rejected — better a Needs Review than a confidently wrong source.

    Fails closed: returns False on any network or parsing error.
    """
    _ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/122.0.0.0 Safari/537.36")
    try:
        async with session.get(url, headers={"User-Agent": _ua}, timeout=8) as r:
            if r.status != 200:
                return False
            raw  = await r.content.read(400_000)
            text = raw.decode("utf-8", errors="ignore")
            stripped = ean.lstrip("0")

            # ── STRONG: labeled GTIN context ─────────────────────────────────
            # JSON-LD / microdata / meta gtin-sku fields
            if re.search(rf'(gtin\d*|"sku"|itemprop=["\']gtin)[\s"\':=]+["\']?0*{re.escape(stripped)}',
                         text, re.IGNORECASE):
                return True
            # Visible label within 30 chars before the number:
            # "EAN: 4260614723382", "GTIN Stück: ...", "Strichcode ..."
            if re.search(rf'(ean|gtin|barcode|strichcode|streepjescode|'
                         rf'codice\s*a\s*barre|c[oó]digo\s*de\s*barras|upc)'
                         rf'[^0-9]{{0,30}}0*{re.escape(stripped)}',
                         text, re.IGNORECASE):
                return True

            # ── WEAK: bare occurrence — requires title cross-check ──────────
            variants = {ean, stripped, ean.zfill(13), ean.zfill(14)}
            if not any(v and v in text for v in variants):
                return False
            if not ground_truth:
                return False   # bare digits with nothing to cross-check → reject
            m = (re.search(r'<meta[^>]+og:title[^>]+content=["\']([^"\']+)', text, re.I)
                 or re.search(r'<title[^>]*>([^<]+)</title>', text, re.I))
            page_title = m.group(1).lower() if m else ""
            gt_tokens  = {t.lower() for t in ground_truth.split() if len(t) > 3}
            if not gt_tokens or not page_title:
                return False
            overlap = sum(1 for t in gt_tokens if t in page_title) / len(gt_tokens)
            return overlap >= 0.4
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
    Returns True for any real, clickable source URL that is not junk.

    Filters out:
    - Vertex AI grounding redirects  (these 404 — the actual bug we were solving)
    - Google internal search/redirect URLs  (Gemini's own search queries leaking out)
    - Hard-excluded marketplace domains  (Amazon, eBay, OpenFoodFacts)

    Does NOT apply a domain whitelist.  Brand websites, regional retailers,
    niche food directories, and any other real page Gemini used are all valid
    sources and should be shown.  The previous whitelist was blocking every URL
    that wasn't a major supermarket — including the actual product pages for
    brand websites (hellmanns.com, mcvities.co.uk, nutella.com, etc.).
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


# ============================================================================
# OUTPUT LINK QUALITY GATES
# Ephemeral CDN URLs render fine during the session, then die — Google
# Shopping thumbnails (encrypted-tbn*.gstatic.com) expire within hours/days.
# They are fine as Gemini INPUT but must never be exported as OUTPUT links.
# ============================================================================

_EPHEMERAL_URL_TOKENS = (
    "encrypted-tbn",                 # Google Shopping / Images thumbnails — expire
    "gstatic.com/shopping",
    "googleusercontent.com/img",
    "tbn:", "tbn0.", "tbn1.", "tbn2.", "tbn3.",
)


def _is_durable_url(url: str) -> bool:
    """True if the URL is safe to export as a long-lived link."""
    if not url or not url.startswith("http"):
        return False
    u = url.lower()
    return not any(t in u for t in _EPHEMERAL_URL_TOKENS)


async def _url_alive(session, url: str) -> bool:
    """
    Lightweight liveness check applied to final output links only.
    HEAD first; some CDNs block HEAD (403/405), so fall back to a GET whose
    body we never read. Fails open on network errors is NOT wanted here —
    a link we cannot verify is a link we should not ship, so fail closed.
    """
    if not url or not url.startswith("http"):
        return False
    _ua = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/122.0.0.0 Safari/537.36")}
    try:
        async with session.head(url, headers=_ua, timeout=6,
                                allow_redirects=True) as r:
            if r.status < 400:
                return True
            if r.status in (403, 405):        # HEAD blocked — verify with GET
                async with session.get(url, headers=_ua, timeout=6,
                                       allow_redirects=True) as g:
                    return g.status < 400
            return False
    except Exception:
        return False


async def _false_coro() -> bool:
    """Awaitable False — placeholder for empty slots in gather() batches."""
    return False


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

# Tier-1 ("goldmine") domains for reliability grading and identity-match
# gating. Kept in sync with the GOLDMINE query dict above and its mirror in
# food_pipeline.py — every domain referenced in a GOLDMINE site: clause must
# also appear here.
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


def _domain_of(url: str) -> str:
    return url.split("/")[2] if url.startswith("http") and "/" in url[8:] else \
        (url.split("/")[2] if url.startswith("http") else "")


def _is_goldmine_domain(url: str) -> bool:
    dom = _domain_of(url).lower()
    return any(dom == g or dom.endswith("." + g) for g in _GOLDMINE_DOMAINS)


def _is_trusted_source(url: str, brand: str = "") -> bool:
    """
    Tier-1 trust = goldmine (approved retailer) domain OR the brand's own
    official site. Used to gate which identity-match methods are acceptable
    (rule: EAN-only / name+EAN / name+weight-only all acceptable ONLY when
    trusted; untrusted domains require an EAN actually found on the page).
    """
    if _is_goldmine_domain(url):
        return True
    if brand:
        dom = _domain_of(url)
        if _brand_matches_domain(brand, dom):
            return True
    return False

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



# Global SerpAPI concurrency cap. Each EAN fires 4-7 SerpAPI calls; with the
# per-EAN semaphore at 5 that meant 20-35 concurrent SerpAPI requests and
# silent 429s that surfaced as "Not Found". Lazily created inside the running
# event loop (asyncio.run creates a fresh loop per batch).
_SERP_SEM: asyncio.Semaphore | None = None
_SERP_SEM_LOOP: object = None


def _serp_error(data) -> int | None:
    """Return the HTTP error code if `data` is a _serp_get error sentinel."""
    if isinstance(data, dict) and "_error" in data:
        return data["_error"]
    return None


async def _serp_get(session, url: str, params: dict, timeout: int = 15,
                    max_retries: int = 2) -> dict | None:
    """
    SerpAPI GET with automatic backoff-retry on 429 AND 5xx, behind a global
    concurrency semaphore.

    Returns:
      dict                  — parsed JSON on success
      {"_error": <status>}  — sentinel after exhausted retries / HTTP error,
                              so callers can distinguish "search failed"
                              from "search returned no results".
                              (.get("organic_results", []) etc. still safely
                              return [] on the sentinel.)
      None                  — network-level failure (timeout, DNS, ...)

    Backoff schedule: 3 s after attempt 1, 6 s after attempt 2.
    """
    global _SERP_SEM, _SERP_SEM_LOOP
    loop = asyncio.get_running_loop()
    if _SERP_SEM is None or _SERP_SEM_LOOP is not loop:
        _SERP_SEM      = asyncio.Semaphore(8)
        _SERP_SEM_LOOP = loop

    async with _SERP_SEM:
        last_status = None
        for attempt in range(max_retries + 1):
            try:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    last_status = resp.status
                    if resp.status == 429 or resp.status >= 500:
                        if attempt < max_retries:
                            await asyncio.sleep(3 * (attempt + 1))   # 3 s, 6 s
                            continue
                        return {"_error": resp.status}
                    # Other HTTP error (403, 404…) — don't retry
                    return {"_error": resp.status}
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                return None
            except Exception:
                return None
        return {"_error": last_status} if last_status else None


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
                            ean_on_page = await page_contains_ean(session, link, ean, _clean_ground_truth)
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

    # ── ATTEMPT 2.5: Combined ground-truth + EAN query ────────────────────────
    # The single most effective manual query ("Kokos Curry Suppe 4260614723382")
    # surfaces niche retailers and GTIN databases that EAN-only goldmine queries
    # miss. Accept a result when the EAN appears in the SNIPPET (Google indexed
    # the GTIN on-page — Tier A-grade evidence) or the page itself confirms it.
    if not product_name and serp_key and _clean_ground_truth:
        diagnostic_log.append("🔍 Combined name+EAN query...")
        try:
            data_c = await _serp_get(session, serp_url,
                params={"q": f"{_clean_ground_truth} {ean}", "gl": gl, "api_key": serp_key})
            organic_c = data_c.get("organic_results", []) if data_c else []
            for res in organic_c[:5]:
                link = res.get("link", "")
                if not link or _is_excluded_domain(link):
                    continue
                title   = res.get("title", "").split("-")[0].split("|")[0].strip()
                snippet = res.get("snippet", "") or ""
                if is_garbage_name(title):
                    continue
                ean_in_snippet = ean in snippet or ean.lstrip("0") in snippet
                if ean_in_snippet or await page_contains_ean(session, link, ean, _clean_ground_truth):
                    product_name = title
                    ean_verified = True
                    if link not in retailer_urls:
                        retailer_urls.insert(0, link)
                    diagnostic_log.append(
                        f"✅ Name+EAN query verified ({'snippet' if ean_in_snippet else 'page'}): {title}")
                    break
        except Exception as e:
            diagnostic_log.append(f"⚠️ Combined name+EAN query failed: {e}")

    # Track SerpAPI hard failures so "search failed" is never reported as
    # "not found". process_ean inspects this flag via the returned diag log.
    search_failed = False

    # ── ATTEMPT 3: SerpAPI goldmine (tier-1 retailers for this market) ──
    if not product_name and serp_key:
        diagnostic_log.append("🔍 Goldmine Google Search for Name...")
        goldmine = f"{GOLDMINE.get(market_upper, '')} OR {GLOBAL_SITES}".strip(" OR")
        try:
            data = await _serp_get(session, serp_url,
                params={"q": f"{goldmine} {ean}", "gl": gl, "api_key": serp_key})
            if _serp_error(data):
                search_failed = True
                diagnostic_log.append(f"❌ SERP_FAILED Goldmine: HTTP {_serp_error(data)}")
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
        except Exception as e:
            diagnostic_log.append(f"⚠️ Google text search failed: {e}")

    # ── ATTEMPT 4: Bare GTIN global fallback (low-confidence) ──
    # Previously nested inside "goldmine returned zero results", which meant it
    # never ran when goldmine returned junk. Now fires whenever the name is
    # still unresolved after Attempt 3.
    if not product_name and serp_key:
        diagnostic_log.append("⚠️ Name still unresolved — bare GTIN global fallback (low confidence)...")
        try:
            data2 = await _serp_get(session, serp_url,
                params={"q": str(ean), "gl": gl, "api_key": serp_key})
            if _serp_error(data2):
                search_failed = True
                diagnostic_log.append(f"❌ SERP_FAILED Bare GTIN: HTTP {_serp_error(data2)}")
            organic2 = data2.get("organic_results", []) if data2 else []
            if organic2:
                candidate = organic2[0].get("title", "").split("-")[0].split("|")[0].strip()
                top_link  = organic2[0].get("link", "")
                # Same policy as Attempt 5: a snippet-level EAN association with no
                # domain trust and no weight check is too weak to name the product
                # from — only accept when the hit is on a trusted (goldmine/brand)
                # domain and the weight/volume doesn't conflict.
                if (not is_garbage_name(candidate)
                        and _result_matches_gt(organic2[0], _clean_ground_truth)
                        and _is_trusted_source(top_link, _extract_brand(_clean_ground_truth))
                        and not _weights_conflict([_clean_ground_truth], candidate)):
                    product_name = candidate
                    diagnostic_log.append(
                        f"✅ Found Name via bare GTIN fallback (trusted source, unverified EAN): {product_name}")
                new_urls2 = [res.get("link") for res in organic2[:4] if "link" in res
                             and not _is_excluded_domain(res.get("link", ""))
                             and _result_matches_gt(res, _clean_ground_truth)]
                for u in new_urls2:
                    if u not in retailer_urls:
                        retailer_urls.append(u)
        except Exception as e:
            diagnostic_log.append(f"⚠️ Bare GTIN search failed: {e}")

    # ── ATTEMPT 5: Name-based fallback (mirrors the manual workflow) ──
    # EAN-keyed queries found nothing, but the user supplied ground truth.
    # Search by product NAME, then verify candidates by looking for the EAN
    # on the page. Verified hits earn Tier-2 trust (ean_verified=True);
    # fuzzy title matches are accepted UNVERIFIED and are routed to
    # Needs Review by the existing reliability gate downstream.
    if not product_name and serp_key and _clean_ground_truth:
        diagnostic_log.append(f"🔍 Name-based fallback: searching by ground truth '{_clean_ground_truth[:60]}'...")
        _name_goldmine = GOLDMINE.get(market_upper, "").strip()
        _name_queries  = [q for q in (
            f'{_name_goldmine} "{_clean_ground_truth}"' if _name_goldmine else "",
            f'"{_clean_ground_truth}"',
        ) if q]
        try:
            for _nq in _name_queries:
                data3 = await _serp_get(session, serp_url,
                    params={"q": _nq, "gl": gl, "api_key": serp_key})
                if _serp_error(data3):
                    search_failed = True
                    diagnostic_log.append(f"❌ SERP_FAILED Name fallback: HTTP {_serp_error(data3)}")
                organic3 = data3.get("organic_results", []) if data3 else []
                for res in organic3[:4]:
                    link = res.get("link", "")
                    if not link or _is_excluded_domain(link):
                        continue
                    title = res.get("title", "").split("-")[0].split("|")[0].strip()
                    if is_garbage_name(title):
                        continue
                    if await page_contains_ean(session, link, ean, _clean_ground_truth):
                        product_name = title
                        ean_verified = True
                        if link not in retailer_urls:
                            retailer_urls.insert(0, link)
                        diagnostic_log.append(f"✅ Name-fallback EAN-VERIFIED on page: {title}")
                        break
                    # Name(+weight)-only acceptance — NO EAN on page. Per policy this
                    # is only acceptable evidence from a TRUSTED source (goldmine
                    # retailer or the brand's own site); an untrusted/general-web
                    # domain with no EAN confirmation is too weak and is skipped, so
                    # a same-brand-different-SKU page can no longer silently become
                    # the resolved identity.
                    if not _is_trusted_source(link, _extract_brand(_clean_ground_truth)):
                        continue
                    gt_tokens = {t.lower() for t in _clean_ground_truth.split() if len(t) > 3}
                    if gt_tokens:
                        hit_ratio = sum(1 for t in gt_tokens if t in title.lower()) / len(gt_tokens)
                        if hit_ratio >= 0.6 and not _weights_conflict([_clean_ground_truth], title):
                            product_name = title      # trusted, name(+weight)-matched
                            if link not in retailer_urls:
                                retailer_urls.append(link)
                            diagnostic_log.append(
                                f"⚠️ Name-fallback trusted-source match ({hit_ratio:.0%} token overlap, "
                                f"no weight conflict): {title}")
                            break
                if product_name:
                    break
        except Exception as e:
            diagnostic_log.append(f"⚠️ Name-based fallback failed: {e}")

    if search_failed:
        diagnostic_log.append("SERP_FAILED_FLAG")   # machine-readable marker for process_ean

    if not product_name:
        diagnostic_log.append("⚠️ Name not found via any source — Gemini will attempt EAN-grounded search.")
        product_name = f"Product with EAN {ean}"
    elif not product_name.startswith("Product with EAN"):
        # Change 5: guard resolved name against obviously non-food titles
        if not _title_looks_like_food(product_name, ""):
            diagnostic_log.append(
                f"⚠️ Resolved name '{product_name}' appears non-food — name discarded."
            )
            product_name = f"Product with EAN {ean}"
            # NOTE: ean_verified is deliberately NOT reset here. A barcode-registry
            # or on-page EAN verification is a source-level fact; a title heuristic
            # disliking the NAME must not destroy source trust (the user also
            # processes non-food SKUs like pet food, which this gate misfires on).

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

    # Stage C: Google Images — FALLBACK tier only.
    # Bare-number image searches routinely return unrelated junk (toys,
    # cosmetics, random hits). These must never be presented to Gemini as
    # "the product" unless the product identity is independently verified —
    # a wrong image actively poisons Gemini's validation gate.
    fallback_image_urls = []
    if serp_key:
        diagnostic_log.append("🖼️ Google Images (fallback tier for Gemini)...")
        _mkt_excl_a = ("-site:amazon.com -site:amazon.co.uk -site:amazon.de "
                       "-site:amazon.fr -site:ebay.com -site:ebay.co.uk")
        for img_q in [f'"{ean}" {_mkt_excl_a}',
                      f'site:barcodelookup.com OR site:go-upc.com "{ean}"']:
            img_data = await _serp_get(session, serp_url,
                params={"q": img_q, "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10)
            if img_data:
                for item in img_data.get("images_results", [])[:6]:
                    url = item.get("original", "")
                    if (_is_valid_image_url(url) and url not in candidate_image_urls
                            and url not in fallback_image_urls):
                        fallback_image_urls.append(url)

    if registry_image_url and _is_valid_image_url(registry_image_url) and registry_image_url not in candidate_image_urls:
        candidate_image_urls.append(registry_image_url)

    # ── DOWNLOAD, SIZE, AND DEDUPLICATE IMAGES (for Gemini) ──
    # Trusted tier (Go-UPC, Shopping, retailer OG, registry) always eligible.
    # Fallback tier (Google Images) only eligible when the identity is
    # EAN-verified — otherwise Gemini receives ZERO images rather than junk.
    final_downloaded_images = []
    seen_b64_prefixes = []

    _download_order = list(candidate_image_urls)
    if ean_verified:
        _download_order += fallback_image_urls
    elif fallback_image_urls:
        diagnostic_log.append(
            f"🛡️ {len(fallback_image_urls)} fallback image(s) WITHHELD from Gemini "
            f"(identity not EAN-verified — junk images would poison validation).")

    for url in _download_order:
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
    return product_name, final_downloaded_images, "\n".join(diagnostic_log), ean_verified, retailer_urls


def run_categorization_sync(ean, product_name, market_code, gemini_key,
                            taxonomy_text, food_fields, user_ground_truth,
                            go_upc_data=None):
    """
    Categorization + tagging ONLY. This is the half of the old run_gemini_sync
    that stays a Gemini call — the food-data search-and-extract half now lives
    in food_pipeline.extract_food_data (deterministic-first, verify-then-read).

    Gemini here does NOT search. It is fed the deterministically-extracted food
    data (ingredients, allergens, nutrition, brand, weight, origin) as context
    and classifies + tags the product, plus derives the inferential packaging
    attributes (uom, packaging, fragile, organic flag, customer-facing weight).

    Returns a dict with categorization/tagging/attribute fields, or
    {"error": ...} on hard failure.  It NEVER returns nutrition numbers,
    ingredients, allergens, manufacturer, or origin — those come from the
    deterministic pipeline and must not be overwritten by an LLM guess.
    """
    market_upper = market_code.upper()
    ff = food_fields or {}

    # Compact food-data context block (the model classifies FROM this, no search).
    def _g(k):
        v = ff.get(k, "")
        return str(v).strip() if v not in (None, "null", "None") else ""

    _ctx_lines = []
    for label, key in [
        ("Brand", "brand"), ("Ingredients", "ingredients"),
        ("Allergens", "allergens"), ("May contain", "may_contain"),
        ("Net weight/volume", "net_weight"), ("Place of origin", "place_of_origin"),
        ("Energy (kJ/100)", "energy_kj"), ("Fat g/100", "fat_g"),
        ("Saturates g/100", "saturates_g"), ("Carbohydrate g/100", "carbohydrates_g"),
        ("Sugars g/100", "sugars_g"), ("Protein g/100", "protein_g"),
        ("Fibre g/100", "fiber_g"), ("Salt g/100", "salt_g"),
    ]:
        val = _g(key)
        if val:
            _ctx_lines.append(f"    - {label}: {val}")
    if go_upc_data and go_upc_data.get("category"):
        _ctx_lines.append(f"    - Barcode-registry category hint: {go_upc_data['category']}")
    _food_ctx = "\n".join(_ctx_lines) if _ctx_lines else "    (no structured food data was extracted)"

    prompt = f"""
    You are a Food Product Categorisation & Tagging specialist.
    You do NOT have web access and you must NOT invent product data. Classify and
    tag the product using ONLY the information below. Do not guess nutrition,
    ingredients, allergens, or manufacturer — those are already handled elsewhere.

    PRODUCT NAME: {product_name}
    USER INPUT (GROUND TRUTH): {user_ground_truth if user_ground_truth else "None provided"}
    TARGET MARKET: {market_code}
    EAN: {ean}

    EXTRACTED FOOD DATA (your evidence — classify and tag from this):
{_food_ctx}

    DIRECTIVES:
    5. TAXONOMY MAPPING: Classify the product into the 6-level taxonomy provided below. You MUST use EXACT matches from the provided taxonomy. Do not invent categories. If a variant (Level 6) doesn't exist for the item category, return "None". Explain your reasoning in the "categorization_reasoning" field.
    5b. TAXONOMY FIRST-PRINCIPLES (INTENDED USE RULE): Before mapping, determine the product's primary intended use from its name, category keywords, and ingredients:
        - If the product is a LIQUID, SYRUP, CONCENTRATE, or MIXER of any kind -> it MUST be classified under Drinks (L1).
        - If the product is a SYRUP specifically (e.g. Monin, Torani) -> Drinks > Soft Drinks > Adult > Mixers.
        - For POWDERS and other ambiguous formats, determine intended use from name and context:
            * "Protein Powder / Shake Powder / Weight Gainer" -> Drinks > Soft Drinks > ...
            * "Cocoa Powder / Baking Powder / Flour" -> Food > Pantry > ...
            * "Protein Supplement / Creatine / Pre-workout" -> Food > Health > ...
            * "Powdered Drink Mix / Instant Drink" -> Drinks > ...
        - Solid food items, snacks, and meals -> Food.
    9. EXHAUSTIVE TAGGING (CONSISTENCY RULE): Evaluate the product against EVERY tag in the exact lists below independently. Treat this as a mandatory True/False checklist for every single tag. Base each decision on the EXTRACTED FOOD DATA above (e.g. read the ingredients for Vegan/Vegetarian, the sugar figure for Low Sugar, the protein figure for High protein). Do not assign a tag whose evidence is absent.
        DIETARY TAGS (EU Regulation 1169/2011 & 432/2012):
        - "Vegetarian": ONLY if no meat, fish, or seafood ingredients present. Dairy and eggs are permitted.
        - "Vegan": ONLY if zero animal-derived ingredients AND no animal cross-contamination advisory (or a certified vegan label).
        - "Organic": ONLY if the product carries an EU Organic logo / national certification (e.g. DE-ÖKO-001, FR-BIO-01). Do NOT infer from ingredients alone.
        - "Halal": ONLY if a recognised Halal certification is indicated.
        - "Kosher": ONLY if a recognised Kosher certification is indicated.
        - "Dairy Free": ONLY if no milk or milk-derived ingredients AND no "contains milk" declaration.
        - "Nut Free": ONLY if no tree nuts or peanuts in ingredients AND no "contains nuts/peanuts" declaration.
        - "Low Sugar": ONLY if <=5g sugars per 100g (solid) or <=2.5g per 100ml (liquid).
        - "High protein": ONLY if protein contributes >20% of total energy value.
        - "Gluten-free": ONLY if labelled gluten-free OR all ingredients gluten-free AND gluten <20 mg/kg.
        - "Low Fat": ONLY if <=3g fat per 100g (solid) or <=1.5g per 100ml (liquid).
        OCCASION TAGS — apply based on product type and primary usage context. Do NOT over-assign:
        - "Breakfast": Cereals, porridge, breakfast biscuits, morning drinks, pastries.
        - "Lunchbox": Individually portioned snacks, sandwich accompaniments, small-format items.
        - "BBQ": Condiments, marinades, grillable meats, charcoal, disposable BBQ accessories.
        - "Party": Multi-serve sharing formats, party snack packs, celebration cakes, large-format mixers/soft drinks.
        - "Christmas": Only if explicitly Christmas-themed or a recognised Christmas food/drink tradition.
        - "Ramadan": Only if marketed for Ramadan or a traditional Ramadan food (e.g. dates, harira).
        - "Meal prep": Bulk staples, dry goods in large quantities, ingredient-focused products.
        - "Quick dinner": Ready meals, instant noodles, stir-in sauces, <15 min prep.
        - "Kids snack": Products explicitly marketed at children OR inherently child-targeted by format/size/packaging.
        SEASONAL TAGS — apply ONLY if the SKU is specifically marketed/packaged for that season. Default empty:
        - "Christmas": Limited-edition Christmas packaging or explicitly seasonal SKU.
        - "Easter": Limited-edition Easter packaging.
        - "Back to School": Explicitly back-to-school themed.
        - "Valentines Day": Explicitly Valentine's Day themed.
        - "Mothers Day": Explicitly Mother's Day themed.
        - "Halloween": Limited-edition Halloween packaging.
        - "Other": A seasonal angle exists but fits none of the above.
        - If NONE apply, return "" (empty string). Do NOT default to "Other".
    ATTRIBUTES: Derive the packaging/handling attributes from the product name and extracted data:
        - uom: strictly "g" (solids) or "ml" (liquids). Never "gram"/"grams"/"gr".
        - packaging: e.g. Box, Bottle, Wrapper, Can, Jar, Pouch, Carton.
        - fragile_item: "Yes" for glass bottles/jars or easily-crushed items, else "No".
        - organic_product: "Yes" only if organic certification is evident, else "No".
        - net_weight_customer_facing: how the size is shown on pack (e.g. "400 ml", "6 x 42 g").
        - gross_weight: only if genuinely known, else null.
        - organic_certification_id: e.g. DE-ÖKO-001 if evident, else null.

    --- START TAXONOMY REFERENCE (CSV FORMAT) ---
    {taxonomy_text}
    --- END TAXONOMY REFERENCE ---

    CRITICAL JSON RULES:
    - YOUR ENTIRE RESPONSE MUST BE A SINGLE VALID JSON OBJECT. NO EXCEPTIONS.
    - NEVER write conversational text outside the JSON object. Put reasoning inside the fields.
    - Use double quotes for keys and string values; single quotes inside values.
    - Do not use literal newlines/tabs inside strings.
    - The 6 taxonomy categories AND the Tags MUST remain exactly as they appear in the English lists above.

    SCHEMA:
    {{
        "chain_of_thought": "Brief reasoning: how you classified and which tags you set from the evidence.",
        "category_1": "Level 1 Category (English)",
        "category_2": "Level 2 Category (English)",
        "category_3": "Level 3 Category (English)",
        "category_4": "Level 4 Category (English)",
        "category_5": "Level 5 Category (English)",
        "category_6": "Level 6 Variant or None (English)",
        "categorization_reasoning": "Brief explanation of the category choice",
        "dietary_tags": "Comma-separated tags from the exact Dietary list (English)",
        "occasion_tags": "Comma-separated tags from the exact Occasion list (English)",
        "seasonal_tags": "Comma-separated tags from the exact Seasonal list (English)",
        "tagging_reasoning": "Brief explanation for the assigned tags only",
        "brand": "Brand name (echo the extracted brand if present, else best guess from name)",
        "uom": "g or ml",
        "packaging": "Packaging type",
        "fragile_item": "Yes or No",
        "gross_weight": "Gross weight if genuinely known, else null",
        "organic_product": "Yes or No",
        "net_weight_customer_facing": "How weight/volume is displayed on pack",
        "organic_certification_id": "e.g. DE-ÖKO-001 or null"
    }}
    """

    client = genai.Client(api_key=gemini_key)
    last_error = "Unknown error"

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=EXTRACTION_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE),
                    ],
                ),
            )

            if not response.candidates:
                raise Exception(f"Request blocked entirely. Raw: {str(response)[:300]}")

            raw_text = ""
            if response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, "text", None):
                        raw_text += part.text + "\n"
            if not raw_text.strip() and getattr(response, "text", None):
                raw_text = response.text
            raw_text = raw_text.strip() if raw_text else ""
            if not raw_text:
                raise Exception(f"Empty text. Reason: {response.candidates[0].finish_reason}")

            start_idx = raw_text.find("{")
            end_idx = raw_text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return json.loads(raw_text[start_idx:end_idx + 1], strict=False)
            raise Exception(f"No JSON object. AI wrote: {raw_text[:200]}...")

        except json.JSONDecodeError as e:
            last_error = f"JSON Error: {str(e)}"
        except Exception as e:
            last_error = str(e)
        if attempt < 2:
            time.sleep(3)

    return {"error": f"Categorization API Error (3 attempts). Last error: {last_error}"}


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


def _result_matches_gt(res: dict, ground_truth: str) -> bool:
    """
    Relevance filter for organic search results found via bare-number queries.
    Bare-GTIN searches match ANY page where the digit string appears — article
    numbers and order codes on unrelated shops (electronics, welding supplies,
    catering catalogues) are the classic false positives. When ground truth is
    available, require ≥30% of its significant tokens in the result's
    title+snippet before treating the link as a candidate source.
    Without ground truth, all results pass (nothing to compare against).
    """
    if not ground_truth:
        return True
    gt_tokens = {t.lower() for t in ground_truth.split() if len(t) > 3}
    if not gt_tokens:
        return True
    haystack = (res.get("title", "") + " " + res.get("snippet", "")).lower()
    return sum(1 for t in gt_tokens if t in haystack) / len(gt_tokens) >= 0.3


# ============================================================================
# ROUTE A / ROUTE B PAGE VERIFICATION
# A page qualifies as a trusted source for the link, food data, and image if:
#   Route A — the EAN is confirmed on the page (labelled GTIN context)   → strong
#   Route B — the product name matches a trusted anchor (user input OR
#             the EAN-database/registry name) by >=80% token overlap,
#             with no conflicting weight/size                            → flagged
# Route B pages are shown/used but flagged and cap the row's reliability at M.
# ============================================================================

_NAME_MATCH_THRESHOLD = 0.80   # per feedback: name must match >=80%


def _sig_tokens(s: str) -> set:
    """Significant (>3 char) lowercased tokens, punctuation stripped."""
    if not s:
        return set()
    cleaned = re.sub(r"[^\w\s]", " ", s.lower())
    return {t for t in cleaned.split() if len(t) > 3}


def _name_match_ratio(anchor: str, text: str) -> float:
    """Fraction of the anchor's significant tokens present in `text`."""
    a = _sig_tokens(anchor)
    if not a:
        return 0.0
    hay = text.lower()
    return sum(1 for t in a if t in hay) / len(a)


def _best_name_ratio(anchors: list[str], text: str) -> float:
    """Best token-overlap across all trusted anchors (user input, registry)."""
    return max((_name_match_ratio(a, text) for a in anchors if a), default=0.0)


def _weights_conflict(anchors: list[str], text: str) -> bool:
    """
    True only when an anchor states a weight AND the page states a DIFFERENT
    weight. Missing weight on either side is not a conflict (feedback: match
    other details only when present). Prevents 5kg-vs-500g variant mix-ups.
    """
    text_w = _extract_weight_hint(text)
    if not text_w:
        return False
    for a in anchors:
        aw = _extract_weight_hint(a)
        if aw and aw != text_w:
            return True
    return False


async def classify_page(session, url: str, ean: str, anchors: list[str]):
    """
    Fetch a candidate page ONCE and classify it as a trusted source.

    Returns (route, images) where:
      route  = "A"  → EAN confirmed on page (strong)
             = "B"  → name matches a trusted anchor >=80%, no weight conflict
             = None → not trustworthy; caller should skip it
      images = list of (source_tag, image_url) scraped from the page (og/JSON-LD),
               so the caller can reuse the same verified page for the image.

    Fails closed (returns (None, [])) on any network/parse error.
    """
    if not url or not url.startswith("http") or _is_excluded_domain(url):
        return None, []
    _ua = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
           "AppleWebKit/537.36 (KHTML, like Gecko) "
           "Chrome/122.0.0.0 Safari/537.36")
    try:
        async with session.get(url, headers={"User-Agent": _ua}, timeout=8) as r:
            if r.status != 200:
                return None, []
            raw  = await r.content.read(400_000)
            text = raw.decode("utf-8", errors="ignore")
    except Exception:
        return None, []

    stripped = ean.lstrip("0")
    route = None

    # ── Route A: EAN in a labelled GTIN context ──────────────────────────────
    if re.search(rf'(gtin\d*|"sku"|itemprop=["\']gtin)[\s"\':=]+["\']?0*{re.escape(stripped)}',
                 text, re.IGNORECASE) or \
       re.search(rf'(ean|gtin|barcode|strichcode|streepjescode|'
                 rf'codice\s*a\s*barre|c[oó]digo\s*de\s*barras|upc)'
                 rf'[^0-9]{{0,30}}0*{re.escape(stripped)}', text, re.IGNORECASE):
        route = "A"
    else:
        # ── Route B: name match against trusted anchors ──────────────────────
        # Compare anchors to the page title / og:title / h1 (identity fields),
        # not the whole body (which contains menus, related products, etc.).
        title_bits = []
        m = re.search(r'<meta[^>]+og:title[^>]+content=["\']([^"\']+)', text, re.I)
        if m: title_bits.append(m.group(1))
        m = re.search(r'<title[^>]*>([^<]+)</title>', text, re.I)
        if m: title_bits.append(m.group(1))
        m = re.search(r'<h1[^>]*>(.*?)</h1>', text, re.I | re.S)
        if m: title_bits.append(re.sub(r"<[^>]+>", " ", m.group(1)))
        page_identity = " ".join(title_bits)
        if page_identity and anchors:
            ratio = _best_name_ratio(anchors, page_identity)
            if ratio >= _NAME_MATCH_THRESHOLD and not _weights_conflict(anchors, page_identity):
                route = "B"

    if route is None:
        return None, []

    # Scrape images from the (now-trusted) page so the caller can reuse it.
    try:
        images = await display_extract_from_page(session, url)
    except Exception:
        images = []
    return route, images


def is_garbage_name(name: str) -> bool:
    if not name or len(name.strip()) < 3:
        return True
    name_lower = name.lower().strip()
    # URLs / domains are never product names (search-result titles are
    # sometimes the bare URL — these were leaking into the Product Name column)
    if name_lower.startswith("http") or "://" in name_lower or "www." in name_lower:
        return True
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
                                ean_on_page = await page_contains_ean(session, link, ean, _clean_gt)
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
                    top_link  = organic[0].get("link", "")
                    if (not is_garbage_name(candidate)
                            and _result_matches_gt(organic[0], _clean_gt)
                            and _is_trusted_source(top_link, brand_for_verify)
                            and not _weights_conflict([_clean_gt], candidate)):
                        product_name = candidate
                        diag.log(f"✅ Global GTIN name (trusted source, unverified EAN): {product_name}")
                new_urls = [
                    res.get("link") for res in organic[:5]
                    if "link" in res and not _is_excluded_domain(res.get("link", ""))
                    and _result_matches_gt(res, _clean_gt)
                ]
                for u in new_urls:
                    if u not in retailer_urls:
                        retailer_urls.append(u)
            elif data is None:
                diag.log("   ⚠️ Global GTIN: SerpAPI returned no data (rate-limited or quota).")
        except Exception as e:
            diag.log(f"⚠️ Global search failed: {e}")

    # ── Stage 1D: Name-based fallback (mirrors manual workflow) ──────────────
    # EAN-keyed queries produced neither a name nor retailer pages, but ground
    # truth exists. Search by NAME restricted to the market goldmine; verified
    # pages (EAN found on page) become high-trust jsonld_retailer sources,
    # fuzzy matches become og_retailer (lower trust, higher vision threshold).
    if serp_key and _clean_gt and (not product_name or not retailer_urls):
        diag.log(f"🔍 Name-based fallback: '{_clean_gt[:60]}'...")
        _nb_goldmine = GOLDMINE.get(market_upper, "").strip()
        _nb_queries  = [q for q in (
            f'{_clean_gt} {ean}',                                  # combined — strongest
            f'{_nb_goldmine} "{_clean_gt}"' if _nb_goldmine else "",
            f'"{_clean_gt}"',
        ) if q]
        try:
            for _nq in _nb_queries:
                data_nb = await _serp_get(session, serp_url,
                    params={"q": _nq, "gl": gl, "api_key": serp_key})
                organic_nb = data_nb.get("organic_results", []) if data_nb else []
                found_here = False
                for res in organic_nb[:4]:
                    link = res.get("link", "")
                    if not link or _is_excluded_domain(link):
                        continue
                    title = res.get("title", "").split("-")[0].split("|")[0].strip()
                    if is_garbage_name(title):
                        continue
                    if await page_contains_ean(session, link, ean, _clean_gt):
                        if not product_name:
                            product_name = title
                        if link not in retailer_urls:
                            retailer_urls.insert(0, link)
                        diag.log(f"✅ Name-fallback page EAN-VERIFIED: {link[:80]}")
                        found_here = True
                        break
                    # Name(+weight)-only match, no EAN on page: only acceptable from
                    # a trusted (goldmine/brand) domain, and only when the weight/
                    # volume doesn't conflict — otherwise a different-variant page
                    # can silently become the identity used for image verification.
                    if not _is_trusted_source(link, brand_for_verify):
                        continue
                    gt_tokens = {t.lower() for t in _clean_gt.split() if len(t) > 3}
                    if gt_tokens:
                        hit_ratio = sum(1 for t in gt_tokens if t in title.lower()) / len(gt_tokens)
                        if hit_ratio >= 0.6 and not _weights_conflict([_clean_gt], title):
                            if not product_name:
                                product_name = title
                            if link not in retailer_urls:
                                retailer_urls.append(link)
                            diag.log(f"⚠️ Name-fallback trusted-source match "
                                     f"({hit_ratio:.0%}, no weight conflict): {link[:80]}")
                            found_here = True
                            break
                if found_here:
                    break
        except Exception as e:
            diag.log(f"⚠️ Name-based fallback failed: {e}")

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
    _name_known = bool(_pname_for_verify) and not _pname_for_verify.startswith("Product with EAN")

    # When product name is unknown, only trust Go-UPC images (barcode-keyed).
    # Without a name, the vision verifier can't distinguish the right product
    # from a tin of foil, so we skip unverifiable candidates entirely.
    if not _name_known:
        deduped = [(s, o, t, ti, sd) for s, o, t, ti, sd in deduped
                   if s == "go_upc_tier1"]
        if deduped:
            diag.log("ℹ️ Product name unknown — only Go-UPC Tier 1 image shown.")
        else:
            diag.log("ℹ️ Product name unknown and no Go-UPC image — no images displayed.")

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

    # ── Stage 7: Unified Route A / Route B source verification ────────────────
    # A page becomes a trusted source (for its link AND, if we still need one,
    # its image) when EITHER the EAN is confirmed on it (Route A) OR its name
    # matches a trusted anchor — user input OR the EAN-database/registry name —
    # by >=80% (Route B). Route B pages are flagged; the row caps at M.
    #
    # This fixes: name-matched pages (e.g. Ökostern Nudeln on eekenhof-shop,
    # which carry NO EAN) now supply both the link and the image, exactly as
    # they already supply the food data.
    _registry_name = ""
    if go_upc_data and go_upc_data.get("name") and not is_garbage_name(go_upc_data["name"]):
        _registry_name = go_upc_data["name"]
    _anchors = [a for a in (_clean_gt, _registry_name) if a]

    source_links      = []          # list[str] (order preserved)
    source_routes     = {}          # url -> "A" | "B"
    seen_link_domains = set()
    for page_url in retailer_urls:
        if len(source_links) >= 2:
            break
        if not page_url or _is_excluded_domain(page_url):
            continue
        dom = page_url.split("/")[2] if page_url.startswith("http") else ""
        if not dom or dom in seen_link_domains:
            continue
        route, page_imgs = await classify_page(session, page_url, ean, _anchors)
        if route is None:
            diag.log(f"   ⚠️ Source link skipped — neither EAN nor name match: {dom}")
            continue
        source_links.append(page_url)
        source_routes[page_url] = route
        seen_link_domains.add(dom)
        diag.log(f"✅ Source link ({'EAN-verified' if route=='A' else 'name-matched'}): {dom}")

        # If we have no image yet, mine THIS verified page's own image.
        if not selected and page_imgs:
            for src_tag, img_url in page_imgs:
                if not _is_valid_image_url(img_url) or _is_excluded_domain(img_url):
                    continue
                payload = await display_fetch_image_bytes(session, img_url)
                if not payload or "error" in payload:
                    continue
                inspection = display_inspect_image(payload["data"])
                if not inspection["ok"]:
                    continue
                safe_mime = _safe_mime(payload["mime"], payload["data"]) or "image/jpeg"
                # Route A page image → trust (page identity is the verification).
                # Route B page image → vision-verify (name match is weaker; a
                # wrong-variant page could carry a plausible-but-wrong picture).
                if route == "B" and gemini_key and product_name and \
                        not str(product_name).startswith("Product with EAN"):
                    vscore = await asyncio.to_thread(
                        verify_image_with_gemini, payload["data"], safe_mime,
                        product_name, brand_for_verify, gemini_key, weight_for_verify)
                    if vscore is not None and 0 <= vscore < IMAGE_MATCH_THRESHOLD_TRUSTED:
                        diag.log(f"   ⚠️ Route-B page image rejected by vision ({vscore:.2f}): {dom}")
                        continue
                selected.append({
                    "url": img_url, "mime": safe_mime, "data": payload["data"],
                    "source": f"route{route}_page_{src_tag}", "inspection": inspection,
                    "phash": display_compute_phash(payload["data"]),
                    "display": True, "match": 0.5,
                })
                diag.log(f"   ✅ Image sourced from {'EAN' if route=='A' else 'name'}-matched page: {dom}")
                break

    return selected, diag, source_links, retailer_urls, source_routes


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
        cached = cache_get(ean, market, ground_truth)
        if cached:
            # Always reflect the CURRENT request's input, never the input the
            # row was first cached under — prevents a stale User Input string
            # (e.g. an old "120x …") showing on a hit.
            cached["Cached"]      = "✅ Cached"
            cached["User Input"]  = ground_truth
            cached["GTIN / EAN"]  = ean
            empty_diag = ImageDiagnostics(ean)
            empty_diag.log("✅ Loaded from cache.")
            return {"row": cached, "image_diag": empty_diag, "food_diag": None}
    else:
        cache_delete(ean, market, ground_truth)

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

        (name, gemini_images, food_diag, ean_verified, food_retailer_urls),         (display_images, image_diag, source_links, pipeline_urls, source_routes) = await asyncio.gather(food_task, image_task)

        # Path A discovered and verified its own retailer pages (e.g. the Byodo
        # oil on bioaufvorrat.de, or the Ökostern Nudeln on eekenhof-shop found
        # by NAME) that Path B's separate search may never have seen. Share and
        # classify them with the SAME Route A/B logic so a page found by name
        # (Route B) still supplies its link and image — flagged, capping M.
        for _u in (food_retailer_urls or []):
            if _u and _u not in (pipeline_urls or []):
                pipeline_urls.append(_u)

        _registry_name_pa = ""
        if go_upc_data and go_upc_data.get("name") and not is_garbage_name(go_upc_data["name"]):
            _registry_name_pa = go_upc_data["name"]
        _clean_gt_pa = _strip_pack_notation(ground_truth) if ground_truth else ground_truth
        _anchors_pa = [a for a in (_clean_gt_pa, _registry_name_pa) if a]

        _existing_src_domains = {s.split("/")[2] for s in source_links if s.startswith("http")}
        for _u in (food_retailer_urls or []):
            if len(source_links) >= 2:
                break
            if not _u or not _u.startswith("http") or _is_excluded_domain(_u):
                continue
            _dom = _u.split("/")[2]
            if _dom in _existing_src_domains:
                continue
            _route_pa, _page_imgs_pa = await classify_page(session, _u, ean, _anchors_pa)
            if _route_pa is None:
                continue
            source_links.append(_u)
            source_routes[_u] = _route_pa
            _existing_src_domains.add(_dom)
            image_diag.log(f"✅ Promoted Path-A page ({'EAN' if _route_pa=='A' else 'name'}-matched): {_dom}")
            # If we still have no image, mine this verified page for one.
            if not display_images and _page_imgs_pa:
                for _tag, _img in _page_imgs_pa:
                    if not _is_valid_image_url(_img) or _is_excluded_domain(_img):
                        continue
                    _pl = await display_fetch_image_bytes(session, _img)
                    if not _pl or "error" in _pl:
                        continue
                    _insp = display_inspect_image(_pl["data"])
                    if not _insp["ok"]:
                        continue
                    _smime = _safe_mime(_pl["mime"], _pl["data"]) or "image/jpeg"
                    # Route B image → vision-verify (name match is weaker).
                    if _route_pa == "B" and gemini_key and name and \
                            not str(name).startswith("Product with EAN"):
                        _vs = await asyncio.to_thread(
                            verify_image_with_gemini, _pl["data"], _smime,
                            name, "", gemini_key, "")
                        if _vs is not None and 0 <= _vs < IMAGE_MATCH_THRESHOLD_TRUSTED:
                            image_diag.log(f"   ⚠️ Route-B Path-A image rejected by vision: {_dom}")
                            continue
                    display_images.append({
                        "url": _img, "mime": _smime, "data": _pl["data"],
                        "source": f"pathA_route{_route_pa}_{_tag}", "inspection": _insp,
                        "phash": display_compute_phash(_pl["data"]),
                        "display": True, "match": 0.5,
                    })
                    image_diag.log(f"   ✅ Image from Path-A {'EAN' if _route_pa=='A' else 'name'}-matched page: {_dom}")
                    break

        # ── Change 1: Early exit when no approved source found anything ───────
        no_sources = (
            name.startswith("Product with EAN") and
            not ean_verified and
            len(gemini_images) == 0 and
            go_upc_data is None
        )
        if no_sources:
            # If any SerpAPI call hard-failed (429/5xx after retries), this row
            # is INCONCLUSIVE, not "Not Found" — surface that so users re-run
            # instead of trusting a quota failure as a genuine miss.
            if "SERP_FAILED_FLAG" in (food_diag or ""):
                nf_status = "⚠️ Search Failed — quota/rate-limit hit, please re-run"
            else:
                nf_status = "⚠️ Not Found — no approved source indexed this EAN"
                if not ground_truth:
                    nf_status += " (tip: add the product name to the input line to enable name-based fallback)"
            row = {
                "Image 1": "", "Image 2": "", "Image Source Link": "",
                "GTIN / EAN": ean, "User Input": ground_truth,
                "Status": nf_status,
                "Cached": "🔄 Fresh",
            }
            return {"row": row, "image_diag": image_diag, "food_diag": food_diag}

        # ── Path B output (display images) ────────────────────────────────────
        # Ephemeral CDN thumbnails (encrypted-tbn*.gstatic.com etc.) render now
        # but expire within days — the main cause of broken links in exports.
        # Blank them; the verified Image Source Link still gives a working page.
        display_urls = []
        for img in display_images:
            u = img["url"]
            if _is_durable_url(u):
                display_urls.append(u)
            else:
                image_diag.log(f"⚠️ Image URL dropped from output (ephemeral CDN): {u[:80]}")
        imgs = (display_urls + ["", ""])[:2]

        # ── Liveness gate: a page being VERIFIED at extraction time doesn't
        # guarantee it's still reachable when someone views/exports this row.
        # Re-check every candidate Image Source Link right now and drop dead
        # ones so an exported link is never a 404.
        if source_links:
            _alive = await asyncio.gather(
                *[_url_alive(session, u) for u in source_links])
            _dead_img_srcs = [u for u, ok in zip(source_links, _alive) if not ok]
            if _dead_img_srcs:
                image_diag.log(
                    f"⚠️ {len(_dead_img_srcs)} Image Source Link candidate(s) dropped "
                    f"— no longer live: {', '.join(_domain_of(u) for u in _dead_img_srcs[:5])}")
            source_links = [u for u, ok in zip(source_links, _alive) if ok]
        img_src_links = (source_links + ["", ""])[:2]

        # ── Path A: deterministic-first food extraction (verify-then-read) ────
        # Food fields now come from the SAME verified pages that fill the Source
        # columns. extract_food_data verifies each candidate (Route A EAN-on-page
        # / Route B name>=80%) and reads it deterministically (JSON-LD ->
        # microdata/OG -> labelled HTML table); Gemini is a text-only last resort.
        # There is NO grounded search for food data any more.
        if not FOOD_PIPELINE_OK:
            # Hard dependency: without the module we cannot extract food data.
            row = {
                "Image 1": imgs[0], "Image 2": imgs[1],
                "Image Source Link": img_src_links[0],
                "GTIN / EAN": ean, "User Input": ground_truth,
                "Status": "⚠️ food_pipeline.py missing — cannot extract food data",
                "Cached": "🔄 Fresh",
            }
            return {"row": row, "image_diag": image_diag, "food_diag": food_diag}

        _registry_name = ""
        if go_upc_data and go_upc_data.get("name") and not is_garbage_name(go_upc_data["name"]):
            _registry_name = go_upc_data["name"]
        _food_keys = {"serp": serp_key, "gemini": gemini_key,
                      "ean_search": ean_token, "go_upc": go_upc_key}
        # Reuse the pages the pipeline already discovered (verified links first)
        # so the module does NOT re-run the search ladder in production.
        _candidate_pages = list(dict.fromkeys(
            [u for u in (source_links or []) if u]
            + [u for u in (food_retailer_urls or []) if u]
            + [u for u in (pipeline_urls or []) if u]
        ))
        food = await extract_food_data(
            session, ean, ground_truth, market, _registry_name, _food_keys,
            go_upc_data=go_upc_data, candidate_pages=_candidate_pages,
            ean_verified_hint=ean_verified,
        )
        food_fields     = food.get("fields", {})
        food_status     = food.get("status", "no_source")
        food_reliab     = food.get("reliability", "L")
        food_srcs       = food.get("source_links", [])
        food_routes     = food.get("source_routes", {})
        data_provenance = food.get("provenance", "")
        for _dl in food.get("diagnostics", []):
            image_diag.log(f"[food] {_dl}")

        # ── Categorization + tagging (Gemini, fed the extracted data; NO search)
        cat = await asyncio.to_thread(
            run_categorization_sync, ean, name, market, gemini_key,
            taxonomy_text, food_fields, ground_truth, go_upc_data
        )
        if "error" in cat:
            # Categorisation is non-fatal: keep the food data we extracted and
            # leave categories/tags blank rather than dropping the whole row.
            image_diag.log(f"⚠️ Categorization failed: {cat['error']} — "
                           f"food data preserved, categories left blank.")
            cat = {}

        # ── Merge: deterministic food fields OVERRIDE Gemini-derived attributes.
        # Numbers, ingredients, allergens, manufacturer and origin are read from
        # the page and must never be overwritten by an LLM guess. The
        # categorization prompt is told not to emit these keys, but an LLM can
        # still add them despite instruction — strip them out before merging so
        # a hallucinated value can never survive with no source link behind it
        # (setdefault alone doesn't protect against a key the categorizer
        # already populated).
        data = {k: v for k, v in cat.items() if k not in ALL_FIELD_KEYS}
        for _k in ALL_FIELD_KEYS:
            data[_k] = food_fields.get(_k, "")

        # ── Source 1-N = the VERIFIED pages the food data was read from ───────
        _verified_srcs = list(dict.fromkeys(
            [u for u in food_srcs if u and _is_displayable_url(u)]))

        # ── Liveness gate: same reasoning as the Image Source Link check above
        # — a page verified during extraction can still have gone dead by the
        # time this row is viewed or exported. Drop any Source link that no
        # longer resolves rather than shipping a link that 404s.
        if _verified_srcs:
            _alive_food = await asyncio.gather(
                *[_url_alive(session, u) for u in _verified_srcs])
            _dead_food_srcs = [u for u, ok in zip(_verified_srcs, _alive_food) if not ok]
            if _dead_food_srcs:
                image_diag.log(
                    f"⚠️ {len(_dead_food_srcs)} food Source link(s) dropped — no "
                    f"longer live: {', '.join(_domain_of(u) for u in _dead_food_srcs[:5])}")
            _verified_srcs = [u for u, ok in zip(_verified_srcs, _alive_food) if ok]

        srcs = (_verified_srcs + ["", "", "", "", ""])[:5]

        # ── Link Basis — how each shown link was verified (A=EAN, B=name) ────
        _routes_shown = [food_routes.get(u) for u in _verified_srcs]
        _has_A = "A" in _routes_shown
        _has_B = "B" in _routes_shown
        _lb_parts = []
        if _has_A:
            _lb_parts.append("EAN-verified")
        if _has_B:
            _lb_parts.append("Name-matched ⚠️")
        link_basis = " + ".join(_lb_parts)

        # ── Audit candidates: discovered pages that did NOT become sources ───
        _audit_candidates = list(dict.fromkeys(
            [u for u in (food.get("retailer_urls") or [])
             if u and _is_displayable_url(u) and u not in _verified_srcs]
            + [u for u in (pipeline_urls or [])
               if u and _is_displayable_url(u) and u not in _verified_srcs]
        ))[:8]
        audit_sources = " | ".join(_audit_candidates)

        # ── Status + reliability come from the PIPELINE verification, not Gemini
        #   success           → H: a trusted (brand/goldmine) page contributed;
        #                       M: only non-tier-1 sources, but >=2 of them agree;
        #                       L: a single non-tier-1 source, uncorroborated.
        #   page_not_readable → verified but JS/cookie-walled: Failed Validation/L
        #   no_source         → no verified page at all: Failed Validation/L
        if food_status == "success":
            final_status = "Success"
            reliability  = food_reliab
            if reliability == "H":
                reliability_reasoning = (
                    f"Confirmed by a trusted source (the brand's own site or a "
                    f"tier-1 retailer) — {', '.join(_verified_srcs[:2])}.")
            elif reliability == "M":
                reliability_reasoning = (
                    f"No trusted (brand/tier-1) source found; the value(s) shown "
                    f"are corroborated across >=2 independent retailer pages — "
                    f"{', '.join(_verified_srcs[:2])}.")
            else:
                reliability_reasoning = (
                    f"Only a single non-tier-1 source backs this data, with no "
                    f"corroboration from another retailer — {', '.join(_verified_srcs[:2])}. "
                    f"Verify manually before upload.")
        else:
            final_status = "Failed Validation"
            reliability  = "L"
            if food_status == "page_not_readable":
                reliability_reasoning = (
                    "Verified page(s) found but not readable (JS/cookie wall) — "
                    "no field is shown without a readable source behind it.")
            else:
                reliability_reasoning = (
                    "No verified source page could be read for this EAN — "
                    "no field is shown without a readable source behind it.")

        # ── Assemble the row ──────────────────────────────────────────────────
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
            "Link Basis": link_basis,
            "Reliability Reasoning": reliability_reasoning,
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
            "Source Candidates (audit)": audit_sources,
            "Data Provenance": data_provenance,
            "Cached": "🔄 Fresh",
        }

        # ── Cache only confirmed successes ────────────────────────────────────
        if should_cache(final_status):
            cache_set(ean, market, row, status=final_status, confidence=1.0,
                      ground_truth=ground_truth)

        # Sanitise: replace Python None and the string "null" with ""
        # so Streamlit never renders the word "None" in any cell.
        row = {k: ("" if (v is None or v == "null" or v == "None") else v)
               for k, v in row.items()}
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
    st.caption(f"build: {BUILD_VERSION}")
    _market_options = [
        "Belgium (BE)", "Denmark (DK)", "Germany (DE)", "Austria (AT)",
        "Netherlands (NL)", "France (FR)", "Italy (IT)", "Spain (ES)",
        "United Kingdom (UK)", "Poland (PL)"
    ]
    # Remember the last market used (persisted in cache.db, survives across
    # sessions/reloads — plain st.session_state would not, since a fresh
    # browser session gets a fresh session_state).
    _last_market = get_setting("last_market", _market_options[0])
    _default_idx = (_market_options.index(_last_market)
                    if _last_market in _market_options else 0)
    market_selection = st.selectbox(
        "Target Market", _market_options, index=_default_idx
    )
    if market_selection != _last_market:
        set_setting("last_market", market_selection)
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

# ── Always-visible legend (shown on the landing page, above the input) ─────────
with st.expander("ℹ️ How to read the results — reliability grades, status & links", expanded=True):
    st.markdown(
        "**Info Reliability**\n"
        "- **H (High)** — details came from the brand's own website or a tier-1 trusted "
        "retailer (Goldmine). Strong, trustworthy source. Safe to upload.\n"
        "- **M (Medium)** — details came from other (non-tier-1) retailers, but the same "
        "value was independently confirmed on 2+ of them. Reasonable, cross-checked.\n"
        "- **L (Low)** — details came from only one non-tier-1 source, with no "
        "corroboration. Thin — verify against the source link before upload.\n\n"
        "**Status**\n"
        "- **Success** — food data was read from a verified source page; use the H/M/L grade to judge confidence.\n"
        "- **Failed Validation** — a page was found but couldn't be read (JS/cookie wall) or no verified page "
        "was found. No field is ever shown without a readable source behind it — a blank cell here means "
        "genuinely nothing was found, not a hidden fallback value.\n"
        "- **Search Failed** — a lookup was rate-limited; re-run the row.\n\n"
        "**Links** — Source links are shown **only** when the pipeline fetched the page, confirmed it "
        "belongs to the product, AND re-confirmed the page is still live right now — so a shown link "
        "always points at the correct item and shouldn't 404. The food data in the row is read from "
        "those same verified pages, never from a barcode-registry API (Go-UPC/EAN-Search are only used "
        "to help resolve an identity that sharpens the search — never as a data source for a shown field). "
        "The **Link Basis** column tells you how it was confirmed:\n"
        "- **EAN-verified** — the barcode was found on the page (strongest).\n"
        "- **Name-matched ⚠️** — the page matched the product name (≥80%, no weight/variant conflict) but "
        "carried no barcode. Only accepted from a trusted (brand/tier-1) domain — on any other domain, "
        "identity requires the barcode to actually be on the page. Common for brand/organic shops that "
        "don't publish EANs.\n"
        "Unverified candidates are kept in the hidden *Source Candidates (audit)* column "
        "(toggle it on above the results table).\n\n"
        "**Data Provenance** — where the food data was read from:\n"
        "- **Verified page** — every field was read directly from the linked source page(s). "
        "The numbers match the Source columns.\n"
        "- **Verified page + fallback ⚠️** — most fields were read from the linked page; a few absent "
        "ones were filled by a text-only Gemini extraction fed that same page's text (never a search). "
        "Those cells are weaker.\n"
        "- **Nutrition failed plausibility check ⚠️** — extracted nutrition was physically impossible "
        "(e.g. macros summing above 100 g/100 g) or disagreed with an internal reference; it was "
        "blanked and the row dropped to **L**.\n"
        "- **Page not readable** — a page was verified as the right product but couldn't be read "
        "(JS/cookie wall); no field is shown from it.\n"
        "- **No verified source** — no verified page could be read at all; "
        "data is unconfirmed (these rows grade **L** and show *Failed Validation*)."
    )

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
        "Link Basis",
        "Data Provenance",
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
        "Source Candidates (audit)",
    ]

    existing_ordered = [c for c in column_order if c in df.columns]
    remaining        = [c for c in df.columns if c not in column_order]
    df               = df[existing_ordered + remaining]

    st.subheader("📊 Results")

    # ── Audit-column visibility toggle (feedback #2) ──────────────────────────
    show_audit = st.checkbox(
        "🔍 Show link / audit columns (Image Source Link, source candidates)",
        value=False,
        help="Off by default for a cleaner DMF view. Turn on to see and click "
             "the underlying image and source URLs.",
    )
    _audit_cols = ["Image Source Link", "Source Candidates (audit)"]
    if not show_audit:
        df = df.drop(columns=[c for c in _audit_cols if c in df.columns], errors="ignore")

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
            **({"Image Source Link": st.column_config.LinkColumn(
                "Image Source Link",
                display_text="🔗 Image Source",
                help="Direct URL to the product image. Click to open if the image above could not be rendered.",
            )} if show_audit else {}),
            **({"Source Candidates (audit)": st.column_config.TextColumn(
                "Source Candidates (audit)",
                help="Unverified discovery candidates — NOT confirmed to match the EAN. For audit only.",
            )} if show_audit else {}),
            "Source 1": st.column_config.LinkColumn(display_text="Link 1"),
            "Source 2": st.column_config.LinkColumn(display_text="Link 2"),
            "Source 3": st.column_config.LinkColumn(display_text="Link 3"),
            "Source 4": st.column_config.LinkColumn(display_text="Link 4"),
            "Source 5": st.column_config.LinkColumn(display_text="Link 5"),
        },
        width='stretch',
        hide_index=True
    )

    # ── DMF-ready Excel export ────────────────────────────────────────────────
    # Two-sheet workbook: "DMF Upload" (color-coded clusters, audit columns
    # hidden but expandable) + "Legend" (color key, status & reliability tiers).
    if XLSX_EXPORT_OK:
        try:
            _export_df = st.session_state["results_df"].copy()
            _export_df = _export_df.drop(columns=["Re-run?"], errors="ignore")
            xlsx_bytes = build_dmf_workbook(_export_df)
            st.download_button(
                "⬇️ Download DMF-ready Excel (color-coded + legend)",
                data=xlsx_bytes,
                file_name=f"dmf_export_{market_code}_{time.strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as _e:
            st.warning(f"Excel export unavailable: {_e}")
    else:
        st.info("💡 Add xlsx_export.py (and `pip install openpyxl`) to enable the "
                "color-coded DMF Excel download.")

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
