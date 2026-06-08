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
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ean_cache (
            ean TEXT NOT NULL,
            market TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ean, market)
        )
    """)
    con.commit()
    con.close()

def cache_get(ean: str, market: str):
    try:
        con = sqlite3.connect(DB_PATH)
        row = con.execute(
            "SELECT result_json FROM ean_cache WHERE ean=? AND market=?",
            (ean, market)
        ).fetchone()
        con.close()
        return json.loads(row[0]) if row else None
    except Exception:
        return None

def cache_set(ean: str, market: str, result_dict: dict):
    try:
        con = sqlite3.connect(DB_PATH)
        con.execute(
            "INSERT OR REPLACE INTO ean_cache (ean, market, result_json) VALUES (?,?,?)",
            (ean, market, json.dumps(result_dict))
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
# SHARED CONSTANTS (used by both pipelines)
# ============================================================================

GOLDMINE = {
    "FR": "site:carrefour.fr OR site:auchan.fr OR site:coursesu.com",
    "UK": "site:ocado.com OR site:waitrose.com OR site:asda.com OR site:tesco.com",
    "NL": "site:ah.nl OR site:jumbo.com OR site:plus.nl",
    "BE": "site:delhaize.be OR site:colruyt.be OR site:carrefour.be",
    "DE": "site:rewe.de OR site:edeka.de OR site:kaufland.de OR site:dm.de OR site:rossmann.de",
    "AT": "site:billa.at OR site:spar.at OR site:gurkerl.at OR site:hofer.at",
    "DK": "site:nemlig.com OR site:matsmart.dk OR site:rema1000.dk",
    "IT": "site:carrefour.it OR site:conad.it OR site:coop.it",
    "ES": "site:carrefour.es OR site:mercadona.es OR site:dia.es",
    "SE": "site:ica.se OR site:coop.se OR site:willys.se",
    "NO": "site:oda.com OR site:meny.no OR site:holdbart.no",
    "FI": "site:k-ruoka.fi OR site:s-kaupat.fi",
    "PL": "site:carrefour.pl OR site:auchan.pl OR site:frisco.pl",
}
GLOBAL_SITES = "site:billigkaffee.eu OR site:fivestartrading-holland.eu"

# ============================================================================
# PATH A: ORIGINAL FOOD EXTRACTION PIPELINE (UNCHANGED FROM YOUR ORIGINAL)
# ============================================================================
# Everything in this section is your original code, byte-for-byte.
# It handles: name lookup, image gathering for Gemini, the Gemini call.
# DO NOT modify this section - it's working well for food info extraction.

BAD_IMAGE_EXTENSIONS = {".svg", ".gif", ".ico", ".webmanifest", ".json", ".xml"}
BAD_IMAGE_PATTERNS = [
    "logo", "icon", "banner", "placeholder", "spinner", "loading",
    "payment", "paypal", "mastercard", "visa", "flag", "star",
    "cart", "account", "arrow", "check", "tick", "social",
    "openfoodfacts", "pinterest", "ebay", "tiktok", "facebook",
    "instagram", "twitter", "youtube", "amazon-ads", "ad_",
    "s192", "width=250", "160x160", "200x200", "250x250", "300x300",
    "50x50", "75x30", "100x100", "128x128", "150x150", "_xs", "_xxs", "thumbnail"
]

def _is_valid_image_url(url: str) -> bool:
    if not url or not url.startswith("http"):
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
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    try:
        async with session.get(url, headers=headers, timeout=5) as resp:
            if resp.status == 200:
                html = await resp.text()
                match = re.search(r'<meta[^>]*property=[\'"]og:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]', html, re.IGNORECASE)
                if match:
                    return match.group(1)
    except Exception:
        pass
    return None

GEMINI_SUPPORTED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

MAGIC_BYTES_MAP = [
    (b"\xff\xd8\xff",       "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"RIFF",               "image/webp"),   # needs extra check but good enough
    (b"GIF87a",             "image/gif"),
    (b"GIF89a",             "image/gif"),
]

def _sniff_mime(data: bytes) -> str | None:
    """Detect image type from magic bytes. Returns None if unrecognised."""
    for magic, mime in MAGIC_BYTES_MAP:
        if data[:len(magic)] == magic:
            return mime
    return None

def _safe_mime(raw_mime: str, data: bytes) -> str | None:
    """
    Return a Gemini-safe MIME type or None if the image should be skipped.
    1. Strip content-type parameters (e.g. 'image/jpeg; charset=utf-8' → 'image/jpeg')
    2. If the cleaned type is supported, use it.
    3. If it's octet-stream / unknown, sniff from magic bytes.
    4. If still unresolvable, return None so the caller can skip the image.
    """
    clean = raw_mime.split(";")[0].strip().lower() if raw_mime else ""
    if clean in GEMINI_SUPPORTED_MIMES:
        return clean
    # Fall back to magic-byte sniffing
    sniffed = _sniff_mime(data)
    return sniffed  # may be None → caller should skip

async def fetch_image_bytes_simple(session, url):
    """Original image fetcher used by the food extraction pipeline. Lenient by design."""
    try:
        async with session.get(url, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.read()
                if len(data) < 8000:
                    return None
                raw_mime = resp.headers.get("content-type", "")
                mime = _safe_mime(raw_mime, data)
                if mime is None:
                    return None  # skip — Gemini would reject this
                return {"url": url, "mime": mime, "data": data}
    except Exception:
        pass
    return None

async def fetch_basic_info(session, ean, serp_key, ean_token, market_code, user_ground_truth=""):
    """ORIGINAL: Retrieves exact product name and performs deduplicated, size-verified image gathering for Gemini."""
    gl = market_code.lower()
    market_upper = market_code.upper()
    diagnostic_log = []

    product_name = None
    retailer_urls = []
    candidate_image_urls = []
    registry_image_url = None

    # ATTEMPT 1: EAN-Search API
    if ean_token:
        diagnostic_log.append("🔍 Attempt 1: EAN-Search.org API...")
        ean_url = f"https://api.ean-search.org/api?token={ean_token}&op=barcode-lookup&ean={ean}&format=json"
        try:
            async with session.get(ean_url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) > 0 and "error" not in data[0]:
                        product_name = data[0].get("name")
                        registry_image_url = data[0].get("image")
                        diagnostic_log.append(f"✅ Found Name via EAN-Search: {product_name}")
        except Exception as e:
            diagnostic_log.append(f"⚠️ EAN-Search failed: {e}")

    # ATTEMPT 2: Brand-site lookup (if brand provided in ground_truth)
    # This runs BEFORE goldmine so direct brand pages are prioritised as sources.
    brand_domain = None
    serp_url = "https://serpapi.com/search"
    if serp_key and user_ground_truth:
        # Extract a single-word brand candidate: first capitalised token in ground_truth
        import re as _re
        brand_tokens = [t for t in user_ground_truth.split() if t[0].isupper() and len(t) > 2]
        if brand_tokens:
            brand_candidate = brand_tokens[0]
            diagnostic_log.append(f"🔍 Brand-site lookup for '{brand_candidate}'...")
            try:
                async with session.get(serp_url, params={
                    "q": f"{brand_candidate} {ean}",
                    "gl": gl, "api_key": serp_key
                }, timeout=10) as resp:
                    data = await resp.json()
                    organic = data.get("organic_results", [])
                    # Look for a result whose domain contains the brand name (case-insensitive)
                    for res in organic[:5]:
                        link = res.get("link", "")
                        domain = link.split("/")[2] if link.startswith("http") else ""
                        if brand_candidate.lower() in domain.lower():
                            if not product_name:
                                product_name = res.get("title", "").split("-")[0].split("|")[0].strip()
                                diagnostic_log.append(f"✅ Found Name via brand-site: {product_name}")
                            if link not in retailer_urls:
                                retailer_urls.insert(0, link)  # brand page goes FIRST
                            brand_domain = domain
                            diagnostic_log.append(f"✅ Brand domain found: {brand_domain}")
                            break
            except Exception as e:
                diagnostic_log.append(f"⚠️ Brand-site lookup failed: {e}")

    # ATTEMPT 3: SerpAPI Text Search (Goldmine retailers)
    if not product_name and serp_key:
        diagnostic_log.append("🔍 Goldmine Google Search for Name...")
        goldmine = f"{GOLDMINE.get(market_upper, '')} OR {GLOBAL_SITES}".strip(" OR")

        try:
            async with session.get(serp_url, params={"q": f"{goldmine} {ean}", "gl": gl, "api_key": serp_key}, timeout=15) as resp:
                data = await resp.json()
                organic = data.get("organic_results", [])
                if organic:
                    product_name = organic[0].get("title", "").split("-")[0].split("|")[0].strip()
                    diagnostic_log.append(f"✅ Found Name via Goldmine: {product_name}")
                    new_urls = [res.get("link") for res in organic[:4] if "link" in res]
                    for u in new_urls:
                        if u not in retailer_urls:
                            retailer_urls.append(u)
                else:
                    diagnostic_log.append("⚠️ Goldmine failed, falling back to global bare GTIN search...")
                    async with session.get(serp_url, params={"q": str(ean), "gl": gl, "api_key": serp_key}, timeout=15) as resp2:
                        data2 = await resp2.json()
                        organic2 = data2.get("organic_results", [])
                        if organic2:
                            product_name = organic2[0].get("title", "").split("-")[0].split("|")[0].strip()
                            diagnostic_log.append(f"✅ Found Name via Global Search: {product_name}")
                            new_urls2 = [res.get("link") for res in organic2[:4] if "link" in res]
                            for u in new_urls2:
                                if u not in retailer_urls:
                                    retailer_urls.append(u)
        except Exception as e:
            diagnostic_log.append(f"⚠️ Google text search failed: {e}")

    if not product_name:
        diagnostic_log.append("⚠️ Name not found via databases. Relying entirely on Gemini...")
        product_name = f"Product with EAN {ean}"

    # GATHERING CANDIDATE IMAGES (for Gemini extraction - lenient, original logic)

    if retailer_urls:
        diagnostic_log.append("🌐 Scraping OG images directly from Retailers...")
        tasks = [fetch_og_image(session, url) for url in retailer_urls]
        og_images = await asyncio.gather(*tasks)
        for img in og_images:
            if img and _is_valid_image_url(img) and img not in candidate_image_urls:
                candidate_image_urls.append(img)

    if serp_key:
        diagnostic_log.append("🖼️ Searching high-res images via Google Images...")
        serp_url = "https://serpapi.com/search"
        try:
            r1, r2 = await asyncio.gather(
                session.get(serp_url, params={"q": f'"{ean}"', "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10),
                session.get(serp_url, params={"q": f'site:barcodelookup.com OR site:go-upc.com "{ean}"', "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10),
                return_exceptions=True
            )
            for resp in [r1, r2]:
                if not isinstance(resp, Exception) and resp.status == 200:
                    img_data = await resp.json()
                    for item in img_data.get("images_results", []):
                        url = item.get("original", "")
                        if _is_valid_image_url(url) and url not in candidate_image_urls:
                            candidate_image_urls.append(url)
        except Exception as e:
            pass

    if registry_image_url and _is_valid_image_url(registry_image_url) and registry_image_url not in candidate_image_urls:
        candidate_image_urls.append(registry_image_url)

    # DOWNLOADING, SIZING, AND DEDUPLICATING IMAGES (for Gemini)
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

    diagnostic_log.append(f"✅ Secured {len(final_downloaded_images)} distinct image(s) for Gemini extraction.")
    return product_name, final_downloaded_images, "\n".join(diagnostic_log)


def run_gemini_sync(ean, product_name, market_code, gemini_key, taxonomy_text, image_bytes_list, user_ground_truth):
    """ORIGINAL: Gemini call with the original prompt - untouched from your working version."""
    market_upper = market_code.upper()
    goldmine_sites = GOLDMINE.get(market_upper, "Major Tier-1 Supermarkets")

    prompt = f"""
    You are the Lead Food Product Researcher.
    TARGET EAN: {ean}
    ONLINE PRODUCT NAME FOUND: {product_name}
    USER INPUT (GROUND TRUTH): {user_ground_truth if user_ground_truth else "None provided (Proceed normally)"}
    MARKET: {market_code}

    CORE DIRECTIVES:
    0. VALIDATION GATE (THE BOUNCER): Look at the "USER INPUT (GROUND TRUTH)". If it is empty, proceed normally. If it contains text (like a brand, name, or weight), compare it to the product data you found online for this EAN. They MUST be the same product — but allow for minor formatting differences such as abbreviations, capitalisation, punctuation, and language translation. Only set "is_exact_match" to false if there is a CLEAR, MEANINGFUL mismatch: e.g. user says 'Strawberry 100g' but you found 'Vanilla 50g', or the brand is completely different, or the weight differs by more than 20%. Do NOT fail validation for cosmetic name differences (e.g. 'da 100 gr.' vs '100g', or 'al Cioccolato' vs 'Al Cioccolato Fondente'). If it matches or is a cosmetic difference, set "is_exact_match" to true and extract the data.
    1. ACCURACY: You have access to Google Search. You MUST prioritize official brand websites and major tier-1 retailers.
    2. SOURCE EXCLUSION: AVOID openfoodfacts.org, wikis, or open-source databases. Only use them as an absolute last resort.
    3. TARGET MARKET LANGUAGE: You MUST translate and output ALL product text (Ingredients, Allergens, May Contain, Dietary Info, Nutritional Context) into the native language of the TARGET MARKET ({market_code}). Write it verbatim. EXCEPTION: The 6 taxonomy categories AND the Tags (Dietary, Occasion, Seasonal) MUST remain exactly as they appear in the English lists below to ensure database consistency.
    4. MISSING DATA & SOURCE CASCADE: You MUST try to fill every field. Follow this cascade:
        STEP 1 — Official brand website (highest priority). Extract everything available.
        STEP 2 — If ANY field is still null after Step 1, search the Tier-1 retailers for {market_code}: {goldmine_sites}. Cross-reference and fill any remaining nulls.
        STEP 3 — If ANY field is still null after Step 2, search global databases (barcodelookup.com, go-upc.com, Open Food Facts as last resort).
        STEP 4 — If a field is genuinely not available anywhere after all 3 steps, return "null". Do NOT guess nutritional values. Do NOT invent ingredients.
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
    8. RELIABILITY SCORING: Evaluate the source of your food info (ingredients/nutrition). Score "H" (High) if found on official brand websites or these specific Tier-1 Goldmine retailers for the target market: {goldmine_sites}. Score "M" (Medium) if found on other retailers but consistent across multiple sites. Score "L" (Low) if found on only a single non-tier-1 site. Explain your choice in the reliability_reasoning field.
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
        "chain_of_thought": "Step-by-step reasoning. If validation failed, explain why. If passed, briefly explain how you found the data, translated it, and read the images.",
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
        "sources": ["Array of full URLs (starting with https://) you visited to find this data"]
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
                model='gemini-2.5-flash',
                contents=contents_payload,
                config=types.GenerateContentConfig(
                    temperature=0.25,
                    tools=[{"google_search": {}}],
                    max_output_tokens=8192,
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
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason
                raise Exception(f"Empty text extracted. Reason: {finish_reason}")

            working_urls = []
            try:
                if response.candidates and response.candidates[0].grounding_metadata:
                    metadata = response.candidates[0].grounding_metadata
                    if metadata.grounding_chunks:
                        for chunk in metadata.grounding_chunks:
                            if chunk.web and chunk.web.uri:
                                working_urls.append(chunk.web.uri)
            except Exception:
                pass

            unique_urls = list(dict.fromkeys(working_urls))

            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean_json = raw_text[start_idx:end_idx+1]
            else:
                rogue_preview = raw_text[:200].replace('\n', ' ')
                raise Exception(f"Could not find JSON object. AI wrote: {rogue_preview}...")

            data = json.loads(clean_json, strict=False)

            if unique_urls:
                data["sources"] = unique_urls
            elif isinstance(data.get("sources"), str):
                data["sources"] = [s.strip() for s in data.get("sources").split(",") if s.strip()]
            elif not isinstance(data.get("sources"), list):
                data["sources"] = []

            return data

        except json.JSONDecodeError as e:
            last_error = f"JSON Error: {str(e)}"
        except Exception as e:
            last_error = str(e)

        if attempt < 2:
            time.sleep(3)

    return {"error": f"API Error (Failed after 3 attempts). Last error: {last_error}"}


# ============================================================================
# PATH B: NEW IMAGE DISPLAY PIPELINE (separate from food extraction)
# ============================================================================
# This pipeline ONLY produces the 2 images shown in the results table + diagnostics.
# It does NOT touch food extraction in any way.
# Runs in parallel with Path A so total time is unchanged.

DISPLAY_MIN_LONG_EDGE_PX = 250
DISPLAY_ASPECT_MIN = 0.3
DISPLAY_ASPECT_MAX = 3.0
DISPLAY_PHASH_DUPLICATE_THRESHOLD = 5
MAX_DISPLAY_IMAGES = 2

DISPLAY_BAD_PATH_TOKENS = {
    "logo", "icon", "banner", "placeholder", "spinner", "loading",
    "payment", "paypal", "mastercard", "visa", "social",
    "pinterest", "tiktok", "facebook", "instagram", "twitter", "youtube",
    "thumbnail", "thumb", "avatar", "favicon", "sprite",
}
DISPLAY_BAD_SUBSTRINGS = {
    "openfoodfacts", "amazon-ads", "ad_servlet", "doubleclick",
    "_xs.", "_xxs.", "/icons/", "/logos/",
}

DISPLAY_SOURCE_PRIORITY = {
    "jsonld_retailer": 100,
    "og_retailer": 80,
    "twitter_retailer": 70,
    "serpapi_strict": 60,
    "serpapi_barcode": 55,
    "serpapi_name": 50,
    "serpapi_hailmary": 45,
    "ean_search_registry": 40,
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
    """URL filter for the DISPLAY pipeline only. Returns (is_valid, reject_reason)."""
    if not url or not url.startswith("http"):
        return False, "Not a valid http(s) URL"
    url_lower = url.lower()
    path = url_lower.split("?")[0]

    for ext in BAD_IMAGE_EXTENSIONS:
        if path.endswith(ext):
            return False, f"Bad extension: {ext}"
    for substr in DISPLAY_BAD_SUBSTRINGS:
        if substr in url_lower:
            return False, f"Substring blacklist: '{substr}'"
    tokens = set(re.split(r"[/_\-.]+", path))
    for token in tokens:
        if token in DISPLAY_BAD_PATH_TOKENS:
            return False, f"Path token blacklist: '{token}'"
    if "media-amazon.com" in url_lower and ("," in url_lower or "_bo" in url_lower):
        return False, "Amazon UI composite (commas / _bo modifier)"
    return True, ""


class ImageDiagnostics:
    """Per-EAN diagnostics for the display image pipeline."""

    def __init__(self, ean):
        self.ean = ean
        self.text_log = []
        self.candidates = []
        self.final_selected = []
        self.image_2_failure = ""

    def log(self, msg):
        self.text_log.append(msg)

    def log_candidate(self, source, url, status, reason="", width=None, height=None):
        self.candidates.append({
            "source": source,
            "url": url[:120] + "..." if len(url) > 120 else url,
            "full_url": url,
            "status": status,
            "reason": reason,
            "width": width,
            "height": height,
        })

    def status_counts(self):
        counts = {}
        for c in self.candidates:
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        return counts

    def summary_string(self):
        counts = self.status_counts()
        parts = [f"{k}={v}" for k, v in counts.items()]
        return f"Selected {len(self.final_selected)}/{MAX_DISPLAY_IMAGES} images from {len(self.candidates)} candidates. " + ", ".join(parts)

    def to_dict_list(self):
        return [{
            "Source": c["source"],
            "Status": c["status"],
            "Reason": c["reason"],
            "Width": c["width"],
            "Height": c["height"],
            "URL": c["url"],
        } for c in self.candidates]


async def display_extract_from_page(session, url):
    """Scrape JSON-LD + OG + Twitter images from a retailer page."""
    images = []
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
                    data = json.loads(match.group(1).strip())
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
                        item_type = item.get("@type", "")
                        is_product = ("Product" in item_type) if isinstance(item_type, list) else (item_type == "Product")
                        if is_product:
                            img = item.get("image")
                            if isinstance(img, list):
                                for i in img:
                                    if isinstance(i, str):
                                        images.append(("jsonld_retailer", i))
                                    elif isinstance(i, dict) and i.get("url"):
                                        images.append(("jsonld_retailer", i["url"]))
                            elif isinstance(img, str):
                                images.append(("jsonld_retailer", img))
                            elif isinstance(img, dict) and img.get("url"):
                                images.append(("jsonld_retailer", img["url"]))
                except Exception:
                    continue

            og = re.search(r'<meta[^>]*property=[\'"]og:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]', html, re.I)
            if og:
                images.append(("og_retailer", og.group(1)))
            tw = re.search(r'<meta[^>]*name=[\'"]twitter:image[\'"][^>]*content=[\'"]([^\'"]+)[\'"]', html, re.I)
            if tw:
                images.append(("twitter_retailer", tw.group(1)))
    except Exception:
        pass

    seen = set()
    unique = []
    for src, u in images:
        if u not in seen:
            seen.add(u)
            unique.append((src, u))
    return unique


async def display_fetch_image_bytes(session, url):
    """Display pipeline image fetcher with UA rotation on 403/429."""
    last_status = None
    last_error = None
    for attempt, ua in enumerate(DISPLAY_USER_AGENTS):
        headers = {
            "User-Agent": ua,
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
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
        img = Image.open(BytesIO(data))
        w, h = img.size
        if h == 0:
            return {"width": w, "height": h, "ok": False, "reason": "Zero height"}
        aspect = w / h
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


async def display_evaluate_candidate(session, source, url, diag,
                                      gemini_key="", product_name="", brand=""):
    """Download + inspect a candidate display image."""
    is_valid, reject_reason = _check_display_url(url)
    if not is_valid:
        diag.log_candidate(source, url, "rejected_url", reject_reason)
        return None

    payload = await display_fetch_image_bytes(session, url)
    if not payload or "error" in payload:
        err = payload.get("error", "unknown") if payload else "no response"
        diag.log_candidate(source, url, "rejected_download", err)
        return None

    inspection = display_inspect_image(payload["data"])
    if not inspection["ok"]:
        diag.log_candidate(source, url, "rejected_dimensions", inspection["reason"],
                           width=inspection.get("width"), height=inspection.get("height"))
        return None

    safe_mime = _safe_mime(payload["mime"], payload["data"]) or "image/jpeg"

    # Vision verification — only for untrusted sources (serpapi image results).
    # Trusted sources (brand/retailer product pages) are pre-verified by origin.
    if source not in TRUSTED_SOURCES and gemini_key and product_name:
        is_correct = await asyncio.to_thread(
            verify_image_with_gemini,
            payload["data"], safe_mime, product_name, brand, gemini_key
        )
        if not is_correct:
            diag.log_candidate(source, url, "rejected_vision",
                               f"Gemini: image does not match '{brand} {product_name}'",
                               width=inspection.get("width"), height=inspection.get("height"))
            return None
        diag.log(f"   ✅ Vision verified: {url[:80]}")

    return {
        "url": url, "mime": safe_mime, "data": payload["data"],
        "source": source, "inspection": inspection,
        "phash": display_compute_phash(payload["data"]),
    }


# Sources that come directly from a brand/retailer product page —
# these are trusted without AI verification (they are the OG/JSON-LD image tag
# on the actual product URL, so they must be the right product).
TRUSTED_SOURCES = {"jsonld_retailer", "og_retailer", "twitter_retailer", "ean_search_registry"}


def verify_image_with_gemini(image_data: bytes, mime: str, product_name: str, brand: str, gemini_key: str) -> bool:
    """
    Lightweight Gemini Vision check: does this image show the right product?
    Returns True (keep) or False (reject).
    Runs synchronously — called via asyncio.to_thread to avoid blocking.
    """
    if not gemini_key or not image_data:
        return True  # can't verify — allow through
    try:
        client = genai.Client(api_key=gemini_key)
        label = f"{brand} {product_name}".strip() if brand else product_name
        prompt = (
            f"Look at this image carefully. Does it show retail food product packaging "
            f"for a product called '{label}'? "
            f"Answer with only one word: YES if it is a food/drink product package or "
            f"product photo that plausibly matches this description, "
            f"NO if it is something completely unrelated (hardware, furniture, clothing, "
            f"electronics, blank barcode, human, animal, landscape, etc.). "
            f"One word only."
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",  # cheapest/fastest model for binary check
            contents=[
                prompt,
                types.Part.from_bytes(data=image_data, mime_type=mime)
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=5,
            )
        )
        answer = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if getattr(part, "text", None):
                    answer += part.text
        answer = answer.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return True  # on error, allow through rather than drop everything


def display_select(candidates, diag):
    """Pick top MAX_DISPLAY_IMAGES, dedupe near-identical."""
    if not candidates:
        diag.image_2_failure = "No candidate images survived quality checks."
        return []

    def score(c):
        base = DISPLAY_SOURCE_PRIORITY.get(c["source"], 30)
        long_edge = c["inspection"].get("long_edge") or 0
        return base + min(20, long_edge / 75)

    sorted_cands = sorted(candidates, key=score, reverse=True)

    selected = []
    selected_hashes = []
    for c in sorted_cands:
        if any(display_phash_distance(c["phash"], h) < DISPLAY_PHASH_DUPLICATE_THRESHOLD for h in selected_hashes):
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
            f"Display image #{len(selected)} (score={score(c):.0f}, {c['inspection']['width']}x{c['inspection']['height']})",
            width=c["inspection"]["width"], height=c["inspection"]["height"]
        )
        if len(selected) >= MAX_DISPLAY_IMAGES:
            break

    diag.final_selected = [c["url"] for c in selected]

    if len(selected) < MAX_DISPLAY_IMAGES:
        if len(candidates) > 1:
            diag.image_2_failure = "Multiple candidates found but only 1 unique image (rest were near-duplicates)."
        else:
            diag.image_2_failure = "Only 1 candidate image survived quality checks."

    return selected


async def fetch_display_images(session, ean, serp_key, ean_token, market_code, ground_truth="", gemini_key=""):
    """
    NEW IMAGE PIPELINE - completely separate from food extraction.
    Returns: (display_images, diagnostics).
    Does its own name lookup and retailer URL discovery so it doesn't depend on Path A.
    """
    gl = market_code.lower()
    market_upper = market_code.upper()
    diag = ImageDiagnostics(ean)
    serp_url = "https://serpapi.com/search"

    product_name = None
    retailer_urls = []
    candidate_image_urls = []
    registry_image_url = None

    # Stage 0: Brand-site lookup (mirrors fetch_basic_info logic)
    # If a brand is provided in ground_truth, search for brand.com/{ean} first.
    # Brand pages have highest-quality product images — prioritise them above everything.
    if serp_key and ground_truth:
        brand_tokens = [t for t in ground_truth.split() if t[0].isupper() and len(t) > 2]
        if brand_tokens:
            brand_candidate = brand_tokens[0]
            diag.log(f"🔍 Brand-site image lookup for '{brand_candidate}'...")
            try:
                async with session.get(serp_url, params={
                    "q": f"{brand_candidate} {ean}",
                    "gl": gl, "api_key": serp_key
                }, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        organic = data.get("organic_results", [])
                        for res in organic[:5]:
                            link = res.get("link", "")
                            domain = link.split("/")[2] if link.startswith("http") else ""
                            if brand_candidate.lower() in domain.lower():
                                if not product_name:
                                    product_name = res.get("title", "").split("-")[0].split("|")[0].strip()
                                    diag.log(f"✅ Brand-site name: {product_name}")
                                if link not in retailer_urls:
                                    retailer_urls.insert(0, link)  # brand page first
                                diag.log(f"✅ Brand domain for images: {domain}")
                                break
            except Exception as e:
                diag.log(f"⚠️ Brand-site image lookup failed: {e}")

    # Derive brand label for vision verification
    brand_for_verify = brand_tokens[0] if (ground_truth and [t for t in ground_truth.split() if t[0].isupper() and len(t) > 2]) else ""

    # Stage 1: Name lookup
    if ean_token:
        diag.log("🔍 EAN-Search.org API...")
        ean_url = f"https://api.ean-search.org/api?token={ean_token}&op=barcode-lookup&ean={ean}&format=json"
        try:
            async with session.get(ean_url, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data and len(data) > 0 and "error" not in data[0]:
                        candidate_name = data[0].get("name", "")
                        registry_image_url = data[0].get("image")
                        if is_garbage_name(candidate_name):
                            diag.log(f"⚠️ EAN-Search returned placeholder: '{candidate_name}' - discarded")
                        else:
                            product_name = candidate_name
                            diag.log(f"✅ EAN-Search name: {product_name}")
        except Exception as e:
            diag.log(f"⚠️ EAN-Search failed: {e}")

    if serp_key:
        diag.log("🔍 Goldmine search...")
        goldmine = f"{GOLDMINE.get(market_upper, '')} OR {GLOBAL_SITES}".strip(" OR")
        try:
            async with session.get(serp_url, params={"q": f"{goldmine} {ean}", "gl": gl, "api_key": serp_key}, timeout=15) as resp:
                data = await resp.json()
                organic = data.get("organic_results", [])
                if organic:
                    if not product_name:
                        product_name = organic[0].get("title", "").split("-")[0].split("|")[0].strip()
                        diag.log(f"✅ Goldmine name: {product_name}")
                    retailer_urls.extend([res.get("link") for res in organic[:4] if "link" in res])
        except Exception as e:
            diag.log(f"⚠️ Goldmine failed: {e}")

    if serp_key and (not retailer_urls or not product_name):
        diag.log("🔍 Global GTIN search for retailer URLs...")
        try:
            async with session.get(serp_url, params={"q": str(ean), "gl": gl, "api_key": serp_key}, timeout=15) as resp:
                data = await resp.json()
                organic = data.get("organic_results", [])
                if organic:
                    if not product_name:
                        product_name = organic[0].get("title", "").split("-")[0].split("|")[0].strip()
                        diag.log(f"✅ Global name: {product_name}")
                    new_urls = [res.get("link") for res in organic[:5] if "link" in res]
                    for u in new_urls:
                        if u not in retailer_urls:
                            retailer_urls.append(u)
        except Exception as e:
            diag.log(f"⚠️ Global search failed: {e}")

    # Stage 2: Gather candidate image URLs from all sources
    if retailer_urls:
        diag.log(f"🌐 Scraping JSON-LD/OG/Twitter from {min(len(retailer_urls), 5)} retailer pages...")
        scrape_tasks = [display_extract_from_page(session, url) for url in retailer_urls[:5]]
        per_page_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
        scraped_count = 0
        for result in per_page_results:
            if isinstance(result, Exception):
                continue
            for source_tag, img_url in result:
                candidate_image_urls.append((source_tag, img_url))
                scraped_count += 1
        diag.log(f"   Scraped {scraped_count} URLs from retailer pages.")

    if serp_key:
        diag.log("🖼️ SerpAPI image search by EAN...")
        try:
            r1, r2 = await asyncio.gather(
                session.get(serp_url, params={"q": f'"{ean}"', "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10),
                session.get(serp_url, params={"q": f'site:barcodelookup.com OR site:go-upc.com "{ean}"', "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10),
                return_exceptions=True
            )
            for resp, tag in [(r1, "serpapi_strict"), (r2, "serpapi_barcode")]:
                if not isinstance(resp, Exception) and resp.status == 200:
                    img_data = await resp.json()
                    for item in img_data.get("images_results", [])[:8]:
                        url = item.get("original", "")
                        if url:
                            candidate_image_urls.append((tag, url))
        except Exception as e:
            diag.log(f"⚠️ SerpAPI EAN search failed: {e}")

    if serp_key and product_name and not product_name.startswith("Product with EAN"):
        diag.log("🖼️ SerpAPI image search by name+EAN (anchored)...")
        # Use EAN as anchor + first 5 words of name to avoid off-topic image results.
        # Restrict to known food/product databases to prevent hardware/homeware contamination.
        name_words = " ".join(product_name.split()[:5])
        anchored_query = f'"{ean}" {name_words} site:barcodelookup.com OR site:go-upc.com OR site:open.fda.gov OR "{ean}" {name_words}'
        hailmary_query = f'"{ean}" {name_words} -site:aliexpress.com -site:ebay.com -site:alibaba.com'
        for q_tag, q_str in [("serpapi_strict", anchored_query), ("serpapi_hailmary", hailmary_query)]:
            try:
                async with session.get(serp_url, params={"q": q_str, "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10) as resp:
                    if resp.status == 200:
                        img_data = await resp.json()
                        name_hits = 0
                        for item in img_data.get("images_results", [])[:6]:
                            url = item.get("original", "")
                            if url:
                                candidate_image_urls.append((q_tag, url))
                                name_hits += 1
                        diag.log(f"   [{q_tag}] Found {name_hits} candidates.")
            except Exception as e:
                diag.log(f"⚠️ SerpAPI name search [{q_tag}] failed: {e}")

    if registry_image_url:
        candidate_image_urls.append(("ean_search_registry", registry_image_url))

    # Stage 3: Dedup URLs
    seen_urls = set()
    deduped = []
    for src, u in candidate_image_urls:
        if u not in seen_urls:
            seen_urls.add(u)
            deduped.append((src, u))

    diag.log(f"📊 {len(deduped)} unique candidate URLs. Evaluating...")

    # Stage 4: Evaluate
    _pname_for_verify = product_name or ""
    eval_tasks = [
        display_evaluate_candidate(
            session, src, url, diag,
            gemini_key=gemini_key,
            product_name=_pname_for_verify,
            brand=brand_for_verify
        )
        for src, url in deduped[:16]
    ]
    eval_results = await asyncio.gather(*eval_tasks, return_exceptions=True)
    valid_candidates = [r for r in eval_results if r and not isinstance(r, Exception)]

    diag.log(f"✅ {len(valid_candidates)} candidates passed quality checks.")

    # Stage 5: Hail mary if zero viable
    if len(valid_candidates) == 0 and serp_key:
        diag.log("🆘 HAIL MARY: zero viable - generic search...")
        generic_query = product_name if product_name and not product_name.startswith("Product with EAN") else str(ean)
        try:
            async with session.get(serp_url, params={"q": generic_query, "tbm": "isch", "gl": gl, "api_key": serp_key}, timeout=10) as resp:
                if resp.status == 200:
                    img_data = await resp.json()
                    hail = []
                    for item in img_data.get("images_results", [])[:10]:
                        url = item.get("original", "")
                        if url and url not in seen_urls:
                            hail.append(("serpapi_hailmary", url))
                            seen_urls.add(url)
                    diag.log(f"   Hail mary: {len(hail)} new candidates.")
                    if hail:
                        h_eval = await asyncio.gather(
                            *[display_evaluate_candidate(session, s, u, diag,
                                gemini_key=gemini_key,
                                product_name=_pname_for_verify,
                                brand=brand_for_verify) for s, u in hail],
                            return_exceptions=True
                        )
                        h_valid = [r for r in h_eval if r and not isinstance(r, Exception)]
                        valid_candidates.extend(h_valid)
                        diag.log(f"   Rescued {len(h_valid)} viable images.")
        except Exception as e:
            diag.log(f"⚠️ Hail mary failed: {e}")

    # Stage 6: Select top MAX_DISPLAY_IMAGES
    selected = display_select(valid_candidates, diag)

    return selected, diag


# ============================================================================
# MERGE: orchestrate Path A (food) + Path B (display images) in parallel
# ============================================================================

async def process_ean(sem, session, item, serp_key, gemini_key, ean_token, market, taxonomy_text):
    ean = item["ean"]
    ground_truth = item.get("ground_truth", "")
    force_refresh = item.get("force_refresh", False)  # <-- new flag

    # Cache check
    if not force_refresh:
        cached = cache_get(ean, market)
        if cached:
            cached["Cached"] = "✅ Cached"
            empty_diag = ImageDiagnostics(ean)
            empty_diag.log("✅ Loaded from cache.")
            return {"row": cached, "image_diag": empty_diag, "food_diag": None}


    async with sem:
        # Run BOTH pipelines concurrently - food extraction is unaffected by image pipeline
        food_task = fetch_basic_info(session, ean, serp_key, ean_token, market, user_ground_truth=ground_truth)
        image_task = fetch_display_images(session, ean, serp_key, ean_token, market, ground_truth=ground_truth, gemini_key=gemini_key)
        (name, gemini_images, food_diag), (display_images, image_diag) = await asyncio.gather(food_task, image_task)

        # PATH A continues: send Gemini the food-extraction images (original logic)
        data = await asyncio.to_thread(
            run_gemini_sync, ean, name, market, gemini_key, taxonomy_text, gemini_images, ground_truth
        )

        # PATH B output: display_images for the table
        display_urls = [img["url"] for img in display_images]
        imgs = (display_urls + ["", ""])[:2]

        if "error" in data:
            row = {
                "Image 1": imgs[0],
                "Image 2": imgs[1],
                "GTIN / EAN": ean,
                "User Input": ground_truth,
                "Status": f"{data['error']}",
                "Cached": "🔄 Fresh"
            }
            cache_set(ean, market, row)    # <-- here, inside the if block
            return {
                "row": row,
                "image_diag": image_diag,
                "food_diag": food_diag,
            }

        sources = data.get("sources", [])
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",") if s.strip()]
        srcs = (sources + ["", "", "", "", ""])[:5]

        # Validation Gate
        if data.get("is_exact_match") is False:
            row = {
                "Image 1": imgs[0],
                "Image 2": imgs[1],
                "Image 2 Failure Reason": image_diag.image_2_failure,
                "Status": "Failed Validation",
                "GTIN / EAN": ean,
                "User Input": ground_truth,
                "Product Name": name,
                "Categorization Diagnosis": "Error: EAN does not correspond to the product found.",
                "Info Reliability": "",
                "Reliability Reasoning": data.get("chain_of_thought", ""),
                "Chain of Thought": data.get("chain_of_thought", ""),
                "Category L1": "", "Category L2": "", "Category L3": "", "Category L4": "", "Category L5": "", "Category L6": "",
                "Dietary Tags": "", "Occasion Tags": "", "Seasonal Tags": "", "Tagging Reasoning": "",
                "Brand": "", "UoM": "", "Packaging": "", "Fragile Item": "", "Net Weight (g) / Volume": "", "Gross Weight (g)": "",
                "Organic Product": "", "Net Weight/ Volume (Customer Facing)": "", "Ingredients": "", "Allergens": "", "May Contain": "",
                "Nutritional Info": "", "Manufacturer Name": "", "Manufacturer Address": "", "Place of Origin": "", "Organic Certification ID": "",
                "Energy (kJ)": "", "Fat (g)": "", "Of Which Saturated Fatty Acids (g)": "", "Carbohydrates (g)": "", "Of Which Sugars (g)": "",
                "Protein (g)": "", "Fiber (g)": "", "Salt (g)": "",
                "Source 1": srcs[0], "Source 2": srcs[1], "Source 3": srcs[2], "Source 4": srcs[3], "Source 5": srcs[4],
                "Cached": "🔄 Fresh"
            }
            cache_set(ean, market, row)
            return {"row": row, "image_diag": image_diag, "food_diag": food_diag}

        # Passed Validation - full populated row
        # "No result > wrong result": if reliability is Low AND no sources found,
        # flag explicitly rather than silently passing as Success.
        reliability = data.get("food_info_reliability", "")
        all_sources = [srcs[i] for i in range(5) if srcs[i]]
        if reliability == "L" and not all_sources:
            final_status = "⚠️ Low Confidence — Review Required"
        else:
            final_status = "Success"
        row = {
            "Image 1": imgs[0],
            "Image 2": imgs[1],
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
        cache_set(ean, market, row)
        return {"row": row, "image_diag": image_diag, "food_diag": food_diag}


async def run_main(parsed_inputs, serp_key, gemini_key, ean_token, market, taxonomy_text, progress_bar, status_text):
    sem = asyncio.Semaphore(5)
    total = len(parsed_inputs)
    completed = 0

    async with aiohttp.ClientSession() as session:
        # Wrap each task to update progress as each one completes,
        # then gather with return_exceptions=True to preserve input order.
        async def tracked(item):
            nonlocal completed
            res = await process_ean(sem, session, item, serp_key, gemini_key, ean_token, market, taxonomy_text)
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Processed {completed}/{total} items...")
            return res

        all_results = await asyncio.gather(*[tracked(item) for item in parsed_inputs])

        results     = [r["row"]        for r in all_results]
        diagnostics = [r["image_diag"] for r in all_results]

        return results, diagnostics


# ============================================================================
# UI APP (STREAMLIT)
# ============================================================================

st.title("🔬 Food Data Researcher PRO")

SERP_KEY = os.environ.get("SERPAPI_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
EAN_TOKEN = os.environ.get("EAN_SEARCH_TOKEN", "")

taxonomy_text = load_taxonomy()
init_cache()  # ensures table exists on every startup

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
            ean = match.group(0)
            ground_truth = line.replace(ean, "").strip()
            ground_truth = re.sub(r'\s+', ' ', ground_truth)
            parsed_inputs.append({
                "ean": ean,
                "ground_truth": ground_truth,
                "force_refresh": force_refresh
            })
        else:
            st.warning(f"⚠️ No valid 8-14 digit EAN in line: '{line}' - Skipping.")

    if parsed_inputs:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        with st.spinner(f"Analyzing {len(parsed_inputs)} products concurrently..."):
            all_data, all_diags = asyncio.run(
                run_main(parsed_inputs, SERP_KEY, GEMINI_KEY, EAN_TOKEN, market_code,
                         taxonomy_text, progress_bar, status_text)
            )

            st.session_state["results_df"] = pd.DataFrame(all_data)

if "results_df" in st.session_state:
    df = st.session_state["results_df"].copy()
    df = df.drop(columns=["Image 2 Failure Reason", "Packaging"], errors="ignore")

     # Ensure Cached column exists for all rows
    if "Cached" not in df.columns:
        df["Cached"] = "🔄 Fresh"

        # Add Re-run checkbox column at the front
    df.insert(0, "Re-run?", False)

    # Define preferred column order
    column_order = [
        "Re-run?",
        "Cached",
        "Image 1",
        "Image 2",
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
    remaining = [c for c in df.columns if c not in column_order]
    df = df[existing_ordered + remaining]

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
            "Image 1": st.column_config.ImageColumn(),
            "Image 2": st.column_config.ImageColumn(),
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
                    "ean": row["GTIN / EAN"],
                    "ground_truth": str(row.get("User Input", "") or "").strip(),
                    "force_refresh": True
                }
                for _, row in rerun_rows.iterrows()
            ]
            rerun_progress = st.progress(0.0)
            rerun_status = st.empty()
            with st.spinner(f"Re-running {len(rerun_inputs)} EAN(s)..."):
                rerun_data, _ = asyncio.run(
                    run_main(rerun_inputs, SERP_KEY, GEMINI_KEY, EAN_TOKEN,
                             market_code, taxonomy_text, rerun_progress, rerun_status)
                )
                # Merge rerun results back into the display
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
