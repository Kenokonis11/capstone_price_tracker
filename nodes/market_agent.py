"""
AssetTrack — Market Data Scraper (LangGraph Worker Node 1: "The Quants")

This worker is responsible for hunting hard numbers: completed sales,
auction results, and dealer listings across category-specific marketplaces.

Architecture Role: WORKER 1
-----------------------------
Receives search_keywords and category from the Vision Agent.  Uses a
dynamically-bound tool registry to scrape the right marketplaces for
the asset type.  Outputs cleaned, schema-mapped comparables and a full
audit trail for downstream RAGAS evaluation.

Agent Write Permissions
-----------------------
MAY WRITE  : asset.comparables (append only), errors (append only),
             agent_logs (append only), pipeline_stage
FORBIDDEN  : asset.name, asset.category, asset.condition,
             asset.current_value, asset.news
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import date, datetime
from typing import Callable, List, Optional

import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from schemas import (
    AssetCategory,
    AssetState,
    Comparable,
    ItemCondition,
)

logger = logging.getLogger(__name__)

# ── API Keys ──
# Read lazily so load_dotenv() in main.py has already run.
def get_ebay_access_token() -> str | None:
    """
    Generate an OAuth 2.0 Access Token using eBay Client Credentials.
    """
    client_id = os.getenv("EBAY_CLIENT_ID")
    client_secret = os.getenv("EBAY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        return None

    credentials = f"{client_id}:{client_secret}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()

    try:
        response = requests.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json().get("access_token")
        else:
            logger.warning(
                "[eBay OAuth] Failed to get token: %d %s", 
                response.status_code, 
                response.text
            )
            return None
    except Exception as exc:
        logger.warning("[eBay OAuth] Exception during token generation: %s", exc)
        return None


def _get_serpapi_key() -> str | None: return os.getenv("SERPAPI_KEY")
def _get_tavily_key() -> str | None: return os.getenv("TAVILY_API_KEY")

FALLBACK_WARNING_PREFIX = (
    "WARNING: API failure/Missing Key. Proceeding with simulated fallback data."
)


def _attach_fallback_warning(
    results: list[dict],
    *,
    tool_label: str,
    reason: str,
) -> list[dict]:
    warning = f"{FALLBACK_WARNING_PREFIX} (Tool={tool_label}; Reason={reason})"
    for r in results:
        if isinstance(r, dict):
            r["_fallback_warning"] = warning
    return results


# ══════════════════════════════════════════════
#  STEP 1 — TOOL REGISTRY (Real APIs + Mock Fallbacks)
# ══════════════════════════════════════════════
#
# Each tool attempts a real HTTP request first.  If the API key is
# missing or the request fails, it falls back to mock data so the
# pipeline never hard-crashes.
#
# Raw result schema (per listing):
#   title, price, currency, date_sold, condition_raw, marketplace, url, notes


# ── MOCK FALLBACK DATA ─────────────────────────

def _mock_ebay(keyword: str) -> list[dict]:
    return [
        {"title": f"{keyword} — Listing A", "price": 325.00, "currency": "USD",
         "date_sold": "2025-04-10", "condition_raw": "Used - Good",
         "marketplace": "eBay", "url": "https://www.ebay.com/itm/mock-a",
         "notes": "Free shipping, 23 bids"},
        {"title": f"{keyword} — Listing B", "price": 289.99, "currency": "USD",
         "date_sold": "2025-04-08", "condition_raw": "Used - Acceptable",
         "marketplace": "eBay", "url": "https://www.ebay.com/itm/mock-b",
         "notes": "Buy It Now, minor scratches"},
        {"title": f"{keyword} — Listing C (EMPTY BOX ONLY)", "price": 12.50,
         "currency": "USD", "date_sold": "2025-04-05", "condition_raw": "Used",
         "marketplace": "eBay", "url": "https://www.ebay.com/itm/mock-c",
         "notes": "BOX ONLY — no item included"},
    ]


def _mock_google_shopping(keyword: str) -> list[dict]:
    return [
        {"title": f"{keyword} — Shopping Result", "price": 299.00,
         "currency": "USD", "date_sold": datetime.now().strftime("%Y-%m-%d"),
         "condition_raw": "Varies", "marketplace": "Google Shopping",
         "url": "https://shopping.google.com/mock",
         "notes": "Aggregated shopping result"},
    ]


def _mock_tavily(keyword: str, site: str = "") -> list[dict]:
    label = site or "Web"
    return [
        {"title": f"{keyword} — {label} Result", "price": 0,
         "currency": "USD", "date_sold": datetime.now().strftime("%Y-%m-%d"),
         "condition_raw": "See content", "marketplace": label,
         "url": f"https://{site or 'search.com'}/mock",
         "notes": f"Mock web search result for {keyword}"},
    ]


# ── REAL API HELPERS ───────────────────────────

def _ebay_api_search(keyword: str, token: str) -> list[dict]:
    """
    Hit the eBay Browse API for ACTIVE listings matching the keyword.
    """
    scrape_date = datetime.now().strftime("%Y-%m-%d")
    resp = requests.get(
        "https://api.ebay.com/buy/browse/v1/item_summary/search",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        },
        params={"q": keyword, "limit": "10"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("itemSummaries", []):
        price_info = item.get("price", {})
        buying_options = item.get("buyingOptions", [])
        results.append({
            "title": item.get("title", ""),
            "price": float(price_info.get("value", 0)),
            "currency": price_info.get("currency", "USD"),
            "date_sold": None,
            "scrape_date": scrape_date,
            "condition_raw": item.get("condition", "Unknown"),
            "marketplace": "eBay",
            "url": item.get("itemWebUrl", ""),
            "notes": (
                "ACTIVE ASKING PRICE - NOT SOLD | "
                f"Scraped {scrape_date} | "
                f"Buying: {buying_options}"
            ),
        })
    return results


def _serpapi_search(keyword: str) -> list[dict]:
    """Hit the SerpApi Google Shopping endpoint."""
    resp = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google_shopping",
            "q": keyword,
            "api_key": _get_serpapi_key(),
            "num": "10",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("shopping_results", []):
        results.append({
            "title": item.get("title", ""),
            "price": float(item.get("extracted_price", 0)),
            "currency": "USD",
            "date_sold": datetime.now().strftime("%Y-%m-%d"),
            "condition_raw": "See listing",
            "marketplace": item.get("source", "Google Shopping"),
            "url": item.get("link", ""),
            "notes": item.get("snippet", ""),
        })
    return results


def _tavily_api_search(keyword: str, site: str | None = None) -> list[dict]:
    """Hit the Tavily Search API with optional site targeting."""
    query = f"{keyword} site:{site}" if site else keyword
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": _get_tavily_key(),
            "query": query,
            "search_depth": "advanced",
            "max_results": 8,
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    source_label = site or "Web Search"
    for item in data.get("results", []):
        results.append({
            "title": item.get("title", ""),
            "price": 0,  # Tavily returns text — LLM will extract prices
            "currency": "USD",
            "date_sold": datetime.now().strftime("%Y-%m-%d"),
            "condition_raw": "See content",
            "marketplace": source_label,
            "url": item.get("url", ""),
            "notes": (item.get("content", "") or "")[:500],
        })
    return results


# ── @tool FUNCTIONS WITH GRACEFUL DEGRADATION ──

@tool
def search_ebay_completed(keyword: str) -> list[dict]:
    """
    Search eBay for listings matching the keyword via the Browse API.

    IMPORTANT: The Browse API returns ACTIVE listings (asking prices), not
    completed/sold transactions. Returned items are explicitly labeled
    "ACTIVE ASKING PRICE - NOT SOLD", include ``scrape_date``, and do NOT set
    ``date_sold``.

    Falls back to mock data if eBay OAuth fails or keys are missing.

    Args:
        keyword: A market-ready search string (e.g., "Rolex Submariner 116610LN").
    """
    logger.info("[Tool:eBay] Searching for: %s", keyword)
    
    token = get_ebay_access_token()
    if not token:
        logger.warning("[Tool:eBay] OAuth failed or missing keys — returning mock data")
        return _attach_fallback_warning(
            _mock_ebay(keyword),
            tool_label="eBay",
            reason="OAuth failure or missing EBAY_CLIENT_ID/SECRET",
        )
    
    try:
        results = _ebay_api_search(keyword, token)
        logger.info("[Tool:eBay] Got %d real results", len(results))
        if results:
            return results
        return _attach_fallback_warning(
            _mock_ebay(keyword),
            tool_label="eBay",
            reason="API returned zero results (simulated fallback)",
        )
    except Exception as exc:
        logger.warning("[Tool:eBay] API failed (%s) — returning mock data", exc)
        return _attach_fallback_warning(
            _mock_ebay(keyword),
            tool_label="eBay",
            reason=f"API exception: {exc}",
        )


@tool
def search_google_shopping(keyword: str) -> list[dict]:
    """
    Search Google Shopping via SerpApi for current retail and resale prices.

    Provides broad market pricing across multiple retailers and marketplaces.
    Falls back to mock data if SERPAPI_KEY is missing.

    Args:
        keyword: A market-ready search string.
    """
    logger.info("[Tool:GoogleShopping] Searching for: %s", keyword)
    if not _get_serpapi_key():
        logger.warning("[Tool:GoogleShopping] SERPAPI_KEY not set — returning mock data")
        return _attach_fallback_warning(
            _mock_google_shopping(keyword),
            tool_label="GoogleShopping",
            reason="Missing SERPAPI_KEY",
        )
    try:
        results = _serpapi_search(keyword)
        logger.info("[Tool:GoogleShopping] Got %d real results", len(results))
        if results:
            return results
        return _attach_fallback_warning(
            _mock_google_shopping(keyword),
            tool_label="GoogleShopping",
            reason="API returned zero results (simulated fallback)",
        )
    except Exception as exc:
        logger.warning("[Tool:GoogleShopping] API failed (%s) — mock data", exc)
        return _attach_fallback_warning(
            _mock_google_shopping(keyword),
            tool_label="GoogleShopping",
            reason=f"API exception: {exc}",
        )


@tool
def search_pricecharting_web(keyword: str) -> list[dict]:
    """
    Search PriceCharting.com via Tavily for collectible and game price data.

    PriceCharting aggregates completed sales into historical price databases.
    Essential for trading cards, retro games, and graded collectibles.

    Args:
        keyword: A product name (e.g., "Pokemon Base Set Charizard Holo").
    """
    logger.info("[Tool:PriceCharting] Searching via Tavily for: %s", keyword)
    if not _get_tavily_key():
        logger.warning("[Tool:PriceCharting] TAVILY_API_KEY not set — mock data")
        return _attach_fallback_warning(
            _mock_tavily(keyword, "pricecharting.com"),
            tool_label="PriceCharting",
            reason="Missing TAVILY_API_KEY",
        )
    try:
        results = _tavily_api_search(keyword, "pricecharting.com")
        logger.info("[Tool:PriceCharting] Got %d results", len(results))
        if results:
            return results
        return _attach_fallback_warning(
            _mock_tavily(keyword, "pricecharting.com"),
            tool_label="PriceCharting",
            reason="API returned zero results (simulated fallback)",
        )
    except Exception as exc:
        logger.warning("[Tool:PriceCharting] Tavily failed (%s) — mock data", exc)
        return _attach_fallback_warning(
            _mock_tavily(keyword, "pricecharting.com"),
            tool_label="PriceCharting",
            reason=f"API exception: {exc}",
        )


@tool
def search_kbb_web(keyword: str) -> list[dict]:
    """
    Search Kelley Blue Book via Tavily for vehicle valuations.

    KBB provides trade-in, private-party, and dealer retail price points.

    Args:
        keyword: Vehicle identifier (e.g., "2018 Honda Civic EX Sedan value").
    """
    logger.info("[Tool:KBB] Searching via Tavily for: %s", keyword)
    if not _get_tavily_key():
        logger.warning("[Tool:KBB] TAVILY_API_KEY not set — mock data")
        return _attach_fallback_warning(
            _mock_tavily(keyword, "kbb.com"),
            tool_label="KBB",
            reason="Missing TAVILY_API_KEY",
        )
    try:
        results = _tavily_api_search(keyword, "kbb.com")
        logger.info("[Tool:KBB] Got %d results", len(results))
        if results:
            return results
        return _attach_fallback_warning(
            _mock_tavily(keyword, "kbb.com"),
            tool_label="KBB",
            reason="API returned zero results (simulated fallback)",
        )
    except Exception as exc:
        logger.warning("[Tool:KBB] Tavily failed (%s) — mock data", exc)
        return _attach_fallback_warning(
            _mock_tavily(keyword, "kbb.com"),
            tool_label="KBB",
            reason=f"API exception: {exc}",
        )


@tool
def search_chrono24_web(keyword: str) -> list[dict]:
    """
    Search Chrono24 via Tavily for luxury watch listings and recent sales.

    Chrono24 is the world's largest marketplace for luxury watches.

    Args:
        keyword: Watch reference or model (e.g., "Rolex 116610LN").
    """
    logger.info("[Tool:Chrono24] Searching via Tavily for: %s", keyword)
    if not _get_tavily_key():
        logger.warning("[Tool:Chrono24] TAVILY_API_KEY not set — mock data")
        return _attach_fallback_warning(
            _mock_tavily(keyword, "chrono24.com"),
            tool_label="Chrono24",
            reason="Missing TAVILY_API_KEY",
        )
    try:
        results = _tavily_api_search(keyword, "chrono24.com")
        logger.info("[Tool:Chrono24] Got %d results", len(results))
        if results:
            return results
        return _attach_fallback_warning(
            _mock_tavily(keyword, "chrono24.com"),
            tool_label="Chrono24",
            reason="API returned zero results (simulated fallback)",
        )
    except Exception as exc:
        logger.warning("[Tool:Chrono24] Tavily failed (%s) — mock data", exc)
        return _attach_fallback_warning(
            _mock_tavily(keyword, "chrono24.com"),
            tool_label="Chrono24",
            reason=f"API exception: {exc}",
        )


@tool
def targeted_web_search(keyword: str) -> list[dict]:
    """
    General-purpose web search via Tavily for any asset category.

    Used as a fallback for categories without dedicated marketplace tools.

    Args:
        keyword: A market-ready search string with pricing intent.
    """
    logger.info("[Tool:WebSearch] Searching via Tavily for: %s", keyword)
    if not _get_tavily_key():
        logger.warning("[Tool:WebSearch] TAVILY_API_KEY not set — mock data")
        return _attach_fallback_warning(
            _mock_tavily(keyword),
            tool_label="WebSearch",
            reason="Missing TAVILY_API_KEY",
        )
    try:
        results = _tavily_api_search(keyword)
        logger.info("[Tool:WebSearch] Got %d results", len(results))
        if results:
            return results
        return _attach_fallback_warning(
            _mock_tavily(keyword),
            tool_label="WebSearch",
            reason="API returned zero results (simulated fallback)",
        )
    except Exception as exc:
        logger.warning("[Tool:WebSearch] Tavily failed (%s) — mock data", exc)
        return _attach_fallback_warning(
            _mock_tavily(keyword),
            tool_label="WebSearch",
            reason=f"API exception: {exc}",
        )


# ══════════════════════════════════════════════
#  STEP 2 — DYNAMIC TOOL BINDING
# ══════════════════════════════════════════════

_GLOBAL_TOOLS = [search_ebay_completed, search_google_shopping]

_CATEGORY_TOOL_MAP: dict[AssetCategory, list] = {
    AssetCategory.COLLECTIBLES: [search_pricecharting_web, *_GLOBAL_TOOLS],
    AssetCategory.MEDIA:        [search_pricecharting_web, *_GLOBAL_TOOLS],
    AssetCategory.VEHICLES:     [search_kbb_web, *_GLOBAL_TOOLS],
    AssetCategory.JEWELRY:      [search_chrono24_web, *_GLOBAL_TOOLS],
}


def get_tools_for_category(category: AssetCategory) -> list:
    """
    Return the ordered list of scraping tools relevant to this asset category.

    Specialized Tavily site-targeted tools come first, global tools last.
    Categories without dedicated tools get eBay + Google Shopping + generic web.
    """
    return _CATEGORY_TOOL_MAP.get(
        category,
        [*_GLOBAL_TOOLS, targeted_web_search],
    )


# ══════════════════════════════════════════════
#  STEP 3 — EVAL-READY STRUCTURED OUTPUT
# ══════════════════════════════════════════════

class MarketScrapeReport(BaseModel):
    """
    Structured output the analysis LLM must return.

    Two critical arrays:
    - ``audit_trail``: Full log of what was searched, how many results were
      found, which were kept/rejected and why.  This is the backbone of our
      RAGAS faithfulness evaluation — every claim must be traceable.
    - ``valid_comparables``: Cleaned, deduplicated, USD-normalised sales
      mapped to our ``Comparable`` schema.
    """

    audit_trail: List[str] = Field(
        ...,
        description=(
            "Step-by-step log of the scraping and analysis process. "
            "Each entry should be a single sentence documenting: which tool "
            "was called, how many results it returned, which results were "
            "kept vs rejected, and the reason for each rejection."
        ),
    )
    valid_comparables: List[Comparable] = Field(
        ...,
        description=(
            "Final list of cleaned, validated market comparables mapped to "
            "the Comparable schema.  All prices must be in USD.  Outliers, "
            "empty-box listings, and accessory-only results must be excluded."
        ),
    )


# ══════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════

MARKET_SYSTEM_PROMPT = """\
You are the Market Data Analyst for AssetTrack, a financial ledger and \
portfolio tracker for physical assets.

Your role is strictly analytical: you receive raw marketplace search \
results from multiple scraping tools and must clean, validate, and \
structure them into reliable market comparables.

═══════════════════════════════════════════════
 1.  YOUR INPUTS
═══════════════════════════════════════════════
You will receive:
  • ASSET CONTEXT: The item's name, category, condition (strict enum),
    and the search keywords that were used.
  • RAW SCRAPE RESULTS: Grouped by tool/marketplace, each result containing
    a title, price, currency, date_sold, condition_raw, marketplace, url, notes.

═══════════════════════════════════════════════
 2.  ANALYSIS DIRECTIVES  (STRICT)
═══════════════════════════════════════════════

  A) CONDITION MATCHING
     The user's asset has a specific ItemCondition (Sealed, Mint, Graded,
     Good, Fair, Used, Needs Repair, For Parts).
     • PRIORITISE comps matching the exact condition of the asset.
     • If exact matches are scarce, include comps from adjacent conditions
       (±1 tier) but LOG the discrepancy in the audit_trail.
     • Map the raw condition strings from scraped results to the closest
       ItemCondition enum value when building the Comparable output.

  B) OUTLIER REJECTION (CRITICAL)
     Reject and log any result that is:
     • An empty box, case, or packaging only (no actual item)
     • An accessory, replacement part, or add-on (not the main item)
     • A lot/bundle where the price covers multiple items
     • Priced more than 3x or less than 0.3x the median of other results
       for similar condition (statistical outlier)
     • A different variant, model, or year than the target asset
     • Clearly a scam, fake, or replica listing

     Every rejection MUST appear in the audit_trail with a reason.

  C) CURRENCY NORMALISATION
     ALL prices in valid_comparables must be in USD.
     If a result is in a foreign currency, convert it using these
     approximate rates (update as needed):
       EUR → USD: ×1.08  |  GBP → USD: ×1.27  |  JPY → USD: ×0.0067
       CAD → USD: ×0.74  |  AUD → USD: ×0.65
     Log any currency conversion in the audit_trail.

  D) DATE HANDLING
     Map date_sold strings to YYYY-MM-DD format for the Comparable schema.
     Prefer recent sales (last 90 days).  Older data is acceptable when
     recent data is thin, but note the age in the audit_trail.

     IMPORTANT: If a result is explicitly labeled "ACTIVE ASKING PRICE - NOT SOLD"
     (often with a scrape date), it is NOT a completed transaction and MUST NOT
     be included in valid_comparables. Log the rejection reason.

  E) DEDUPLICATION
     If the same sale appears from multiple tools (e.g., eBay result also
     in PriceCharting), keep only one and log the dedup.

═══════════════════════════════════════════════
 3.  OUTPUT REQUIREMENTS
═══════════════════════════════════════════════

  audit_trail — one entry per significant action:
    ✓ "Searched eBay with keyword 'X' — returned 3 results"
    ✓ "Rejected eBay result 'EMPTY BOX ONLY...' — box/packaging only"
    ✓ "Converted Chrono24 result from EUR 8400 to USD 9072"
    ✓ "Kept eBay result 'Listing A' at $325.00 — condition maps to Good"
    ✓ "Note: 2 of 5 comps are Fair condition vs asset's Mint — logged discrepancy"

  valid_comparables — each entry must have:
    • event_date: YYYY-MM-DD
    • marketplace: source platform name
    • condition: one of the ItemCondition enum values (Sealed, Mint, Graded,
      Good, Fair, Used, Needs Repair, For Parts)
    • price: float in USD
    • notes: any relevant context (bid count, box/papers, grade, etc.)

═══════════════════════════════════════════════
 4.  EDGE CASES
═══════════════════════════════════════════════
  • ZERO VALID RESULTS: If all results are rejected, return an empty
    valid_comparables list and explain in audit_trail.
  • SINGLE RESULT: Keep it but note low data confidence in audit_trail.
  • ALL FOREIGN CURRENCY: Convert everything, note the conversion risk.
"""


# ══════════════════════════════════════════════
#  MODEL CONFIGURATION
# ══════════════════════════════════════════════

_market_model: Optional[BaseChatModel] = None


def configure_market_model(model: BaseChatModel) -> None:
    """
    Set the LLM used by the Market Data Agent.

    Does not require vision capabilities — text-only is fine since this
    node processes structured scrape results, not images.
    """
    global _market_model
    _market_model = model


def _get_market_model() -> BaseChatModel:
    """Retrieve the configured model or raise a clear startup error."""
    if _market_model is None:
        raise RuntimeError(
            "Market model not configured. Call configure_market_model() "
            "with a LangChain chat model before running the pipeline."
        )
    return _market_model


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def _execute_tools(
    tools: list,
    keywords: List[str],
) -> tuple[list[dict], list[str]]:
    """
    Execute every tool against every keyword (deterministic fan-out).

    Returns:
        (all_raw_results, execution_log) — raw result dicts grouped with
        their source tool name, plus a log of what was called.
    """
    all_results: list[dict] = []
    exec_log: list[str] = []

    for t in tools:
        tool_name = t.name
        for kw in keywords:
            try:
                results = t.invoke(kw)
                count = len(results) if isinstance(results, list) else 0
                exec_log.append(
                    f"Called {tool_name} with '{kw}' — {count} result(s)"
                )
                if isinstance(results, list):
                    warning_entries = sorted({
                        r.get("_fallback_warning")
                        for r in results
                        if isinstance(r, dict) and r.get("_fallback_warning")
                    })
                    for w in warning_entries:
                        exec_log.append(w)
                    for r in results:
                        if isinstance(r, dict):
                            r.pop("_fallback_warning", None)

                    for r in results:
                        r["_source_tool"] = tool_name
                    all_results.extend(results)
            except Exception as exc:
                err = f"Tool {tool_name} failed on '{kw}': {exc}"
                exec_log.append(err)
                logger.warning(err)

    return all_results, exec_log


def _format_raw_results_for_llm(
    raw_results: list[dict],
    state: AssetState,
) -> str:
    """
    Format the asset context and raw scrape results into a structured
    text block the analysis LLM can reason over.
    """
    asset = state.asset

    # Asset context
    lines = [
        "ASSET CONTEXT:",
        f"  Name:       {asset.name}",
        f"  Category:   {asset.category.value}",
        f"  Condition:  {asset.condition.value}",
        f"  Keywords:   {asset.search_keywords}",
        "",
        f"RAW SCRAPE RESULTS ({len(raw_results)} total):",
    ]

    for i, r in enumerate(raw_results, 1):
        notes = r.get("notes", "") or ""
        if (not r.get("date_sold")) and ("ACTIVE ASKING PRICE - NOT SOLD" in notes):
            sold_display = f"NOT SOLD (SCRAPED {r.get('scrape_date', '?')})"
        else:
            sold_display = r.get("date_sold", "?")
        lines.append(
            f"  [{i}] {r.get('marketplace', '?')} | "
            f"\"{r.get('title', '?')}\" | "
            f"{r.get('currency', 'USD')} {r.get('price', 0):.2f} | "
            f"Sold: {sold_display} | "
            f"Condition: {r.get('condition_raw', '?')} | "
            f"Notes: {r.get('notes', '-')}"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════
#  STEP 4 — LANGGRAPH NODE
# ══════════════════════════════════════════════

def market_data_node(state: AssetState) -> AssetState:
    """
    LangGraph node: Market Data Scraper (Worker 1 — "The Quants").

    Executes category-specific scraping tools against the asset's search
    keywords, then passes the raw results to an analysis LLM that cleans,
    filters, and structures them into validated Comparables.

    State mutations
    ---------------
    - ``asset.comparables``  ← appended with validated market comps
    - ``pipeline_stage``     ← set to ``"market_complete"``
    - ``agent_logs``         ← appended with audit trail
    - ``errors``             ← appended if tools fail or zero comps found
    """
    asset = state.asset

    # 1. Resolve the tools for this category
    tools = get_tools_for_category(asset.category)
    tool_names = [t.name for t in tools]
    logger.info(
        "Market Agent: category=%s → tools=%s, keywords=%s",
        asset.category.value,
        tool_names,
        asset.search_keywords,
    )

    # 2. Execute all tools against all keywords (deterministic fan-out)
    raw_results, exec_log = _execute_tools(tools, asset.search_keywords)

    logger.info(
        "Market Agent: collected %d raw results from %d tool calls",
        len(raw_results),
        len(exec_log),
    )

    # 3. Feed raw results to the analysis LLM for cleaning/structuring
    model = _get_market_model()
    structured_model = model.with_structured_output(MarketScrapeReport)

    briefing = _format_raw_results_for_llm(raw_results, state)

    messages = [
        SystemMessage(content=MARKET_SYSTEM_PROMPT),
        HumanMessage(content=briefing),
    ]

    logger.info("Market Agent: invoking LLM for analysis and filtering...")
    report: MarketScrapeReport = structured_model.invoke(messages)
    logger.info(
        "Market Agent: %d valid comps from %d raw results",
        len(report.valid_comparables),
        len(raw_results),
    )

    # 4. State Mutation — append only, never overwrite existing comps
    merged_comparables = [*asset.comparables, *report.valid_comparables]

    updated_asset = asset.model_copy(update={
        "comparables": merged_comparables,
    })

    # Build the combined log (execution log + LLM audit trail)
    combined_log = [
        f"[Market Agent] Tool routing: {asset.category.value} → {tool_names}",
        *[entry if entry.startswith("WARNING:") else f"[Market Agent] {entry}" for entry in exec_log],
        f"[Market Agent] ── LLM Analysis ──",
        *[f"[Market Agent] {entry}" for entry in report.audit_trail],
        f"[Market Agent] Final: {len(report.valid_comparables)} valid "
        f"comp(s) from {len(raw_results)} raw result(s)",
    ]

    # Check for zero-comp edge case
    new_errors = list(state.errors)
    if len(report.valid_comparables) == 0:
        zero_msg = (
            "[Market Agent] WARNING: Zero valid comparables found. "
            "Supervisor will not be able to produce a data-backed valuation."
        )
        new_errors.append(zero_msg)
        logger.warning(zero_msg)

    return AssetState(
        asset=updated_asset,
        pipeline_stage="market_complete",
        confidence_score=state.confidence_score,
        needs_human_review=state.needs_human_review,
        agent_logs=[*state.agent_logs, *combined_log],
        errors=new_errors,
    )
