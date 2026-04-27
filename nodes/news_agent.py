"""
AssetTrack — News & Context Agent (LangGraph Worker Node 2: "The Analyst")

This worker hunts for qualitative market signals: product news, industry
trends, cultural moments, and macro events that could affect an asset's
trajectory.

Architecture Role: WORKER 2
-----------------------------
Receives the identified asset from the Vision Agent.  Searches for
relevant news and trend data, then filters it through an LLM to produce
clean, schema-mapped NewsEvent records.

Agent Write Permissions
-----------------------
MAY WRITE  : asset.news (append only), agent_logs (append only),
             pipeline_stage
FORBIDDEN  : asset.name, asset.category, asset.condition,
             asset.comparables, asset.current_value
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

import requests as http_requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from schemas import AssetState, NewsEvent

logger = logging.getLogger(__name__)

# ── API Key ──
# Read lazily so load_dotenv() in main.py has already run.
def _get_news_api_key() -> str | None: return os.getenv("NEWS_API_KEY")

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
#  TOOL — NEWS SEARCH (Real API + Mock Fallback)
# ══════════════════════════════════════════════

def _mock_news(query: str) -> list[dict]:
    """Fallback mock data when NEWS_API_KEY is missing or request fails."""
    return [
        {
            "headline": f"Market Update: {query.split()[0]} segment sees steady demand in 2025",
            "source": "CollectorBuzz",
            "date": "2025-04-10",
            "summary": (
                f"Industry analysts report stable pricing across the "
                f"{query.split()[0]} market, with a slight uptick in premium-"
                f"condition items.  Auction volumes remain healthy."
            ),
        },
        {
            "headline": "New Authentication Standards Announced for Graded Collectibles",
            "source": "PSA News",
            "date": "2025-04-08",
            "summary": (
                "Professional Sports Authenticator (PSA) has announced updated "
                "grading criteria effective Q3 2025.  Industry experts expect "
                "short-term volatility as the market adjusts to the new scale."
            ),
        },
        {
            "headline": "Celebrity Endorsement Drives Viral Interest in Vintage Items",
            "source": "TrendWatch",
            "date": "2025-04-05",
            "summary": (
                "A recent social media post by a major celebrity has driven a "
                "surge of interest in vintage collectibles, with related search "
                "volume up 340% week-over-week.  Pricing impact remains unclear."
            ),
        },
        {
            "headline": "Supply Chain Update: Electronics Tariffs May Affect Resale Market",
            "source": "Reuters",
            "date": "2025-04-02",
            "summary": (
                "New import tariffs on consumer electronics could indirectly "
                "raise prices in the secondary market as new-product costs "
                "increase.  Analysts predict a 5-10% uplift on used electronics."
            ),
        },
        {
            "headline": "Weekend Sports Recap: Local Team Wins Championship",
            "source": "ESPN",
            "date": "2025-04-12",
            "summary": (
                "The local basketball team clinched their division title in a "
                "thrilling overtime victory.  This is unrelated to asset markets."
            ),
        },
    ]


def _newsapi_search(query: str) -> list[dict]:
    """Hit the NewsAPI /v2/everything endpoint."""
    resp = http_requests.get(
        "https://newsapi.org/v2/everything",
        params={
            "q": query,
            "apiKey": _get_news_api_key(),
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": "10",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for article in data.get("articles", []):
        pub_date = article.get("publishedAt", "")
        try:
            parsed = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            date_str = parsed.strftime("%Y-%m-%d")
        except (ValueError, AttributeError):
            date_str = datetime.now().strftime("%Y-%m-%d")

        results.append({
            "headline": article.get("title", ""),
            "source": (article.get("source") or {}).get("name", "Unknown"),
            "date": date_str,
            "summary": article.get("description", "") or "",
        })
    return results


@tool
def search_market_news(query: str) -> list[dict]:
    """
    Search financial and collector news sources for headlines relevant
    to a specific asset or asset category.

    Uses the NewsAPI /v2/everything endpoint for real news data.
    Falls back to mock headlines if NEWS_API_KEY is missing or the
    request fails.

    Args:
        query: A natural-language search query combining the asset name
               and category (e.g., "Pokemon Base Set Charizard Collectibles market").

    Returns:
        List of raw news dicts with keys: headline, source, date, summary.
    """
    logger.info("[Tool:News] Searching market news for: %s", query)

    if not _get_news_api_key():
        logger.warning("[Tool:News] NEWS_API_KEY not set — returning mock data")
        return _attach_fallback_warning(
            _mock_news(query),
            tool_label="News",
            reason="Missing NEWS_API_KEY",
        )

    try:
        results = _newsapi_search(query)
        logger.info("[Tool:News] Got %d real articles", len(results))
        if results:
            return results
        return _attach_fallback_warning(
            _mock_news(query),
            tool_label="News",
            reason="API returned zero results (simulated fallback)",
        )
    except Exception as exc:
        logger.warning("[Tool:News] NewsAPI failed (%s) — returning mock data", exc)
        return _attach_fallback_warning(
            _mock_news(query),
            tool_label="News",
            reason=f"API exception: {exc}",
        )


# ══════════════════════════════════════════════
#  STRUCTURED OUTPUT MODEL
# ══════════════════════════════════════════════

class NewsExtraction(BaseModel):
    """
    Schema the analysis LLM must return via ``.with_structured_output()``.

    Contains only the filtered, relevant news events mapped to our
    canonical ``NewsEvent`` schema.  Irrelevant or fluff articles must
    be discarded by the LLM before reaching this output.
    """

    extracted_news: List[NewsEvent] = Field(
        ...,
        max_length=3,
        description=(
            "1-3 highly relevant news events mapped to the NewsEvent schema. "
            "Each must have is_user_update=False since these are AI-sourced. "
            "Discard irrelevant or off-topic articles entirely."
        ),
    )
    filtering_rationale: List[str] = Field(
        ...,
        description=(
            "One entry per raw article explaining whether it was kept or "
            "discarded and why.  Essential for audit and RAGAS evaluation."
        ),
    )


# ══════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════

NEWS_SYSTEM_PROMPT = """\
You are the Market Context Analyst for AssetTrack, a financial ledger \
and portfolio tracker for physical assets.

Your role is to review raw news headlines and summaries, then filter \
them down to the 1-3 most relevant signals that could materially \
affect the valuation of a specific asset.

═══════════════════════════════════════════════
 1.  YOUR INPUTS
═══════════════════════════════════════════════
You will receive:
  • ASSET CONTEXT: The item's name and category.
  • RAW NEWS RESULTS: A list of headlines with sources, dates, and summaries
    from various news aggregators.

═══════════════════════════════════════════════
 2.  FILTERING DIRECTIVES  (STRICT)
═══════════════════════════════════════════════

  A) RELEVANCE TEST
     A news item is relevant ONLY if it could plausibly affect the
     buying, selling, or valuation of this specific asset or its
     immediate category.  Examples of relevant signals:
       • Grading authority changes (PSA, BGS, CGC policy updates)
       • Celebrity/influencer-driven demand spikes
       • Tariff or regulation changes affecting the asset's market
       • Manufacturer recalls or discontinuations
       • Major auction results setting new price records
       • Cultural moments (movie releases, anniversaries) tied to the asset

  B) DISCARD CRITERIA
     Reject articles that are:
       • General sports scores, weather, politics with no market connection
       • Vague "market is doing well" fluff with no actionable data
       • About a completely different asset category
       • Duplicate of another kept article

  C) QUANTITY CONTROL
     Keep a MAXIMUM of 3 articles.  If only 1 is truly relevant, return 1.
     If none are relevant, return an empty list.  Never pad with marginal
     articles just to fill the quota.

═══════════════════════════════════════════════
 3.  OUTPUT REQUIREMENTS
═══════════════════════════════════════════════

  extracted_news — each entry must have:
    • event_date: The article's publication date in YYYY-MM-DD format
    • source: The publication or outlet name
    • description: A concise, factual summary of the event and its
      potential impact on asset valuation (2-3 sentences max)
    • is_user_update: ALWAYS set to false (these are AI-sourced)

  filtering_rationale — one entry per raw article:
    ✓ "KEPT: 'PSA announces new grading scale' — directly affects graded \
       collectible valuations"
    ✓ "DISCARDED: 'Local team wins championship' — sports score with no \
       connection to asset market"
"""


# ══════════════════════════════════════════════
#  MODEL CONFIGURATION
# ══════════════════════════════════════════════

_news_model: Optional[BaseChatModel] = None


def configure_news_model(model: BaseChatModel) -> None:
    """
    Set the LLM used by the News Agent.

    Text-only models work fine here — no vision required.
    """
    global _news_model
    _news_model = model


def _get_news_model() -> BaseChatModel:
    """Retrieve the configured model or raise a clear startup error."""
    if _news_model is None:
        raise RuntimeError(
            "News model not configured. Call configure_news_model() "
            "with a LangChain chat model before running the pipeline."
        )
    return _news_model


# ══════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════

def _build_news_query(state: AssetState) -> str:
    """Build a natural-language news search query from the asset context."""
    parts = [state.asset.name, state.asset.category.value, "market"]
    return " ".join(parts)


def _format_news_for_llm(
    raw_news: list[dict],
    state: AssetState,
) -> str:
    """Format the asset context and raw news into an LLM briefing."""
    asset = state.asset

    lines = [
        "ASSET CONTEXT:",
        f"  Name:     {asset.name}",
        f"  Category: {asset.category.value}",
        "",
        f"RAW NEWS RESULTS ({len(raw_news)} articles):",
    ]

    for i, article in enumerate(raw_news, 1):
        lines.append(
            f"  [{i}] {article.get('date', '?')} | "
            f"{article.get('source', '?')} | "
            f"\"{article.get('headline', '?')}\"\n"
            f"      Summary: {article.get('summary', 'N/A')}"
        )

    return "\n".join(lines)


# ══════════════════════════════════════════════
#  LANGGRAPH NODE
# ══════════════════════════════════════════════

def news_data_node(state: AssetState) -> AssetState:
    """
    LangGraph node: News & Context Agent (Worker 2 — "The Analyst").

    Searches for market news relevant to the asset, filters through
    an LLM to extract 1-3 actionable signals, and appends them as
    NewsEvent records.

    State mutations
    ---------------
    - ``asset.news``       ← appended with filtered news events
    - ``pipeline_stage``   ← set to ``"news_complete"``
    - ``agent_logs``       ← appended with filtering rationale
    """
    asset = state.asset

    # 1. Build search query and execute the news tool
    query = _build_news_query(state)
    logger.info("News Agent: searching for '%s'", query)

    raw_news = search_market_news.invoke(query)
    result_count = len(raw_news) if isinstance(raw_news, list) else 0
    logger.info("News Agent: received %d raw articles", result_count)

    fallback_warnings: list[str] = []
    if isinstance(raw_news, list):
        fallback_warnings = sorted({
            a.get("_fallback_warning")
            for a in raw_news
            if isinstance(a, dict) and a.get("_fallback_warning")
        })
        for a in raw_news:
            if isinstance(a, dict):
                a.pop("_fallback_warning", None)

    # 2. Pass raw news to the analysis LLM
    model = _get_news_model()
    structured_model = model.with_structured_output(NewsExtraction)

    briefing = _format_news_for_llm(raw_news, state)

    messages = [
        SystemMessage(content=NEWS_SYSTEM_PROMPT),
        HumanMessage(content=briefing),
    ]

    logger.info("News Agent: invoking LLM for relevance filtering...")
    extraction: NewsExtraction = structured_model.invoke(messages)
    logger.info(
        "News Agent: kept %d/%d articles",
        len(extraction.extracted_news),
        result_count,
    )

    # 3. State Mutation — append only, never overwrite existing news
    merged_news = [*asset.news, *extraction.extracted_news]

    updated_asset = asset.model_copy(update={
        "news": merged_news,
    })

    # Build the log
    log_entries = [
        *fallback_warnings,
        f"[News Agent] Searched: '{query}' — {result_count} raw articles",
        *[f"[News Agent] {r}" for r in extraction.filtering_rationale],
        f"[News Agent] Final: {len(extraction.extracted_news)} relevant "
        f"event(s) appended to asset news",
    ]

    return AssetState(
        asset=updated_asset,
        pipeline_stage="news_complete",
        confidence_score=state.confidence_score,
        needs_human_review=state.needs_human_review,
        agent_logs=[*state.agent_logs, *log_entries],
        errors=list(state.errors),
    )
