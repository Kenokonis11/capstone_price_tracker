"""
AssetTrack — Supervisor / Executive Pricing Engine (LangGraph Node)

This is the FINAL analytical node in the pipeline.  It runs after the
Market Scraper and News Analyst workers have populated the state with
comparables and news events.  Its sole purpose is to synthesize all
available data into a mathematically sound ``current_value``.

Architecture Role: SUPERVISOR (Executive)
------------------------------------------
The Supervisor sits at the top of the hierarchy.  It does NOT scrape,
does NOT identify — it only *thinks*.  It receives:

  • asset.condition    (from Vision Agent)
  • asset.category     (from Vision Agent)
  • asset.comparables  (from Market Scraper, cleaned by Data Validator)
  • asset.news         (from News Analyst)

And produces:
  • asset.current_value  (the final USD valuation)
  • agent_logs           (appended with full rationale)
  • pipeline_stage       (set to "valuation_complete")

Agent Write Permissions
-----------------------
MAY WRITE  : asset.current_value, agent_logs, pipeline_stage
MAY READ   : entire state (all fields)
FORBIDDEN  : asset.name, asset.category, asset.condition,
             asset.comparables (no mutation), asset.news (no mutation)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from schemas import AssetState, ItemCondition

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  STRUCTURED OUTPUT MODEL
# ──────────────────────────────────────────────

class ValuationDecision(BaseModel):
    """
    Schema the Supervisor LLM is forced to return via
    ``.with_structured_output()``.

    This is the final analytical output of the entire pipeline — the
    executive decision on what this asset is worth right now.
    """

    estimated_value: float = Field(
        ...,
        ge=0.0,
        description=(
            "The Supervisor's best estimate of the asset's current fair "
            "market value in USD.  Must be a concrete number, not a range."
        ),
    )
    valuation_rationale: str = Field(
        ...,
        description=(
            "A detailed, multi-sentence explanation of how the value was "
            "derived.  Must reference specific comparables used, any "
            "adjustments made for condition differences, and how news/"
            "trends influenced the final figure."
        ),
    )
    comps_considered: int = Field(
        ...,
        ge=0,
        description="Total number of comparables that were evaluated.",
    )
    comps_used: int = Field(
        ...,
        ge=0,
        description=(
            "Number of comparables that materially influenced the final "
            "valuation (after discounting outliers or condition mismatches)."
        ),
    )
    trend_direction: str = Field(
        ...,
        description=(
            "One of: 'rising', 'stable', 'declining', or 'insufficient_data'. "
            "Indicates the market trajectory based on comp dates and news."
        ),
    )


# ──────────────────────────────────────────────
#  CONDITION ORDINAL MAP
# ──────────────────────────────────────────────

# Numeric ranking so the prompt can reference relative condition levels.
# Higher = better condition.  "Graded" is special — it sits above Good
# because professional grading inherently adds market premium, but the
# actual grade nuance is captured in search_keywords / comp notes.
CONDITION_RANK = {
    ItemCondition.SEALED: 8,
    ItemCondition.MINT: 7,
    ItemCondition.GRADED: 6,
    ItemCondition.GOOD: 5,
    ItemCondition.FAIR: 4,
    ItemCondition.USED: 3,
    ItemCondition.NEEDS_REPAIR: 2,
    ItemCondition.FOR_PARTS: 1,
}


# ──────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """\
You are the Executive Pricing Engine for AssetTrack, a financial ledger \
and portfolio tracker for physical assets.

Your role is that of a senior analyst at a valuation firm.  You do NOT \
identify items or scrape data — that work has already been done by \
specialist agents.  You receive their outputs and make the FINAL \
executive decision on the asset's current fair market value in USD.

═══════════════════════════════════════════════
 1.  YOUR INPUTS
═══════════════════════════════════════════════
You will receive a structured briefing containing:

  • ASSET IDENTITY: Name, category, and the Vision Agent's condition
    assessment (one of: Sealed, Mint, Graded, Good, Fair, Used,
    Needs Repair, For Parts).

  • MARKET COMPARABLES: An array of recent sales/transactions, each with:
    - event_date, marketplace, condition, price (USD), notes

  • NEWS & TRENDS: An array of relevant market events that may affect
    value (product announcements, recalls, cultural moments, etc.)

═══════════════════════════════════════════════
 2.  VALUATION METHODOLOGY
═══════════════════════════════════════════════
Follow this analytical framework in strict order:

  Step 1 — CONDITION FILTERING
    Compare each comparable's condition to the asset's condition.
    The condition hierarchy (best → worst) is:
      Sealed > Mint > Graded > Good > Fair > Used > Needs Repair > For Parts

    • Same condition: Use the comparable's price at full weight.
    • 1 tier difference: Apply a ±10-15% adjustment.
    • 2+ tiers: Apply a steeper adjustment (±20-40%) or consider
      discounting the comp entirely if the gap is too large.
    • GRADED is special: A "Graded PSA 9" is NOT the same as "Graded PSA 4".
      Check the comp's notes for the specific grade and adjust accordingly.

  Step 2 — OUTLIER REJECTION
    Identify and discount prices that are statistical anomalies:
    • Prices more than 2x or less than 0.5x the median of same-condition comps
    • Listings that are clearly for accessories, empty boxes, or different
      variants (check the notes field)
    • Auction snipes with only 1-2 bidders vs. healthy auction competition

    If you reject a comp, note it in your rationale.

  Step 3 — RECENCY WEIGHTING
    More recent sales are more relevant.  Weight the last 30 days of data
    more heavily than data from 60-90+ days ago.  If the market is moving
    (per news events), recency matters even more.

  Step 4 — NEWS/TREND ADJUSTMENT
    Factor in qualitative signals from the news array:
    • Product recall or controversy → bearish adjustment (-5% to -20%)
    • Cultural moment or viral trend → bullish adjustment (+5% to +20%)
    • New model announced → existing model may depreciate
    • Scarcity event (discontinued, limited run) → premium
    If no news is available, state "no significant trend signals" and
    rely purely on the comparables.

  Step 5 — FINAL SYNTHESIS
    Compute a weighted average of the condition-adjusted, outlier-cleaned
    comparables.  Apply the trend adjustment.  Output a single concrete
    USD value — NOT a range.

═══════════════════════════════════════════════
 3.  OUTPUT REQUIREMENTS
═══════════════════════════════════════════════
  • estimated_value: A single float in USD.  Be precise (e.g., 347.50,
    not "around 350").
  • valuation_rationale: 3-8 sentences explaining your methodology.
    Reference specific comp prices, condition adjustments, and any
    news influence.  This text is shown directly to the end user.
  • comps_considered: Total comps you received.
  • comps_used: How many actually influenced the valuation.
  • trend_direction: One of "rising", "stable", "declining", or
    "insufficient_data".

═══════════════════════════════════════════════
 4.  EDGE CASES
═══════════════════════════════════════════════
  • ZERO COMPARABLES: If no comps are available, set estimated_value to 0.0
    and explain that valuation requires market data.  Set trend_direction
    to "insufficient_data".
  • ALL COMPS REJECTED: Same as zero — output 0.0 with explanation.
  • SINGLE COMP: Use it but note the low confidence in the rationale.
    Consider applying a conservative 10% discount for thin data.
  • CONFLICTING NEWS: If news signals conflict, note the ambiguity and
    lean toward the conservative (lower) valuation.
"""


# ──────────────────────────────────────────────
#  MODEL CONFIGURATION
# ──────────────────────────────────────────────

_supervisor_model: Optional[BaseChatModel] = None


def configure_supervisor_model(model: BaseChatModel) -> None:
    """
    Set the LLM used by the Supervisor node.

    Does not require vision capabilities — text-only models work fine here
    since this node operates purely on structured data, not images.
    """
    global _supervisor_model
    _supervisor_model = model


def _get_supervisor_model() -> BaseChatModel:
    """Retrieve the configured model or raise a clear startup error."""
    if _supervisor_model is None:
        raise RuntimeError(
            "Supervisor model not configured. Call configure_supervisor_model() "
            "with a LangChain chat model before running the pipeline."
        )
    return _supervisor_model


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _format_comparables_briefing(state: AssetState) -> str:
    """
    Format the asset's comparables into a structured text block
    the Supervisor can reason over.
    """
    comps = state.asset.comparables
    if not comps:
        return "MARKET COMPARABLES: None available."

    lines = ["MARKET COMPARABLES:"]
    for i, comp in enumerate(comps, 1):
        line = (
            f"  [{i}] {comp.event_date} | {comp.marketplace} | "
            f"Condition: {comp.condition.value} | "
            f"Price: ${comp.price:,.2f}"
        )
        if comp.notes:
            line += f" | Notes: {comp.notes}"
        lines.append(line)

    return "\n".join(lines)


def _format_news_briefing(state: AssetState) -> str:
    """Format the asset's news events into a structured text block."""
    news = state.asset.news
    if not news:
        return "NEWS & TRENDS: No news events available."

    lines = ["NEWS & TRENDS:"]
    for i, event in enumerate(news, 1):
        source_type = "User Update" if event.is_user_update else event.source
        lines.append(
            f"  [{i}] {event.event_date} | {source_type} | "
            f"{event.description}"
        )

    return "\n".join(lines)


def _build_supervisor_briefing(state: AssetState) -> str:
    """
    Assemble the full analytical briefing for the Supervisor from the
    current pipeline state.
    """
    asset = state.asset

    identity_block = (
        f"ASSET IDENTITY:\n"
        f"  Name:      {asset.name}\n"
        f"  Category:  {asset.category.value}\n"
        f"  Condition: {asset.condition.value} "
        f"(rank {CONDITION_RANK.get(asset.condition, '?')}/8)\n"
        f"  Keywords:  {', '.join(asset.search_keywords) if asset.search_keywords else 'N/A'}"
    )

    comps_block = _format_comparables_briefing(state)
    news_block = _format_news_briefing(state)

    return f"{identity_block}\n\n{comps_block}\n\n{news_block}"


# ──────────────────────────────────────────────
#  LANGGRAPH NODE
# ──────────────────────────────────────────────

def valuation_supervisor_node(state: AssetState) -> AssetState:
    """
    LangGraph node: Executive Pricing Engine (Supervisor).

    Synthesizes all worker outputs (comparables, news, condition) into a
    final USD valuation for the asset.

    State mutations
    ---------------
    - ``asset.current_value``  ← final USD valuation
    - ``pipeline_stage``       ← set to ``"valuation_complete"``
    - ``agent_logs``           ← appended with valuation rationale
    """
    model = _get_supervisor_model()
    structured_model = model.with_structured_output(ValuationDecision)

    # Build the analytical briefing
    briefing = _build_supervisor_briefing(state)

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=briefing),
    ]

    # Invoke the LLM
    logger.info("Supervisor: invoking LLM for valuation synthesis...")
    decision: ValuationDecision = structured_model.invoke(messages)
    logger.info(
        "Supervisor: valued '%s' at $%.2f (%s) — used %d/%d comps",
        state.asset.name,
        decision.estimated_value,
        decision.trend_direction,
        decision.comps_used,
        decision.comps_considered,
    )

    # ── State Mutation ────────────────────────
    # Only touch current_value — never mutate name, category, condition,
    # comparables, or news (forbidden for this node).
    updated_asset = state.asset.model_copy(update={
        "current_value": decision.estimated_value,
    })

    # Build the log entry
    log_entry = (
        f"[Supervisor] Valuation: ${decision.estimated_value:,.2f} "
        f"| Trend: {decision.trend_direction} "
        f"| Comps used: {decision.comps_used}/{decision.comps_considered}\n"
        f"[Supervisor] Rationale: {decision.valuation_rationale}"
    )

    return AssetState(
        asset=updated_asset,
        pipeline_stage="valuation_complete",
        confidence_score=state.confidence_score,
        needs_human_review=state.needs_human_review,
        agent_logs=[*state.agent_logs, log_entry],
        errors=list(state.errors),
    )
