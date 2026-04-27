"""
AssetTrack — Verification Agent (The Honesty Filter)

This agent acts as a fraud/data-integrity analyst for manually submitted
social proof comparables.  It reviews user-submitted marketplace data
from Facebook Marketplace, Craigslist, OfferUp, etc., and:

  1. Validates whether the listing is a real, comparable data point
  2. Adjusts the price if there are discrepancies ("listed as $0 but
     description says $500 firm")
  3. Assigns a confidence_weight (0.0–1.0) based on listing quality
  4. Provides clear reasoning for every decision

Architecture Role: VERIFICATION WORKER
--------------------------------------
Sits outside the main LangGraph pipeline.  Called on-demand from the
/api/evaluate-social-proof endpoint when a user submits social proof.

Agent Write Permissions
-----------------------
MAY WRITE  : asset.comparables (append only), agent_logs (append only)
FORBIDDEN  : asset.name, asset.category, asset.condition,
             asset.current_value, asset.news
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from schemas import AssetState, Comparable, ItemCondition

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  STRUCTURED OUTPUT MODEL
# ──────────────────────────────────────────────

class SocialCompValidation(BaseModel):
    """
    Schema the Verification LLM must return.

    Determines whether a manually submitted social comp should be
    included in the asset's valuation, and with what confidence.
    """

    is_valid: bool = Field(
        ...,
        description=(
            "Should this listing be included as a comparable at all? "
            "False if the listing is clearly fake, spam, for a different "
            "item, or provides no usable price signal."
        ),
    )
    adjusted_price: float = Field(
        ...,
        ge=0.0,
        description=(
            "The corrected price in USD.  If the user reported $0 but the "
            "notes say 'asking $500 firm', the adjusted price is 500.0.  "
            "If the reported price looks accurate, use it unchanged."
        ),
    )
    confidence_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How much to trust this comp in valuation calculations.  "
            "1.0 = high quality, detailed listing with clear photos and "
            "verified seller.  0.5 = typical social listing with some "
            "missing detail.  0.1-0.3 = sketchy, vague, or suspicious.  "
            "0.0 = should not influence valuation at all."
        ),
    )
    mapped_condition: str = Field(
        ...,
        description=(
            "The user-selected condition mapped to the closest ItemCondition "
            "enum value.  One of: Sealed, Mint, Graded, Good, Fair, Used, "
            "Needs Repair, For Parts."
        ),
    )
    reasoning: str = Field(
        ...,
        description=(
            "2-4 sentence explanation of the validation decision.  Must "
            "reference specific evidence (price discrepancies, red flags "
            "mentioned in notes, listing quality signals)."
        ),
    )


# ──────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────

VERIFICATION_SYSTEM_PROMPT = """\
You are the Data Integrity Analyst for AssetTrack, a financial ledger \
and portfolio tracker for physical assets.

Your role is to act as a fraud detective.  You receive manually submitted \
marketplace listings from the user (e.g., from Facebook Marketplace, \
Craigslist, OfferUp) and must validate them before they enter the \
valuation pipeline.

═══════════════════════════════════════════════
 1.  YOUR INPUTS
═══════════════════════════════════════════════
You will receive:
  • ASSET CONTEXT: The item's name, category, condition, and current
    estimated value (if available).
  • MANUAL LISTING: A user-submitted comp with marketplace, title,
    reported price, condition, and any notes/red flags.

═══════════════════════════════════════════════
 2.  VALIDATION CHECKS (in order)
═══════════════════════════════════════════════

  A) PRICE INTEGRITY
     • "$0" or "$1" listings: Check the notes — many social sellers
       list at $0 but state their real price in the description.
       If the notes reveal an actual asking price, use that.
     • "Free" listings that mention "taking offers" → invalid signal
       (the seller doesn't know what they want; weight should be very low)
     • Prices wildly different from the existing comparables (if available)
       should receive low confidence, not automatic rejection.

  B) LISTING QUALITY ASSESSMENT
     HIGH confidence (0.7-1.0):
       • Detailed title matching the asset
       • Specific price (not "$0" or "make offer")
       • Notes mention condition details, original packaging, etc.
     MEDIUM confidence (0.4-0.6):
       • Generic title but correct product category
       • Price present but seems unreasonable (too high or too low)
       • No red flags but sparse detail
     LOW confidence (0.1-0.3):
       • Vague listing ("selling stuff, make offer")
       • Price discrepancy between listed price and notes
       • Red flags noted by user (stock photos, new account, etc.)

  C) RELEVANCE CHECK
     • Is this listing for the SAME item type, or something different?
     • If it's clearly a different product → is_valid = false

  D) RED FLAG DETECTION
     • "Stock photos" → lower confidence
     • "New account / no reviews" → lower confidence
     • "Firm on price" → slight confidence boost (seller knows value)
     • "OBO" / "or best offer" → neutral, price may be aspirational

═══════════════════════════════════════════════
 3.  OUTPUT REQUIREMENTS
═══════════════════════════════════════════════
  • is_valid: true/false — should this enter the comparable pool?
  • adjusted_price: The corrected USD price after reading the full context
  • confidence_weight: 0.0-1.0 trust score
  • mapped_condition: Map to the closest ItemCondition enum value
  • reasoning: Clear explanation referencing specific evidence
"""


# ──────────────────────────────────────────────
#  MODEL CONFIGURATION
# ──────────────────────────────────────────────

_verification_model: Optional[BaseChatModel] = None


def configure_verification_model(model: BaseChatModel) -> None:
    """Set the LLM used by the Verification Agent."""
    global _verification_model
    _verification_model = model


def _get_verification_model() -> BaseChatModel:
    """Retrieve the configured model or raise a clear startup error."""
    if _verification_model is None:
        raise RuntimeError(
            "Verification model not configured. Call "
            "configure_verification_model() before using the social proof endpoint."
        )
    return _verification_model


# ──────────────────────────────────────────────
#  CONDITION MAPPING HELPER
# ──────────────────────────────────────────────

_CONDITION_MAP = {v.value.lower(): v for v in ItemCondition}


def _map_condition(raw: str) -> ItemCondition:
    """Map a raw condition string to the ItemCondition enum."""
    normalized = raw.strip().lower()
    if normalized in _CONDITION_MAP:
        return _CONDITION_MAP[normalized]
    # Fuzzy fallback
    for key, val in _CONDITION_MAP.items():
        if key in normalized or normalized in key:
            return val
    return ItemCondition.GOOD


# ──────────────────────────────────────────────
#  CORE FUNCTION
# ──────────────────────────────────────────────

def process_manual_comp(state: AssetState, manual_comp: dict) -> AssetState:
    """
    Process a manually submitted social proof comparable.

    1. Sends the comp to the Verification LLM for fraud/quality analysis
    2. If valid, creates a Comparable with confidence_weight and source_type
    3. Appends to state.asset.comparables
    4. Logs the full reasoning to agent_logs

    Args:
        state: The current asset state from the frontend.
        manual_comp: Dict with keys: marketplace, title, price, condition, notes.

    Returns:
        Updated AssetState with the verified comp (or rejection log).
    """
    model = _get_verification_model()
    structured_model = model.with_structured_output(SocialCompValidation)

    # Build the briefing
    asset = state.asset
    existing_prices = [c.price for c in asset.comparables if c.price > 0]
    avg_existing = (
        sum(existing_prices) / len(existing_prices) if existing_prices else 0
    )

    briefing = (
        f"ASSET CONTEXT:\n"
        f"  Name:          {asset.name}\n"
        f"  Category:      {asset.category.value}\n"
        f"  Condition:     {asset.condition.value}\n"
        f"  Current Value: ${asset.current_value:,.2f}\n"
        f"  Avg Comp Price: ${avg_existing:,.2f} ({len(existing_prices)} existing comps)\n"
        f"\n"
        f"MANUAL LISTING:\n"
        f"  Marketplace:  {manual_comp.get('marketplace', 'Unknown')}\n"
        f"  Title:        {manual_comp.get('title', 'No title')}\n"
        f"  Listed Price: ${manual_comp.get('price', 0):,.2f}\n"
        f"  Condition:    {manual_comp.get('condition', 'Unknown')}\n"
        f"  Notes/Flags:  {manual_comp.get('notes', 'None')}"
    )

    messages = [
        SystemMessage(content=VERIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=briefing),
    ]

    logger.info(
        "Verification Agent: analyzing manual comp '%s' at $%.2f from %s",
        manual_comp.get("title", "?"),
        manual_comp.get("price", 0),
        manual_comp.get("marketplace", "?"),
    )

    validation: SocialCompValidation = structured_model.invoke(messages)

    logger.info(
        "Verification Agent: valid=%s, adjusted_price=$%.2f, "
        "confidence=%.2f, condition=%s",
        validation.is_valid,
        validation.adjusted_price,
        validation.confidence_weight,
        validation.mapped_condition,
    )

    # Build log entries
    log_entries = [
        f"[Verification Agent] Reviewed: '{manual_comp.get('title', '?')}' "
        f"from {manual_comp.get('marketplace', '?')}",
        f"[Verification Agent] Valid: {validation.is_valid} | "
        f"Adjusted Price: ${validation.adjusted_price:,.2f} | "
        f"Confidence: {validation.confidence_weight:.2f}",
        f"[Verification Agent] Reasoning: {validation.reasoning}",
    ]

    if validation.is_valid:
        # Create the verified Comparable
        verified_comp = Comparable(
            event_date=date.today(),
            marketplace=manual_comp.get("marketplace", "Social Media"),
            condition=_map_condition(validation.mapped_condition),
            price=validation.adjusted_price,
            notes=(
                f"[Social Proof] {manual_comp.get('title', '')} | "
                f"Original: ${manual_comp.get('price', 0):,.2f} | "
                f"Weight: {validation.confidence_weight:.2f} | "
                f"{manual_comp.get('notes', '')}"
            ),
            source_type="manual_social",
            confidence_weight=validation.confidence_weight,
        )

        # Append to comparables
        merged_comparables = [*asset.comparables, verified_comp]
        updated_asset = asset.model_copy(update={
            "comparables": merged_comparables,
        })

        log_entries.append(
            f"[Verification Agent] ✓ Comp accepted and appended "
            f"(weight={validation.confidence_weight:.2f})"
        )
    else:
        # Rejected — state unchanged
        updated_asset = asset
        log_entries.append(
            f"[Verification Agent] ✗ Comp REJECTED — not included in valuation"
        )

    return AssetState(
        asset=updated_asset,
        pipeline_stage="verification_complete",
        confidence_score=state.confidence_score,
        needs_human_review=state.needs_human_review,
        agent_logs=[*state.agent_logs, *log_entries],
        errors=list(state.errors),
    )
