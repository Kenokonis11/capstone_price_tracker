"""
AssetTrack — Vision & Identification Agent (LangGraph Node)

This is the FIRST node in the pipeline.  It receives raw multimodal user
input (images + free-text) and produces a strictly typed, market-ready
asset record.

Agent Write Permissions
-----------------------
MAY WRITE  : asset.name, asset.category, asset.condition,
             asset.search_keywords, confidence_score
FORBIDDEN  : asset.comparables, asset.current_value
"""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from schemas import AssetCategory, AssetState, ItemCondition

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  STRUCTURED OUTPUT MODEL
# ──────────────────────────────────────────────

class VisionExtraction(BaseModel):
    """
    Schema the LLM is forced to return via ``.with_structured_output()``.

    This acts as the "exit contract" of the Vision Agent — every field here
    maps directly to a permitted state mutation on ``AssetState``.
    """

    name: str = Field(
        ...,
        description="The definitive, market-standard name for this asset.",
    )
    category: AssetCategory = Field(
        ...,
        description="Strictly one of the AssetCategory enum values.",
    )
    condition: ItemCondition = Field(
        ...,
        description="Strictly one of the ItemCondition enum values.",
    )
    search_keywords: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description=(
            "3-5 keyword strings ordered most-specific to broadest, "
            "structured like an eBay power-seller search query."
        ),
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this identification (0.0-1.0).",
    )


# ──────────────────────────────────────────────
#  SYSTEM PROMPT
# ──────────────────────────────────────────────

VISION_SYSTEM_PROMPT = """\
You are the Vision & Identification Agent for AssetTrack, a financial ledger \
and portfolio tracker for physical assets.  Your job is to analyze \
user-submitted images and descriptions, then produce a precisely classified \
asset record with market-ready search terms.

═══════════════════════════════════════════════
 1.  CATEGORY TAXONOMY
═══════════════════════════════════════════════
You MUST classify the item into exactly ONE of these categories:

  Electronics        — Consumer electronics, computers, phones, cameras, consoles
  Collectibles       — Trading cards, coins, stamps, figurines, memorabilia
  Vehicles           — Cars, motorcycles, boats, recreational vehicles
  Tools              — Power tools, hand tools, workshop equipment
  Media              — Vinyl records, books, movies, video games (software only)
  Jewelry            — Watches, rings, necklaces, precious metals & stones
  Clothing           — Sneakers, designer apparel, vintage clothing, accessories
  Household          — Kitchen appliances, home goods, general household items
  Furniture          — Tables, chairs, credenzas, desks, mid-century modern
  Art                — Paintings, prints, sculptures, photography
  Instruments        — Guitars, pianos, drums, amplifiers, pro audio equipment
  Sporting Equipment — Golf clubs, bikes, skis, fitness equipment
  Other              — Use ONLY when nothing above fits

If the user provides a category hint, use it to GUIDE your decision but always
map to the closest enum above.  "Pokemon cards" → Collectibles.
"My old Fender" → Instruments.  Never default to Other when a real match exists.

═══════════════════════════════════════════════
 2.  SEARCH KEYWORD ENGINEERING  (CRITICAL)
═══════════════════════════════════════════════
The `search_keywords` list is fed DIRECTLY into marketplace scraping APIs
(eBay, Heritage Auctions, Chrono24, COMC, Reverb, etc.).
Poor keywords → garbage price data → worthless valuations.

Generate 3–5 keyword strings ordered from MOST SPECIFIC → BROADEST.

Rules:
  • String 1 (most specific): Include EVERY identifying detail — year, brand,
    model number, variant, edition, grade/condition, and serial identifiers.
  • Strings 2-3 (mid-range): Progressively drop one detail at a time
    (e.g., remove grade, then remove year).
  • String 4-5 (broadest fallback): Brand + model + key distinguishing feature.
  • NEVER include subjective words ("nice", "rare", "great"), seller language
    ("free shipping", "fast delivery"), or condition descriptors in the BROAD
    fallback strings.

Examples of correct output:
  Trading card →
    ["1999 Pokemon Base Set Charizard Holo 4/102 PSA 8",
     "Pokemon Base Set Charizard Holo 4/102 PSA",
     "Pokemon Base Set Charizard 4/102",
     "Charizard Base Set Holo"]

  Watch →
    ["Rolex Submariner Date 116610LN Black Dial 2018 Box Papers",
     "Rolex Submariner 116610LN 2018",
     "Rolex Submariner 116610LN",
     "Rolex Submariner Date Black"]

  Furniture →
    ["Herman Miller Eames Lounge Chair 670 Rosewood Black Leather 2015",
     "Herman Miller Eames Lounge Chair 670 Rosewood",
     "Eames Lounge Chair 670",
     "Eames Lounge Chair"]

  Guitar →
    ["Fender Stratocaster American Professional II 2022 Olympic White Rosewood",
     "Fender Stratocaster American Professional II Olympic White",
     "Fender Stratocaster American Professional II",
     "Fender Stratocaster USA"]

═══════════════════════════════════════════════
 3.  CONDITION ASSESSMENT  (STRICT ENUM)
═══════════════════════════════════════════════
You MUST classify the item's condition into exactly ONE of these values:

  Sealed       — Factory sealed, unopened, shrink-wrapped, never removed from packaging
  Mint         — Opened but essentially perfect; no visible wear, marks, or defects
  Graded       — Item is in a sealed plastic slab with a NUMERIC GRADE from a
                 professional authority (PSA, BGS, CGC, AFA, etc.).  Any slab = Graded,
                 regardless of the number on it.  Put the specific grade (e.g. "PSA 8")
                 in the search_keywords, not here.
  Good         — Above average; light wear, minor cosmetic imperfections, fully functional
  Fair         — Average condition; moderate wear, visible scratches or patina,
                 all core functions intact
  Used         — Below average; noticeable wear, scuffs, dings; still functional
  Needs Repair — Functional issues or significant cosmetic damage requiring restoration
  For Parts    — Non-functional or incomplete; value is in salvageable components

Mapping guidance:
  • A "PSA 8" trading card in a slab                       → Graded
  • A watch described as "recently serviced, light wear"   → Good
  • A "New in Box" sneaker                                 → Sealed
  • A guitar with "plays great, some fret wear"            → Good
  • A console that "turns on but disc drive broken"        → Needs Repair
  • An antique with "beautiful original patina"            → Fair

If no images are provided, infer from the user's text.  When truly ambiguous,
err toward the LOWER condition — conservative assessments protect the user
from overvaluation.

═══════════════════════════════════════════════
 4.  CONFIDENCE SCORE
═══════════════════════════════════════════════
Rate your identification confidence from 0.0 to 1.0:

  0.90 – 1.00 : Clear images + detailed description, item unambiguously identified
  0.75 – 0.89 : Strong identification, missing minor details (year, sub-variant)
  0.60 – 0.74 : Reasonable guess with significant uncertainty (blurry image, vague text)
  Below 0.60  : Cannot reliably identify — flag for human review

Be CONSERVATIVE.  A false positive (high confidence on a wrong ID) is far
worse than requesting human review.
"""


# ──────────────────────────────────────────────
#  MODEL CONFIGURATION
# ──────────────────────────────────────────────

_vision_model: Optional[BaseChatModel] = None

# Confidence threshold below which the graph pauses for human review
CONFIDENCE_THRESHOLD = 0.70


def configure_vision_model(model: BaseChatModel) -> None:
    """
    Set the vision-capable LLM used by this node.

    Call this once during application startup before running the graph.
    Any LangChain chat model with vision support works
    (ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, etc.).
    """
    global _vision_model
    _vision_model = model


def _get_vision_model() -> BaseChatModel:
    """Retrieve the configured model or raise a clear startup error."""
    if _vision_model is None:
        raise RuntimeError(
            "Vision model not configured. Call configure_vision_model() "
            "with a vision-capable LangChain chat model before running "
            "the pipeline."
        )
    return _vision_model


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def _encode_image_to_data_uri(image_ref: str) -> Optional[str]:
    """
    Convert an image reference to a ``data:`` URI for multimodal messages.

    Handles three input formats:
      1. Already a data URI  → pass through
      2. A local file path   → read + base64-encode
      3. An HTTP(S) URL      → pass through (the model fetches it)
    """
    if image_ref.startswith("data:"):
        return image_ref

    if image_ref.startswith(("http://", "https://")):
        return image_ref

    # Treat as a local file path
    path = Path(image_ref)
    if not path.is_file():
        logger.warning("Image file not found, skipping: %s", image_ref)
        return None

    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    raw_bytes = path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def _build_multimodal_content(state: AssetState) -> list[dict]:
    """
    Assemble the ``content`` list for a ``HumanMessage`` from the current
    asset state — text description first, then every available image.
    """
    content: list[dict] = []

    # --- Textual context ---
    text_parts: list[str] = []

    if state.asset.raw_user_category:
        text_parts.append(
            f"User-provided category: {state.asset.raw_user_category}"
        )

    if state.asset.description:
        text_parts.append(f"User description: {state.asset.description}")

    if state.asset.name:
        text_parts.append(f"User-provided name: {state.asset.name}")

    if not text_parts:
        text_parts.append(
            "No text description provided. Identify this item from the "
            "images alone."
        )

    content.append({"type": "text", "text": "\n\n".join(text_parts)})

    # --- Image payloads ---
    for image_ref in state.asset.images:
        uri = _encode_image_to_data_uri(image_ref)
        if uri is None:
            continue

        if uri.startswith("data:"):
            content.append({
                "type": "image_url",
                "image_url": {"url": uri},
            })
        else:
            # HTTP(S) URL — let the model fetch it directly
            content.append({
                "type": "image_url",
                "image_url": {"url": uri},
            })

    return content


# ──────────────────────────────────────────────
#  LANGGRAPH NODE
# ──────────────────────────────────────────────

def identify_asset_node(state: AssetState) -> AssetState:
    """
    LangGraph node: Vision & Identification Agent.

    Takes raw multimodal user input and produces a cleanly classified asset
    with high-signal market search keywords.

    State mutations
    ---------------
    - ``asset.name``            ← AI-generated market-standard name
    - ``asset.category``        ← Strict ``AssetCategory`` enum
    - ``asset.condition``       ← Marketplace-standard condition string
    - ``asset.search_keywords`` ← 3-5 API-ready keyword strings
    - ``confidence_score``      ← 0.0–1.0 identification confidence
    - ``pipeline_stage``        ← set to ``"vision_complete"``
    - ``needs_human_review``    ← ``True`` if confidence < threshold
    - ``agent_logs``            ← appended with identification summary
    - ``errors``                ← appended if confidence is low
    """
    model = _get_vision_model()
    structured_model = model.with_structured_output(VisionExtraction)

    # Build the multimodal message
    content = _build_multimodal_content(state)

    messages = [
        SystemMessage(content=VISION_SYSTEM_PROMPT),
        HumanMessage(content=content),
    ]

    # Invoke the LLM
    logger.info("Vision Agent: invoking LLM for asset identification...")
    extraction: VisionExtraction = structured_model.invoke(messages)
    logger.info(
        "Vision Agent: identified '%s' (%s) — confidence %.2f",
        extraction.name,
        extraction.category.value,
        extraction.confidence_score,
    )

    # ── State Mutation ────────────────────────
    # Create a shallow copy of the asset with updated fields.
    # We never touch comparables, current_value, or news (forbidden).
    updated_asset = state.asset.model_copy(update={
        "name": extraction.name,
        "category": extraction.category,
        "condition": extraction.condition,
        "search_keywords": extraction.search_keywords,
    })

    # Build the log entry
    log_entry = (
        f"[Vision Agent] Identified as '{extraction.name}' "
        f"| Category: {extraction.category.value} "
        f"| Condition: {extraction.condition.value} "
        f"| Confidence: {extraction.confidence_score:.0%} "
        f"| Keywords: {extraction.search_keywords}"
    )

    # Check confidence and set routing flags
    needs_review = extraction.confidence_score < CONFIDENCE_THRESHOLD
    new_errors = list(state.errors)

    if needs_review:
        low_conf_msg = (
            f"[Vision Agent] Low confidence ({extraction.confidence_score:.0%}) "
            f"identifying '{extraction.name}'. Flagged for human review."
        )
        new_errors.append(low_conf_msg)
        logger.warning(low_conf_msg)

    # Return the fully updated state
    return AssetState(
        asset=updated_asset,
        pipeline_stage="vision_complete",
        confidence_score=extraction.confidence_score,
        needs_human_review=needs_review,
        agent_logs=[*state.agent_logs, log_entry],
        errors=new_errors,
    )
