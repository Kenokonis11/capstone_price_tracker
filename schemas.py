"""
AssetTrack — Global State Management Blueprint

This module defines the single source of truth for all data flowing through
the LangGraph Supervisor-Worker pipeline.  Every agent reads from and writes
to these models according to the strict Agent Write Permissions below.

Architecture (Supervisor-Worker Hierarchy)
------------------------------------------
Gatekeeper  : Vision / Identification Agent   → classify & generate keywords
Worker 1    : Market Scraper (The Quants)      → hard numbers, sales history
Worker 2    : News / Trends Analyst            → qualitative signals
Worker 3    : Data Validator (Outlier Rejector) → clean scraped data
Supervisor  : Executive Pricing Engine         → synthesize final valuation
SLM         : Chat / Update Agent              → user interface & ledger edits

Agent Write Permissions (Immutable Architecture Rules)
------------------------------------------------------
Vision / Identification Agent
    READ  : raw inputs (images, user text)
    WRITE : asset.name, asset.category (mapping raw input → Enum),
            asset.condition, asset.search_keywords, confidence_score
    FORBIDDEN : comparables, current_value

Market Agent
    WRITE : asset.comparables (append only), errors (append only)

Speculation Agent
    WRITE : asset.news (append only)

Data Validator
    WRITE : asset.comparables (filter / remove outliers), agent_logs

Supervisor (Executive Pricing Engine)
    WRITE : asset.current_value (final valuation), agent_logs
    READ  : all state (comparables, news, condition, category)
"""

from datetime import date
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
#  TAXONOMIES
# ──────────────────────────────────────────────

class ItemStatus(str, Enum):
    """Allowed lifecycle states for any tracked asset."""

    VAULTED = "Vaulted"
    MONITORING = "Monitoring"
    TARGET_SET = "Target Set"
    LISTED = "Listed"
    SOLD = "Sold"


class AssetCategory(str, Enum):
    """
    Strict category buckets the Vision Agent maps user input into.

    The user's free-text category is preserved in ``Asset.raw_user_category``
    before the AI routes it to one of these values.
    """

    ELECTRONICS = "Electronics"
    COLLECTIBLES = "Collectibles"
    VEHICLES = "Vehicles"
    TOOLS = "Tools"
    MEDIA = "Media"
    JEWELRY = "Jewelry"
    CLOTHING = "Clothing"
    HOUSEHOLD = "Household"
    FURNITURE = "Furniture"
    ART = "Art"
    INSTRUMENTS = "Instruments"
    SPORTING_EQUIPMENT = "Sporting Equipment"
    OTHER = "Other"


class ItemCondition(str, Enum):
    """
    Strict condition taxonomy used across Asset records and Comparables.

    The Vision Agent maps raw user descriptions and visual evidence to one
    of these values.  The Supervisor uses these to weight comparables
    against the asset's own condition when computing valuations.
    """

    SEALED = "Sealed"
    MINT = "Mint"
    GRADED = "Graded"
    GOOD = "Good"
    FAIR = "Fair"
    USED = "Used"
    NEEDS_REPAIR = "Needs Repair"
    FOR_PARTS = "For Parts"


# ──────────────────────────────────────────────
#  HISTORICAL LEDGER EVENTS
# ──────────────────────────────────────────────

class Comparable(BaseModel):
    """A single market-comparable transaction logged by the Market Agent or user."""

    event_date: date = Field(
        ...,
        description="Date of the historical sale or purchase. YYYY-MM-DD",
    )
    marketplace: str = Field(
        ...,
        description="Where the transaction occurred (e.g., eBay, Goldin, User Input)",
    )
    condition: ItemCondition = Field(
        ...,
        description="Condition of the item at the time of sale (strict enum).",
    )
    price: float = Field(
        ...,
        description="The transaction value STRICTLY in base USD.",
    )
    notes: Optional[str] = Field(
        None,
        description="Contextual notes regarding this specific comp",
    )


class NewsEvent(BaseModel):
    """A news or update event appended by the Speculation Agent or user."""

    event_date: date = Field(
        ...,
        description="Date of the news or update. YYYY-MM-DD",
    )
    source: str = Field(
        ...,
        description="The publisher or origin of the event",
    )
    description: str = Field(
        ...,
        description="Summary of the event or user update",
    )
    is_user_update: bool = Field(
        default=False,
        description="True if a human updated the asset context; False if AI scraped.",
    )


# ──────────────────────────────────────────────
#  BASE ASSET SCHEMA
# ──────────────────────────────────────────────

class Asset(BaseModel):
    """
    The canonical record for a single tracked physical asset.

    All pricing is stored in base USD.  Currency conversion is a
    presentation concern handled exclusively by the frontend.
    """

    id: Optional[str] = Field(
        None,
        description="Unique identifier (assigned by DB)",
    )
    name: str = Field(
        ...,
        description="The common identifying name of the asset",
    )
    description: Optional[str] = Field(
        None,
        description="Detailed physical description",
    )

    # Classification (User Context → AI Strict Enum)
    raw_user_category: Optional[str] = Field(
        None,
        description="The free-text category the user typed before AI classification.",
    )
    category: AssetCategory = Field(default=AssetCategory.OTHER)

    status: ItemStatus = Field(default=ItemStatus.VAULTED)
    condition: ItemCondition = Field(
        ...,
        description="Current condition of the asset (strict enum).",
    )

    # Multimodal inputs
    images: List[str] = Field(
        default_factory=list,
        description="List of Base64 strings or file paths",
    )

    # The historical arrays
    comparables: List[Comparable] = Field(default_factory=list)
    news: List[NewsEvent] = Field(default_factory=list)

    # Engine-specific identifiers
    search_keywords: List[str] = Field(
        default_factory=list,
        description="Targeted keywords generated by Vision Agent for Market scraping",
    )
    current_value: float = Field(
        0.0,
        description="The most recent USD value, updated strictly by the engine",
    )


# ──────────────────────────────────────────────
#  FRONTEND → BACKEND INTAKE CONTRACT
# ──────────────────────────────────────────────

class RawAssetInput(BaseModel):
    """
    Payload received from the Add Asset frontend modal.

    This is the unprocessed user submission that the LangGraph ingestion
    pipeline receives as its initial input before any AI enrichment.
    """

    raw_name: str = Field(
        ...,
        description="User-provided title",
    )
    raw_description: Optional[str] = Field(
        None,
        description="User's detailed description",
    )
    raw_category: Optional[str] = Field(
        None,
        description="Free-text category the user typed (preserved in Asset.raw_user_category)",
    )
    acquisition_source: Optional[str] = Field(
        None,
        description="Where it was acquired",
    )
    purchase_price: Optional[float] = Field(
        None,
        description="Original cost basis in USD",
    )
    owner_condition: Optional[str] = Field(
        None,
        description="User's own description of item condition",
    )
    intended_status: ItemStatus = Field(
        default=ItemStatus.VAULTED,
        description="User's pipeline status selection",
    )
    image_paths: List[str] = Field(
        default_factory=list,
        description="Paths to uploaded multimodal images",
    )


# ──────────────────────────────────────────────
#  LANGGRAPH STATE ENVELOPE
# ──────────────────────────────────────────────

class AssetState(BaseModel):
    """
    The top-level state object that flows through every LangGraph node.

    Each agent reads and writes to specific fields according to the
    Agent Write Permissions documented at the top of this module.
    """

    # The core asset being processed
    asset: Asset

    # Graph routing & logic
    pipeline_stage: str = Field(
        "ingestion",
        description="Current node in the LangGraph (e.g., vision_agent, market_agent)",
    )

    # Fallback and human-in-the-loop triggers
    confidence_score: float = Field(
        1.0,
        description="Overall AI confidence (0.0 to 1.0). Below threshold triggers human review.",
    )
    needs_human_review: bool = Field(
        False,
        description="Router flag. If True, graph pauses and asks frontend for input.",
    )

    # Observability and debugging
    agent_logs: List[str] = Field(
        default_factory=list,
        description="Internal system logs for the frontend History tab",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Any scraping or API failures encountered during this run",
    )
