"""
AssetTrack — FastAPI Bridge (main.py)

The API layer connecting the static HTML/JS frontend to the compiled
LangGraph multi-agent pipeline.

Endpoints
---------
  POST /api/evaluate-asset   → Run the full valuation pipeline
  GET  /health               → Server health check

Startup
-------
Configure LLM models for all agents via environment variables:
  GOOGLE_API_KEY   — for Google Gemini models (default provider)
  OPENAI_API_KEY   — for OpenAI GPT models (alternative)
  LLM_PROVIDER     — "google" (default) | "openai"
  LLM_MODEL_NAME   — model identifier (default: "gemini-2.0-flash")
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import List, Optional
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

import models
from database import engine, get_db

from schemas import Asset, AssetState, ItemCondition, RawAssetInput

# Load .env file (GOOGLE_API_KEY, LLM_PROVIDER, etc.)
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("assettrack.api")


def _persist_asset_state(db: Session, state: AssetState) -> AssetState:
    """Upsert a full AssetState document into SQLite."""
    if not state.asset.id:
        state.asset.id = str(uuid4())

    state_dump = state.model_dump_json()
    existing_asset = (
        db.query(models.DBAsset)
        .filter(models.DBAsset.id == state.asset.id)
        .first()
    )

    if existing_asset:
        existing_asset.name = state.asset.name
        existing_asset.current_value = state.asset.current_value
        existing_asset.state_json = state_dump
    else:
        db.add(
            models.DBAsset(
                id=state.asset.id,
                name=state.asset.name,
                current_value=state.asset.current_value,
                state_json=state_dump,
            )
        )

    db.commit()
    return state


def _load_asset_state_or_404(db: Session, asset_id: str) -> AssetState:
    """Load the canonical persisted AssetState for an asset id."""
    db_asset = (
        db.query(models.DBAsset)
        .filter(models.DBAsset.id == asset_id)
        .first()
    )
    if not db_asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    try:
        return AssetState.model_validate_json(db_asset.state_json)
    except ValueError as exc:
        logger.exception("Stored asset state is unreadable for '%s': %s", asset_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Persisted asset state is unreadable",
        ) from exc


# ──────────────────────────────────────────────
#  LLM CONFIGURATION (runs at startup)
# ──────────────────────────────────────────────

def _configure_all_models() -> None:
    """
    Read environment variables and configure every agent's LLM.

    Supported providers:
      - "google"  → ChatGoogleGenerativeAI (requires GOOGLE_API_KEY)
      - "openai"  → ChatOpenAI (requires OPENAI_API_KEY)

    All agents share the same model by default.  To use different models
    per agent, extend this function with agent-specific env vars.
    """
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")

    logger.info("Configuring LLMs: provider=%s, model=%s", provider, model_name)

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not set. Set it in your .env or environment "
                "before starting the API."
            )

        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.2,
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Set it in your .env or environment "
                "before starting the API."
            )

        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.2,
        )

    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'google' or 'openai'."
        )

    # Wire the same model into every agent
    from nodes.vision_agent import configure_vision_model
    from nodes.market_agent import configure_market_model
    from nodes.news_agent import configure_news_model
    from nodes.supervisor_agent import configure_supervisor_model
    from nodes.verification_agent import configure_verification_model

    configure_vision_model(model)
    configure_market_model(model)
    configure_news_model(model)
    configure_supervisor_model(model)
    configure_verification_model(model)

    logger.info("All agent models configured successfully (including Verification Agent).")


# ──────────────────────────────────────────────
#  APP LIFESPAN
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Startup: configure LLMs.  Shutdown: clean up."""
    models.Base.metadata.create_all(bind=engine)
    _configure_all_models()
    yield
    logger.info("AssetTrack API shutting down.")


# ──────────────────────────────────────────────
#  FASTAPI APP
# ──────────────────────────────────────────────

app = FastAPI(
    title="AssetTrack API",
    description=(
        "Multi-agent valuation engine for physical assets. "
        "Accepts raw user input, runs Vision → Market → News → Supervisor "
        "pipeline, and returns a fully enriched AssetState."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — wide open for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  REQUEST / RESPONSE MODELS
# ──────────────────────────────────────────────

class ProcessAssetRequest(RawAssetInput):
    """Payload sent from the frontend to kick off the valuation pipeline."""


class ChatHistoryEntry(BaseModel):
    """Single chat message exchanged in the frontend drawer."""

    role: str = Field(
        ...,
        description="Message role: 'user' or 'assistant'.",
    )
    content: str = Field(
        ...,
        description="Message text content.",
    )


class ChatRequest(BaseModel):
    """Payload for the local SLM chat endpoint."""

    message: str = Field(
        ...,
        description="The latest user message.",
    )
    history: List[ChatHistoryEntry] = Field(
        default_factory=list,
        description="Prior chat turns from the drawer UI.",
    )
    current_state: dict = Field(
        default_factory=dict,
        description="The currently selected asset state from the frontend.",
    )


class ChatResponse(BaseModel):
    """Response returned by the local SLM chat endpoint."""

    response: str = Field(
        ...,
        description="Assistant reply text.",
    )


# The response is just AssetState — FastAPI serialises Pydantic natively.


# ──────────────────────────────────────────────
#  ENDPOINTS
# ──────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Basic health check — confirms the server is up."""
    return {"status": "ok", "service": "AssetTrack API"}


@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    return FileResponse("index.html")


@app.get("/api/portfolio", response_model=List[AssetState])
async def get_portfolio(db: Session = Depends(get_db)):
    """Load the persisted portfolio for frontend bootstrapping."""
    db_assets = db.query(models.DBAsset).all()
    portfolio: List[AssetState] = []

    for db_asset in db_assets:
        try:
            portfolio.append(AssetState.model_validate_json(db_asset.state_json))
        except ValueError as exc:
            logger.warning("Skipping unreadable asset '%s': %s", db_asset.id, exc)

    return portfolio


@app.put("/api/asset/{asset_id}", response_model=AssetState)
async def update_asset(
    asset_id: str,
    state: AssetState,
    db: Session = Depends(get_db),
):
    """Persist a manually edited asset state."""
    existing_asset = (
        db.query(models.DBAsset)
        .filter(models.DBAsset.id == asset_id)
        .first()
    )
    if not existing_asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    state.asset.id = asset_id
    return _persist_asset_state(db, state)


@app.delete("/api/asset/{asset_id}")
async def delete_asset(asset_id: str, db: Session = Depends(get_db)):
    """Delete an asset from the persisted portfolio."""
    existing_asset = (
        db.query(models.DBAsset)
        .filter(models.DBAsset.id == asset_id)
        .first()
    )
    if not existing_asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    db.delete(existing_asset)
    db.commit()
    return {"status": "ok", "message": f"Deleted asset {asset_id}"}


@app.post("/api/evaluate-asset", response_model=AssetState)
async def evaluate_asset(
    request: ProcessAssetRequest,
    db: Session = Depends(get_db),
):
    """
    Run the full multi-agent valuation pipeline.

    1. Builds an initial ``AssetState`` from the raw frontend input.
    2. Invokes the compiled LangGraph (Vision → Market → News → Supervisor).
    3. Returns the enriched ``AssetState`` with valuations, comps, and news.

    If the Vision Agent flags low confidence, the pipeline halts early
    and returns the partial state with ``needs_human_review=True`` so the
    frontend can prompt the user for clarification.
    """
    # Lazy import to avoid circular imports at module level
    from graph import app as graph_app

    # ── Build the initial state ──────────────
    initial_asset = Asset(
        name=request.raw_text or "Unidentified Asset",
        description=request.raw_text,
        raw_user_category=request.raw_category,
        images=request.images,
        # Placeholder — Vision Agent will overwrite both of these
        condition=ItemCondition.GOOD,
    )

    initial_state = AssetState(
        asset=initial_asset,
        pipeline_stage="ingestion",
    )

    logger.info(
        "Pipeline starting: raw_text='%s', images=%d, raw_category='%s'",
        request.raw_text,
        len(request.images),
        request.raw_category,
    )

    # ── Execute the graph ────────────────────
    try:
        final_state = graph_app.invoke(initial_state)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {exc}",
        ) from exc

    # LangGraph may return a dict or AssetState depending on version
    if isinstance(final_state, dict):
        final_state = AssetState(**final_state)

    _persist_asset_state(db, final_state)

    logger.info(
        "Pipeline complete: stage=%s, value=$%.2f, human_review=%s",
        final_state.pipeline_stage,
        final_state.asset.current_value,
        final_state.needs_human_review,
    )

    return final_state


class SocialProofRequest(BaseModel):
    """Payload from the Mission Briefing modal with manual comp data."""

    state: AssetState = Field(
        ...,
        description="The current asset state from the frontend.",
    )
    manual_comp: dict = Field(
        ...,
        description=(
            "Manual comparable data from the user, containing: "
            "marketplace, title, price, condition, notes."
        ),
    )


@app.post("/api/evaluate-social-proof", response_model=AssetState)
async def evaluate_social_proof(
    request: SocialProofRequest,
    db: Session = Depends(get_db),
):
    """
    Process manual social proof through the Verification Agent,
    then re-run the Supervisor to recalculate the asset's value.

    Mini-pipeline: Verify → Recalculate
    """
    from nodes.verification_agent import process_manual_comp
    from nodes.supervisor_agent import valuation_supervisor_node

    logger.info(
        "Social Proof received: marketplace=%s, title='%s', price=$%.2f, condition=%s",
        request.manual_comp.get("marketplace", "?"),
        request.manual_comp.get("title", "?"),
        request.manual_comp.get("price", 0),
        request.manual_comp.get("condition", "?"),
    )
    if request.manual_comp.get("notes"):
        logger.info("  Notes/Red Flags: %s", request.manual_comp["notes"])

    try:
        asset_id = request.state.asset.id
        if not asset_id:
            raise HTTPException(status_code=400, detail="Asset id is required")

        canonical_state = _load_asset_state_or_404(db, asset_id)

        # Step 1: Run through the Honesty Filter
        verified_state = process_manual_comp(canonical_state, request.manual_comp)

        # Step 2: Re-run the Supervisor to recalculate value with the new comp
        final_state = valuation_supervisor_node(verified_state)

        logger.info(
            "Social Proof pipeline complete: value=$%.2f, stage=%s",
            final_state.asset.current_value,
            final_state.pipeline_stage,
        )

        _persist_asset_state(db, final_state)
        return final_state

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Social proof pipeline failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Social proof processing failed: {exc}",
        ) from exc


# ──────────────────────────────────────────────
#  ENTRYPOINT
# ──────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_asset_assistant(request: ChatRequest):
    """
    Query the local SLM companion for an asset-grounded response.
    """
    from nodes.chat_agent import generate_chat_response

    logger.info(
        "Chat request received: history=%d, has_current_state=%s",
        len(request.history),
        bool(request.current_state),
    )

    try:
        response_text = generate_chat_response(
            user_message=request.message,
            chat_history=[entry.model_dump() for entry in request.history],
            current_state=request.current_state,
        )
    except Exception as exc:
        logger.exception("Chat endpoint failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Chat request failed: {exc}",
        ) from exc

    return ChatResponse(response=response_text)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
