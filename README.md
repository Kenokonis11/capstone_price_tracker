# AssetTrack — AI-Powered Physical Asset Valuation Engine

AssetTrack is a multi-agent valuation platform for physical assets. You describe an item, optionally upload photos, and the system identifies it, scrapes comparable sales across category-specific marketplaces, analyzes market trends, and synthesizes a defensible fair-market valuation — fully automated, end to end.

Built on a **LangGraph supervisor-worker architecture** with a FastAPI backend, SQLite persistence, and a responsive single-page frontend.

---

## What It Does

- **Vision-first identification** — A multimodal AI gatekeeper classifies your item, assigns a condition tier, generates marketplace search keywords, and scores its own confidence. Low-confidence identifications pause the pipeline and ask for human confirmation before spending API calls on garbage data.
- **Dynamic marketplace scraping** — A category-aware tool registry routes to the right marketplaces per asset type: PriceCharting for collectibles, Kelley Blue Book for vehicles, Chrono24 for watches, eBay + Google Shopping for everything else.
- **Market news analysis** — A dedicated news agent ingests recent headlines and extracts trend signals that the supervisor uses to adjust the final valuation.
- **Weighted valuation synthesis** — The supervisor applies outlier rejection, recency weighting, condition matching, and news adjustment to produce a single USD figure with full audit trail.
- **Manual Bridge (Social Proof)** — Users can submit listings they find on Facebook Marketplace, Craigslist, or OfferUp. A standalone verification agent scores the listing for fraud signals and assigns a confidence weight before the supervisor incorporates it.
- **Conversational Q&A** — A local Ollama-backed chat agent answers questions about any asset in the portfolio.
- **MCP Server** — All scraping and news tools are exposed as a Model Context Protocol server for external AI client integrations.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI  (main.py)                            │
│  POST /api/evaluate-asset     → LangGraph pipeline trigger       │
│  POST /api/evaluate-social-proof → Verification + re-valuation  │
│  GET  /api/portfolio          → Load all persisted assets        │
│  PUT  /api/asset/{id}         → Persist edited state             │
│  DELETE /api/asset/{id}       → Remove from portfolio            │
│  POST /api/chat               → Asset-grounded Q&A              │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  LangGraph App  │
                    │   (graph.py)    │
                    └────────┬────────┘
                             │
          ┌──────────────────▼──────────────────┐
          │         VISION AGENT  (Gatekeeper)   │
          │  • Multimodal identification         │
          │  • Category + condition assignment   │
          │  • Search keyword generation         │
          │  • Confidence scoring (0.0–1.0)      │
          └──────────────┬──────────────────────┘
                         │
               ┌─────────▼──────────┐
               │  Confidence Router  │
               │  score < 0.70 ──────┼──→  END  (human review)
               │  score ≥ 0.70 ──────┼──→  continue
               └─────────────────────┘
                         │
          ┌──────────────▼──────────────────────┐
          │       MARKET AGENT  (Worker 1)       │
          │  • Category-mapped tool selection    │
          │  • Multi-marketplace scraping        │
          │  • LLM-based comparable validation   │
          │  • Schema-mapped Comparable objects  │
          └──────────────┬──────────────────────┘
                         │
          ┌──────────────▼──────────────────────┐
          │        NEWS AGENT  (Worker 2)        │
          │  • NewsAPI + Tavily trend search     │
          │  • Relevance filtering               │
          │  • Trend signal extraction           │
          └──────────────┬──────────────────────┘
                         │
          ┌──────────────▼──────────────────────┐
          │       SUPERVISOR  (Executive)        │
          │  • Condition-matched weighting       │
          │  • Outlier rejection (2×/0.5× median)│
          │  • Recency decay                     │
          │  • News-driven ±5–20% adjustment     │
          │  • Confidence-weight provenance      │
          │  • Final USD valuation               │
          └─────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│            VERIFICATION AGENT  (out-of-pipeline)        │
│  Called by POST /api/evaluate-social-proof              │
│  • Fraud / price integrity detection                    │
│  • Listing quality scoring                              │
│  • confidence_weight assignment (0.0–1.0)               │
│  → Supervisor re-runs to recalculate valuation          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐
│      CHAT AGENT  (standalone)   │
│  Local Ollama (llama3)          │
│  Asset-grounded conversational  │
└─────────────────────────────────┘
```

---

## Multi-Phase Search Tool Strategy

The Market Agent is the core data collection engine. It operates in three sequential phases.

### Phase 1 — Tool Selection (Dynamic Binding)

There is no single marketplace that covers all asset types. Before any scraping begins, the agent resolves a category-specific tool list:

```
Category          Tools (in priority order)
─────────────────────────────────────────────────────────────────
COLLECTIBLES      search_pricecharting_web  →  search_ebay_completed  →  search_google_shopping
MEDIA             search_pricecharting_web  →  search_ebay_completed  →  search_google_shopping
VEHICLES          search_kbb_web            →  search_ebay_completed  →  search_google_shopping
JEWELRY           search_chrono24_web       →  search_ebay_completed  →  search_google_shopping
All others        search_ebay_completed  →  search_google_shopping  →  targeted_web_search
```

Each tool is a `@tool`-decorated function bound to the LLM as a LangChain tool call. The agent fans out across `N tools × M keywords` and aggregates all raw results before the analysis phase.

| Tool | Source | Auth |
|---|---|---|
| `search_ebay_completed` | eBay Browse API (active listings) | OAuth 2.0 Client Credentials |
| `search_google_shopping` | SerpApi | API key |
| `search_pricecharting_web` | PriceCharting (via Tavily) | Tavily API key |
| `search_kbb_web` | Kelley Blue Book (via Tavily) | Tavily API key |
| `search_chrono24_web` | Chrono24 (via Tavily) | Tavily API key |
| `targeted_web_search` | General web (via Tavily) | Tavily API key |

Every tool has a **graceful fallback to mock data** if the API key is absent or the request fails — the pipeline never hard-crashes during demos.

### Phase 2 — LLM-Validated Analysis

Raw listings (~30+ entries) are passed to the LLM with a structured-output schema. The model:

- Deduplicates and cleans listing text
- Rejects irrelevant variants, empty boxes, damaged samples, and obvious outliers
- Maps free-text condition strings to the strict `ItemCondition` enum
- Converts non-USD prices
- Returns a `MarketScrapeReport` (validated Pydantic model)

Using `llm.with_structured_output()` eliminates JSON parsing bugs entirely — the LLM must conform to the schema or raise a validation error.

### Phase 3 — Schema Mapping

Each validated listing becomes a `Comparable`:

```python
Comparable(
    event_date:        date,           # YYYY-MM-DD
    marketplace:       str,            # "eBay", "PriceCharting", …
    condition:         ItemCondition,  # strict enum
    price:             float,          # USD
    notes:             str,            # bid count, graded cert, etc.
    source_type:       str,            # "automated" | "manual_social"
    confidence_weight: float,          # 0.0–1.0
)
```

Automated comparables always carry `confidence_weight=1.0`. Manual social proof entries receive a weight assigned by the Verification Agent.

---

## Manual Bridge — Social Proof Validation

Facebook Marketplace and Craigslist often have the most accurate local pricing for niche items, but their data can't be scraped reliably. The Manual Bridge lets users submit listings by hand while the system validates them intelligently.

**Workflow:**

1. User finds a comparable listing on a social marketplace
2. Opens the **Mission Briefing** modal in the asset detail view
3. Enters: marketplace name, listing title, price, condition, and optional notes (e.g., "stock photos", "new account", "no serial number shown")
4. `POST /api/evaluate-social-proof` dispatches to the Verification Agent
5. Verification Agent scores:
   - **Price integrity** — does the stated price match the description?
   - **Listing quality** — specific photos, detailed description, seller reputation signals
   - **Red flags** — stock images, suspiciously vague terms, new account indicators, missing provenance
   - Assigns `confidence_weight` (high = 0.7–1.0, typical = 0.4–0.6, suspicious = 0.1–0.3)
6. A new `Comparable` is created with `source_type="manual_social"` and the assigned weight
7. The Supervisor immediately re-runs over all comparables (old + new) and recalculates the final valuation
8. The frontend displays the updated value alongside the Verification Agent's reasoning

This preserves data integrity: sketchy listings contribute less signal, not zero, and the audit trail shows exactly why.

---

## LangGraph State & Architectural Constraints

State flows immutably through the pipeline as `AssetState`. Each agent has explicit write permissions to prevent workers from overwriting each other's outputs.

```python
AssetState(
    asset:               Asset,       # Full asset record
    pipeline_stage:      str,         # Active node name
    confidence_score:    float,       # Vision Agent output (0.0–1.0)
    needs_human_review:  bool,        # Router flag
    agent_logs:          List[str],   # Append-only audit trail
    errors:              List[str],   # Append-only error log
)
```

| Agent | May Write | Forbidden |
|---|---|---|
| Vision Agent | `name`, `category`, `condition`, `search_keywords`, `confidence_score`, `needs_human_review` | `comparables`, `current_value`, `news` |
| Market Agent | `comparables` (append), `errors` (append), `agent_logs` (append) | `name`, `category`, `condition`, `current_value`, `news` |
| News Agent | `news` (append), `agent_logs` (append) | `name`, `category`, `condition`, `comparables`, `current_value` |
| Supervisor | `current_value`, `agent_logs` (append) | All identification and collection fields |

---

## Data Model

**Asset:**

```python
Asset(
    id:                str,             # UUID
    name:              str,             # AI-generated market name
    description:       str,             # Original user description
    raw_user_category: str,             # Pre-classification hint
    category:          AssetCategory,   # 13-value enum
    status:            ItemStatus,      # VAULTED | MONITORING | TARGET_SET | LISTED | SOLD
    condition:         ItemCondition,   # SEALED | MINT | GRADED | GOOD | FAIR | USED | NEEDS_REPAIR | FOR_PARTS
    images:            List[str],       # Base64 or file paths
    comparables:       List[Comparable],
    news:              List[NewsEvent],
    search_keywords:   List[str],       # 3–5 eBay-style strings
    current_value:     float,           # USD
)
```

**Asset Categories:** `ELECTRONICS`, `COLLECTIBLES`, `VEHICLES`, `TOOLS`, `MEDIA`, `JEWELRY`, `CLOTHING`, `HOUSEHOLD`, `FURNITURE`, `ART`, `INSTRUMENTS`, `SPORTING_EQUIPMENT`, `OTHER`

---

## Project Structure

```
capstone_price_tracker/
├── main.py                   # FastAPI app, endpoints, LLM provider config
├── graph.py                  # LangGraph workflow (nodes, edges, router)
├── schemas.py                # Pydantic models — AssetState, Asset, Comparable, etc.
├── models.py                 # SQLAlchemy ORM (DBAsset)
├── database.py               # SQLite session management
├── mcp_server.py             # MCP stdio server exposing all tools
├── evaluate.py               # RAGAS-compatible evaluation framework
├── index.html                # Single-page frontend (Vanilla JS + Tailwind CSS)
├── requirements.txt
├── .env                      # API keys (not committed)
└── nodes/
    ├── vision_agent.py       # Multimodal gatekeeper
    ├── market_agent.py       # Scraper + comparable extraction
    ├── news_agent.py         # Market context + trend signals
    ├── supervisor_agent.py   # Valuation synthesis
    ├── verification_agent.py # Social proof fraud detection
    └── chat_agent.py         # Ollama-backed Q&A
```

---

## Setup

### Prerequisites

- Python 3.11+
- (Optional) [Ollama](https://ollama.ai) running locally with `llama3` for the chat feature

### Install

```bash
git clone <repo-url>
cd capstone_price_tracker
pip install -r requirements.txt
```

### Configure

Copy `.env.example` to `.env` and fill in your keys:

```bash
# Required — choose one LLM provider
GOOGLE_API_KEY=<your-google-ai-studio-key>
LLM_PROVIDER=google
LLM_MODEL_NAME=gemini-2.0-flash

# Optional — OpenAI instead of Google
# OPENAI_API_KEY=<your-openai-key>
# LLM_PROVIDER=openai
# LLM_MODEL_NAME=gpt-4o

# Market data (any combination works; missing keys fall back to mock data)
EBAY_CLIENT_ID=<ebay-app-client-id>
EBAY_CLIENT_SECRET=<ebay-app-client-secret>
SERPAPI_KEY=<serpapi-key>
TAVILY_API_KEY=<tavily-api-key>
NEWS_API_KEY=<newsapi-org-key>
```

### Run

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) — the frontend is served directly by FastAPI.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/api/portfolio` | Load all assets |
| `POST` | `/api/evaluate-asset` | Run full valuation pipeline |
| `POST` | `/api/evaluate-social-proof` | Validate + incorporate manual comp |
| `PUT` | `/api/asset/{id}` | Save edited asset state |
| `DELETE` | `/api/asset/{id}` | Remove asset |
| `POST` | `/api/chat` | Asset-grounded Q&A |

**Evaluate Asset — Request Body:**
```json
{
  "raw_text": "1999 Pokémon Base Set Charizard Holo",
  "raw_category": "Trading Cards",
  "images": ["data:image/jpeg;base64,..."]
}
```

**Evaluate Social Proof — Request Body:**
```json
{
  "state": { "...full AssetState..." },
  "manual_comp": {
    "marketplace": "Facebook Marketplace",
    "title": "Charizard holo, played condition",
    "price": 280.00,
    "condition": "Good",
    "notes": "seller has photos, been active 3 years"
  }
}
```

---

## Evaluation Framework

`evaluate.py` runs a RAGAS-compatible assessment against all persisted assets:

- **Faithfulness** — LLM-judges whether the agent audit logs match the actual data collected (comparables, news, keywords)
- **Fraud Detection** — LLM-judges whether the Verification Agent correctly identified suspicious vs. legitimate social proof listings
- **Overall Quality** — Human-readable valuation quality assessment per asset

Results are written to `capstone_evaluation_report.txt` with per-metric 1–10 scores and rationale.

---

## MCP Integration

`mcp_server.py` exposes all scraping and news tools as a **Model Context Protocol** server over stdio. This allows Claude Desktop and other MCP-compatible clients to invoke AssetTrack's data pipeline directly from a conversation.

```bash
python mcp_server.py
```

---

## LLM Provider Support

| Provider | Models | Set via env |
|---|---|---|
| Google Gemini | `gemini-2.0-flash`, `gemini-1.5-pro`, etc. | `LLM_PROVIDER=google` |
| OpenAI | `gpt-4o`, `gpt-4-turbo`, `gpt-4`, etc. | `LLM_PROVIDER=openai` |
| Ollama (local) | `llama3` | Chat agent only (auto-detected) |

All agents are initialized through a single `_configure_all_models()` call in `main.py` — switching providers requires only env var changes.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| LangGraph over raw LangChain agents | Deterministic routing, explicit state transitions, built-in audit trail |
| `llm.with_structured_output()` everywhere | Eliminates JSON parsing bugs; agents either return valid schemas or raise loudly |
| Dynamic tool binding by category | No single marketplace is authoritative across all asset classes |
| Confidence gating at Vision Agent | Prevents garbage identifications from cascading through expensive scraping calls |
| Single `state_json` column in SQLite | Schema evolution without migrations; full Pydantic model round-trips perfectly |
| Confidence weighting on comparables | Treats automated API data and user-submitted social proof differently without discarding either |
| Graceful mock data fallback | Pipeline works for demos with zero API keys configured |
| Vanilla JS + Tailwind frontend | No build step, no framework lock-in, instant startup |
