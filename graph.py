"""
AssetTrack — LangGraph Orchestrator

Defines the complete Supervisor-Worker pipeline as a compiled LangGraph.

Pipeline Flow
─────────────
    START
      │
      ▼
  ┌─────────────────────┐
  │  Vision Agent        │  (Gatekeeper — identify & classify)
  └──────────┬──────────┘
             │
        ┌────┴────┐
        │ Router  │  confidence < 0.70 ?
        └────┬────┘
         yes │         no
          ┌──┘          └──┐
          ▼                ▼
        END          ┌─────────────────────┐
    (human review)   │  Market Agent        │  (Worker 1 — scrape prices)
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  News Agent          │  (Worker 2 — scrape context)
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │  Supervisor          │  (Executive — synthesize value)
                     └──────────┬──────────┘
                                │
                                ▼
                              END
"""

from langgraph.graph import END, START, StateGraph

from schemas import AssetState

from nodes.vision_agent import identify_asset_node
from nodes.market_agent import market_data_node
from nodes.news_agent import news_data_node
from nodes.supervisor_agent import valuation_supervisor_node


# ──────────────────────────────────────────────
#  CONDITIONAL ROUTER
# ──────────────────────────────────────────────

def route_after_vision(state: AssetState) -> str:
    """
    Router function for the conditional edge after the Vision Agent.

    If the Vision Agent flagged low confidence (needs_human_review=True),
    the graph terminates early so the frontend can prompt the user
    to confirm or correct the identification before proceeding.

    Otherwise, the pipeline continues to the Market Agent.
    """
    if state.needs_human_review:
        return END
    return "market_agent"


# ──────────────────────────────────────────────
#  GRAPH DEFINITION
# ──────────────────────────────────────────────

workflow = StateGraph(AssetState)

# Register all nodes
workflow.add_node("vision_agent", identify_asset_node)
workflow.add_node("market_agent", market_data_node)
workflow.add_node("news_agent", news_data_node)
workflow.add_node("supervisor", valuation_supervisor_node)

# Wire the edges
workflow.add_edge(START, "vision_agent")

workflow.add_conditional_edges(
    "vision_agent",
    route_after_vision,
    {
        "market_agent": "market_agent",
        END: END,
    },
)

workflow.add_edge("market_agent", "news_agent")
workflow.add_edge("news_agent", "supervisor")
workflow.add_edge("supervisor", END)

# ──────────────────────────────────────────────
#  COMPILED APPLICATION
# ──────────────────────────────────────────────

app = workflow.compile()
