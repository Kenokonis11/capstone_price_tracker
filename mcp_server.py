"""
AssetTrack — MCP Tool Server

Exposes the Market and News scraping tools as a Model Context Protocol
(MCP) server so that any MCP-compatible client (Claude Desktop, LangChain
MCP adapter, custom agents) can invoke them over stdio.

Start:
    python mcp_server.py

This satisfies the Gen AI capstone requirement for an MCP server with
real tool calls routed through the protocol.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()  # ensure API keys are available

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("AssetTrack Tools")


# ── Market Tools ────────────────────────────────

@mcp.tool()
def search_ebay(keyword: str) -> list[dict]:
    """Search eBay for active listings matching the keyword.
    Falls back to mock data if EBAY_TOKEN is missing."""
    from nodes.market_agent import search_ebay_completed
    return search_ebay_completed.invoke(keyword)


@mcp.tool()
def search_google_shopping(keyword: str) -> list[dict]:
    """Search Google Shopping via SerpApi for retail/resale prices.
    Falls back to mock data if SERPAPI_KEY is missing."""
    from nodes.market_agent import search_google_shopping as _tool
    return _tool.invoke(keyword)


@mcp.tool()
def search_pricecharting(keyword: str) -> list[dict]:
    """Search PriceCharting.com via Tavily for collectible price data.
    Falls back to mock data if TAVILY_API_KEY is missing."""
    from nodes.market_agent import search_pricecharting_web
    return search_pricecharting_web.invoke(keyword)


@mcp.tool()
def search_kbb(keyword: str) -> list[dict]:
    """Search Kelley Blue Book via Tavily for vehicle valuations.
    Falls back to mock data if TAVILY_API_KEY is missing."""
    from nodes.market_agent import search_kbb_web
    return search_kbb_web.invoke(keyword)


@mcp.tool()
def search_chrono24(keyword: str) -> list[dict]:
    """Search Chrono24 via Tavily for luxury watch listings.
    Falls back to mock data if TAVILY_API_KEY is missing."""
    from nodes.market_agent import search_chrono24_web
    return search_chrono24_web.invoke(keyword)


@mcp.tool()
def web_search(keyword: str) -> list[dict]:
    """General-purpose web search via Tavily for any asset category.
    Falls back to mock data if TAVILY_API_KEY is missing."""
    from nodes.market_agent import targeted_web_search
    return targeted_web_search.invoke(keyword)


# ── News Tools ──────────────────────────────────

@mcp.tool()
def search_news(query: str) -> list[dict]:
    """Search for market news relevant to an asset or category.
    Falls back to mock data if NEWS_API_KEY is missing."""
    from nodes.news_agent import search_market_news
    return search_market_news.invoke(query)


if __name__ == "__main__":
    mcp.run()
