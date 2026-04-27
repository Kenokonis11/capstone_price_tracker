"""
AssetTrack - Local SLM Chat Companion

Provides a lightweight chat bridge for the frontend assistant drawer.
This module uses a local Ollama model to answer questions about the
currently selected asset based only on the asset state supplied by the UI.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = """\
You are the AssetTrack Portfolio Assistant.
You have direct access to the user's current asset data.
Answer their questions based ONLY on the provided asset state.
Be concise and conversational.

Rules:
- If the answer is not supported by the asset state, say so plainly.
- Prefer concrete references to comparables, news, status, and valuation data.
- Do not invent marketplace activity, prices, or events that are not present.
- If the user wants to update an item, explain what change they described,
  but do not claim the system has already saved it unless the UI confirms that.
"""


def _normalize_history(chat_history: list[dict[str, Any]] | None) -> list[BaseMessage]:
    """Map frontend chat history into LangChain message objects."""
    messages: list[BaseMessage] = []

    for entry in chat_history or []:
        role = str(entry.get("role", "user")).lower()
        content = str(entry.get("content", "")).strip()
        if not content:
            continue

        if role == "assistant":
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    return messages


def _format_state(current_state: dict[str, Any] | None) -> str:
    """Serialize current asset state for prompt injection."""
    if not current_state:
        return "No asset is currently selected."

    try:
        return json.dumps(current_state, indent=2, sort_keys=True, default=str)
    except TypeError:
        logger.warning("Chat Agent: current_state was not JSON serializable")
        return json.dumps({"raw_state": str(current_state)}, indent=2)


def generate_chat_response(
    user_message: str,
    chat_history: list[dict[str, Any]] | None,
    current_state: dict[str, Any] | None,
) -> str:
    """
    Generate a local SLM response grounded in the active asset state.

    Returns a user-safe fallback message if Ollama or the required Python
    package is unavailable on the machine.
    """
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        logger.exception("Chat Agent: langchain_community is not installed")
        return (
            "Local chat is not ready yet because `langchain_community` is not "
            "installed in the backend environment."
        )

    state_block = _format_state(current_state)
    history_messages = _normalize_history(chat_history)

    messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                f"{CHAT_SYSTEM_PROMPT}\n\n"
                f"Current asset state:\n{state_block}"
            )
        ),
        *history_messages,
        HumanMessage(content=user_message),
    ]

    try:
        model = ChatOllama(model="llama3", temperature=0.2)
        response = model.invoke(messages)
    except Exception as exc:
        logger.exception("Chat Agent: Ollama request failed: %s", exc)
        return (
            "I couldn't reach the local Ollama model. Make sure Ollama is "
            "installed, running, and that the `llama3` model is available."
        )

    content = getattr(response, "content", "")
    if isinstance(content, list):
        return " ".join(str(part) for part in content if part).strip()

    return str(content).strip() or "I wasn't able to generate a response."
