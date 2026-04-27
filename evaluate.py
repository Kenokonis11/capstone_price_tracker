from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, List, Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "assettrack.db"
REPORT_PATH = ROOT / "capstone_evaluation_report.txt"


class MetricScore(BaseModel):
    score: int = Field(..., ge=1, le=10)
    rationale: str = Field(..., min_length=1)


class AssetEvaluation(BaseModel):
    faithfulness: MetricScore
    fraud_detection: MetricScore | None = None
    overall_summary: str = Field(..., min_length=1)


class EvaluationFailure(BaseModel):
    error: str = Field(..., min_length=1)


class CorruptStateError(Exception):
    """Raised when a persisted asset state cannot be decoded."""


def load_evaluator_model():
    """Create an evaluator LLM using the project's provider conventions."""
    load_dotenv(override=True)

    provider = os.getenv("LLM_PROVIDER", "google").lower()
    model_name = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set in .env.")

        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env.")

        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LLM_PROVIDER=openai but langchain-openai is not installed. "
                "Add it to requirements.txt before running evaluate.py."
            ) from exc

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
        )

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER '{provider}'. Use 'google' or 'openai'."
    )


def fetch_assets() -> List[dict[str, Any]]:
    """Load all persisted assets and parse each state_json document."""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT id, name, current_value, state_json FROM assets ORDER BY name"
        ).fetchall()
    finally:
        connection.close()

    assets: List[dict[str, Any]] = []
    for row in rows:
        try:
            state = json.loads(row["state_json"])
        except json.JSONDecodeError as exc:
            raise CorruptStateError(
                f"Asset '{row['id']}' has unreadable state_json: {exc}"
            ) from exc

        assets.append(
            {
                "id": row["id"],
                "db_name": row["name"],
                "db_current_value": row["current_value"],
                "state": state,
            }
        )

    return assets


def _extract_supervisor_logs(agent_logs: Iterable[str]) -> List[str]:
    return [log for log in agent_logs if "[Supervisor]" in log]


def _extract_manual_social_comps(comparables: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    return [comp for comp in comparables if comp.get("source_type") == "manual_social"]


def evaluate_asset(evaluator_llm, asset_record: dict[str, Any]) -> AssetEvaluation:
    """Ask the evaluator LLM to judge the persisted asset state."""
    state = asset_record.get("state") or {}
    asset = state.get("asset") or {}
    comparables = asset.get("comparables") or []
    agent_logs = state.get("agent_logs") or []
    supervisor_logs = _extract_supervisor_logs(agent_logs)
    manual_social_comps = _extract_manual_social_comps(comparables)

    has_manual_social = bool(manual_social_comps)

    system_prompt = """You are an evaluation judge for AssetTrack, an AI valuation pipeline.

Your job is to audit whether the final valuation is grounded in the evidence already present in the saved asset state.

Score each metric from 1 to 10:
- 10 means excellent, fully supported, and careful.
- 1 means poor, invented, unsupported, or clearly wrong.

Metric definitions:
1. Faithfulness
   Judge whether the final current_value appears strictly derived from the available comparables and reasoning in agent_logs.
   Penalize invented numbers, unsupported leaps, contradictions between logs and final value, or rationales that cite evidence absent from the state.

2. Fraud Detection
   Only score this if manual_social comparables are present.
   Judge whether the pipeline appears to have applied appropriate skepticism or a confidence penalty to manual_social comps.
   Look for confidence_weight usage, skeptical reasoning, and whether a low-confidence social listing was prevented from dominating the final value.

Important rules:
- Evaluate only from the provided JSON evidence.
- Do not assume access to hidden chain-of-thought or external context.
- Be concrete and concise in rationales.
- If a metric is not applicable, return null for fraud_detection.
"""

    evidence = {
        "asset_id": asset_record.get("id"),
        "asset_name": asset.get("name") or asset_record.get("db_name"),
        "pipeline_stage": state.get("pipeline_stage"),
        "current_value": asset.get("current_value", asset_record.get("db_current_value")),
        "comparables": comparables,
        "manual_social_present": has_manual_social,
        "manual_social_comps": manual_social_comps,
        "agent_logs": agent_logs,
        "supervisor_logs": supervisor_logs,
        "errors": state.get("errors") or [],
    }

    human_prompt = (
        "Evaluate this asset state and return structured output.\n\n"
        f"{json.dumps(evidence, indent=2, default=str)}"
    )

    structured_model = evaluator_llm.with_structured_output(AssetEvaluation)
    return structured_model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )


def _format_metric(name: str, metric: MetricScore | None) -> List[str]:
    if metric is None:
        return [f"{name}: N/A"]
    return [
        f"{name}: {metric.score}/10",
        f"{name} Rationale: {metric.rationale}",
    ]


def _report_block(
    asset_record: dict[str, Any],
    evaluation: AssetEvaluation | EvaluationFailure,
) -> str:
    state = asset_record.get("state") or {}
    asset = state.get("asset") or {}
    asset_name = asset.get("name") or asset_record.get("db_name") or "Unknown Asset"
    current_value = asset.get("current_value", asset_record.get("db_current_value", 0.0))

    lines = [
        "=" * 88,
        f"Asset: {asset_name}",
        f"ID: {asset_record.get('id')}",
        f"Current Value: ${float(current_value):,.2f}",
    ]
    if isinstance(evaluation, EvaluationFailure):
        lines.append(f"Evaluation Error: {evaluation.error}")
        return "\n".join(lines)

    lines.extend(_format_metric("Faithfulness", evaluation.faithfulness))
    lines.extend(_format_metric("Fraud Detection", evaluation.fraud_detection))
    lines.append(f"Summary: {evaluation.overall_summary}")
    return "\n".join(lines)


def _summarize_scores(
    results: List[Tuple[dict[str, Any], AssetEvaluation | EvaluationFailure]]
) -> str:
    successful_results = [
        evaluation for _, evaluation in results if isinstance(evaluation, AssetEvaluation)
    ]
    failed_results = [
        evaluation for _, evaluation in results if isinstance(evaluation, EvaluationFailure)
    ]

    faithfulness_scores = [evaluation.faithfulness.score for evaluation in successful_results]
    fraud_scores = [
        evaluation.fraud_detection.score
        for evaluation in successful_results
        if evaluation.fraud_detection is not None
    ]

    summary_lines = [
        "=" * 88,
        "OVERALL SUMMARY",
        f"Assets Evaluated: {len(results)}",
        f"Successful Evaluations: {len(successful_results)}",
        f"Failed Evaluations: {len(failed_results)}",
        (
            f"Average Faithfulness: {mean(faithfulness_scores):.2f}/10"
            if faithfulness_scores
            else "Average Faithfulness: N/A"
        ),
        (
            f"Average Fraud Detection: {mean(fraud_scores):.2f}/10"
            if fraud_scores
            else "Average Fraud Detection: N/A"
        ),
    ]
    return "\n".join(summary_lines)


def main() -> None:
    try:
        assets = fetch_assets()
    except CorruptStateError as exc:
        message = f"Evaluation aborted: {exc}"
        print(message)
        REPORT_PATH.write_text(message + "\n", encoding="utf-8")
        return

    if not assets:
        message = "No assets found in assettrack.db."
        print(message)
        REPORT_PATH.write_text(message + "\n", encoding="utf-8")
        return

    evaluator_llm = load_evaluator_model()
    results: List[Tuple[dict[str, Any], AssetEvaluation | EvaluationFailure]] = []

    for asset_record in assets:
        try:
            evaluation = evaluate_asset(evaluator_llm, asset_record)
        except Exception as exc:
            evaluation = EvaluationFailure(error=str(exc))
        results.append((asset_record, evaluation))

    report_blocks = [_report_block(asset_record, evaluation) for asset_record, evaluation in results]
    overall_summary = _summarize_scores(results)
    report_text = "\n\n".join([*report_blocks, overall_summary]) + "\n"

    print(report_text)
    REPORT_PATH.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
