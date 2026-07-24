# -*- coding: utf-8 -*-
"""
Metrics Finalization
Finalizes performance metrics
"""

from typing import Dict, Any
from dataclasses import replace
from datetime import datetime

from ..state import KnowledgeAgentState


def finalize_metrics(state: KnowledgeAgentState) -> Dict[str, Any]:
    """
    Finalize performance metrics
    
    Metrics Calculated:
        - Total duration (milliseconds)
        - Estimated cost (based on model and tokens)
    
    Args:
        state: Current agent state
        
    Returns:
        State update with finalized metrics
    """
    metrics = state["metrics"]
    config = state["config"]

    print("\n[Metrics] Finalizing metrics")

    try:
        end_time = datetime.now() if metrics.start_time else metrics.end_time
        total_duration = (
            (end_time - metrics.start_time).total_seconds() * 1000
            if metrics.start_time else metrics.total_duration_ms
        )

        # Estimate cost from SUPPORTED_MODELS config
        from app.core.config import SUPPORTED_MODELS
        model_info = SUPPORTED_MODELS.get(config.model, {})
        cost_per_1k_tokens = model_info.get("cost_per_1k_tokens", 0.001)
        updated_metrics = replace(
            metrics,
            end_time=end_time,
            total_duration_ms=total_duration,
            estimated_cost=(metrics.total_tokens / 1000) * cost_per_1k_tokens,
        )

        print(f"[Metrics] Duration: {updated_metrics.total_duration_ms:.2f}ms")
        print(f"[Metrics] Tokens: {updated_metrics.total_tokens}")
        print(f"[Metrics] Cost: ${updated_metrics.estimated_cost:.4f}")

        return {"metrics": updated_metrics}
    
    except Exception as e:
        print(f"[Metrics] Error: {str(e)}")
        return {
            "all_errors": [f"Metrics finalization failed: {str(e)}"]
        }
