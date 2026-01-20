"""
Cost Tracking Module

Tracks API usage and costs during LLM extraction steps.
Provides functions for recording calls, calculating costs, and formatting displays.
"""

import os
import json
from datetime import datetime
from typing import Optional

import streamlit as st

# =============================================================================
# Model Pricing ($ per 1M tokens)
# =============================================================================

MODEL_PRICING = {
    # Claude 3.5 models
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    # Claude 4 models
    "claude-haiku-4-5-20251218": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-5-20251101": {"input": 15.00, "output": 75.00},
    # Fallbacks for older models
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}

# Default pricing for unknown models (use Haiku pricing as conservative estimate)
DEFAULT_PRICING = {"input": 1.00, "output": 5.00}


# =============================================================================
# Cost Calculation Functions
# =============================================================================

def get_model_pricing(model_id: str) -> dict:
    """
    Get pricing for a model.

    Args:
        model_id: The model ID (e.g., "claude-3-5-haiku-20241022")

    Returns:
        Dict with "input" and "output" prices per 1M tokens
    """
    # Direct lookup
    if model_id in MODEL_PRICING:
        return MODEL_PRICING[model_id]

    # Pattern matching for model families
    for pattern, pricing in MODEL_PRICING.items():
        # Match by model family prefix (e.g., "claude-3-5-haiku" matches "claude-3-5-haiku-20241022")
        base_pattern = "-".join(pattern.split("-")[:-1])  # Remove date suffix
        if model_id.startswith(base_pattern):
            return pricing

    return DEFAULT_PRICING


def calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of an API call.

    Args:
        model_id: The model ID used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in dollars
    """
    pricing = get_model_pricing(model_id)

    # Convert from per-million to per-token
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


# =============================================================================
# Session State Management
# =============================================================================

def _has_session_state() -> bool:
    """Check if we're in a Streamlit context with session state available."""
    try:
        # This will raise an error if not in Streamlit context
        _ = st.session_state
        return True
    except Exception:
        return False


def init_cost_tracker():
    """Initialize cost tracker in session state if not present."""
    if not _has_session_state():
        return
    if "cost_tracker" not in st.session_state:
        st.session_state.cost_tracker = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost": 0.0,
            "by_step": {},
            "calls": []
        }


def track_api_call(step_name: str, model_id: str, usage: dict):
    """
    Record an API call to the cost tracker.

    Args:
        step_name: Name of the step (e.g., "identify_chapters", "format_blocks")
        model_id: The model ID used
        usage: Usage dict from stream_message with input_tokens, output_tokens

    Note: Silently skips tracking if called from a thread without Streamlit context.
    """
    if not _has_session_state():
        return  # Skip tracking when called from worker threads

    init_cost_tracker()

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    cost = calculate_cost(model_id, input_tokens, output_tokens)

    tracker = st.session_state.cost_tracker

    # Update totals
    tracker["total_input_tokens"] += input_tokens
    tracker["total_output_tokens"] += output_tokens
    tracker["total_cost"] += cost

    # Update by-step aggregation
    if step_name not in tracker["by_step"]:
        tracker["by_step"][step_name] = {
            "input": 0,
            "output": 0,
            "cost": 0.0,
            "call_count": 0
        }

    step_data = tracker["by_step"][step_name]
    step_data["input"] += input_tokens
    step_data["output"] += output_tokens
    step_data["cost"] += cost
    step_data["call_count"] += 1

    # Record individual call
    tracker["calls"].append({
        "step": step_name,
        "model": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "timestamp": datetime.now().isoformat()
    })


def get_session_summary() -> dict:
    """
    Get summary of session costs and usage.

    Returns:
        Dict with total_input_tokens, total_output_tokens, total_cost, by_step
    """
    if not _has_session_state():
        return {"total_input_tokens": 0, "total_output_tokens": 0, "total_cost": 0.0, "by_step": {}}

    init_cost_tracker()
    tracker = st.session_state.cost_tracker

    return {
        "total_input_tokens": tracker["total_input_tokens"],
        "total_output_tokens": tracker["total_output_tokens"],
        "total_cost": tracker["total_cost"],
        "by_step": tracker["by_step"]
    }


def get_step_cost(step_name: str) -> dict:
    """
    Get cost data for a specific step.

    Args:
        step_name: Name of the step

    Returns:
        Dict with input, output, cost, call_count for the step
    """
    default = {"input": 0, "output": 0, "cost": 0.0, "call_count": 0}

    if not _has_session_state():
        return default

    init_cost_tracker()
    return st.session_state.cost_tracker["by_step"].get(step_name, default)


def reset_cost_tracker():
    """Reset the cost tracker (e.g., when starting a new session)."""
    if not _has_session_state():
        return
    st.session_state.cost_tracker = {
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cost": 0.0,
        "by_step": {},
        "calls": []
    }


# =============================================================================
# Display Formatting Functions
# =============================================================================

def format_tokens(tokens: int) -> str:
    """Format token count for display (e.g., 125000 -> '125k')."""
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}k"
    else:
        return str(tokens)


def format_cost(cost: float) -> str:
    """Format cost for display (e.g., 0.0825 -> '$0.08')."""
    if cost >= 1.0:
        return f"${cost:.2f}"
    elif cost >= 0.01:
        return f"${cost:.2f}"
    else:
        return f"${cost:.4f}"


def format_cost_display(cost: float, input_tokens: int, output_tokens: int) -> str:
    """
    Format a complete cost display line.

    Args:
        cost: Cost in dollars
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Formatted string like "125k in / 45k out | $0.25"
    """
    return f"{format_tokens(input_tokens)} in / {format_tokens(output_tokens)} out | {format_cost(cost)}"


def format_session_cost_display() -> str:
    """Format the session cost summary for display."""
    summary = get_session_summary()
    return format_cost_display(
        summary["total_cost"],
        summary["total_input_tokens"],
        summary["total_output_tokens"]
    )


# =============================================================================
# Persistence Functions
# =============================================================================

def get_cost_tracking_file(output_dir: str) -> str:
    """Get the path to the cost tracking JSON file."""
    return os.path.join(output_dir, "cost_tracking.json")


def save_cost_tracking(output_dir: str):
    """
    Save cost tracking data to JSON file.

    Args:
        output_dir: Output directory for the current textbook
    """
    init_cost_tracker()

    os.makedirs(output_dir, exist_ok=True)

    data = {
        "total_input_tokens": st.session_state.cost_tracker["total_input_tokens"],
        "total_output_tokens": st.session_state.cost_tracker["total_output_tokens"],
        "total_cost": st.session_state.cost_tracker["total_cost"],
        "by_step": st.session_state.cost_tracker["by_step"],
        "calls": st.session_state.cost_tracker["calls"],
        "last_updated": datetime.now().isoformat()
    }

    with open(get_cost_tracking_file(output_dir), "w") as f:
        json.dump(data, f, indent=2)


def load_cost_tracking(output_dir: str):
    """
    Load cost tracking data from JSON file.

    Args:
        output_dir: Output directory for the current textbook
    """
    cost_file = get_cost_tracking_file(output_dir)

    if os.path.exists(cost_file):
        with open(cost_file) as f:
            data = json.load(f)

        st.session_state.cost_tracker = {
            "total_input_tokens": data.get("total_input_tokens", 0),
            "total_output_tokens": data.get("total_output_tokens", 0),
            "total_cost": data.get("total_cost", 0.0),
            "by_step": data.get("by_step", {}),
            "calls": data.get("calls", [])
        }
    else:
        init_cost_tracker()
