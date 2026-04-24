"""
Observability and analytics tracking.
Logs token usage, search modes, and timing for endpoints.
"""
import logging
import time

logger = logging.getLogger("app.metrics")

def log_request_metrics(endpoint: str, query: str, duration_ms: float, mode: str = "general", results_count: int = 0):
    """Log performance and usage metrics for observability dashboards."""
    logger.info(
        "Request metric",
        extra={
            "metric": "request_latency",
            "endpoint": endpoint,
            "query_length": len(query),
            "mode": mode,
            "duration_ms": int(duration_ms),
            "results_count": results_count,
        }
    )

def log_agent_step(session_id: str, step_id: int, tool: str, duration_ms: float):
    logger.info(
        "Agent step metric",
        extra={
            "metric": "agent_step_latency",
            "session_id": session_id,
            "step": step_id,
            "tool": tool,
            "duration_ms": int(duration_ms)
        }
    )
