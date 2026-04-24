"""
Executor — dispatches a single plan step to the correct tool.
"""
import logging
import time
from typing import Any, Dict, Optional

from app.agents.planner import Step
from app.agents.tool_registry import tool_registry

logger = logging.getLogger(__name__)


class StepResult:
    def __init__(
        self,
        step_id: int,
        tool: str,
        success: bool,
        observation: Any,
        error: Optional[str] = None,
        duration_ms: float = 0,
    ):
        self.step_id = step_id
        self.tool = tool
        self.success = success
        self.observation = observation
        self.error = error
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "success": self.success,
            "observation": self.observation,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


async def execute_step(step: Step) -> StepResult:
    """
    Dispatch a step to its designated tool and return a StepResult.
    Never raises — errors are captured in the result.
    """
    logger.info(
        "Executing step %d: tool=%s action=%s",
        step.step_id, step.tool, step.action[:80],
    )
    t0 = time.monotonic()
    try:
        result = await tool_registry.execute(step.tool, step.tool_input)
        duration_ms = (time.monotonic() - t0) * 1000
        success = result.get("success", True)
        error = result.get("error") if not success else None
        return StepResult(
            step_id=step.step_id,
            tool=step.tool,
            success=success,
            observation=result,
            error=error,
            duration_ms=round(duration_ms, 1),
        )
    except Exception as e:
        duration_ms = (time.monotonic() - t0) * 1000
        logger.exception("Step %d execution failed: tool=%s", step.step_id, step.tool)
        return StepResult(
            step_id=step.step_id,
            tool=step.tool,
            success=False,
            observation=None,
            error=str(e),
            duration_ms=round(duration_ms, 1),
        )
