import json
import logging
import time
from typing import AsyncGenerator, Dict, Any, Optional, List

from app.core.config import settings
from app.agents.planner import plan, PlanResult, Step
from app.agents.executor import execute_step, StepResult
from app.agents.reviewer import review_step, ReviewResult
from app.agents.tool_registry import tool_registry

logger = logging.getLogger(__name__)


def _memory_summary(observations: List[Dict]) -> str:
    """Compress prior observations into a short memory string."""
    if not observations:
        return ""
    lines = []
    for obs in observations[-5:]:  # Keep last 5
        step_id = obs.get("step_id", "?")
        tool = obs.get("tool", "?")
        result_preview = str(obs.get("observation", ""))[:300]
        lines.append(f"Step {step_id} ({tool}): {result_preview}")
    return "\n".join(lines)


async def run_agent(
    goal: str,
    session_id: str,
    workspace_id: Optional[str] = None,
    max_iterations: Optional[int] = None,
) -> AsyncGenerator[str, None]:
    """
    Main agent controller loop.
    Yields SSE data strings for:
      - plan
      - step_start
      - tool_call
      - observation
      - review
      - done
      - error
    """
    max_iterations = max_iterations or settings.AGENT_MAX_ITERATIONS
    iterations = 0
    all_observations: List[Dict] = []
    tools_description = "\n".join(
        f"- {s['name']}: {s['description']}"
        for s in tool_registry.list_schemas()
    )

    if workspace_id:
        pass

    try:
        while iterations < max_iterations:
            iterations += 1
            memory = _memory_summary(all_observations)

            # ── 1. PLAN ───────────────────────────────────────────────────────
            logger.info("Agent iteration %d: planning goal=%s", iterations, goal[:80])
            plan_result: PlanResult = await plan(goal, tools_description, memory)

            yield f"data: {json.dumps({'type': 'plan', 'iteration': iterations, 'steps': [s.dict() for s in plan_result.steps], 'reasoning': plan_result.reasoning})}\n\n"

            if not plan_result.steps:
                yield f"data: {json.dumps({'type': 'done', 'reason': 'No further steps required', 'iterations': iterations})}\n\n"
                return

            # ── 2. EXECUTE EACH STEP ─────────────────────────────────────────
            goal_complete = False
            for step in plan_result.steps:
                # Inject workspace_id if tool is rag_query and not specified
                if step.tool == "rag_query" and workspace_id and "workspace_id" not in step.tool_input:
                    step.tool_input["workspace_id"] = workspace_id

                yield f"data: {json.dumps({'type': 'step_start', 'step': step.dict()})}\n\n"
                yield f"data: {json.dumps({'type': 'tool_call', 'tool': step.tool, 'input': step.tool_input})}\n\n"

                # Execute
                step_result: StepResult = await execute_step(step)
                all_observations.append(step_result.to_dict())

                yield f"data: {json.dumps({'type': 'observation', 'step_id': step.step_id, 'result': step_result.to_dict()})}\n\n"

                # Review
                review: ReviewResult = await review_step(
                    goal=goal,
                    action=step.action,
                    tool=step.tool,
                    observation=step_result.observation,
                )
                yield f"data: {json.dumps({'type': 'review', 'step_id': step.step_id, 'review': review.to_dict()})}\n\n"

                if review.next_action == "complete":
                    goal_complete = True
                    break
                if review.next_action == "abort":
                    yield f"data: {json.dumps({'type': 'error', 'message': f'Aborted: {review.reason}'})}\n\n"
                    return
                # retry/continue → proceed to next step

            if goal_complete:
                break

        # ── 3. FINAL SYNTHESIS ────────────────────────────────────────────────
        logger.info("Agent complete: iterations=%d", iterations)
        final_summary = await _synthesize_final(goal, all_observations)
        yield f"data: {json.dumps({'type': 'done', 'summary': final_summary, 'iterations': iterations, 'steps_executed': len(all_observations)})}\n\n"

    except Exception as e:
        logger.exception("Agent controller error: goal=%s", goal[:80])
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


async def _synthesize_final(goal: str, observations: List[Dict]) -> str:
    """Produce a concise final answer from all gathered observations."""
    if not settings.GROQ_API_KEY or not observations:
        return "Agent completed. See step observations for details."

    obs_text = ""
    for obs in observations:
        obs_text += f"\n\nStep {obs['step_id']} ({obs['tool']}):\n{json.dumps(obs.get('observation', ''), default=str)[:1500]}"

    from langchain_groq import ChatGroq
    model = ChatGroq(
        temperature=0.2,
        model_name=settings.GROQ_SYNTHESIS_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
    )
    prompt = (
        f"Based on the following research and actions performed to achieve the goal:\n"
        f"Goal: {goal}\n\nObservations:{obs_text[:6000]}\n\n"
        f"Write a comprehensive, structured final answer in Markdown."
    )
    try:
        result = await model.ainvoke(prompt)
        return result.content
    except Exception as e:
        logger.exception("Final synthesis failed")
        return f"Agent completed {len(observations)} steps. Synthesis failed: {str(e)[:100]}"
