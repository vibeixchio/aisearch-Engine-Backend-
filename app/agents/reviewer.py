"""
Reviewer / Critic — validates step output and decides next action.
"""
import json
import logging
from typing import Any, Dict

from langchain_groq import ChatGroq
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)

REVIEWER_PROMPT = """You are a critical AI reviewer. Evaluate whether the step result satisfies the goal.

Goal: {goal}
Step action: {action}
Step tool: {tool}
Observation: {observation}

Answer ONLY with valid JSON:
{{
  "approved": true|false,
  "reason": "brief explanation",
  "next_action": "retry|continue|complete|abort"
}}

Rules:
- approved=true → move to next step
- next_action="complete" → the overall goal is fully achieved
- next_action="retry" → re-run this step (e.g., tool error)
- next_action="abort" → unrecoverable error
- next_action="continue" → step done, proceed"""


class ReviewResult:
    def __init__(self, approved: bool, reason: str, next_action: str):
        self.approved = approved
        self.reason = reason
        self.next_action = next_action  # retry | continue | complete | abort

    def to_dict(self) -> Dict:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "next_action": self.next_action,
        }


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=3), reraise=False)
async def _llm_review(prompt_text: str) -> Dict:
    model = ChatGroq(
        temperature=0,
        model_name=settings.GROQ_ROUTER_MODEL,  # Use faster small model for review
        groq_api_key=settings.GROQ_API_KEY,
    )
    response = await model.ainvoke(prompt_text)
    raw = response.content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


async def review_step(
    goal: str,
    action: str,
    tool: str,
    observation: Any,
) -> ReviewResult:
    """Review a step result. Falls back to approval on LLM error."""
    if not settings.GROQ_API_KEY:
        return ReviewResult(approved=True, reason="No API key — auto-approve", next_action="continue")

    obs_str = json.dumps(observation, default=str)[:2000]
    prompt_text = REVIEWER_PROMPT.format(
        goal=goal, action=action, tool=tool, observation=obs_str
    )

    try:
        data = await _llm_review(prompt_text)
        result = ReviewResult(
            approved=data.get("approved", True),
            reason=data.get("reason", ""),
            next_action=data.get("next_action", "continue"),
        )
        logger.info(
            "Review: approved=%s next=%s action=%s",
            result.approved, result.next_action, action[:60],
        )
        return result
    except Exception as e:
        logger.exception("Reviewer failed — defaulting to approve")
        return ReviewResult(approved=True, reason=f"Review error: {str(e)[:80]}", next_action="continue")
