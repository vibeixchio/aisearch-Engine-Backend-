"""
Planner — converts a goal into structured steps using chain-of-thought.
"""
import json
import logging
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """You are a senior AI agent planner. Your job is to decompose a complex goal into a precise, ordered list of steps.

Guidelines:
- Each step should be atomic (one action, one tool).
- Prefer web_search for current information, rag_query for document knowledge, calculator for math, summarize for condensing text.
- Never generate steps that require human interaction.
- Maximum 10 steps.

Available tools:
{tools_description}

Memory context:
{memory}

Return ONLY valid JSON in this exact format:
{{
  "steps": [
    {{"step_id": 1, "action": "brief action description", "tool": "tool_name", "tool_input": {{...}}}},
    ...
  ],
  "reasoning": "brief explanation of your approach"
}}"""


class Step(BaseModel):
    step_id: int
    action: str
    tool: str
    tool_input: dict = Field(default_factory=dict)


class PlanResult(BaseModel):
    steps: List[Step]
    reasoning: str = ""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=4), reraise=True)
async def _llm_plan(prompt_text: str) -> PlanResult:
    model = ChatGroq(
        temperature=0,
        model_name=settings.GROQ_SYNTHESIS_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
    )
    response = await model.ainvoke(prompt_text)
    raw = response.content.strip()
    # Extract JSON block if wrapped in markdown
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    steps = [Step(**s) for s in data.get("steps", [])]
    return PlanResult(steps=steps, reasoning=data.get("reasoning", ""))


async def plan(
    goal: str,
    tools_description: str,
    memory: str = "",
) -> PlanResult:
    """Generate a structured action plan for the given goal."""
    if not settings.GROQ_API_KEY:
        # Fallback: single web search step
        return PlanResult(
            steps=[Step(step_id=1, action=f"Search for: {goal}", tool="web_search", tool_input={"query": goal})],
            reasoning="Fallback plan — no API key configured",
        )

    prompt = ChatPromptTemplate.from_template(
        f"{PLANNER_SYSTEM_PROMPT}\n\nGoal: {{goal}}"
    )
    prompt_text = prompt.format(
        tools_description=tools_description,
        memory=memory or "No prior context.",
        goal=goal,
    )

    try:
        result = await _llm_plan(prompt_text)
        logger.info("Plan generated: steps=%d goal=%s", len(result.steps), goal[:80])
        return result
    except Exception as e:
        logger.exception("Planner failed, using fallback")
        return PlanResult(
            steps=[Step(step_id=1, action=f"Search for: {goal}", tool="web_search", tool_input={"query": goal})],
            reasoning=f"Fallback due to error: {str(e)[:100]}",
        )
