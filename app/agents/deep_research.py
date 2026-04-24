import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.models.deep_research import ResearchState
from app.agents.search_agent import process_search_queries

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PLANNER_PROMPT = """You are an expert researcher. Given a complex query, break it down into 3-5 specific sub-queries that need to be searched on the web to form a comprehensive answer.
Return a JSON object with a single key 'queries' containing a list of strings.
Query: {query}"""

REFLECTOR_PROMPT = """You are reviewing research materials for the query: {original_query}
Current gathered information:
{context}

Analyze if this is sufficient to write a comprehensive report. If not, generate up to 2 NEW search queries to fill any gaps.
DO NOT repeat queries that have already been executed: {executed_queries}

Return a JSON object:
{{
  "is_sufficient": boolean,
  "new_queries": [list of strings or empty]
}}"""

WRITER_PROMPT = """You are an expert analyst writing a final comprehensive report based on the following gathered research.
Use Markdown formatting. Include citations in the form of URLs where appropriate.
Format it as a clean, structured document with sections and subsections.

Original Query: {original_query}
Research Material:
{context}

Report:"""


# ---------------------------------------------------------------------------
# LLM Helpers
# ---------------------------------------------------------------------------
def get_graph_model():
    return ChatGroq(
        temperature=0.1,
        model_name=settings.GROQ_SYNTHESIS_MODEL,
        groq_api_key=settings.GROQ_API_KEY,
    )


class SubQueriesOutput(BaseModel):
    queries: list[str]


class ReflectionOutput(BaseModel):
    is_sufficient: bool
    new_queries: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Nodes (with per-node error recovery)
# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, max=3), reraise=True)
async def _llm_plan(query: str) -> list[str]:
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    model = get_graph_model().with_structured_output(SubQueriesOutput)
    result = await (prompt | model).ainvoke({"query": query})
    return result.queries


async def plan_node(state: ResearchState) -> dict:
    try:
        queries = await _llm_plan(state["original_query"])
        logger.info("Plan generated", extra={"sub_queries": len(queries)})
        return {"sub_queries": queries}
    except Exception as e:
        logger.exception("Plan node failed — falling back to raw query")
        return {"sub_queries": [state["original_query"]]}


async def execute_search_node(state: ResearchState) -> dict:
    queries_to_run = [q for q in state["sub_queries"] if q not in state["executed_queries"]]
    if not queries_to_run:
        return {}

    try:
        extracted_data = await process_search_queries(queries_to_run, max_results_per_query=2)
        logger.info("Search executed", extra={"queries": len(queries_to_run), "results": len(extracted_data)})
    except Exception as e:
        logger.exception("Search execution failed")
        extracted_data = []

    return {
        "executed_queries": state["executed_queries"] + queries_to_run,
        "gathered_context": state["gathered_context"] + extracted_data,
    }


async def reflect_node(state: ResearchState) -> dict:
    context_str = ""
    for item in state["gathered_context"]:
        context_str += f"- Source: {item.get('url')}\n  Snippet: {item.get('extracted_content', '')[:1000]}\n"

    try:
        prompt = ChatPromptTemplate.from_template(REFLECTOR_PROMPT)
        model = get_graph_model().with_structured_output(ReflectionOutput)
        result = await (prompt | model).ainvoke({
            "original_query": state["original_query"],
            "context": context_str,
            "executed_queries": state["executed_queries"],
        })
        new_queries = result.new_queries
        is_sufficient = result.is_sufficient
    except Exception as e:
        logger.exception("Reflect node failed — marking as sufficient")
        new_queries = []
        is_sufficient = True

    return {
        "sub_queries": new_queries,
        "iteration_count": state["iteration_count"] + 1,
        "research_summary": f"Iteration {state['iteration_count']} complete. Sufficient: {is_sufficient}",
    }


async def draft_report_node(state: ResearchState) -> dict:
    context_str = ""
    for item in state["gathered_context"]:
        context_str += f"- Source: {item.get('url')}\n  Content: {item.get('extracted_content', '')[:1500]}\n\n"

    try:
        prompt = ChatPromptTemplate.from_template(WRITER_PROMPT)
        model = get_graph_model()
        result = await (prompt | model).ainvoke({
            "original_query": state["original_query"],
            "context": context_str,
        })
        report = result.content
        logger.info("Report drafted", extra={"length": len(report)})
    except Exception as e:
        logger.exception("Draft report failed")
        report = f"# Research Report\n\nFailed to generate report: {str(e)}\n\n## Raw Sources\n\n{context_str[:3000]}"

    return {"final_report": report}


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
def should_continue(state: ResearchState):
    if state["iteration_count"] >= state["max_iterations"]:
        return "draft"
    if len(state["sub_queries"]) == 0:
        return "draft"
    return "execute"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------
def build_research_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("execute_search", execute_search_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("draft_report", draft_report_node)

    workflow.set_entry_point("plan")

    workflow.add_edge("plan", "execute_search")
    workflow.add_edge("execute_search", "reflect")

    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {"execute": "execute_search", "draft": "draft_report"},
    )

    workflow.add_edge("draft_report", END)

    return workflow.compile()
