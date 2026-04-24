from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

class ResearchState(TypedDict):
    """
    Represents the state of our deep research LangGraph execution.
    """
    original_query: str
    sub_queries: list[str]
    executed_queries: list[str]
    gathered_context: list[dict] # The raw extracted text from urls
    research_summary: str # Internal scratchpad
    final_report: str # The resulting markdown document
    iteration_count: int
    max_iterations: int
