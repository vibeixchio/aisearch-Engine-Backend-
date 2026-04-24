from pydantic import BaseModel, Field
from typing import Literal

class IntentClassification(BaseModel):
    mode: Literal["direct_answer", "quick_search", "deep_research", "file_query", "action"] = Field(
        description="The primary mode required to handle the user query."
    )
    search_queries: list[str] = Field(
        description="Optimized search queries to use if mode is quick_search or deep_research.",
        default_factory=list
    )
    required_plugins: list[str] = Field(
        description="List of plugins needed if mode is action.",
        default_factory=list
    )
    reasoning: str = Field(
        description="Brief explanation of why this mode was chosen."
    )
