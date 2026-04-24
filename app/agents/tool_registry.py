"""
Agent Tool Registry — defines the Tool interface and built-in tools.
"""
import ast
import math
import asyncio
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool interface
# ---------------------------------------------------------------------------
class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    async def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool and return a result dict."""
        ...

    def schema(self) -> Dict:
        return {"name": self.name, "description": self.description}


# ---------------------------------------------------------------------------
# Safe Math Evaluator (replaces eval() — NO CODE EXECUTION)
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Walk the AST and evaluate only arithmetic operations — no code execution."""
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if not op_func:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        # Prevent absurd exponents (DoS via 10**10000000)
        if isinstance(node.op, ast.Pow) and right > 1000:
            raise ValueError("Exponent too large (max 1000)")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        op_func = _SAFE_OPS.get(type(node.op))
        if not op_func:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval_node(node.operand))
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCS:
            raise ValueError(f"Unsupported function: {func_name}")
        func = _SAFE_FUNCS[func_name]
        if callable(func):
            args = [_safe_eval_node(arg) for arg in node.args]
            return func(*args)
        return func  # constants like pi, e
    elif isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            val = _SAFE_FUNCS[node.id]
            if not callable(val):
                return val  # pi, e
        raise ValueError(f"Unknown variable: {node.id}")
    else:
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


def safe_math_eval(expression: str) -> float:
    """Parse and evaluate a math expression using AST — zero code execution risk."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return _safe_eval_node(tree)
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid expression: {e}")


# ---------------------------------------------------------------------------
# Built-in Tools
# ---------------------------------------------------------------------------
class WebSearchTool(Tool):
    name = "web_search"
    description = (
        "Search the web for up-to-date information. "
        "Input: {'query': str, 'max_results': int (optional, default 3)}. "
        "Output: list of {title, url, snippet, content}."
    )

    async def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        from app.agents.search_agent import process_search_queries
        query = input.get("query", "")
        max_results = input.get("max_results", 3)
        try:
            results = await process_search_queries([query], max_results_per_query=max_results)
            return {"success": True, "results": results[:max_results]}
        except Exception as e:
            logger.exception("WebSearchTool failed")
            return {"success": False, "error": str(e), "results": []}


class RAGQueryTool(Tool):
    name = "rag_query"
    description = (
        "Query uploaded documents in a workspace using RAG. "
        "Input: {'query': str, 'workspace_id': str}. "
        "Output: {'answer': str, 'sources': list}."
    )

    async def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        from app.services.rag import rag_query
        query = input.get("query", "")
        workspace_id = input.get("workspace_id")
        if not workspace_id:
            return {"success": False, "error": "workspace_id is required", "answer": ""}
        try:
            generator, sources = await rag_query(query, workspace_id=workspace_id)
            chunks = []
            async for chunk in generator:
                chunks.append(chunk)
            return {"success": True, "answer": "".join(chunks), "sources": sources}
        except Exception as e:
            logger.exception("RAGQueryTool failed")
            return {"success": False, "error": str(e), "answer": ""}


class CalculatorTool(Tool):
    name = "calculator"
    description = (
        "Evaluate safe arithmetic expressions (no code execution). "
        "Supports: +, -, *, /, **, %, sqrt(), log(), sin(), cos(), pi, e. "
        "Input: {'expression': str}. "
        "Output: {'result': float}."
    )

    async def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        expr = input.get("expression", "")
        try:
            result = safe_math_eval(expr)
            return {"success": True, "result": result}
        except (ValueError, ZeroDivisionError) as e:
            return {"success": False, "error": str(e), "result": None}


class SummarizeTool(Tool):
    name = "summarize"
    description = (
        "Summarize a long piece of text into a short paragraph. "
        "Input: {'text': str, 'max_words': int (optional)}. "
        "Output: {'summary': str}."
    )

    async def execute(self, input: Dict[str, Any]) -> Dict[str, Any]:
        from langchain_groq import ChatGroq
        from app.core.config import settings

        text = input.get("text", "")[:8000]
        max_words = input.get("max_words", 150)
        if not settings.GROQ_API_KEY:
            return {"success": False, "error": "GROQ_API_KEY not set", "summary": ""}
        try:
            model = ChatGroq(
                temperature=0.1,
                model_name=settings.GROQ_SYNTHESIS_MODEL,
                groq_api_key=settings.GROQ_API_KEY,
            )
            prompt = f"Summarize the following text in at most {max_words} words:\n\n{text}"
            result = await model.ainvoke(prompt)
            return {"success": True, "summary": result.content}
        except Exception as e:
            logger.exception("SummarizeTool failed")
            return {"success": False, "error": str(e), "summary": ""}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_schemas(self) -> List[Dict]:
        return [t.schema() for t in self._tools.values()]

    async def execute(self, name: str, input: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.get(name)
        if not tool:
            return {"success": False, "error": f"Unknown tool: {name}"}
        return await tool.execute(input)


# Singleton registry with built-in tools pre-registered
tool_registry = ToolRegistry()
tool_registry.register(WebSearchTool())
tool_registry.register(RAGQueryTool())
tool_registry.register(CalculatorTool())
tool_registry.register(SummarizeTool())
