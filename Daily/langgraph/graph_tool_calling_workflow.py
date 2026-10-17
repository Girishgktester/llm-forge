import ast
import math
import operator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()


class ToolCallingState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class ToolCallingAssistant:
    """Build a LangGraph assistant that can call utility tools."""

    def __init__(self) -> None:
        self.tools = [
            StructuredTool.from_function(self.calculate),
            StructuredTool.from_function(self.get_current_utc_time),
        ]
        self.model = ChatOpenAI(model="gpt-6-luna").bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)

        builder = StateGraph(ToolCallingState)
        builder.add_node("assistant", self._assistant_node)
        builder.add_node("tools", self.tool_node)
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
            {"tools": "tools", END: END},
        )
        builder.add_edge("tools", "assistant")
        self.graph = builder.compile()

    @staticmethod
    def calculate(expression: str) -> str:
        """Evaluate a basic arithmetic expression with +, -, *, /, and parentheses."""
        if len(expression) > 200:
            raise ValueError("Expression must be 200 characters or fewer.")

        operations = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        def evaluate(node: ast.AST) -> float:
            if isinstance(node, ast.Expression):
                return evaluate(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = evaluate(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp) and type(node.op) in operations:
                return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
            raise ValueError("Use only numbers, parentheses, and +, -, *, / operators.")

        try:
            result = evaluate(ast.parse(expression, mode="eval"))
        except (SyntaxError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(f"Cannot calculate expression: {exc}") from exc

        if not math.isfinite(result):
            raise ValueError("The result must be a finite number.")
        return str(result)

    @staticmethod
    def get_current_utc_time() -> str:
        """Return the current time in UTC as an ISO 8601 timestamp."""
        return datetime.now(timezone.utc).isoformat()

    def _assistant_node(self, state: ToolCallingState) -> dict[str, list[Any]]:
        system_message = {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use the calculator for arithmetic "
                "and the UTC-time tool when the user asks for the current time."
            ),
        }
        response = self.model.invoke([system_message, *state["messages"]])
        return {"messages": [response]}


assistant = ToolCallingAssistant()
graph = assistant.graph
