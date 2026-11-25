"""Safe arithmetic evaluator: a whitelist over the AST, never ``eval``."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from forgeai.tools.base import ToolError

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "round": round, "min": min, "max": max, "abs": abs, "sum": lambda *a: sum(a),
    "ceil": math.ceil, "floor": math.floor, "sqrt": math.sqrt,
}
MAX_EXPONENT = 64
MAX_EXPRESSION_CHARS = 500


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"exponent too large (max {MAX_EXPONENT})")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_eval(arg) for arg in node.args])
    raise ToolError(f"unsupported syntax: {ast.dump(node)[:60]}")


def evaluate(expression: str) -> float | int:
    """Evaluate an arithmetic expression. Raises ``ToolError`` on anything else."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    try:
        return _eval(tree)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"cannot evaluate: {exc}") from exc


def format_number(value: float | int) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}" if isinstance(value, float) else str(value)


class Calculator:
    name = "calculator"

    def run(self, args: dict[str, Any]) -> str:
        expression = args.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ToolError("calculator needs a string argument 'expression'")
        return format_number(evaluate(expression))
