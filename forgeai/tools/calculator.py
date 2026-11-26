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


_COMPARE: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
}


def _eval(node: ast.AST, names: dict[str, float] | None = None) -> Any:
    names = names or {}
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.Name) and node.id in names:  # only caller-supplied numeric variables
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left, names), _eval(node.right, names)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError(f"exponent too large (max {MAX_EXPONENT})")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand, names))
    if isinstance(node, ast.Compare) and all(type(op) in _COMPARE for op in node.ops):
        values = [_eval(node.left, names)] + [_eval(c, names) for c in node.comparators]
        return all(_COMPARE[type(op)](values[i], values[i + 1]) for i, op in enumerate(node.ops))
    if isinstance(node, ast.BoolOp):
        results = [_eval(v, names) for v in node.values]
        return all(results) if isinstance(node.op, ast.And) else any(results)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*[_eval(arg, names) for arg in node.args])
    raise ToolError(f"unsupported syntax: {ast.dump(node)[:60]}")


def evaluate(expression: str, names: dict[str, float] | None = None) -> Any:
    """Evaluate an arithmetic expression (``names`` supplies variables). Raises ``ToolError`` otherwise."""
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise ToolError("expression too long")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {exc.msg}") from exc
    try:
        return _eval(tree, names)
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except (ValueError, OverflowError, TypeError) as exc:
        raise ToolError(f"cannot evaluate: {exc}") from exc


def evaluate_condition(expression: str, names: dict[str, float]) -> bool:
    """Evaluate a comparison such as ``subtotal >= 5000 and subtotal <= 14999`` over named numbers."""
    return bool(evaluate(expression, names))


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
