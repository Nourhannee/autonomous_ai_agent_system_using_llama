"""
Tool: Calculator
Evaluates mathematical expressions safely using Python's ast module.
"""

import ast
import operator
from langchain_core.tools import tool

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

def _safe_eval(expr: str) -> float:
    expr = expr.replace("^", "**").replace("%", "/100").strip()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}") from e

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant: {node.value}")
        elif isinstance(node, ast.BinOp):
            op = _OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator")
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            op = _OPERATORS.get(type(node.op))
            return op(_eval(node.operand))
        else:
            raise ValueError(f"Unsupported node: {type(node).__name__}")
    return _eval(tree)


@tool
def calculator_tool(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Input: a math expression e.g. (3.14 * 5^2) or 5000 * 1.12^7.
    Supports +, -, *, /, ^(power), %(percent), parentheses.
    """
    try:
        result = _safe_eval(expression)
        if isinstance(result, float) and result.is_integer():
            return f"Result: {int(result)}"
        return f"Result: {result:.6g}"
    except ZeroDivisionError:
        return "Error: Division by zero."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"