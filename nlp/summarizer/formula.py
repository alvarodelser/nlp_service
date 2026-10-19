# nlp_service/nlp/summarizer/formula.py
"""The criterion_formula profile's output: a scoring formula as an expression a client can parse,
render and evaluate, with every symbol declared.

Expression language (what the prompt asks for and what `check` accepts):
  OUT = expr                      one output symbol, then the expression
  numbers 0.5 · symbols [A-Za-z][A-Za-z0-9_]* · + - * / ^ ( )
  comparisons < <= > >= == != · if(cond, then, else) · min(a, b, ...) · max(a, b, ...)
  abs(x) · sqrt(x) · ln(x) · log10(x) · exp(x) · floor(x) · ceil(x)
"""
from __future__ import annotations

import ast
import re

FUNCTIONS = {"if_", "min", "max", "abs", "sqrt", "ln", "log10", "exp", "floor", "ceil"}
_OK = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant, ast.Load,
       ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
       ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)


def symbols(expression: str) -> tuple[str, set[str]]:
    """(output symbol, symbols used on the right). Raises ValueError when it does not parse."""
    if "=" not in expression.replace("==", "").replace("<=", "").replace(">=", "").replace("!=", ""):
        raise ValueError("no 'OUT = expr'")
    m = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*=(?!=)(.*)$", expression, re.S)
    if not m:
        raise ValueError("the left side must be one symbol")
    rhs = re.sub(r"\bif\s*\(", "if_(", m.group(2)).replace("^", "**")
    tree = ast.parse(rhs.strip(), mode="eval")
    used = set()
    for node in ast.walk(tree):
        if not isinstance(node, _OK):
            raise ValueError(f"not allowed: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ValueError("unknown function")
        elif isinstance(node, ast.Name) and node.id not in FUNCTIONS:
            used.add(node.id)
        elif isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError("only numbers as constants")
    return m.group(1), used


def check(result: dict) -> tuple[bool, str | None]:
    """A formula that parses and declares every symbol it uses; or none at all."""
    if not result.get("has_formula"):
        return True, None
    try:
        out, used = symbols(result.get("expression") or "")
    except (ValueError, SyntaxError) as e:
        return False, f"expression: {e}"
    declared = {v.get("symbol") for v in result.get("variables") or []}
    missing = (used | {out}) - declared
    return (not missing), (f"undeclared symbols: {sorted(missing)}" if missing else None)
