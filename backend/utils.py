

"""
utils.py
--------
Pure utility functions for the chatbot.

This module intentionally has NO dependency on:
- Discord
- database code
- NLU / intent detection
- API clients

It provides small, deterministic interfaces for:
- Dice rolling
- Coin flipping
- Safe arithmetic
- Random jokes
- Simple text helpers

Python version:
    3.10+

External libraries:
    None.

Everything here uses Python's standard library.
"""

from __future__ import annotations

import ast
import math
import operator
import random
import re
from typing import Any


# ============================================================================
# Configuration
# ============================================================================

MAX_DICE_COUNT = 20
MAX_DICE_SIDES = 1000
MAX_MATH_LENGTH = 200
MAX_ABS_NUMBER = 1_000_000_000
MAX_EXPONENT = 100


# ============================================================================
# Dice
# ============================================================================

# Examples:
#   d20
#   2d6
#   10D100
#   2d6+3
#   1d20 - 2
#
# The modifier is optional.
_DICE_RE = re.compile(
    r"""
    (?P<count>\d{1,2})?
    \s*[dD]\s*
    (?P<sides>\d{1,4})
    \s*
    (?P<modifier>[+-]\s*\d{1,5})?
    """,
    re.VERBOSE,
)

_COUNT_ONLY_RE = re.compile(
    r"""
    \b
    (?P<count>\d{1,2})
    \s*
    (?:dice|die)
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def roll_dice(text: str) -> dict[str, Any]:
    """
    Roll dice described in free-form text.

    Supported examples:
        "roll a d20"
        "roll 2d6"
        "give me 4 dice"
        "roll 2d6+3"
        "roll 1d20 - 2"

    If no dice notation is found, one d6 is rolled.

    Returns:
        {
            "count": 2,
            "sides": 6,
            "rolls": [3, 5],
            "modifier": 3,
            "total": 11,
            "notation": "2d6+3"
        }
    """

    if not isinstance(text, str):
        text = str(text)

    match = _DICE_RE.search(text)

    if match:
        count = int(match.group("count") or 1)
        sides = int(match.group("sides"))

        modifier_text = match.group("modifier")
        modifier = 0

        if modifier_text:
            modifier = int(modifier_text.replace(" ", ""))

    else:
        count_match = _COUNT_ONLY_RE.search(text)

        if count_match:
            count = int(count_match.group("count"))
        else:
            count = 1

        sides = 6
        modifier = 0

    # Prevent ridiculous requests.
    count = max(1, min(count, MAX_DICE_COUNT))
    sides = max(2, min(sides, MAX_DICE_SIDES))
    modifier = max(-MAX_ABS_NUMBER, min(modifier, MAX_ABS_NUMBER))

    rolls = [random.randint(1, sides) for _ in range(count)]

    subtotal = sum(rolls)
    total = subtotal + modifier

    notation = f"{count}d{sides}"

    if modifier > 0:
        notation += f"+{modifier}"
    elif modifier < 0:
        notation += str(modifier)

    return {
        "count": count,
        "sides": sides,
        "rolls": rolls,
        "modifier": modifier,
        "subtotal": subtotal,
        "total": total,
        "notation": notation,
    }


def format_dice_result(result: dict[str, Any]) -> str:
    """
    Convert a roll_dice() result into a Discord-friendly message.

    Example:
        🎲 2d6+3 → [4, 5] + 3 = 12
    """

    rolls = result["rolls"]
    modifier = result["modifier"]
    total = result["total"]
    notation = result["notation"]

    roll_text = ", ".join(str(value) for value in rolls)

    if modifier > 0:
        return f"🎲 **{notation}** → `{roll_text}` + {modifier} = **{total}**"

    if modifier < 0:
        return f"🎲 **{notation}** → `{roll_text}` {modifier} = **{total}**"

    return f"🎲 **{notation}** → `{roll_text}` = **{total}**"


# ============================================================================
# Coin
# ============================================================================

COIN_SIDES = ("heads", "tails")


def flip_coin() -> str:
    """
    Flip one coin.

    Returns:
        "heads" or "tails"
    """

    return random.choice(COIN_SIDES)


def flip_coins(count: int = 1) -> list[str]:
    """
    Flip multiple coins.

    The number of flips is capped at 100.
    """

    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1

    count = max(1, min(count, 100))

    return [flip_coin() for _ in range(count)]


def format_coin_result(results: list[str]) -> str:
    """
    Format one or multiple coin flips.

    Examples:
        "🪙 Heads!"
        "🪙 Heads: 3 | Tails: 2"
    """

    if not results:
        return "🪙 Nothing to flip."

    if len(results) == 1:
        return f"🪙 **{results[0].capitalize()}!**"

    heads = results.count("heads")
    tails = results.count("tails")

    return (
        f"🪙 **{len(results)} coin flips**\n"
        f"Heads: **{heads}**\n"
        f"Tails: **{tails}**"
    )


# ============================================================================
# Math
# ============================================================================

_WORD_OPERATORS = {
    "multiplied by": "*",
    "divided by": "/",
    "plus": "+",
    "minus": "-",
    "times": "*",
    "over": "/",
}

_ALLOWED_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# Matches arithmetic-looking sections of text.
_MATH_EXPRESSION_RE = re.compile(
    r"""
    (?:
        \d+(?:\.\d+)?
        |
        [+\-*/%()]
    )+
    """,
    re.VERBOSE,
)


def _safe_number(value: Any) -> float | int:
    """
    Validate a numeric value before it is used in calculations.
    """

    if not isinstance(value, (int, float)):
        raise ValueError("invalid number")

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("number is not finite")

    if abs(value) > MAX_ABS_NUMBER:
        raise ValueError("number is too large")

    return value


def _safe_eval(node: ast.AST) -> float | int:
    """
    Recursively evaluate an AST using only explicitly allowed operations.

    This is deliberately NOT Python's eval().
    """

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise ValueError("booleans are not allowed")

        if isinstance(node.value, (int, float)):
            return _safe_number(node.value)

        raise ValueError("unsupported constant")

    if isinstance(node, ast.BinOp):
        operation = _ALLOWED_BINARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("unsupported operator")

        left = _safe_eval(node.left)
        right = _safe_eval(node.right)

        # Protect against giant exponent calculations.
        if isinstance(node.op, ast.Pow):
            if abs(right) > MAX_EXPONENT:
                raise ValueError("exponent is too large")

            if abs(left) > 1000 and abs(right) > 10:
                raise ValueError("power calculation is too large")

        # Explicit division-by-zero protection.
        if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
            raise ZeroDivisionError("division by zero")

        result = operation(left, right)

        return _safe_number(result)

    if isinstance(node, ast.UnaryOp):
        operation = _ALLOWED_UNARY_OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("unsupported unary operator")

        value = _safe_eval(node.operand)

        return _safe_number(operation(value))

    raise ValueError("unsupported expression")


def _normalise_math_text(text: str) -> str:
    """
    Convert natural-language math into normal arithmetic syntax.

    Examples:
        "12 plus 4" -> "12 + 4"
        "10 times 5" -> "10 * 5"
        "20 divided by 4" -> "20 / 4"
        "8 × 2" -> "8 * 2"
    """

    lowered = text.lower()

    # Normalize common symbols.
    lowered = lowered.replace("×", "*")
    lowered = lowered.replace("÷", "/")
    lowered = lowered.replace("−", "-")

    # Replace words longest-first so "divided by" is handled correctly.
    for phrase in sorted(_WORD_OPERATORS, key=len, reverse=True):
        symbol = _WORD_OPERATORS[phrase]

        lowered = re.sub(
            rf"\b{re.escape(phrase)}\b",
            f" {symbol} ",
            lowered,
        )

    return lowered


def _extract_math_expression(text: str) -> str:
    """
    Extract the most likely arithmetic expression from a sentence.
    """

    normalised = _normalise_math_text(text)

    candidates = _MATH_EXPRESSION_RE.findall(normalised)

    if not candidates:
        return ""

    # Prefer the longest expression containing at least one number.
    valid_candidates = [
        candidate.strip()
        for candidate in candidates
        if re.search(r"\d", candidate)
    ]

    if not valid_candidates:
        return ""

    return max(valid_candidates, key=len)


def safe_math_eval(text: str) -> dict[str, Any] | None:
    """
    Safely evaluate arithmetic found inside natural language.

    Examples:
        "what is 12 + 5"
        "calculate 10 times 4"
        "100 divided by 5"
        "2 ** 8"
        "what is (10 + 5) * 2"

    Returns:
        {
            "expression": "10 * 4",
            "result": 40
        }

    Returns None when the input cannot safely be evaluated.
    """

    if not isinstance(text, str):
        return None

    if not text.strip():
        return None

    if len(text) > MAX_MATH_LENGTH:
        return None

    expression = _extract_math_expression(text)

    if not expression:
        return None

    # Reject malformed repeated operators before parsing.
    if re.search(r"[*/%]{2}(?!\*)", expression):
        return None

    try:
        tree = ast.parse(expression, mode="eval")

        result = _safe_eval(tree)

        if isinstance(result, float):
            if not math.isfinite(result):
                return None

            # Make 10.0 look nicer as 10.
            if result.is_integer():
                result = int(result)

        return {
            "expression": expression,
            "result": result,
        }

    except (
        SyntaxError,
        ValueError,
        ZeroDivisionError,
        OverflowError,
        TypeError,
    ):
        return None


def format_math_result(result: dict[str, Any]) -> str:
    """
    Format the result returned by safe_math_eval().
    """

    expression = result["expression"]
    value = result["result"]

    return f"🧮 `{expression}` = **{value}**"


# ============================================================================
# Jokes
# ============================================================================

JOKES = [
    "I'd tell you a UDP joke, but you might not get it.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    (
        "A programmer's partner says, 'Grab a loaf of bread, "
        "and if they have eggs, get a dozen.' They came home with 12 loaves."
    ),
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "My code doesn't have bugs. It just develops random features.",
    "I changed one line of CSS and now I have to rebuild society.",
    "It works on my machine is the developer's version of 'it's not you, it's me.'",
    "Debugging: being the detective in a crime movie where you're also the murderer.",
    "I named my variables so poorly that even I don't trust me anymore.",
    "There's no place like 127.0.0.1.",
    "A SQL query walks into a bar, walks up to two tables, and asks: 'Can I join you?'",
    "Why was the JavaScript developer sad? Because they didn't know how to `null` their feelings.",
    "Git is basically a time machine where every button has consequences.",
    "I don't always test my code, but when I do, I do it in production.",
    "Why did the developer go broke? They used up all their cache.",
    "A programmer puts two glasses on their desk before coding: one for water and one for debugging.",
    "My favorite programming language is whatever works before the deadline.",
    "The code review said 'LGTM.' Five minutes later, everything was on fire.",
    "I have a joke about recursion, but first you need to hear a joke about recursion.",
    "There are two hard things in computer science: cache invalidation, naming things, and off-by-one errors.",
]


def random_joke() -> str:
    """
    Return a random programming joke.
    """

    return random.choice(JOKES)


# ============================================================================
# Small text utilities
# ============================================================================


def clean_text(text: str, max_length: int = 500) -> str:
    """
    Normalize whitespace and limit text length.

    Useful before passing user messages into other chatbot logic.
    """

    if not isinstance(text, str):
        text = str(text)

    cleaned = re.sub(r"\s+", " ", text).strip()

    return cleaned[:max_length]


def choose_response(options: list[str] | tuple[str, ...]) -> str:
    """
    Pick a random response from a sequence.

    Raises:
        ValueError: if options is empty.
    """

    if not options:
        raise ValueError("response list cannot be empty")

    return random.choice(options)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "roll_dice",
    "format_dice_result",
    "flip_coin",
    "flip_coins",
    "format_coin_result",
    "safe_math_eval",
    "format_math_result",
    "random_joke",
    "clean_text",
    "choose_response",
]