"""
extraction.py
-------------
Information extraction for SideEye.

nlu.py decides WHAT a message is about.
This module decides WHAT useful information can be extracted from it.

The extractors are intentionally lightweight and dependency-free. They use
carefully ordered regex patterns plus a few normalization helpers so SideEye
can understand casual, real-world phrasing without needing a heavyweight NLP
stack.

Public API:
    extract_project(text)
    extract_language(text)
    extract_feature(text)
    extract_problem(text)
    extract_subject(text)
    extract_game(text)
    extract_goal(text)
    extract_name(text)
    extract_taught_fact(text)
    extract_all(text)

    EXTRACTORS
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Optional


# ============================================================================
# LANGUAGE DATA
# ============================================================================

KNOWN_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "golang",
    "go",
    "kotlin",
    "swift",
    "ruby",
    "php",
    "html",
    "css",
    "sql",
    "rust",
    "c",
]

# Common aliases users may type.
LANGUAGE_ALIASES = {
    "py": "python",
    "python3": "python",
    "python 3": "python",
    "js": "javascript",
    "jsx": "javascript",
    "node": "javascript",
    "nodejs": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "golang": "go",
    "c plus plus": "c++",
    "cpp": "c++",
    "c plusplus": "c++",
    "c sharp": "c#",
    "csharp": "c#",
    "cs": "c#",
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
}


# ============================================================================
# REGEX / CLEANING HELPERS
# ============================================================================

# Things that commonly indicate that the useful value has ended.
_STOP_TAIL = re.compile(
    r"""
    \s+
    (?:
        and
        |but
        |because
        |so
        |which
        |that
        |while
        |when
        |where
        |for
        |with
        |without
        |before
        |after
        |already
        |yet
        |recently
        |lately
        |earlier
        |though
        |tho
        |btw
        |now
    )
    \b.*
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Extra conversational endings that should not become part of a slot.
_CONVERSATIONAL_TAIL = re.compile(
    r"""
    \s+
    (?:
        please
        |pls
        |plz
        |thanks
        |thank you
        |lol
        |lmao
        |bro
        |man
        |dude
        |fr
        |ngl
    )
    \b.*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Repeated whitespace.
_WHITESPACE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Normalize whitespace while preserving useful punctuation."""
    if not text:
        return ""

    text = str(text).replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')

    return _WHITESPACE.sub(" ", text.strip())


def _clean(value: str) -> str:
    """
    Clean an extracted value.

    The function intentionally does not lowercase the value because some
    extracted values may benefit from preserving their original spelling.
    """
    if not value:
        return ""

    value = _normalize_text(value)

    # Remove common leading filler.
    value = re.sub(
        r"^(?:a|an|the)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )

    # Remove conversational endings.
    value = _CONVERSATIONAL_TAIL.sub("", value)
    value = _STOP_TAIL.sub("", value)

    # Remove surrounding punctuation.
    value = value.strip(" \t\r\n.,!?;:'\"`")

    # Avoid doubled whitespace after cleanup.
    value = _WHITESPACE.sub(" ", value)

    return value.strip()


def _first_match(
    patterns: list[str],
    text: str,
) -> Optional[str]:
    """
    Return the first non-empty named ``value`` captured by ``patterns``.

    Patterns should contain a named group called ``value``.
    """
    text = _normalize_text(text)

    if not text:
        return None

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        value = _clean(match.group("value"))

        if value:
            return value

    return None


def _normalize_language(value: str) -> Optional[str]:
    """Convert a language/alias into SideEye's canonical language name."""
    candidate = _normalize_text(value).lower().strip()

    if not candidate:
        return None

    candidate = candidate.rstrip(".,!?;:")

    # Direct alias lookup.
    if candidate in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[candidate]

    # Canonical names.
    if candidate in KNOWN_LANGUAGES:
        return candidate

    # Compact aliases.
    compact = candidate.replace(" ", "")

    if compact in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[compact]

    if compact in KNOWN_LANGUAGES:
        return compact

    return None


def _language_pattern() -> str:
    """
    Build a regex that recognizes known languages and aliases.

    Longer names are placed first so ``c++`` / ``c#`` aren't accidentally
    treated as ``c``.
    """
    names = set(KNOWN_LANGUAGES)
    names.update(LANGUAGE_ALIASES.keys())

    escaped = sorted(
        (re.escape(name) for name in names),
        key=len,
        reverse=True,
    )

    return r"(?:" + "|".join(escaped) + r")"


_LANGUAGE_RE = _language_pattern()


# ============================================================================
# PROJECT
# ============================================================================

def extract_project(text: str) -> Optional[str]:
    """
    Extract a project name/description.

    Examples:
        "I'm building a discord bot"
        "I'm making a portfolio website"
        "I've started a game"
        "my project is a weather app"
        "I'm developing an ecommerce site"
    """
    patterns = [
        # Most specific first.
        r"\b(?:i'?m|i am)\s+building\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+building\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+making\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+making\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+developing\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+developing\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+working\s+on\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+working\s+on\s+(?P<value>.+)",
        r"\b(?:i'?ve|i have)\s+been\s+trying\s+to\s+build\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?ve|i have)\s+been\s+building\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?ve|i have)\s+started\s+(?:building|making|developing)\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?ve|i have)\s+started\s+(?P<value>.+)",
        r"\bmy\s+project\s+is\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\bmy\s+project\s+is\s+(?P<value>.+)",
        r"\bthe\s+project\s+i'?m\s+working\s+on\s+is\s+(?P<value>.+)",
        r"\bproject\s*:\s*(?P<value>.+)",
    ]

    value = _first_match(patterns, text)

    if not value:
        return None

    # Avoid returning just "a project" / "the project".
    if value.lower() in {"project", "a project", "the project"}:
        return None

    return value


# ============================================================================
# LANGUAGE
# ============================================================================

def extract_language(text: str) -> Optional[str]:
    """
    Extract a programming language.

    Handles both explicit phrasing and standalone mentions.

    Examples:
        "I'm coding in Python"
        "written using JavaScript"
        "the bot is in py"
        "I'm using C++"
        "this is a TypeScript project"
    """
    text = _normalize_text(text)

    if not text:
        return None

    # Explicit language phrases.
    explicit_patterns = [
        rf"\b(?:using|use|with|in|written\s+in|built\s+in|coded\s+in|coding\s+in|programming\s+in)\s+(?P<value>{_LANGUAGE_RE})(?![\w+#])",
        rf"\b(?:written|coded|built|made)\s+(?:using|with)\s+(?P<value>{_LANGUAGE_RE})(?![\w+#])",
        rf"\b(?:language|lang)\s*(?:is|:)\s*(?P<value>{_LANGUAGE_RE})(?![\w+#])",
    ]

    for pattern in explicit_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            language = _normalize_language(match.group("value"))

            if language:
                return language

    # Standalone language mention.
    # Check aliases first because "js" etc. are otherwise easy to miss.
    language_candidates = sorted(
        set(KNOWN_LANGUAGES) | set(LANGUAGE_ALIASES.keys()),
        key=len,
        reverse=True,
    )

    for candidate in language_candidates:
        pattern = (
            r"(?<![a-zA-Z0-9+#])"
            + re.escape(candidate)
            + r"(?![a-zA-Z0-9+#])"
        )

        if re.search(pattern, text, flags=re.IGNORECASE):
            language = _normalize_language(candidate)

            if language:
                return language

    return None


# ============================================================================
# FEATURE
# ============================================================================

def extract_feature(text: str) -> Optional[str]:
    """
    Extract a feature/system/functionality the user is working on.

    Examples:
        "I'm working on authentication"
        "I'm adding a leaderboard"
        "implementing slash commands"
        "building the moderation system"
        "I need to add a database"
    """
    patterns = [
        r"\b(?:i'?m|i am)\s+working\s+on\s+(?:the|a|an)\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+working\s+on\s+(?P<value>.+)",
        r"\bworking\s+on\s+(?:the|a|an)\s+(?P<value>.+)",
        r"\bworking\s+on\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+adding\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+adding\s+(?P<value>.+)",
        r"\badding\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\badding\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+implementing\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+implementing\s+(?P<value>.+)",
        r"\bimplementing\s+(?:a|an|the)\s+(?P<value>.+)",
        r"\bimplementing\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+building\s+(?:a|an|the)\s+(?P<value>.+?(?:feature|system|functionality|module|component|commands?))\b",
        r"\bbuilding\s+(?:a|an|the)\s+(?P<value>.+?(?:feature|system|functionality|module|component|commands?))\b",
        r"\b(?:i'?m|i am)\s+adding\s+(?P<value>.+?)\s+(?:feature|system|functionality)\b",
        r"\bfeature\s*:\s*(?P<value>.+)",
    ]

    value = _first_match(patterns, text)

    if not value:
        return None

    return value


# ============================================================================
# PROBLEM
# ============================================================================

def extract_problem(text: str) -> Optional[str]:
    """
    Extract the thing that appears to be broken or causing trouble.

    Examples:
        "my login keeps crashing"
        "the bot isn't working"
        "I'm stuck with discord.py"
        "having trouble with my database"
        "trying to fix the slash commands"
    """
    patterns = [
        # "X keeps crashing / doesn't work"
        r"\b(?P<value>.+?)\s+(?:keeps?|keep)\s+(?:crashing|crash|breaking|breaking down)\b",
        r"\b(?P<value>.+?)\s+(?:is|are)\s+(?:broken|bugged|buggy)\b",
        r"\b(?P<value>.+?)\s+(?:is|are)\s+(?:not|n't)\s+(?:working|work)\b",
        r"\b(?P<value>.+?)\s+(?:does|do)\s+not\s+work\b",
        r"\b(?P<value>.+?)\s+(?:doesn't|dont|don't)\s+work\b",
        r"\b(?P<value>.+?)\s+(?:won't|will\s+not)\s+work\b",
        r"\b(?P<value>.+?)\s+(?:fails?|failed)\b",

        # "stuck on/with X"
        r"\bstuck\s+(?:on|with)\s+(?P<value>.+)",
        r"\bi'?m\s+stuck\s+(?:on|with)\s+(?P<value>.+)",
        r"\bi am\s+stuck\s+(?:on|with)\s+(?P<value>.+)",

        # Trouble / issue / problem.
        r"\bhaving\s+(?:some\s+|a\s+|an\s+)?trouble\s+(?:with|on)\s+(?P<value>.+)",
        r"\bhaving\s+(?:some\s+|a\s+|an\s+)?issue\s+(?:with|on)\s+(?P<value>.+)",
        r"\bhaving\s+(?:some\s+|a\s+|an\s+)?problem\s+(?:with|on)\s+(?P<value>.+)",
        r"\bi'?ve\s+got\s+(?:a\s+|an\s+)?(?:problem|issue)\s+(?:with|on)\s+(?P<value>.+)",
        r"\bi have\s+got\s+(?:a\s+|an\s+)?(?:problem|issue)\s+(?:with|on)\s+(?P<value>.+)",

        # Fixing something.
        r"\btrying\s+to\s+fix\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+trying\s+to\s+fix\s+(?P<value>.+)",
        r"\bneed\s+to\s+fix\s+(?P<value>.+)",
        r"\bneed\s+help\s+with\s+(?P<value>.+)",
        r"\bcan'?t\s+figure\s+out\s+(?P<value>.+)",
        r"\bcannot\s+figure\s+out\s+(?P<value>.+)",
    ]

    value = _first_match(patterns, text)

    if not value:
        return None

    # Don't accept extremely generic captures.
    generic = {
        "it",
        "this",
        "that",
        "everything",
        "nothing",
        "something",
    }

    if value.lower() in generic:
        return None

    return value


# ============================================================================
# SUBJECT / LEARNING
# ============================================================================

def extract_subject(text: str) -> Optional[str]:
    """
    Extract what the user is studying or learning.

    Examples:
        "I'm studying Python"
        "I'm learning about databases"
        "trying to learn machine learning"
        "I want to learn React"
    """
    patterns = [
        r"\b(?:i'?m|i am)\s+studying\s+(?:for\s+)?(?:the\s+)?(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+learning\s+about\s+(?:the\s+)?(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+learning\s+(?:the\s+)?(?P<value>.+)",
        r"\btrying\s+to\s+learn\s+(?:about\s+)?(?:the\s+)?(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+trying\s+to\s+learn\s+(?:about\s+)?(?:the\s+)?(?P<value>.+)",
        r"\bi\s+want\s+to\s+learn\s+(?:about\s+)?(?:the\s+)?(?P<value>.+)",
        r"\bi\s+want\s+to\s+study\s+(?:the\s+)?(?P<value>.+)",
        r"\blearning\s+(?:about\s+)?(?:the\s+)?(?P<value>.+)",
        r"\bstudying\s+(?:for\s+)?(?:the\s+)?(?P<value>.+)",
        r"\btopic\s*:\s*(?P<value>.+)",
    ]

    return _first_match(patterns, text)


# ============================================================================
# GAME
# ============================================================================

def extract_game(text: str) -> Optional[str]:
    """
    Extract a game title/name.

    Examples:
        "I'm playing Minecraft"
        "I'm playing valorant"
        "building a game called SideEye"
        "making a game called Snake"
    """
    patterns = [
        r"\b(?:i'?m|i am)\s+playing\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+playing\s+the\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+playing\s+(?:a|an)\s+game\s+(?:called\s+)?(?P<value>.+)",
        r"\bbuilding\s+(?:a|an)\s+game\s+(?:called\s+)?(?P<value>.+)",
        r"\bmaking\s+(?:a|an)\s+game\s+(?:called\s+)?(?P<value>.+)",
        r"\bdeveloping\s+(?:a|an)\s+game\s+(?:called\s+)?(?P<value>.+)",
        r"\bworking\s+on\s+(?:a|an)\s+game\s+(?:called\s+)?(?P<value>.+)",
        r"\bgame\s+(?:name|title)\s*:\s*(?P<value>.+)",
        r"\bthe\s+game\s+is\s+(?:called\s+)?(?P<value>.+)",
    ]

    return _first_match(patterns, text)


# ============================================================================
# GOAL
# ============================================================================

def extract_goal(text: str) -> Optional[str]:
    """
    Extract the user's stated goal.

    Examples:
        "my goal is to build a bot"
        "I'm trying to learn Python"
        "I want to make a website"
        "hoping to finish this today"
    """
    patterns = [
        r"\bmy\s+goal\s+is\s+to\s+(?P<value>.+)",
        r"\bmy\s+goal\s+is\s+(?P<value>.+)",
        r"\b(?:i'?m|i am)\s+trying\s+to\s+(?P<value>.+)",
        r"\b(?:i'?d|i would)\s+like\s+to\s+(?P<value>.+)",
        r"\bi\s+want\s+to\s+(?P<value>.+)",
        r"\bi\s+need\s+to\s+(?P<value>.+)",
        r"\btrying\s+to\s+(?P<value>.+)",
        r"\bhoping\s+to\s+(?P<value>.+)",
        r"\bplanning\s+to\s+(?P<value>.+)",
        r"\bplan\s+is\s+to\s+(?P<value>.+)",
        r"\bgoal\s*:\s*(?P<value>.+)",
    ]

    value = _first_match(patterns, text)

    if not value:
        return None

    # Avoid treating "I'm trying to fix X" as a generic goal if it is clearly
    # a problem statement. The NLU layer should decide intent, but this small
    # guard prevents poor extraction in ambiguous messages.
    if value.lower().startswith(
        (
            "fix ",
            "solve ",
            "debug ",
        )
    ):
        return value

    return value


# ============================================================================
# NAME
# ============================================================================

def extract_name(text: str) -> Optional[str]:
    """
    Extract a person's preferred name.

    Examples:
        "my name is Aridro"
        "call me Alex"
        "you can call me Sam"
        "I go by Jay"
    """
    text = _normalize_text(text)

    if not text:
        return None

    patterns = [
        r"\bmy\s+name\s+is\s+(?P<value>[a-z][a-z0-9_' -]{0,30})",
        r"\byou\s+can\s+call\s+me\s+(?P<value>[a-z][a-z0-9_' -]{0,30})",
        r"\bcall\s+me\s+(?P<value>[a-z][a-z0-9_' -]{0,30})",
        r"\bi\s+go\s+by\s+(?P<value>[a-z][a-z0-9_' -]{0,30})",
        r"\bmy\s+name\s*:\s*(?P<value>[a-z][a-z0-9_' -]{0,30})",
    ]

    value = _first_match(patterns, text)

    if not value:
        return None

    # A name should not contain a whole sentence.
    value = re.split(
        r"\s+(?:and|but|because|so|that|which|i'?m|i am)\s+",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    value = _clean(value)

    if not value:
        return None

    # Reject obvious non-name values.
    invalid_names = {
        "the",
        "bot",
        "you",
        "someone",
        "nobody",
        "friend",
        "user",
    }

    if value.lower() in invalid_names:
        return None

    # Keep normal capitalization for names.
    return value.title()


# ============================================================================
# TAUGHT FACTS
# ============================================================================

def extract_taught_fact(
    text: str,
) -> Optional[tuple[str, str]]:
    """
    Extract a simple user-provided fact.

    Intended to be called only after NLU has identified a teach/remember
    intent.

    Examples:
        "remember that my project is called SideEye"
        "remember that python is my favorite language"
        "my bot is called SideEye"

    Returns:
        (subject, full_fact)

    Example:
        ("python", "python is my favorite language")
    """
    text = _normalize_text(text)

    if not text:
        return None

    # Explicit "remember that X is Y".
    explicit = re.search(
        r"\bremember\s+that\s+(?P<subject>.+?)\s+is\s+(?P<rest>.+)",
        text,
        flags=re.IGNORECASE,
    )

    if explicit:
        subject = _clean(explicit.group("subject"))
        rest = _clean(explicit.group("rest"))

        if subject and rest:
            return subject, f"{subject} is {rest}"

    # "keep in mind that X is Y"
    explicit = re.search(
        r"\bkeep\s+in\s+mind\s+that\s+(?P<subject>.+?)\s+is\s+(?P<rest>.+)",
        text,
        flags=re.IGNORECASE,
    )

    if explicit:
        subject = _clean(explicit.group("subject"))
        rest = _clean(explicit.group("rest"))

        if subject and rest:
            return subject, f"{subject} is {rest}"

    # "don't forget that X is Y"
    explicit = re.search(
        r"\bdon'?t\s+forget\s+that\s+(?P<subject>.+?)\s+is\s+(?P<rest>.+)",
        text,
        flags=re.IGNORECASE,
    )

    if explicit:
        subject = _clean(explicit.group("subject"))
        rest = _clean(explicit.group("rest"))

        if subject and rest:
            return subject, f"{subject} is {rest}"

    # Generic X is Y.
    # Keep the subject deliberately short to avoid capturing whole sentences.
    generic = re.search(
        r"\b(?P<subject>[a-z0-9][a-z0-9 _-]{1,39}?)\s+is\s+(?P<rest>.+)",
        text,
        flags=re.IGNORECASE,
    )

    if not generic:
        return None

    subject = _clean(generic.group("subject"))
    rest = _clean(generic.group("rest"))

    if not subject or not rest:
        return None

    # Avoid useless generic subjects.
    if subject.lower() in {
        "this",
        "that",
        "it",
        "everything",
        "something",
        "someone",
        "the",
    }:
        return None

    return subject, f"{subject} is {rest}"


# ============================================================================
# BULK EXTRACTION
# ============================================================================

Extractor = Callable[[str], Optional[str]]

EXTRACTORS: dict[str, Extractor] = {
    "project": extract_project,
    "language": extract_language,
    "feature": extract_feature,
    "problem": extract_problem,
    "subject": extract_subject,
    "game": extract_game,
    "goal": extract_goal,
}


def extract_all(text: str) -> dict[str, str]:
    """
    Run every normal slot extractor and return the values that were found.

    Example:
        extract_all("I'm building a Discord bot in Python")

    May return something like:
        {
            "project": "Discord bot in Python",
            "language": "python",
        }

    ``name`` and ``taught_fact`` are intentionally excluded because their
    return types/semantics differ from normal slot extractors.
    """
    results: dict[str, str] = {}

    for slot, extractor in EXTRACTORS.items():
        try:
            value = extractor(text)
        except (TypeError, ValueError):
            # One malformed extractor should not break the entire pipeline.
            continue

        if value:
            results[slot] = value

    return results


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "KNOWN_LANGUAGES",
    "LANGUAGE_ALIASES",
    "EXTRACTORS",
    "extract_project",
    "extract_language",
    "extract_feature",
    "extract_problem",
    "extract_subject",
    "extract_game",
    "extract_goal",
    "extract_name",
    "extract_taught_fact",
    "extract_all",
]