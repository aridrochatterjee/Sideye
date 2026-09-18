"""
references.py
-------------

Context-aware reference and pronoun resolution for SideEye.

This module handles conversational references such as:

    "who created it?"
    "how do I improve it?"
    "is that broken?"
    "can I change them?"
    "what about this?"
    "how do I fix the project?"

The user's original message is NEVER modified for display.

Instead, this module creates a resolved COPY of the message that
the rest of SideEye's pipeline can use for reasoning.

Example:

    User:
        I'm building a Discord bot.

    User:
        How do I improve it?

Internal resolution:

    "How do I improve the Discord bot?"

The module remains completely deterministic and local.

No LLMs, embeddings, external APIs, or heavy NLP libraries are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ============================================================
# PRONOUN DEFINITIONS
# ============================================================

SINGULAR_PRONOUNS = {
    "it",
    "this",
    "that",
}

PLURAL_PRONOUNS = {
    "they",
    "them",
    "these",
    "those",
}

ALL_PRONOUNS = SINGULAR_PRONOUNS | PLURAL_PRONOUNS


# Possessive references need slightly different handling.
POSSESSIVE_PRONOUNS = {
    "its",
    "their",
}


# ============================================================
# REGEX
# ============================================================

_PRONOUN_RE = re.compile(
    r"\b("
    + "|".join(
        re.escape(word)
        for word in sorted(
            ALL_PRONOUNS | POSSESSIVE_PRONOUNS,
            key=len,
            reverse=True,
        )
    )
    + r")\b",
    re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)


# ============================================================
# THAT / CLAUSE HANDLING
# ============================================================

# "that" is not always a reference.
#
# Examples:
#
#     "remember that SideEye is local"
#     "I think that Python is easier"
#
# In these cases "that" introduces a clause.
#
# But:
#
#     "fix that"
#     "what is that?"
#
# is a genuine reference.

_COMPLEMENTIZER_TRIGGERS = {
    "remember",
    "know",
    "knew",
    "think",
    "thought",
    "believe",
    "believed",
    "said",
    "say",
    "says",
    "mean",
    "means",
    "meant",
    "hope",
    "hoped",
    "hear",
    "heard",
    "notice",
    "noticed",
    "realize",
    "realized",
    "understand",
    "understood",
    "feel",
    "felt",
    "note",
    "noted",
    "imagine",
    "imagined",
    "guess",
    "guessed",
    "assume",
    "assumed",
}


# ============================================================
# REFERENCE TRIGGERS
# ============================================================

# Verbs where a reference is very likely to be an object.

OBJECT_REFERENCE_TRIGGERS = {
    "fix",
    "debug",
    "improve",
    "change",
    "edit",
    "modify",
    "update",
    "delete",
    "remove",
    "build",
    "create",
    "run",
    "test",
    "use",
    "open",
    "close",
    "start",
    "stop",
    "install",
    "configure",
    "deploy",
    "explain",
    "describe",
    "understand",
    "check",
    "review",
    "rewrite",
    "refactor",
    "optimize",
    "compile",
}


# ============================================================
# ENTITY TYPE COMPATIBILITY
# ============================================================

# These groups describe which entity types commonly behave alike.
#
# This prevents things like:
#
#     "Python"
#
# being selected as the answer to:
#
#     "How do I improve it?"
#
# when the conversation is clearly about a project.

ENTITY_TYPE_GROUPS = {
    "project": {
        "project",
        "bot",
        "application",
        "app",
        "website",
        "webapp",
        "game",
        "program",
        "software",
        "system",
    },
    "code": {
        "code",
        "function",
        "class",
        "module",
        "file",
        "script",
        "command",
    },
    "language": {
        "language",
        "programming_language",
    },
    "person": {
        "person",
        "user",
        "developer",
        "creator",
        "author",
    },
    "subject": {
        "subject",
        "topic",
        "concept",
    },
}


def _entity_group(entity_type: str | None) -> str | None:
    """Return the broad compatibility group for an entity type."""

    if not entity_type:
        return None

    normalized = entity_type.lower().strip()

    for group, values in ENTITY_TYPE_GROUPS.items():
        if normalized in values:
            return group

    return normalized


# ============================================================
# DATA MODEL
# ============================================================


@dataclass
class EntityCandidate:
    """
    Internal representation of a possible reference target.
    """

    value: str
    entity_type: str = "unknown"
    plural: bool = False
    score: float = 0.0
    index: int = 0
    source: str = "context"

    def group(self) -> str | None:
        return _entity_group(self.entity_type)


# ============================================================
# BASIC HELPERS
# ============================================================


def _normalize_value(value: Any) -> str:
    """Normalize an entity value into a clean string."""

    if value is None:
        return ""

    value = str(value).strip()

    # Collapse repeated whitespace.
    value = re.sub(r"\s+", " ", value)

    return value


def _normalize_entity_type(entity_type: Any) -> str:
    """Normalize entity type."""

    if entity_type is None:
        return "unknown"

    value = str(entity_type).strip().lower()

    return value or "unknown"


def _previous_words(
    text: str,
    position: int,
    count: int = 5,
) -> list[str]:
    """Return a small window of words immediately before a position."""

    prefix = text[:position]

    words = _WORD_RE.findall(prefix)

    return [word.lower() for word in words[-count:]]


def _is_complementizer_that(
    text: str,
    match: re.Match,
) -> bool:
    """
    Determine whether "that" is introducing a clause instead of
    referring to an entity.
    """

    preceding = _previous_words(
        text,
        match.start(),
        count=5,
    )

    if not preceding:
        return False

    # Direct trigger:
    #
    # "remember that..."
    # "think that..."
    if preceding[-1] in _COMPLEMENTIZER_TRIGGERS:
        return True

    return False


def _pronoun_is_plural(pronoun: str) -> bool:
    return pronoun.lower() in PLURAL_PRONOUNS


def _is_possessive(pronoun: str) -> bool:
    return pronoun.lower() in POSSESSIVE_PRONOUNS


# ============================================================
# PLURALITY
# ============================================================


def guess_plural(value: str) -> bool:
    """
    Cheap noun-phrase plurality heuristic.

    This is intentionally conservative.

    Examples:

        users       -> True
        commands    -> True
        project     -> False
        bot         -> False
        Python      -> False
    """

    value = _normalize_value(value)

    if not value:
        return False

    words = value.lower().split()

    last_word = words[-1]

    # Common irregular plurals.
    irregular = {
        "people",
        "children",
        "men",
        "women",
        "users",
        "developers",
        "commands",
        "files",
        "projects",
        "features",
        "issues",
        "problems",
        "things",
        "messages",
        "entities",
        "questions",
    }

    if last_word in irregular:
        return True

    # Avoid obvious non-plurals.
    if last_word.endswith(
        (
            "ss",
            "us",
            "is",
            "os",
        )
    ):
        return False

    # Common singular words ending in s.
    if last_word in {
        "python",
        "javascript",
        "typescript",
        "css",
        "html",
        "sql",
        "discord",
        "sideeye",
    }:
        return False

    return last_word.endswith("s")


# ============================================================
# ENTITY CONVERSION
# ============================================================


def _to_candidate(
    entity: dict[str, Any],
    index: int,
) -> EntityCandidate | None:
    """
    Convert the existing database entity format into an
    EntityCandidate.

    Existing expected format:

        {
            "value": "...",
            "type": "...",
            "plural": True/False
        }
    """

    value = _normalize_value(entity.get("value"))

    if not value:
        return None

    entity_type = _normalize_entity_type(
        entity.get("type")
    )

    plural = entity.get("plural")

    if plural is None:
        plural = guess_plural(value)

    return EntityCandidate(
        value=value,
        entity_type=entity_type,
        plural=bool(plural),
        index=index,
        source=str(
            entity.get("source", "context")
        ),
    )


# ============================================================
# ENTITY DEDUPLICATION
# ============================================================


def _deduplicate_entities(
    entity_stack: list[dict[str, Any]],
) -> list[EntityCandidate]:
    """
    Remove duplicate entities while keeping the newest occurrence.

    This prevents the same entity from receiving artificial
    recency advantages just because it was pushed multiple times.
    """

    seen: set[tuple[str, str]] = set()
    candidates: list[EntityCandidate] = []

    for index, entity in enumerate(entity_stack):
        candidate = _to_candidate(entity, index)

        if candidate is None:
            continue

        key = (
            candidate.value.casefold(),
            candidate.entity_type,
        )

        if key in seen:
            continue

        seen.add(key)
        candidates.append(candidate)

    return candidates


# ============================================================
# CONTEXT EXTRACTION
# ============================================================


def _context_value(
    context: Any,
    key: str,
) -> Any:
    """
    Read a context value from either:

        dict
        ConversationState-like objects

    This keeps the module compatible with your upgraded
    conversation.py.
    """

    if context is None:
        return None

    if isinstance(context, dict):
        value = context.get(key)

        # Support ContextValue-style objects.
        if hasattr(value, "value"):
            return value.value

        return value

    if hasattr(context, "context"):
        values = getattr(context, "context", {})

        item = values.get(key)

        if hasattr(item, "value"):
            return item.value

        return item

    return None


def _context_entities(
    context: Any,
) -> list[EntityCandidate]:
    """
    Build entity candidates from conversation context.

    This is a fallback in case the database entity stack doesn't
    contain the current context.
    """

    if context is None:
        return []

    candidates: list[EntityCandidate] = []

    for slot in (
        "project",
        "feature",
        "problem",
        "goal",
        "subject",
        "game",
        "language",
    ):
        value = _context_value(context, slot)

        if not value:
            continue

        entity_type = slot

        if slot == "project":
            entity_type = "project"

        elif slot == "language":
            entity_type = "language"

        candidates.append(
            EntityCandidate(
                value=_normalize_value(value),
                entity_type=entity_type,
                plural=guess_plural(str(value)),
                index=len(candidates),
                source="conversation_context",
            )
        )

    return candidates


# ============================================================
# PRONOUN / ENTITY COMPATIBILITY
# ============================================================


def _plurality_score(
    candidate: EntityCandidate,
    plural: bool,
) -> float:
    """
    Score grammatical number compatibility.
    """

    if candidate.plural == plural:
        return 3.0

    return -3.5


def _type_score(
    candidate: EntityCandidate,
    previous_intent: str | None,
) -> float:
    """
    Score whether an entity type makes sense with the previous intent.

    This is deliberately conservative.
    """

    intent = (previous_intent or "").lower()

    group = candidate.group()

    if not group:
        return 0.0

    if intent == "coding_help":
        if group in {"project", "code"}:
            return 2.5

        if group == "language":
            return 0.5

    if intent == "project_share":
        if group == "project":
            return 2.5

    if intent == "language_share":
        if group == "language":
            return 2.5

    if intent == "feature_share":
        if group in {"project", "code"}:
            return 1.5

    return 0.0


# ============================================================
# LOCAL GRAMMATICAL HINTS
# ============================================================


def _verb_before_reference(
    text: str,
    match: re.Match,
) -> str | None:
    """
    Find the nearest likely verb before a pronoun.

    Example:

        "how do I fix it"

    returns:

        "fix"
    """

    words = _previous_words(
        text,
        match.start(),
        count=8,
    )

    if not words:
        return None

    for word in reversed(words):
        if word in OBJECT_REFERENCE_TRIGGERS:
            return word

    return None


def _object_reference_score(
    candidate: EntityCandidate,
    verb: str | None,
) -> float:
    """
    Score whether an entity makes sense as the object of a verb.
    """

    if not verb:
        return 0.0

    group = candidate.group()

    if verb in {
        "fix",
        "debug",
        "improve",
        "change",
        "modify",
        "update",
        "refactor",
        "optimize",
    }:
        if group in {"project", "code"}:
            return 2.5

        if group == "language":
            return -1.5

    if verb in {
        "use",
        "learn",
        "study",
        "install",
    }:
        if group == "language":
            return 2.0

    if verb in {
        "run",
        "execute",
        "compile",
    }:
        if group == "code":
            return 2.0

    return 0.0


# ============================================================
# ENTITY SCORING
# ============================================================


def _score_candidate(
    candidate: EntityCandidate,
    *,
    plural: bool,
    text: str,
    match: re.Match,
    context: Any = None,
    previous_intent: str | None = None,
    current_topic: str | None = None,
) -> float:
    """
    Calculate the total score for a possible reference target.

    Score components:

        + plurality compatibility
        + recency
        + current topic match
        + previous intent compatibility
        + grammatical/object compatibility
        + context-slot relevance
    """

    score = 0.0

    # --------------------------------------------------------
    # 1. GRAMMATICAL NUMBER
    # --------------------------------------------------------

    score += _plurality_score(
        candidate,
        plural,
    )

    # --------------------------------------------------------
    # 2. RECENCY
    # --------------------------------------------------------

    # entity_stack is expected to be newest-first.
    #
    # Newest entity gets the strongest bonus.
    recency_bonus = max(
        0.0,
        3.5 - (candidate.index * 0.35),
    )

    score += recency_bonus

    # --------------------------------------------------------
    # 3. CURRENT TOPIC
    # --------------------------------------------------------

    if current_topic:
        topic = _normalize_value(
            str(current_topic)
        ).casefold()

        value = candidate.value.casefold()

        if value == topic:
            score += 5.0

        elif topic in value or value in topic:
            score += 3.0

    # --------------------------------------------------------
    # 4. PREVIOUS INTENT
    # --------------------------------------------------------

    score += _type_score(
        candidate,
        previous_intent,
    )

    # --------------------------------------------------------
    # 5. VERB / OBJECT COMPATIBILITY
    # --------------------------------------------------------

    verb = _verb_before_reference(
        text,
        match,
    )

    score += _object_reference_score(
        candidate,
        verb,
    )

    # --------------------------------------------------------
    # 6. CONTEXT SLOT MATCH
    # --------------------------------------------------------

    project = _context_value(
        context,
        "project",
    )

    feature = _context_value(
        context,
        "feature",
    )

    problem = _context_value(
        context,
        "problem",
    )

    if project and candidate.value.casefold() == str(
        project
    ).casefold():
        score += 4.0

    if feature and candidate.value.casefold() == str(
        feature
    ).casefold():
        score += 2.5

    if problem and candidate.value.casefold() == str(
        problem
    ).casefold():
        score += 2.5

    return score


# ============================================================
# ENTITY SELECTION
# ============================================================


def _pick_entity(
    entity_stack: list[dict[str, Any]],
    plural: bool,
    *,
    text: str = "",
    match: re.Match | None = None,
    context: Any = None,
    previous_intent: str | None = None,
    current_topic: str | None = None,
) -> tuple[EntityCandidate | None, float, float]:
    """
    Pick the best entity.

    Returns:

        (best_entity, best_score, second_best_score)

    The second-best score is important because it allows the
    caller to detect ambiguity.
    """

    if match is None:
        return None, 0.0, 0.0

    candidates = _deduplicate_entities(
        entity_stack
    )

    # Add conversation-state entities that aren't already present.
    existing = {
        (
            candidate.value.casefold(),
            candidate.entity_type,
        )
        for candidate in candidates
    }

    for candidate in _context_entities(context):
        key = (
            candidate.value.casefold(),
            candidate.entity_type,
        )

        if key not in existing:
            candidates.append(candidate)

    if not candidates:
        return None, 0.0, 0.0

    scored: list[EntityCandidate] = []

    for candidate in candidates:
        candidate.score = _score_candidate(
            candidate,
            plural=plural,
            text=text,
            match=match,
            context=context,
            previous_intent=previous_intent,
            current_topic=current_topic,
        )

        scored.append(candidate)

    scored.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    best = scored[0]

    second_score = (
        scored[1].score
        if len(scored) > 1
        else 0.0
    )

    return (
        best,
        best.score,
        second_score,
    )


# ============================================================
# RESOLUTION CONFIDENCE
# ============================================================


def _resolution_confidence(
    best_score: float,
    second_score: float,
) -> float:
    """
    Convert candidate scores into a rough deterministic confidence.

    This is NOT machine-learning probability.
    It simply represents how strongly one candidate beats another.
    """

    if best_score <= 0:
        return 0.0

    margin = max(
        0.0,
        best_score - second_score,
    )

    confidence = (
        best_score / (best_score + 5.0)
    )

    confidence += min(
        margin / 10.0,
        0.25,
    )

    return round(
        min(confidence, 0.99),
        2,
    )


# ============================================================
# MAIN RESOLVER
# ============================================================


def resolve_references(
    text: str,
    entity_stack: list[dict[str, Any]],
    *,
    context: Any = None,
    previous_intent: str | None = None,
    current_topic: str | None = None,
    ambiguity_threshold: float = 1.25,
) -> dict[str, Any]:
    """
    Resolve conversational pronouns.

    Parameters
    ----------
    text:
        Original user message.

    entity_stack:
        Most-recent-first list of entities.

    context:
        Optional ConversationState/dict.

    previous_intent:
        Previous detected intent.

    current_topic:
        Current conversation topic.

    ambiguity_threshold:
        Minimum score margin required to confidently select
        between competing entities.

    Returns
    -------
    dict

    Example:

        {
            "resolved_text": "how do i improve my bot",
            "replacements": [
                {
                    "pronoun": "it",
                    "replacement": "my bot",
                    "entity_type": "bot",
                    "confidence": 0.86,
                }
            ],
            "ambiguous": False,
        }
    """

    if not text:
        return {
            "resolved_text": text,
            "replacements": [],
            "ambiguous": False,
        }

    replacements: list[dict[str, Any]] = []

    # If there are no entities, there is nothing to resolve.
    if not entity_stack and not _context_entities(context):
        return {
            "resolved_text": text,
            "replacements": [],
            "ambiguous": False,
        }

    global_ambiguous = False

    def _sub(match: re.Match) -> str:
        nonlocal global_ambiguous

        pronoun = match.group(1)
        lower_pronoun = pronoun.lower()

        # ----------------------------------------------------
        # THAT AS CLAUSE INTRODUCER
        # ----------------------------------------------------

        if lower_pronoun == "that":
            if _is_complementizer_that(
                text,
                match,
            ):
                return match.group(0)

        # ----------------------------------------------------
        # PLURALITY
        # ----------------------------------------------------

        plural = _pronoun_is_plural(
            lower_pronoun
        )

        # ----------------------------------------------------
        # FIND ENTITY
        # ----------------------------------------------------

        (
            entity,
            best_score,
            second_score,
        ) = _pick_entity(
            entity_stack,
            plural,
            text=text,
            match=match,
            context=context,
            previous_intent=previous_intent,
            current_topic=current_topic,
        )

        if entity is None:
            return match.group(0)

        # ----------------------------------------------------
        # AMBIGUITY
        # ----------------------------------------------------

        margin = best_score - second_score

        confidence = _resolution_confidence(
            best_score,
            second_score,
        )

        ambiguous = (
            second_score > 0
            and margin < ambiguity_threshold
        )

        if ambiguous:
            global_ambiguous = True

            # Do NOT perform a potentially dangerous replacement.
            return match.group(0)

        # ----------------------------------------------------
        # POSSESSIVE HANDLING
        # ----------------------------------------------------

        replacement = entity.value

        if _is_possessive(lower_pronoun):
            if lower_pronoun == "its":
                replacement = f"{replacement}'s"

            elif lower_pronoun == "their":
                replacement = f"{replacement}'s"

        # ----------------------------------------------------
        # RECORD DEBUG INFORMATION
        # ----------------------------------------------------

        replacements.append(
            {
                "pronoun": pronoun,
                "replacement": replacement,
                "entity_type": entity.entity_type,
                "confidence": confidence,
                "score": round(best_score, 2),
                "second_score": round(
                    second_score,
                    2,
                ),
                "ambiguous": False,
                "source": entity.source,
            }
        )

        return replacement

    resolved = _PRONOUN_RE.sub(
        _sub,
        text,
    )

    return {
        "resolved_text": resolved,
        "replacements": replacements,
        "ambiguous": global_ambiguous,
    }


# ============================================================
# SIMPLE BACKWARDS-COMPATIBLE API
# ============================================================


def resolve_reference(
    text: str,
    entity_stack: list[dict[str, Any]],
) -> str:
    """
    Simple helper for code that only needs the resolved text.

    Existing code can continue doing:

        resolved = resolve_reference(text, entities)
    """

    result = resolve_references(
        text,
        entity_stack,
    )

    return result["resolved_text"]


# ============================================================
# ENTITY STACK HELPERS
# ============================================================


def push_entity(
    entity_stack: list[dict[str, Any]],
    value: str,
    entity_type: str,
    *,
    plural: bool | None = None,
    source: str = "context",
    max_entities: int = 20,
) -> None:
    """
    Add an entity to the front of the entity stack.

    Existing duplicates are removed.

    The stack remains newest-first.
    """

    value = _normalize_value(value)

    if not value:
        return

    entity_type = _normalize_entity_type(
        entity_type
    )

    if plural is None:
        plural = guess_plural(value)

    # Remove existing equivalent entity.
    entity_stack[:] = [
        entity
        for entity in entity_stack
        if not (
            _normalize_value(
                entity.get("value")
            ).casefold()
            == value.casefold()
            and _normalize_entity_type(
                entity.get("type")
            )
            == entity_type
        )
    ]

    entity_stack.insert(
        0,
        {
            "value": value,
            "type": entity_type,
            "plural": bool(plural),
            "source": source,
        },
    )

    # Prevent unlimited growth.
    del entity_stack[max_entities:]


def remove_entity(
    entity_stack: list[dict[str, Any]],
    value: str,
) -> None:
    """Remove all matching entities from the stack."""

    normalized = _normalize_value(
        value
    ).casefold()

    entity_stack[:] = [
        entity
        for entity in entity_stack
        if _normalize_value(
            entity.get("value")
        ).casefold()
        != normalized
    ]


def clear_entities(
    entity_stack: list[dict[str, Any]],
) -> None:
    """Clear the entity stack in-place."""

    entity_stack.clear()


# ============================================================
# DEBUGGING
# ============================================================


def explain_resolution(
    text: str,
    entity_stack: list[dict[str, Any]],
    *,
    context: Any = None,
    previous_intent: str | None = None,
    current_topic: str | None = None,
) -> dict[str, Any]:
    """
    Return a verbose diagnostic representation of reference
    resolution.

    Useful for SideEye's debug mode.
    """

    result = resolve_references(
        text,
        entity_stack,
        context=context,
        previous_intent=previous_intent,
        current_topic=current_topic,
    )

    return {
        "original_text": text,
        "resolved_text": result["resolved_text"],
        "replacements": result["replacements"],
        "ambiguous": result["ambiguous"],
        "entity_count": len(entity_stack),
        "current_topic": current_topic,
        "previous_intent": previous_intent,
    }