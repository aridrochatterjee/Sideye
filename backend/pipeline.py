
"""
pipeline.py
-----------
Main processing pipeline for SideEye.

Processing flow:

    User Message
        ↓
    Normalize
        ↓
    Load Context / State
        ↓
    Resolve References
        ↓
    Detect Intent
        ↓
    Check Pending Slot
        ↓
    Dispatch Intent
        ↓
    Generate Response
        ↓
    Persist Context / State / History
        ↓
    Return Snapshot

The pipeline deliberately stays thin:
- NLU decides what the user probably means.
- extraction pulls structured values out of text.
- references resolves things like "it" / "that project".
- conversation manages follow-up questions.
- personality generates the actual wording.
- database persists state.
- utils performs pure calculations / party tricks.

This module coordinates those systems; it should not contain
database implementation details or Discord-specific code.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from . import conversation
from . import database
from . import extraction
from . import nlu
from . import personality
from . import references
from . import utils


# ============================================================================
# Configuration
# ============================================================================

USER_ID = database.DEFAULT_USER_ID

MAX_INPUT_LENGTH = 4_000
MAX_HISTORY_LIMIT = 200

DEFAULT_CONFIDENCE = 0.0

logger = logging.getLogger(__name__)


# ============================================================================
# Command parsing
# ============================================================================

_COMMAND_RE = re.compile(
    r"""
    ^\s*
    /(?P<command>[a-zA-Z][a-zA-Z0-9_-]*)
    (?:\s+(?P<args>.*?))?
    \s*$
    """,
    re.VERBOSE,
)


# ============================================================================
# "Ask about..." detection
# ============================================================================

_ASK_ABOUT_RE = re.compile(
    r"""
    ^
    (?:
        who\s+(?:created|made|invented)
        |
        what\s+is
        |
        what'?s
        |
        tell\s+me\s+about
        |
        how\s+does
        |
        how\s+do\s+i\s+improve
        |
        how\s+can\s+i\s+improve
        |
        what\s+did\s+i\s+say\s+about
        |
        what\s+did\s+i\s+say\s+regarding
        |
        did\s+i\s+mention
        |
        did\s+i\s+say\s+anything\s+about
        |
        do\s+you\s+remember
        |
        do\s+you\s+recall
        |
        have\s+i\s+told\s+you\s+about
        |
        have\s+i\s+mentioned
        |
        what\s+do\s+you\s+know\s+about
        |
        do\s+you\s+know\s+about
        |
        what\s+did\s+we\s+talk\s+about
        |
        remind\s+me\s+about
        |
        remind\s+me\s+what\s+i\s+said\s+about
    )
    \s+
    (?P<subject>.+?)
    \??$
    """,
    re.IGNORECASE | re.VERBOSE,
)


_LEADING_DETERMINERS_RE = re.compile(
    r"""
    ^
    (?:
        my
        |your
        |our
        |the
        |a
        |an
        |regarding
        |concerning
        |about
    )
    \s+
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================================
# Helpers
# ============================================================================


def _safe_text(value: Any) -> str:
    """
    Convert arbitrary input into safe, normalized text.
    """

    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    value = value.replace("\x00", "")
    value = value.strip()

    if len(value) > MAX_INPUT_LENGTH:
        value = value[:MAX_INPUT_LENGTH]

    return value


def _search_term(subject: str | None) -> str:
    """
    Clean an 'ask about' subject.

    Examples:
        "my project" -> "project"
        "about my project" -> "project"
        "the Discord bot" -> "Discord bot"
    """

    term = _safe_text(subject)

    # Remove trailing punctuation.
    term = term.rstrip("?!.,:;")

    # Remove stacked filler/determiner words.
    for _ in range(3):
        cleaned = _LEADING_DETERMINERS_RE.sub("", term).strip()

        if cleaned == term:
            break

        term = cleaned

    return term


def _slot_intent(slot: str) -> str:
    """
    Convert a context slot name into the corresponding intent name.
    """

    return f"{slot}_share"


def _apply_slot_value(
    user_id: int,
    context: dict[str, Any],
    slot: str,
    value: str,
) -> None:
    """
    Persist a newly extracted context value.
    """

    value = _safe_text(value)

    if not value:
        return

    context[slot] = value

    database.set_context(user_id, slot, value)

    # Entity tracking is useful for later reference resolution:
    # "my project" -> "it" -> "the project"
    try:
        plural = references.guess_plural(value)
    except Exception:
        logger.exception("Failed to guess plural for entity %r", value)
        plural = value

    database.push_entity(user_id, slot, value, plural)


def _current_mood(user_id: int) -> str:
    """
    Safely retrieve the current mood.
    """

    try:
        state = database.get_state(user_id)
    except Exception:
        logger.exception("Failed to load state for user %s", user_id)
        return personality.DEFAULT_MOOD

    mood = state.get("mood", personality.DEFAULT_MOOD)

    if mood not in personality.MOODS:
        return personality.DEFAULT_MOOD

    return mood


def _safe_respond(
    intent: str,
    mood: str,
    **kwargs: Any,
) -> str:
    """
    Call personality.respond() without allowing a broken response template
    to crash the entire chatbot.
    """

    try:
        response = personality.respond(intent, mood, **kwargs)

        if response is None:
            return personality.respond(
                "unclear",
                personality.DEFAULT_MOOD,
            )

        response = _safe_text(response)

        if response:
            return response

    except Exception:
        logger.exception(
            "Personality response failed: intent=%s mood=%s",
            intent,
            mood,
        )

    return "I lost my train of thought. Try that again."


def _persist_turn(
    user_id: int,
    raw_text: str,
    reply: str,
    intent: str,
    confidence: float,
    mood: str,
) -> None:
    """
    Persist a completed user/bot turn.

    Keeping this in one place prevents different branches of the pipeline
    from accidentally storing turns differently.
    """

    try:
        database.log_message(
            user_id,
            "user",
            raw_text,
            intent,
            confidence,
            mood,
        )

        database.log_message(
            user_id,
            "bot",
            reply,
            intent,
            confidence,
            mood,
        )

        database.update_state(
            user_id,
            last_intent=intent,
        )

        database.increment_turn_count(user_id)

    except Exception:
        # Persistence failures should be logged, but they should not destroy
        # the actual chatbot response.
        logger.exception("Failed to persist conversation turn")


def _persist_command(
    user_id: int,
    raw_text: str,
    result: dict[str, Any],
) -> None:
    """
    Persist a command response.
    """

    _persist_turn(
        user_id=user_id,
        raw_text=raw_text,
        reply=result["reply"],
        intent=result["intent"],
        confidence=result["confidence"],
        mood=result["mood"],
    )


# ============================================================================
# Suggestions
# ============================================================================


def generate_suggestions(
    context: dict[str, Any],
    intent: str,
) -> list[str]:
    """
    Generate useful next-message suggestions.

    Suggestions are based on the actual conversation state rather than
    randomly throwing commands at the user.
    """

    suggestions: list[str] = []

    try:
        next_slot = conversation.next_pending_slot(context)
    except Exception:
        logger.exception("Failed to calculate next pending slot")
        next_slot = None

    # ------------------------------------------------------------
    # Context collection
    # ------------------------------------------------------------

    if not context.get("project"):
        suggestions.append("I'm building a Discord bot")

    elif next_slot == "language":
        suggestions.append("I'm using Python")

    elif next_slot == "feature":
        suggestions.append("I'm working on the login feature")

    elif next_slot == "goal":
        suggestions.append("I want to make it faster")

    # ------------------------------------------------------------
    # Intent-specific suggestions
    # ------------------------------------------------------------

    if intent in {"coding_help", "problem_report"}:
        suggestions.append("Here's the error I'm getting")

    elif intent == "math_request":
        suggestions.append("Calculate 25 * 8")

    elif intent == "dice_request":
        suggestions.append("Roll 2d6+3")

    elif intent == "coinflip_request":
        suggestions.append("Flip 10 coins")

    elif intent == "ask_about":
        suggestions.append("What do you remember about my project?")

    # ------------------------------------------------------------
    # Generic useful actions
    # ------------------------------------------------------------

    suggestions.extend(
        [
            "Roll a d20",
            "Tell me a joke",
            "/mood acid",
        ]
    )

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique: list[str] = []

    for suggestion in suggestions:
        if suggestion not in seen:
            seen.add(suggestion)
            unique.append(suggestion)

    return unique[:4]


# ============================================================================
# Snapshot
# ============================================================================


def _snapshot(
    user_id: int,
    reply: str,
    intent: str,
    confidence: float,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Return the standard API response object.
    """

    try:
        context = database.get_context(user_id)
    except Exception:
        logger.exception("Failed to load context for snapshot")
        context = {}

    try:
        state = database.get_state(user_id)
    except Exception:
        logger.exception("Failed to load state for snapshot")
        state = {}

    mood = state.get("mood", personality.DEFAULT_MOOD)

    if mood not in personality.MOODS:
        mood = personality.DEFAULT_MOOD

    try:
        name = database.get_user_name(user_id)
    except Exception:
        logger.exception("Failed to load user name")
        name = None

    return {
        "reply": reply,
        "mood": mood,
        "intent": intent,
        "confidence": round(float(confidence), 4),
        "context": context,
        "name": name,
        "suggestions": generate_suggestions(context, intent),
        "debug": debug or {},
    }


# ============================================================================
# Commands
# ============================================================================


def _handle_command(
    user_id: int,
    command: str,
    args: str,
) -> dict[str, Any]:
    """
    Handle slash-style chatbot commands such as:

        /help
        /mood acid
        /reset
        /forget
    """

    command = _safe_text(command).lower()
    args = _safe_text(args)

    # ------------------------------------------------------------
    # /mood
    # ------------------------------------------------------------

    if command == "mood":
        requested_mood = args.lower().strip()

        if requested_mood in personality.MOODS:
            database.update_state(
                user_id,
                mood=requested_mood,
            )

            reply = _safe_respond(
                "mood_change",
                requested_mood,
                mood_name=requested_mood,
            )

            return _snapshot(
                user_id,
                reply,
                "mood_change",
                1.0,
                {
                    "command": True,
                    "command_name": "mood",
                },
            )

        available = ", ".join(personality.MOODS)

        reply = (
            f"Unknown mood `{requested_mood or 'none'}`. "
            f"Available moods: {available}"
        )

        return _snapshot(
            user_id,
            reply,
            "mood_change",
            1.0,
            {
                "command": True,
                "command_name": "mood",
                "invalid_mood": True,
            },
        )

    # ------------------------------------------------------------
    # /help
    # ------------------------------------------------------------

    if command == "help":
        mood = _current_mood(user_id)

        reply = _safe_respond(
            "help_command",
            mood,
        )

        return _snapshot(
            user_id,
            reply,
            "help_command",
            1.0,
            {
                "command": True,
                "command_name": "help",
            },
        )

    # ------------------------------------------------------------
    # /reset and /forget
    # ------------------------------------------------------------

    if command in {"reset", "forget"}:
        try:
            database.reset_all(user_id)
        except Exception:
            logger.exception("Failed to reset user state")
            return _snapshot(
                user_id,
                "I couldn't wipe the case file. Try again.",
                "reset",
                0.0,
                {
                    "command": True,
                    "reset_failed": True,
                },
            )

        return _snapshot(
            user_id,
            "Case file wiped. Starting fresh.",
            "reset",
            1.0,
            {
                "command": True,
                "command_name": command,
            },
        )

    # ------------------------------------------------------------
    # Unknown command
    # ------------------------------------------------------------

    mood = _current_mood(user_id)

    reply = _safe_respond(
        "unclear",
        mood,
    )

    return _snapshot(
        user_id,
        reply,
        "unclear",
        0.0,
        {
            "command": True,
            "unknown_command": command,
        },
    )


# ============================================================================
# Pending-slot handling
# ============================================================================


def _try_pending_slot(
    user_id: int,
    raw_text: str,
    resolved_text: str,
    context: dict[str, Any],
    state: dict[str, Any],
    intent_result: dict[str, Any],
    mood: str,
    debug: dict[str, Any],
) -> dict[str, Any] | None:
    """
    If the conversation is waiting for a particular piece of information,
    attempt to treat this message as that answer.

    Returns:
        Snapshot if a slot was successfully filled.
        None otherwise.
    """

    pending_slot = state.get("pending_slot")

    if not pending_slot:
        return None

    extractor = extraction.EXTRACTORS.get(pending_slot)

    slot_value: str | None = None

    # ------------------------------------------------------------
    # First: use the dedicated extractor.
    # ------------------------------------------------------------

    if extractor:
        try:
            slot_value = extractor(resolved_text)
        except Exception:
            logger.exception(
                "Slot extractor failed: slot=%s",
                pending_slot,
            )

    # ------------------------------------------------------------
    # Second: allow a short raw answer if NLU did not recognize
    # another meaningful intent.
    # ------------------------------------------------------------

    if not slot_value and not intent_result.get("accepted"):
        try:
            candidate = extraction._clean(raw_text)
        except Exception:
            candidate = raw_text.strip()

        if candidate and len(candidate.split()) <= 8:
            slot_value = candidate

    if not slot_value:
        return None

    slot_value = _safe_text(slot_value)

    if not slot_value:
        return None

    slot_intent = _slot_intent(pending_slot)

    _apply_slot_value(
        user_id,
        context,
        pending_slot,
        slot_value,
    )

    ack = _safe_respond(
        slot_intent,
        mood,
        value=slot_value,
    )

    try:
        next_slot = conversation.next_pending_slot(context)
    except Exception:
        logger.exception("Failed to calculate next pending slot")
        next_slot = None

    question = None

    if next_slot:
        try:
            question = conversation.followup_question_for(next_slot)
        except Exception:
            logger.exception(
                "Failed to generate follow-up question: slot=%s",
                next_slot,
            )

    reply = f"{ack} {question}".strip() if question else ack

    try:
        database.update_state(
            user_id,
            pending_slot=next_slot,
            last_intent=slot_intent,
        )
    except Exception:
        logger.exception("Failed to update pending slot state")

    debug["pending_slot"] = pending_slot
    debug["slot_value"] = slot_value
    debug["slot_answer"] = True

    _persist_turn(
        user_id,
        raw_text,
        reply,
        slot_intent,
        float(intent_result.get("confidence", DEFAULT_CONFIDENCE)),
        mood,
    )

    return _snapshot(
        user_id,
        reply,
        slot_intent,
        float(intent_result.get("confidence", DEFAULT_CONFIDENCE)),
        debug,
    )


# ============================================================================
# Intent handlers
# ============================================================================


def _handle_mood_change(
    user_id: int,
    resolved_text: str,
    mood: str,
) -> tuple[str, str]:
    """
    Determine and apply a new mood.
    """

    target: str | None = None

    for available_mood in personality.MOODS:
        if available_mood in resolved_text.lower():
            target = available_mood
            break

    lowered = resolved_text.lower()

    if target is None:
        if any(
            word in lowered
            for word in ("nicer", "kinder", "gentle", "gentler")
        ):
            target = "soft"

        elif any(
            word in lowered
            for word in ("meaner", "sarcastic", "harsh", "savage")
        ):
            target = "acid"

        else:
            target = "deadpan"

    if target not in personality.MOODS:
        target = personality.DEFAULT_MOOD

    database.update_state(
        user_id,
        mood=target,
    )

    reply = _safe_respond(
        "mood_change",
        target,
        mood_name=target,
    )

    return reply, target


def _handle_share_intent(
    user_id: int,
    context: dict[str, Any],
    resolved_text: str,
    intent: str,
    mood: str,
) -> str:
    """
    Handle context-sharing intents:

        project_share
        language_share
        feature_share
        subject_share
        game_share
        goal_share
    """

    slot = intent.removesuffix("_share")

    extractor = extraction.EXTRACTORS.get(slot)

    if not extractor:
        return _safe_respond(
            "unclear",
            mood,
        )

    try:
        value = extractor(resolved_text)
    except Exception:
        logger.exception(
            "Extractor failed: slot=%s",
            slot,
        )
        value = None

    if not value:
        matched = "that"

        return _safe_respond(
            intent,
            mood,
            value=matched,
        )

    _apply_slot_value(
        user_id,
        context,
        slot,
        value,
    )

    ack = _safe_respond(
        intent,
        mood,
        value=value,
    )

    try:
        next_slot = conversation.next_pending_slot(context)
    except Exception:
        logger.exception("Failed to calculate next slot")
        next_slot = None

    if not next_slot:
        try:
            database.update_state(
                user_id,
                pending_slot=None,
            )
        except Exception:
            logger.exception("Failed to clear pending slot")

        return ack

    try:
        question = conversation.followup_question_for(next_slot)
    except Exception:
        logger.exception("Failed to create follow-up question")
        question = None

    try:
        database.update_state(
            user_id,
            pending_slot=next_slot,
        )
    except Exception:
        logger.exception("Failed to update pending slot")

    return f"{ack} {question}".strip() if question else ack


def _handle_name_share(
    user_id: int,
    resolved_text: str,
    mood: str,
) -> str:
    """
    Store the user's name.
    """

    try:
        value = extraction.extract_name(resolved_text)
    except Exception:
        logger.exception("Name extractor failed")
        value = None

    if not value:
        return _safe_respond(
            "unclear",
            mood,
        )

    value = _safe_text(value)

    if not value:
        return _safe_respond(
            "unclear",
            mood,
        )

    database.set_user_name(
        user_id,
        value,
    )

    return _safe_respond(
        "name_share",
        mood,
        value=value,
    )


def _handle_coding_help(
    user_id: int,
    context: dict[str, Any],
    resolved_text: str,
    intent: str,
    mood: str,
) -> str:
    """
    Handle coding-help and problem-report intents.
    """

    project = context.get("project") or "your code"

    reply = _safe_respond(
        intent,
        mood,
        project=project,
    )

    try:
        problem_value = extraction.extract_problem(resolved_text)
    except Exception:
        logger.exception("Problem extractor failed")
        problem_value = None

    if problem_value:
        _apply_slot_value(
            user_id,
            context,
            "problem",
            problem_value,
        )

    return reply


def _handle_teach_fact(
    user_id: int,
    resolved_text: str,
    mood: str,
) -> str:
    """
    Handle both learned reply patterns and ordinary facts.
    """

    # ------------------------------------------------------------
    # Learned response:
    #
    # when i say hello reply hi there
    # ------------------------------------------------------------

    pattern_match = re.search(
        r"""
        when\s+i\s+say
        \s+
        (?P<trigger>.+?)
        \s+
        reply
        \s+
        (?P<response>.+)
        """,
        resolved_text,
        re.IGNORECASE | re.VERBOSE,
    )

    if pattern_match:
        try:
            trigger = extraction._clean(
                pattern_match.group("trigger")
            )

            response_text = extraction._clean(
                pattern_match.group("response")
            )
        except Exception:
            trigger = pattern_match.group("trigger").strip()
            response_text = pattern_match.group("response").strip()

        if trigger and response_text:
            database.add_learned_pattern(
                user_id,
                trigger,
                response_text,
            )

            return _safe_respond(
                "teach_fact",
                mood,
                value=f"'{trigger}' -> '{response_text}'",
            )

    # ------------------------------------------------------------
    # Ordinary fact:
    #
    # remember that my project is called SideEye
    # ------------------------------------------------------------

    try:
        taught = extraction.extract_taught_fact(
            resolved_text
        )
    except Exception:
        logger.exception("Taught-fact extractor failed")
        taught = None

    if taught:
        subject, fact_text = taught

        subject = _safe_text(subject)
        fact_text = _safe_text(fact_text)

        if subject and fact_text:
            database.add_fact(
                user_id,
                subject,
                fact_text,
            )

            return _safe_respond(
                "teach_fact",
                mood,
                value=fact_text,
            )

    return _safe_respond(
        "unclear",
        mood,
    )


def _handle_ask_about(
    user_id: int,
    context: dict[str, Any],
    resolved_text: str,
    mood: str,
) -> str:
    """
    Search known facts, current context, and recent messages.
    """

    match = _ASK_ABOUT_RE.search(
        resolved_text
    )

    subject = None

    if match:
        subject = _safe_text(
            match.group("subject")
        )

    search_term = _search_term(subject)

    # ------------------------------------------------------------
    # 1. Explicit facts
    # ------------------------------------------------------------

    facts: list[dict[str, Any]] = []

    if subject:
        try:
            facts = database.find_facts(
                user_id,
                subject,
            )
        except Exception:
            logger.exception(
                "Fact search failed: subject=%s",
                subject,
            )

    if (
        not facts
        and search_term
        and search_term != subject
    ):
        try:
            facts = database.find_facts(
                user_id,
                search_term,
            )
        except Exception:
            logger.exception(
                "Fallback fact search failed: term=%s",
                search_term,
            )

    if facts:
        fact_text = facts[0].get(
            "fact_text",
            "",
        )

        if fact_text:
            return _safe_respond(
                "ask_about_known",
                mood,
                value=fact_text,
            )

    # ------------------------------------------------------------
    # 2. Current context
    # ------------------------------------------------------------

    if (
        search_term
        and search_term in database.CONTEXT_KEYS
        and context.get(search_term)
    ):
        return _safe_respond(
            "ask_about_known",
            mood,
            value=(
                f"{search_term} is "
                f"{context[search_term]}"
            ),
        )

    # ------------------------------------------------------------
    # 3. Message history
    # ------------------------------------------------------------

    if search_term and len(search_term) >= 3:
        try:
            history_matches = database.search_messages(
                user_id,
                search_term,
            )
        except Exception:
            logger.exception(
                "Message history search failed: term=%s",
                search_term,
            )
            history_matches = []

        if history_matches:
            content = history_matches[0].get(
                "content",
                "",
            )

            if content:
                return _safe_respond(
                    "ask_about_recalled",
                    mood,
                    value=content,
                )

    # ------------------------------------------------------------
    # 4. Nothing found
    # ------------------------------------------------------------

    return _safe_respond(
        "ask_about",
        mood,
    )


# ============================================================================
# Main pipeline
# ============================================================================


def process_message(
    raw_text: str,
) -> dict[str, Any]:
    """
    Process one complete user message.

    The returned dictionary always follows this shape:

        {
            "reply": str,
            "mood": str,
            "intent": str,
            "confidence": float,
            "context": dict,
            "name": str | None,
            "suggestions": list[str],
            "debug": dict,
        }

    The function is intentionally resilient: a failure in persistence,
    reference resolution, or a personality template should not normally
    prevent the user from receiving a response.
    """

    user_id = USER_ID

    raw_text = _safe_text(raw_text)

    # ------------------------------------------------------------
    # Empty message
    # ------------------------------------------------------------

    if not raw_text:
        mood = _current_mood(user_id)

        reply = _safe_respond(
            "unclear",
            mood,
        )

        return _snapshot(
            user_id,
            reply,
            "unclear",
            0.0,
            {
                "empty_input": True,
            },
        )

    # ------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------

    command_match = _COMMAND_RE.match(
        raw_text
    )

    if command_match:
        command = command_match.group(
            "command"
        )

        args = command_match.group(
            "args"
        ) or ""

        try:
            database.log_message(
                user_id,
                "user",
                raw_text,
            )
        except Exception:
            logger.exception(
                "Failed to log command message"
            )

        try:
            result = _handle_command(
                user_id,
                command,
                args,
            )
        except Exception:
            logger.exception(
                "Command handler crashed: /%s",
                command,
            )

            mood = _current_mood(user_id)

            result = _snapshot(
                user_id,
                "Something went wrong while handling that command.",
                "unclear",
                0.0,
                {
                    "command": True,
                    "command_name": command,
                    "handler_error": True,
                },
            )

        _persist_command(
            user_id,
            raw_text,
            result,
        )

        return result

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------

    try:
        normalized = nlu.normalize(
            raw_text
        )
    except Exception:
        logger.exception(
            "NLU normalization failed"
        )
        normalized = raw_text.lower().strip()

    normalized = _safe_text(normalized)

    # ------------------------------------------------------------
    # Load conversation state
    # ------------------------------------------------------------

    try:
        context = database.get_context(
            user_id
        )
    except Exception:
        logger.exception(
            "Failed to load context"
        )
        context = {}

    try:
        entity_stack = database.get_recent_entities(
            user_id
        )
    except Exception:
        logger.exception(
            "Failed to load recent entities"
        )
        entity_stack = []

    try:
        state = database.get_state(
            user_id
        )
    except Exception:
        logger.exception(
            "Failed to load conversation state"
        )
        state = {
            "mood": personality.DEFAULT_MOOD,
            "pending_slot": None,
        }

    mood = state.get(
        "mood",
        personality.DEFAULT_MOOD,
    )

    if mood not in personality.MOODS:
        mood = personality.DEFAULT_MOOD

    # ------------------------------------------------------------
    # Resolve references
    # ------------------------------------------------------------

    try:
        ref = references.resolve_references(
            normalized,
            entity_stack,
        )
    except Exception:
        logger.exception(
            "Reference resolution failed"
        )

        ref = {
            "resolved_text": normalized,
            "replacements": [],
        }

    resolved_text = _safe_text(
        ref.get(
            "resolved_text",
            normalized,
        )
    )

    # ------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------

    try:
        intent_result = nlu.detect_intent(
            resolved_text
        )
    except Exception:
        logger.exception(
            "Intent detection failed"
        )

        intent_result = {
            "intent": "unclear",
            "confidence": 0.0,
            "accepted": False,
            "matched": [],
            "board": {},
        }

    intent = intent_result.get(
        "intent",
        "unclear",
    )

    confidence = intent_result.get(
        "confidence",
        DEFAULT_CONFIDENCE,
    )

    try:
        confidence = float(
            confidence
        )
    except (
        TypeError,
        ValueError,
    ):
        confidence = DEFAULT_CONFIDENCE

    confidence = max(
        0.0,
        min(1.0, confidence),
    )

    # ------------------------------------------------------------
    # Debug information
    # ------------------------------------------------------------

    board = intent_result.get(
        "board",
        {},
    )

    top_scores: list[tuple[str, float]] = []

    if isinstance(board, dict):
        for key, value in board.items():
            try:
                score = float(
                    value.get(
                        "score",
                        0.0,
                    )
                )
            except (
                AttributeError,
                TypeError,
                ValueError,
            ):
                continue

            top_scores.append(
                (
                    key,
                    round(score, 2),
                )
            )

    top_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    debug: dict[str, Any] = {
        "resolved_text": resolved_text,
        "reference_replacements": ref.get(
            "replacements",
            [],
        ),
        "top_scores": top_scores[:5],
    }

    # ------------------------------------------------------------
    # Pending question / slot answer
    # ------------------------------------------------------------

    pending_result = _try_pending_slot(
        user_id=user_id,
        raw_text=raw_text,
        resolved_text=resolved_text,
        context=context,
        state=state,
        intent_result=intent_result,
        mood=mood,
        debug=debug,
    )

    if pending_result is not None:
        return pending_result

    # ------------------------------------------------------------
    # Not confidently understood
    # ------------------------------------------------------------

    if not intent_result.get(
        "accepted",
        False,
    ):
        learned = None

        try:
            learned = database.find_learned_pattern(
                user_id,
                normalized,
            )
        except Exception:
            logger.exception(
                "Learned-pattern search failed"
            )

        if learned:
            reply = _safe_text(
                learned
            )
            final_intent = "learned_pattern"

        else:
            reply = _safe_respond(
                "unclear",
                mood,
            )
            final_intent = "unclear"

        debug["accepted"] = False

        _persist_turn(
            user_id,
            raw_text,
            reply,
            final_intent,
            confidence,
            mood,
        )

        return _snapshot(
            user_id,
            reply,
            final_intent,
            confidence,
            debug,
        )

    # =========================================================================
    # Intent dispatch
    # =========================================================================

    reply: str | None = None

    # ------------------------------------------------------------
    # Conversation basics
    # ------------------------------------------------------------

    if intent in {
        "greeting",
        "goodbye",
        "thanks",
        "help_command",
    }:
        reply = _safe_respond(
            intent,
            mood,
        )

    # ------------------------------------------------------------
    # Mood
    # ------------------------------------------------------------

    elif intent == "mood_change":
        try:
            reply, mood = _handle_mood_change(
                user_id,
                resolved_text,
                mood,
            )
        except Exception:
            logger.exception(
                "Mood handler failed"
            )

            reply = _safe_respond(
                "unclear",
                mood,
            )

    # ------------------------------------------------------------
    # Joke
    # ------------------------------------------------------------

    elif intent == "joke_request":
        reply = _safe_respond(
            "joke_request",
            mood,
            value=utils.random_joke(),
        )

    # ------------------------------------------------------------
    # Math
    # ------------------------------------------------------------

    elif intent == "math_request":
        try:
            calculation = utils.safe_math_eval(
                resolved_text
            )
        except Exception:
            logger.exception(
                "Math evaluation failed"
            )
            calculation = None

        if calculation:
            reply = _safe_respond(
                "math_request",
                mood,
                expression=calculation["expression"],
                result=calculation["result"],
            )

        else:
            reply = _safe_respond(
                "math_fail",
                mood,
            )

    # ------------------------------------------------------------
    # Dice
    # ------------------------------------------------------------

    elif intent == "dice_request":
        try:
            roll = utils.roll_dice(
                resolved_text
            )

            # Prefer the formatter from the improved utils.py.
            if hasattr(
                utils,
                "format_dice_result",
            ):
                reply = utils.format_dice_result(
                    roll
                )
            else:
                reply = _safe_respond(
                    "dice_request",
                    mood,
                    count=roll["count"],
                    sides=roll["sides"],
                    rolls=", ".join(
                        str(value)
                        for value in roll["rolls"]
                    ),
                    total=roll["total"],
                )

        except Exception:
            logger.exception(
                "Dice handler failed"
            )

            reply = _safe_respond(
                "unclear",
                mood,
            )

    # ------------------------------------------------------------
    # Coin flip
    # ------------------------------------------------------------

    elif intent == "coinflip_request":
        try:
            # If the message contains a number such as
            # "flip 10 coins", use the multi-flip helper.
            count_match = re.search(
                r"\b(\d{1,3})\s*(?:coins?|times?)\b",
                resolved_text,
                re.IGNORECASE,
            )

            count = (
                int(count_match.group(1))
                if count_match
                else 1
            )

            if hasattr(
                utils,
                "flip_coins",
            ) and hasattr(
                utils,
                "format_coin_result",
            ):
                results = utils.flip_coins(
                    count
                )

                reply = utils.format_coin_result(
                    results
                )

            else:
                reply = _safe_respond(
                    "coinflip_request",
                    mood,
                    value=utils.flip_coin(),
                )

        except Exception:
            logger.exception(
                "Coin flip handler failed"
            )

            reply = _safe_respond(
                "unclear",
                mood,
            )

    # ------------------------------------------------------------
    # Context-sharing intents
    # ------------------------------------------------------------

    elif intent in {
        "project_share",
        "language_share",
        "feature_share",
        "subject_share",
        "game_share",
        "goal_share",
    }:
        try:
            reply = _handle_share_intent(
                user_id,
                context,
                resolved_text,
                intent,
                mood,
            )
        except Exception:
            logger.exception(
                "Share-intent handler failed: intent=%s",
                intent,
            )

            reply = _safe_respond(
                "unclear",
                mood,
            )

    # ------------------------------------------------------------
    # Name
    # ------------------------------------------------------------

    elif intent == "name_share":
        try:
            reply = _handle_name_share(
                user_id,
                resolved_text,
                mood,
            )
        except Exception:
            logger.exception(
                "Name-share handler failed"
            )

            reply = _safe_respond(
                "unclear",
                mood,
            )

    # ------------------------------------------------------------
    # Coding / problem reports
    # ------------------------------------------------------------

    elif intent in {
        "coding_help",
        "problem_report",
    }:
        try:
            reply = _handle_coding_help(
                user_id,
                context,
                resolved_text,
                intent,
                mood,
            )
        except Exception:
            logger.exception(
                "Coding-help handler failed"
            )

            reply = _safe_respond(
                "unclear",
                mood,
            )

    # ------------------------------------------------------------
    # Teaching / learning
    # ------------------------------------------------------------

    elif intent == "teach_fact":
        try:
            reply = _handle_teach_fact(
                user_id,
                resolved_text,
                mood,
            )
        except Exception:
            logger.exception(
                "Teach-fact handler failed"
            )

            reply = _safe_respond(
                "unclear",
                mood,
            )

    # ------------------------------------------------------------
    # Memory lookup
    # ------------------------------------------------------------

    elif intent == "ask_about":
        try:
            reply = _handle_ask_about(
                user_id,
                context,
                resolved_text,
                mood,
            )
        except Exception:
            logger.exception(
                "Ask-about handler failed"
            )

            reply = _safe_respond(
                "ask_about",
                mood,
            )

    # ------------------------------------------------------------
    # Unknown recognized intent
    # ------------------------------------------------------------

    else:
        logger.warning(
            "No dispatcher for intent: %s",
            intent,
        )

        reply = _safe_respond(
            "unclear",
            mood,
        )

        intent = "unclear"

    # =========================================================================
    # Final persistence
    # =========================================================================

    if reply is None:
        reply = _safe_respond(
            "unclear",
            mood,
        )
        intent = "unclear"

    reply = _safe_text(reply)

    if not reply:
        reply = "I have absolutely no idea what just happened."

    _persist_turn(
        user_id=user_id,
        raw_text=raw_text,
        reply=reply,
        intent=intent,
        confidence=confidence,
        mood=mood,
    )

    return _snapshot(
        user_id,
        reply,
        intent,
        confidence,
        debug,
    )


# ============================================================================
# API helpers
# ============================================================================


def get_snapshot() -> dict[str, Any]:
    """
    Return the current conversation state.
    """

    user_id = USER_ID

    try:
        state = database.get_state(
            user_id
        )
    except Exception:
        logger.exception(
            "Failed to load state"
        )
        state = {}

    try:
        context = database.get_context(
            user_id
        )
    except Exception:
        logger.exception(
            "Failed to load context"
        )
        context = {}

    try:
        name = database.get_user_name(
            user_id
        )
    except Exception:
        logger.exception(
            "Failed to load user name"
        )
        name = None

    mood = state.get(
        "mood",
        personality.DEFAULT_MOOD,
    )

    if mood not in personality.MOODS:
        mood = personality.DEFAULT_MOOD

    return {
        "mood": mood,
        "context": context,
        "name": name,
        "turn_count": state.get(
            "turn_count",
            0,
        ),
    }


def set_mood(
    mood: str,
) -> dict[str, Any]:
    """
    Set the user's mood and return the updated snapshot.
    """

    mood = _safe_text(
        mood
    ).lower()

    if mood not in personality.MOODS:
        mood = personality.DEFAULT_MOOD

    try:
        database.update_state(
            USER_ID,
            mood=mood,
        )
    except Exception:
        logger.exception(
            "Failed to set mood"
        )

    return get_snapshot()


def get_history(
    limit: int = 50,
) -> list[Any]:
    """
    Return recent conversation history.

    The limit is clamped to prevent accidental huge database reads.
    """

    try:
        limit = int(limit)
    except (
        TypeError,
        ValueError,
    ):
        limit = 50

    limit = max(
        1,
        min(
            limit,
            MAX_HISTORY_LIMIT,
        ),
    )

    try:
        return database.get_recent_messages(
            USER_ID,
            limit=limit,
        )
    except Exception:
        logger.exception(
            "Failed to retrieve history"
        )
        return []


def reset_session() -> dict[str, Any]:
    """
    Completely reset the current user's session.
    """

    try:
        database.reset_all(
            USER_ID
        )
    except Exception:
        logger.exception(
            "Failed to reset session"
        )

    return get_snapshot()


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "process_message",
    "generate_suggestions",
    "get_snapshot",
    "set_mood",
    "get_history",
    "reset_session",
]
