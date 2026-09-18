"""
conversation.py
---------------

Conversation state and context management for SideEye.

Responsibilities:
1. Track the active conversation topic.
2. Track project/language/feature/problem/goal context.
3. Decide what information is still missing.
4. Manage pending follow-up questions.
5. Detect topic switches and corrections.
6. Resolve whether a message is answering a pending question.
7. Keep short-term conversational history.
8. Track confidence and timestamps for context.
9. Prevent stale context from hijacking unrelated messages.

This module intentionally contains NO AI/LLM logic.
Everything is deterministic and local.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

SLOT_FLOW = [
    "project",
    "language",
    "feature",
]

# Slots that can exist in conversation context even if they aren't
# part of the basic follow-up flow.
KNOWN_SLOTS = {
    "project",
    "language",
    "feature",
    "problem",
    "goal",
    "subject",
    "game",
}

FOLLOWUP_QUESTIONS = {
    "language": "What language are you using for that?",
    "feature": "Nice. What feature are you working on right now?",
}

MAX_HISTORY = 12

# How long a topic remains strongly relevant.
TOPIC_DECAY_SECONDS = 60 * 15


# ============================================================
# DATA TYPES
# ============================================================


@dataclass
class ContextValue:
    """A value stored in conversation context."""

    value: Any
    confidence: float = 1.0
    updated_at: float = field(default_factory=time)
    source: str = "user"

    def is_stale(self) -> bool:
        """Return True when this context value is old enough to decay."""
        return (time() - self.updated_at) > TOPIC_DECAY_SECONDS


@dataclass
class ConversationState:
    """
    Stores the user's current conversational state.

    This is intentionally small and serializable so it can easily
    be passed through the existing pipeline/database system.
    """

    context: dict[str, ContextValue] = field(default_factory=dict)

    current_topic: str | None = None
    previous_topic: str | None = None

    pending_slot: str | None = None

    history: list[dict[str, Any]] = field(default_factory=list)

    last_intent: str | None = None
    last_message: str | None = None

    topic_confidence: float = 0.0

    # Whether the user explicitly changed/corrected context.
    last_was_correction: bool = False
    last_was_topic_switch: bool = False


# ============================================================
# CONTEXT MANAGEMENT
# ============================================================


def set_context(
    state: ConversationState,
    slot: str,
    value: Any,
    *,
    confidence: float = 1.0,
    source: str = "user",
) -> None:
    """
    Store or update a context value.

    Existing values are replaced when the new value is more recent.
    """

    if slot not in KNOWN_SLOTS:
        return

    if value is None:
        return

    if isinstance(value, str):
        value = value.strip()

    if not value:
        return

    state.context[slot] = ContextValue(
        value=value,
        confidence=max(0.0, min(1.0, confidence)),
        source=source,
    )


def get_context(
    state: ConversationState,
    slot: str,
    default: Any = None,
) -> Any:
    """Return a stored context value."""

    item = state.context.get(slot)

    if item is None:
        return default

    return item.value


def get_context_confidence(
    state: ConversationState,
    slot: str,
) -> float:
    """Return confidence for a stored context value."""

    item = state.context.get(slot)

    if item is None:
        return 0.0

    return item.confidence


def clear_context(
    state: ConversationState,
    slot: str,
) -> None:
    """Remove a single context value."""

    state.context.pop(slot, None)


def clear_all_context(state: ConversationState) -> None:
    """Reset all conversational context."""

    state.context.clear()

    state.current_topic = None
    state.previous_topic = None
    state.pending_slot = None

    state.last_intent = None
    state.last_message = None

    state.topic_confidence = 0.0

    state.last_was_correction = False
    state.last_was_topic_switch = False


# ============================================================
# TOPIC MANAGEMENT
# ============================================================


def set_topic(
    state: ConversationState,
    topic: str | None,
    *,
    confidence: float = 1.0,
) -> None:
    """
    Set the current conversation topic.

    The old topic is preserved as previous_topic.
    """

    if not topic:
        return

    topic = str(topic).strip()

    if not topic:
        return

    if state.current_topic != topic:
        state.previous_topic = state.current_topic
        state.current_topic = topic

    state.topic_confidence = max(
        0.0,
        min(1.0, confidence),
    )


def switch_topic(
    state: ConversationState,
    topic: str | None,
) -> None:
    """
    Explicitly switch conversation topic.

    Unlike set_topic(), this makes the topic switch obvious.
    """

    state.last_was_topic_switch = True

    if topic:
        set_topic(state, topic)

    # A topic switch should not leave an old follow-up question
    # hanging around.
    state.pending_slot = None


def topic_is_active(state: ConversationState) -> bool:
    """Return whether the current topic is still reasonably fresh."""

    if not state.current_topic:
        return False

    if state.topic_confidence <= 0:
        return False

    return True


# ============================================================
# CORRECTIONS
# ============================================================


def correct_context(
    state: ConversationState,
    slot: str,
    new_value: Any,
    *,
    confidence: float = 1.0,
) -> None:
    """
    Replace an existing context value with a corrected value.

    The old value isn't retained here; database/history code can
    preserve it if historical auditing is required.
    """

    set_context(
        state,
        slot,
        new_value,
        confidence=confidence,
        source="correction",
    )

    state.last_was_correction = True


# ============================================================
# FOLLOW-UP QUESTIONS
# ============================================================


def missing_slots(
    state: ConversationState,
) -> list[str]:
    """
    Return all slots that are currently missing.

    Project is required before the normal follow-up flow begins.
    """

    project = get_context(state, "project")

    if not project:
        return []

    return [
        slot
        for slot in SLOT_FLOW
        if not get_context(state, slot)
    ]


def next_pending_slot(
    state: ConversationState | dict,
) -> str | None:
    """
    Return the next missing slot.

    Supports both the new ConversationState object and the old
    dictionary-based context for backwards compatibility.
    """

    if isinstance(state, dict):
        context = state

        if not context.get("project"):
            return None

        for slot in SLOT_FLOW:
            if not context.get(slot):
                return slot

        return None

    missing = missing_slots(state)

    if not missing:
        state.pending_slot = None
        return None

    state.pending_slot = missing[0]

    return state.pending_slot


def followup_question_for(slot: str) -> str | None:
    """Return the follow-up question for a slot."""

    return FOLLOWUP_QUESTIONS.get(slot)


def get_next_followup(
    state: ConversationState,
) -> tuple[str, str] | None:
    """
    Return:

        (slot, question)

    or None when no follow-up is needed.
    """

    slot = next_pending_slot(state)

    if slot is None:
        return None

    question = followup_question_for(slot)

    if question is None:
        return None

    return slot, question


# ============================================================
# PENDING ANSWERS
# ============================================================


def has_pending_question(
    state: ConversationState,
) -> bool:
    """Return whether SideEye is waiting for a specific answer."""

    return state.pending_slot is not None


def clear_pending_question(
    state: ConversationState,
) -> None:
    """Clear the current pending question."""

    state.pending_slot = None


def pending_answer_matches(
    state: ConversationState,
    slot: str,
) -> bool:
    """
    Check whether a slot is currently being requested.

    This does not extract the value itself.
    Extraction remains the responsibility of extraction.py.
    """

    return state.pending_slot == slot


def accept_pending_answer(
    state: ConversationState,
    slot: str,
    value: Any,
    *,
    confidence: float = 1.0,
) -> bool:
    """
    Accept a value as the answer to the pending question.

    Returns True when accepted.
    """

    if not pending_answer_matches(state, slot):
        return False

    set_context(
        state,
        slot,
        value,
        confidence=confidence,
        source="pending_answer",
    )

    clear_pending_question(state)

    return True


# ============================================================
# MESSAGE HISTORY
# ============================================================


def add_history(
    state: ConversationState,
    role: str,
    message: str,
    *,
    intent: str | None = None,
) -> None:
    """Add a message to short-term conversation history."""

    if not message:
        return

    state.history.append(
        {
            "role": role,
            "message": message,
            "intent": intent,
            "timestamp": time(),
        }
    )

    if len(state.history) > MAX_HISTORY:
        del state.history[:-MAX_HISTORY]


def recent_history(
    state: ConversationState,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Return the most recent conversation messages."""

    if limit <= 0:
        return []

    return state.history[-limit:]


# ============================================================
# MESSAGE STATE
# ============================================================


def update_message_state(
    state: ConversationState,
    message: str,
    *,
    intent: str | None = None,
) -> None:
    """Record the latest message and intent."""

    state.last_message = message
    state.last_intent = intent

    # These flags describe the previous message only.
    # Reset them when processing the next normal message.
    state.last_was_correction = False
    state.last_was_topic_switch = False


# ============================================================
# SNAPSHOT / DEBUGGING
# ============================================================


def context_snapshot(
    state: ConversationState,
) -> dict[str, Any]:
    """
    Return a clean serializable snapshot of the current state.

    Useful for:
    - debug mode
    - frontend context panel
    - database persistence
    """

    return {
        "current_topic": state.current_topic,
        "previous_topic": state.previous_topic,
        "topic_confidence": state.topic_confidence,
        "pending_slot": state.pending_slot,
        "last_intent": state.last_intent,
        "last_message": state.last_message,
        "context": {
            slot: {
                "value": item.value,
                "confidence": item.confidence,
                "source": item.source,
                "updated_at": item.updated_at,
                "stale": item.is_stale(),
            }
            for slot, item in state.context.items()
        },
        "history_size": len(state.history),
    }


# ============================================================
# BACKWARDS-COMPATIBILITY HELPERS
# ============================================================


def context_to_dict(
    state: ConversationState,
) -> dict[str, Any]:
    """Convert ConversationState into the old simple context format."""

    return {
        slot: item.value
        for slot, item in state.context.items()
    }


def dict_to_state(
    context: dict[str, Any],
) -> ConversationState:
    """Create a ConversationState from the old dictionary format."""

    state = ConversationState()

    for slot, value in context.items():
        if slot in KNOWN_SLOTS and value:
            set_context(state, slot, value)

    return state