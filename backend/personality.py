"""
personality.py
--------------

SideEye's response/personality engine.

SideEye has three moods:

    acid
        Mildly sarcastic, teasing, and judgmental.

    deadpan
        Calm, dry, professional, and slightly unimpressed.

    soft
        Supportive, friendly, and still playful.

The personality layer is intentionally separate from NLU, database,
conversation state, and extraction.

Pipeline:

    intent
       ↓
    personality.respond()
       ↓
    final response


Template values
---------------

Templates may use values supplied by the caller:

    {value}
    {name}
    {project}
    {language}
    {feature}
    {problem}
    {subject}
    {game}
    {goal}
    {expression}
    {result}
    {count}
    {sides}
    {rolls}
    {total}
    {mood_name}

Missing values never crash SideEye.

The personality system is deliberately forgiving because response
templates are UI text, not a full templating language.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping
from typing import Any


# ============================================================================
# MOODS
# ============================================================================

MOODS = (
    "acid",
    "deadpan",
    "soft",
)

DEFAULT_MOOD = "deadpan"


# User-friendly aliases.
MOOD_ALIASES = {
    "acid": "acid",
    "acidic": "acid",
    "sarcastic": "acid",
    "sarcasm": "acid",
    "sassy": "acid",
    "roast": "acid",
    "roasty": "acid",

    "deadpan": "deadpan",
    "dry": "deadpan",
    "neutral": "deadpan",
    "normal": "deadpan",
    "serious": "deadpan",

    "soft": "soft",
    "friendly": "soft",
    "nice": "soft",
    "supportive": "soft",
    "kind": "soft",
}


# ============================================================================
# TEMPLATE HELPERS
# ============================================================================

class _SafeDict(dict[str, Any]):
    """
    Dictionary used by str.format_map().

    Missing template values become an empty string instead of raising
    KeyError.
    """

    def __missing__(self, key: str) -> str:
        return ""


# Matches placeholders such as:
#
#     {name}
#     {project}
#     {result}
#
# It intentionally does not try to implement a complete format language.
_PLACEHOLDER_RE = re.compile(
    r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
)


def _clean_value(value: Any) -> str:
    """Convert a template value into clean display text."""
    if value is None:
        return ""

    return str(value).strip()


def _clean_rendered_text(text: str) -> str:
    """
    Clean whitespace and obvious punctuation artifacts caused by missing
    template values.
    """
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Collapse normal whitespace.
    text = " ".join(text.split())

    # Remove spaces before punctuation.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # Avoid repeated punctuation caused by optional values.
    text = re.sub(r"([!?.,]){3,}", r"\1\1", text)

    # Avoid accidental double spaces.
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def render(
    template: str,
    **values: Any,
) -> str:
    """
    Safely render one response template.

    Missing placeholders are replaced with empty strings.
    """
    if not template:
        return ""

    cleaned_values = {
        key: _clean_value(value)
        for key, value in values.items()
    }

    try:
        text = template.format_map(
            _SafeDict(cleaned_values)
        )
    except (ValueError, IndexError):
        # A malformed template should never take down the chatbot.
        text = template

        for match in _PLACEHOLDER_RE.finditer(template):
            key = match.group(1)
            text = text.replace(
                match.group(0),
                cleaned_values.get(key, ""),
            )

    return _clean_rendered_text(text)


# ============================================================================
# RESPONSE TEMPLATES
# ============================================================================

TEMPLATES: dict[str, dict[str, list[str]]] = {

    # ---------------------------------------------------------------- greeting

    "greeting": {
        "acid": [
            "Oh, you're back. Thrilling.",
            "Hey. Try not to break anything today.",
            "Look who decided to show up.",
            "Hello again. What are we causing problems with?",
        ],
        "deadpan": [
            "Hello. I am ready.",
            "Hi there. State your business.",
            "Hello. What can I help with?",
            "Greetings. What are we working on?",
        ],
        "soft": [
            "Hey! Good to see you.",
            "Hi! What are we working on today?",
            "Hey, welcome back. What's up?",
            "Good to see you. What are we building?",
        ],
    },

    # ---------------------------------------------------------------- goodbye

    "goodbye": {
        "acid": [
            "Finally. Go touch grass.",
            "Leaving already? Bold of you.",
            "Alright, escape while you still can.",
            "Fine. I'll survive without you.",
        ],
        "deadpan": [
            "Goodbye. Session logged.",
            "Understood. Ending this turn.",
            "Goodbye. Conversation paused.",
            "Until next time.",
        ],
        "soft": [
            "Bye for now! I'll remember where we left off.",
            "Take care — I'll be here.",
            "See you later!",
            "Catch you next time. Good luck with the project!",
        ],
    },

    # ---------------------------------------------------------------- thanks

    "thanks": {
        "acid": [
            "Don't get used to it.",
            "I know, I'm a delight.",
            "You're welcome. Try not to make this a habit.",
            "Obviously. I'm excellent at this.",
        ],
        "deadpan": [
            "Acknowledged.",
            "You're welcome. Moving on.",
            "No problem.",
            "Assistance provided.",
        ],
        "soft": [
            "Anytime!",
            "Happy to help!",
            "Of course!",
            "Glad I could help.",
        ],
    },

    # ----------------------------------------------------------- help command

    "help_command": {
        "acid": [
            (
                "Fine, here's the tour: I track your project, language, "
                "features, goals, and other context; understand follow-ups "
                "like 'it' and 'them'; roll dice, flip coins, do math, tell "
                "jokes, learn facts, and switch moods. Try to keep up."
            ),
        ],
        "deadpan": [
            (
                "Capabilities: context memory, reference resolution, "
                "intent detection, dice, coin flips, arithmetic, jokes, "
                "mood switching, learned facts, and learned response "
                "patterns."
            ),
        ],
        "soft": [
            (
                "I can keep track of what you're building, understand "
                "follow-ups like 'it' or 'them', do quick math, roll dice, "
                "flip coins, tell jokes, remember facts, learn custom "
                "responses, and switch personality modes."
            ),
        ],
    },

    # ------------------------------------------------------------ mood change

    "mood_change": {
        "acid": [
            "Ugh, fine. {mood_name} mode. Happy now?",
            "Switching to {mood_name}. This should be fun.",
            "Alright. {mood_name} mode activated.",
        ],
        "deadpan": [
            "Mood set to {mood_name}.",
            "Personality parameter updated: {mood_name}.",
            "Mode changed to {mood_name}.",
        ],
        "soft": [
            "Switched to {mood_name} mode for you!",
            "Okay, {mood_name} it is.",
            "Got it. We're going with {mood_name} mode!",
        ],
    },

    # ------------------------------------------------------------ jokes

    "joke_request": {
        "acid": [
            "Fine, one joke. Don't laugh too hard: {value}",
            "{value} ...you're welcome.",
            "Here's your comedy allowance: {value}",
        ],
        "deadpan": [
            "Here: {value}",
            "Requested joke: {value}",
            "Comedy module returned: {value}",
        ],
        "soft": [
            "Here's one for you: {value}",
            "Hope this gets a smile: {value}",
            "Okay, try this one: {value}",
        ],
    },

    # ------------------------------------------------------------- math

    "math_request": {
        "acid": [
            "{expression} = {result}. Yes, I did that in my head. Sort of.",
            "It's {result}. Try a calculator next time, I'm not your app.",
            "{expression} = {result}. Humanity survives another equation.",
        ],
        "deadpan": [
            "{expression} = {result}.",
            "Result: {result}.",
            "Calculated: {result}.",
        ],
        "soft": [
            "That comes out to {result}!",
            "{expression} = {result} — nice and easy.",
            "The answer is {result}!",
        ],
    },

    "math_fail": {
        "acid": [
            "I can't parse that into math. Try numbers and an operator, genius.",
            "That wasn't math. That was vibes.",
            "I asked for arithmetic, not abstract art.",
        ],
        "deadpan": [
            "Could not parse that into a valid mathematical expression.",
            "No parseable arithmetic found.",
            "I couldn't extract a valid expression from that.",
        ],
        "soft": [
            (
                "I couldn't quite pull out the numbers there — "
                "try something like '12 times 4'?"
            ),
            "I couldn't parse that one. Try writing the calculation more clearly.",
        ],
    },

    # ------------------------------------------------------------- dice

    "dice_request": {
        "acid": [
            (
                "Rolled {count}d{sides}: {rolls} for a total of "
                "{total}. Riveting."
            ),
            "{total}. The dice have judged you.",
            "The dice say {total}. I wouldn't argue with them.",
        ],
        "deadpan": [
            "Roll result ({count}d{sides}): {rolls}. Total: {total}.",
            "Total: {total}.",
            "{count}d{sides} → {rolls} → {total}.",
        ],
        "soft": [
            "You rolled {rolls} — that's {total} total! Nice.",
            "Rolled it! Total: {total}.",
            "Your dice came up {rolls}. That's {total} altogether!",
        ],
    },

    # ----------------------------------------------------------- coin flip

    "coinflip_request": {
        "acid": [
            "{value}. Groundbreaking stuff.",
            "It landed on {value}. Riveting, I'm sure.",
            "{value}. The universe has spoken. Apparently.",
        ],
        "deadpan": [
            "Result: {value}.",
            "Coin landed on {value}.",
            "The result is {value}.",
        ],
        "soft": [
            "It's {value}!",
            "Landed on {value} — your call!",
            "The coin says {value}!",
        ],
    },

    # -------------------------------------------------------------- project

    "project_share": {
        "acid": [
            "A {value}, huh? Bold choice.",
            "Noted: {value}. Let's see if it survives.",
            "{value}. Ambitious. Potentially dangerous.",
        ],
        "deadpan": [
            "Project noted: {value}.",
            "Logged. Project: {value}.",
            "Project recorded: {value}.",
        ],
        "soft": [
            "That sounds like a fun project — {value}!",
            "Nice, I'll remember you're building {value}.",
            "Cool project. I'll keep {value} in mind.",
        ],
    },

    # ------------------------------------------------------------ language

    "language_share": {
        "acid": [
            "{value}. Of course.",
            "Ah, {value}. Brave.",
            "{value}? Interesting choice.",
        ],
        "deadpan": [
            "Language noted: {value}.",
            "Logged. Language: {value}.",
            "Language recorded: {value}.",
        ],
        "soft": [
            "{value} is a solid choice!",
            "Got it — {value} it is.",
            "Nice. I'll remember you're using {value}.",
        ],
    },

    # -------------------------------------------------------------- feature

    "feature_share": {
        "acid": [
            "{value}. Sure, why not.",
            "Adding {value}. This'll be interesting.",
            "{value}. Because apparently we needed another thing.",
        ],
        "deadpan": [
            "Feature noted: {value}.",
            "Logged. Feature: {value}.",
            "Feature recorded: {value}.",
        ],
        "soft": [
            "{value} sounds like good progress!",
            "Nice — {value} is a solid next step.",
            "Got it. I'll keep {value} in context.",
        ],
    },

    # ------------------------------------------------------------- subject

    "subject_share": {
        "acid": [
            "Studying {value}? Look at you, being productive.",
            "{value}. Riveting subject choice.",
            "Ah yes, {value}. Education strikes again.",
        ],
        "deadpan": [
            "Subject noted: {value}.",
            "Logged. Subject: {value}.",
            "Study subject recorded: {value}.",
        ],
        "soft": [
            "{value} is a great thing to learn!",
            "Nice, I'll remember you're studying {value}.",
            "Got it. I'll keep {value} in mind.",
        ],
    },

    # ---------------------------------------------------------------- game

    "game_share": {
        "acid": [
            "{value}. Try not to rage quit.",
            "Noted: {value}. We'll see how that goes.",
            "{value}. Excellent excuse to ignore productivity.",
        ],
        "deadpan": [
            "Game noted: {value}.",
            "Logged. Game: {value}.",
            "Game recorded: {value}.",
        ],
        "soft": [
            "{value} sounds fun!",
            "Got it — {value}.",
            "Nice. I'll remember you're playing {value}.",
        ],
    },

    # ---------------------------------------------------------------- goal

    "goal_share": {
        "acid": [
            "Ambitious. Noted: {value}.",
            "Sure, {value}. Good luck with that.",
            "{value}. Bold. Let's see if future-you agrees.",
        ],
        "deadpan": [
            "Goal noted: {value}.",
            "Logged. Goal: {value}.",
            "Goal recorded: {value}.",
        ],
        "soft": [
            "That's a great goal — {value}!",
            "I'll remember that: {value}.",
            "Nice goal. One step at a time — {value}.",
        ],
    },

    # ---------------------------------------------------------------- name

    "name_share": {
        "acid": [
            "Fine, {value}. I'll allow it.",
            "Noted, {value}. Don't make me regret it.",
            "Alright, {value}. Try not to become too memorable.",
        ],
        "deadpan": [
            "Name noted: {value}.",
            "Logged. Name: {value}.",
            "Understood. I'll call you {value}.",
        ],
        "soft": [
            "Nice to meet you, {value}!",
            "Got it, {value} — good to know.",
            "Nice to know your name, {value}!",
        ],
    },

    # ----------------------------------------------------------- coding help

    "coding_help": {
        "acid": [
            "Ah yes, {project} betraying you again. What's actually breaking?",
            "Code trouble. Shocking. What's the error say?",
            "Something's on fire. What exactly is going wrong?",
            "Your code has chosen violence. What's the error?",
        ],
        "deadpan": [
            (
                "Describe the failure: what did you expect, "
                "and what happened instead?"
            ),
            "State the error message or symptom.",
            "What is the code doing incorrectly?",
        ],
        "soft": [
            "Sorry that's giving you trouble — what's the error message?",
            "Let's untangle it together. What exactly is happening?",
            "No worries. Show me the error or describe what's going wrong.",
        ],
    },

    # --------------------------------------------------------- problem report

    "problem_report": {
        "acid": [
            "Something's broken. Groundbreaking. What is 'it', exactly?",
            "Broken how? Give me specifics, not vibes.",
            "Okay, something exploded. What exactly happened?",
        ],
        "deadpan": [
            "Specify what is failing and how.",
            (
                "What is the observed behavior versus "
                "the expected one?"
            ),
            "Describe the problem and what you expected to happen.",
        ],
        "soft": [
            "Let's figure it out — what's it doing that it shouldn't?",
            "No worries, that happens. What's going wrong exactly?",
            "Tell me what happened and we'll work through it.",
        ],
    },

    # ------------------------------------------------------------- ask about

    "ask_about": {
        "acid": [
            (
                "I'm a local chatbot, not a search engine. I've got no idea, "
                "and neither does my SQLite file."
            ),
            (
                "No clue. I only know what you've told me — "
                "this isn't Wikipedia."
            ),
        ],
        "deadpan": [
            (
                "I do not have general knowledge access. "
                "I only recall what you have told me."
            ),
            "Unknown. My knowledge is limited to what you've taught me directly.",
        ],
        "soft": [
            (
                "I don't actually have general knowledge like that — "
                "I only know what you've shared with me. Want to teach me?"
            ),
            (
                "I can't look that up, but if you tell me, "
                "I'll remember it!"
            ),
        ],
    },

    # -------------------------------------------------------- known answer

    "ask_about_known": {
        "acid": [
            "You told me this already: {value}. Pay attention.",
            "According to you: {value}.",
            "My records say: {value}. You gave me that information.",
        ],
        "deadpan": [
            "Recorded answer: {value}.",
            "You previously stated: {value}.",
            "From stored memory: {value}.",
        ],
        "soft": [
            "You mentioned this before: {value}!",
            "From what you told me: {value}.",
            "I remember you telling me: {value}!",
        ],
    },

    # ------------------------------------------------------ recalled message

    "ask_about_recalled": {
        "acid": [
            (
                'You literally said this earlier: "{value}". '
                "Try to keep up."
            ),
            'Pulled from the chat log: "{value}". You said it, not me.',
        ],
        "deadpan": [
            'Found it in the chat log: "{value}".',
            'Earlier, you said: "{value}".',
            'Conversation history says: "{value}".',
        ],
        "soft": [
            'You mentioned this earlier: "{value}"!',
            'I found it — you said: "{value}".',
            'Yep, I found that in our earlier conversation: "{value}".',
        ],
    },

    # -------------------------------------------------------------- teach

    "teach_fact": {
        "acid": [
            "Fine, filed away: {value}. Don't quiz me later.",
            "Noted: {value}. I'll pretend to care.",
            "Stored. Apparently this is important now: {value}.",
        ],
        "deadpan": [
            "Fact stored: {value}.",
            "Recorded: {value}.",
            "Memory updated: {value}.",
        ],
        "soft": [
            "Got it, I'll remember: {value}!",
            "Thanks for teaching me that: {value}.",
            "Got it. I'll keep that in mind!",
        ],
    },

    # ---------------------------------------------------------------- unclear

    "unclear": {
        "acid": [
            "I have genuinely no idea what to do with that.",
            "That didn't parse as anything. Try again, clearer this time.",
            "Cool sentence. Meant nothing to me though.",
            "You've successfully confused the machine.",
        ],
        "deadpan": [
            "Intent not recognized.",
            "Unable to classify that message.",
            "No matching intent found.",
            "I need a little more information.",
        ],
        "soft": [
            "I'm not quite following — could you rephrase that?",
            "Hmm, not sure I caught that one. Try saying it differently?",
            "I didn't quite understand that. Give me another shot?",
            "I'm listening — could you explain that a little differently?",
        ],
    },
}


# ============================================================================
# MOOD FUNCTIONS
# ============================================================================

def get_mood(
    current_mood: str | None,
) -> str:
    """
    Normalize a mood name.

    Examples:

        get_mood("acid")       -> "acid"
        get_mood("sarcastic")  -> "acid"
        get_mood("friendly")   -> "soft"
        get_mood("something")  -> "deadpan"
    """
    if current_mood is None:
        return DEFAULT_MOOD

    normalized = str(current_mood).strip().casefold()

    return MOOD_ALIASES.get(
        normalized,
        DEFAULT_MOOD,
    )


def is_valid_mood(
    mood: str | None,
) -> bool:
    """Return True if the supplied mood is a recognized mood/alias."""
    if mood is None:
        return False

    normalized = str(mood).strip().casefold()

    return normalized in MOOD_ALIASES


def get_available_moods() -> tuple[str, ...]:
    """Return the canonical mood names."""
    return MOODS


# ============================================================================
# INTENT HELPERS
# ============================================================================

def has_intent(
    intent: str,
) -> bool:
    """Return True when an intent has response templates."""
    return bool(
        intent
        and intent in TEMPLATES
    )


def get_available_intents() -> tuple[str, ...]:
    """Return all intents with personality templates."""
    return tuple(TEMPLATES.keys())


def get_templates(
    intent: str,
    mood: str | None = None,
) -> list[str]:
    """
    Return a copy of the templates for an intent/mood.

    Returning a copy prevents callers from accidentally modifying the
    global template dictionary.
    """
    if not intent:
        intent = "unclear"

    bucket = TEMPLATES.get(
        intent,
        TEMPLATES["unclear"],
    )

    normalized_mood = get_mood(mood)

    templates = (
        bucket.get(normalized_mood)
        or bucket.get(DEFAULT_MOOD)
        or next(iter(bucket.values()), [])
    )

    return list(templates)


# ============================================================================
# RESPONSE SELECTION
# ============================================================================

def random_response(
    intent: str,
    mood: str | None = None,
) -> str:
    """
    Return one raw template selected randomly.

    Usually callers should use respond() instead because respond() also
    renders template values.
    """
    templates = get_templates(
        intent,
        mood,
    )

    if not templates:
        return ""

    return random.choice(templates)


def respond(
    intent: str,
    mood: str | None = None,
    **values: Any,
) -> str:
    """
    Generate a personality-aware response.

    Unknown intents fall back to "unclear".

    Invalid moods fall back to the default deadpan mood.

    Missing template values never crash the application.
    """
    if not intent:
        intent = "unclear"

    normalized_mood = get_mood(mood)

    templates = get_templates(
        intent,
        normalized_mood,
    )

    if not templates:
        templates = get_templates(
            "unclear",
            normalized_mood,
        )

    if not templates:
        return "I don't know how to respond to that."

    template = random.choice(templates)

    # Always make the active mood available to templates.
    render_values = dict(values)
    render_values.setdefault(
        "mood",
        normalized_mood,
    )
    render_values.setdefault(
        "mood_name",
        normalized_mood,
    )

    response = render(
        template,
        **render_values,
    )

    # Final safety fallback.
    if not response:
        return "I don't know how to respond to that."

    return response


# ============================================================================
# DEBUG / INSPECTION HELPERS
# ============================================================================

def describe_personality() -> dict[str, Any]:
    """
    Return a compact description of the personality system.

    Useful for debugging, help commands, or a future UI.
    """
    intent_count = len(TEMPLATES)

    template_count = sum(
        len(template_list)
        for mood_bucket in TEMPLATES.values()
        for template_list in mood_bucket.values()
    )

    return {
        "moods": list(MOODS),
        "default_mood": DEFAULT_MOOD,
        "intent_count": intent_count,
        "template_count": template_count,
        "intents": list(TEMPLATES.keys()),
    }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "MOODS",
    "MOOD_ALIASES",
    "DEFAULT_MOOD",
    "TEMPLATES",

    "render",
    "get_mood",
    "is_valid_mood",
    "get_available_moods",

    "has_intent",
    "get_available_intents",
    "get_templates",

    "random_response",
    "respond",

    "describe_personality",
]