"""
nlu.py
------
SideEye's lightweight natural-language understanding layer.

The NLU system does not try to be a full NLP engine. Instead, it combines:

    - keywords
    - weak keywords
    - phrases
    - related expressions
    - negative signals
    - intent priorities
    - contextual hints
    - confidence scoring

This makes SideEye much better at understanding casual messages without
requiring a heavyweight NLP dependency.

Public API
----------
    normalize(text)
    tokenize(text)
    score_intent(text, definition)
    detect_intent(text)
    explain_intent(text)

The structure intentionally remains compatible with the rest of SideEye.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

_WORD_RE = re.compile(r"[a-z0-9+#']+", re.IGNORECASE)

_WHITESPACE_RE = re.compile(r"\s+")

_APOSTROPHE_RE = re.compile(r"[’`]")


def normalize(text: str) -> str:
    """
    Normalize user text into a predictable form.

    Examples:
        "  I'M Building Python!! "
            -> "i'm building python!!"

        "what's   up?"
            -> "what's up?"
    """
    if not text:
        return ""

    text = str(text)

    # Normalize curly/backtick apostrophes.
    text = _APOSTROPHE_RE.sub("'", text)

    # Normalize whitespace.
    text = _WHITESPACE_RE.sub(" ", text)

    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    """Return simple word-like tokens."""
    return _WORD_RE.findall(normalize(text))


# ============================================================================
# MATCHING HELPERS
# ============================================================================

def _contains_word(text: str, word: str) -> bool:
    """
    Check whether a complete word/token exists.

    This prevents things like:

        "java" matching "javascript"
        "go" matching "good"
    """
    word = normalize(word)

    if not word:
        return False

    pattern = (
        r"(?<![a-z0-9+#'])"
        + re.escape(word)
        + r"(?![a-z0-9+#'])"
    )

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _contains_phrase(text: str, phrase: str) -> bool:
    """
    Check whether a phrase occurs as a meaningful phrase.

    Unlike a raw `phrase in text`, this avoids matching fragments inside
    unrelated words.
    """
    phrase = normalize(phrase)

    if not phrase:
        return False

    # Phrases containing special characters such as "C++" still work.
    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    """Count meaningful occurrences of a phrase."""
    phrase = normalize(phrase)

    if not phrase:
        return 0

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return len(
        re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )


# ============================================================================
# LANGUAGE ALIASES
# ============================================================================

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

    "cpp": "c++",
    "c plus plus": "c++",

    "csharp": "c#",
    "c sharp": "c#",
    "cs": "c#",

    "golang": "go",

    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
}


# ============================================================================
# INTENT DEFINITION
# ============================================================================

@dataclass
class IntentDef:
    """
    Definition for one SideEye intent.

    keywords:
        Strong single-word indicators.

    weak_keywords:
        Ambiguous words that should only contribute a little.

    phrases:
        Strong multi-word indicators.

    related:
        Looser synonyms or related terminology.

    negatives:
        Signals that reduce confidence.

    priority:
        Tie-breaking weight for intents that are naturally more specific.

    digit_sensitive:
        Whether numbers should provide an additional signal.
    """

    keywords: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    weak_keywords: list[str] = field(default_factory=list)

    priority: float = 0.0
    digit_sensitive: bool = False


# ============================================================================
# SCORING CONSTANTS
# ============================================================================

KEYWORD_WEIGHT = 1.0

WEAK_KEYWORD_WEIGHT = 0.45

PHRASE_WEIGHT = 2.7

RELATED_WEIGHT = 0.65

NEGATIVE_WEIGHT = 1.8

DIGIT_BONUS = 1.35

# Additional bonus when a phrase is very specific.
SPECIFIC_PHRASE_BONUS = 0.35

# Intent must reach this to be accepted.
ACCEPT_THRESHOLD = 0.95

# Below this, the message is considered unclear.
FLOOR_THRESHOLD = 0.5

# If the top two intents are extremely close, confidence is reduced.
AMBIGUITY_GAP = 0.55

_HAS_DIGIT_RE = re.compile(r"\d")


# ============================================================================
# INTENTS
# ============================================================================

INTENTS: dict[str, IntentDef] = {

    # ---------------------------------------------------------------- greeting
    "greeting": IntentDef(
        keywords=[
            "hi",
            "hello",
            "hey",
            "yo",
            "sup",
            "howdy",
        ],
        phrases=[
            "good morning",
            "good afternoon",
            "good evening",
            "what's up",
            "what is up",
            "how's it going",
            "how is it going",
            "how are you",
            "hey there",
            "hello there",
        ],
        related=[
            "heyo",
            "hiya",
            "ahoy",
        ],
        priority=0.25,
    ),

    # ---------------------------------------------------------------- goodbye
    "goodbye": IntentDef(
        keywords=[
            "bye",
            "goodbye",
            "cya",
            "gtg",
            "later",
        ],
        phrases=[
            "see you",
            "see ya",
            "talk later",
            "gotta go",
            "have to go",
            "catch you later",
            "logging off",
            "heading out",
            "i'm leaving",
            "i am leaving",
        ],
        related=[
            "farewell",
            "peace out",
            "night night",
        ],
        priority=0.2,
    ),

    # ---------------------------------------------------------------- thanks
    "thanks": IntentDef(
        keywords=[
            "thanks",
            "thank",
            "thx",
            "ty",
            "appreciated",
        ],
        phrases=[
            "thank you",
            "much appreciated",
            "appreciate it",
            "you're a lifesaver",
            "you are a lifesaver",
            "thanks a lot",
            "thanks so much",
        ],
        related=[
            "cheers",
            "props",
            "gracias",
        ],
        priority=0.2,
    ),

    # ----------------------------------------------------------- help command
    "help_command": IntentDef(
        keywords=[
            "help",
            "commands",
            "capabilities",
        ],
        phrases=[
            "what can you do",
            "what do you do",
            "how do you work",
            "list of commands",
            "show me your commands",
            "what are your commands",
            "help me out with the basics",
            "what are you capable of",
        ],
        related=[
            "instructions",
            "manual",
            "guide me",
        ],
        priority=0.4,
    ),

    # ------------------------------------------------------------ mood change
    "mood_change": IntentDef(
        keywords=[
            "mood",
            "acid",
            "deadpan",
            "soft",
            "personality",
            "nicer",
            "meaner",
            "kinder",
            "sarcastic",
            "sarcasm",
        ],
        phrases=[
            "be nicer",
            "be meaner",
            "switch mood",
            "change your mood",
            "change mood",
            "change mode",
            "be more sarcastic",
            "be less sarcastic",
            "tone it down",
            "be soft",
            "be acid",
            "be deadpan",
            "switch personality",
            "change personality",
            "change your personality",
        ],
        related=[
            "attitude",
            "vibe",
            "tone",
            "style",
        ],
        negatives=[
            "food",
            "moody teenager",
        ],
        priority=0.7,
    ),

    # ------------------------------------------------------------ joke request
    "joke_request": IntentDef(
        keywords=[
            "joke",
            "jokes",
            "funny",
            "laugh",
        ],
        phrases=[
            "tell me a joke",
            "make me laugh",
            "say something funny",
            "got any jokes",
            "give me a joke",
            "another joke",
        ],
        related=[
            "pun",
            "gag",
            "one-liner",
            "humor",
        ],
        priority=0.5,
    ),

    # ------------------------------------------------------------- math
    "math_request": IntentDef(
        keywords=[
            "calculate",
            "compute",
            "math",
            "sum",
            "plus",
            "minus",
            "times",
            "multiply",
            "divide",
            "divided",
            "sqrt",
            "squared",
            "equation",
            "percentage",
            "percent",
        ],
        phrases=[
            "what is",
            "what's",
            "how much is",
            "can you solve",
            "calculate this",
            "solve this",
            "work this out",
            "do the math",
            "what does this equal",
        ],
        related=[
            "arithmetic",
            "calculation",
            "expression",
        ],
        negatives=[
            "language",
            "python",
            "javascript",
            "typescript",
            "project",
            "who",
            "where",
            "when",
        ],
        priority=0.35,
        digit_sensitive=True,
    ),

    # ------------------------------------------------------------- dice
    "dice_request": IntentDef(
        keywords=[
            "dice",
            "die",
            "roll",
            "d4",
            "d6",
            "d8",
            "d10",
            "d12",
            "d20",
            "d100",
        ],
        phrases=[
            "roll a die",
            "roll the dice",
            "roll a dice",
            "roll some dice",
            "roll dice",
        ],
        related=[
            "rng",
            "randomizer",
            "random roll",
        ],
        priority=1.0,
        digit_sensitive=True,
    ),

    # ------------------------------------------------------------- coin flip
    "coinflip_request": IntentDef(
        keywords=[
            "coin",
            "heads",
            "tails",
        ],
        phrases=[
            "flip a coin",
            "toss a coin",
            "heads or tails",
            "flip the coin",
            "toss the coin",
        ],
        related=[
            "coinflip",
            "coin toss",
        ],
        priority=0.9,
    ),

    # ------------------------------------------------------------- coding
    "coding_help": IntentDef(
        keywords=[
            "bug",
            "bugs",
            "crash",
            "crashing",
            "crashed",
            "error",
            "errors",
            "broken",
            "debug",
            "debugging",
            "exception",
            "traceback",
            "glitch",
            "fix",
            "compile",
            "compilation",
            "syntax",
            "undefined",
        ],
        phrases=[
            "keeps crashing",
            "not working",
            "isn't working",
            "is not working",
            "won't work",
            "will not work",
            "throwing an error",
            "having trouble",
            "having an issue",
            "having a problem",
            "stuck with",
            "stuck on",
            "help me fix",
            "can you help me fix",
            "help me build",
            "help me debug",
            "trying to build",
            "trying to fix",
            "i'm stuck",
            "i am stuck",
            "having trouble programming",
            "this is broken",
            "something's broken",
            "something is broken",
            "code is broken",
            "code isn't working",
            "code is not working",
        ],
        related=[
            "malfunction",
            "misbehaving",
            "acting up",
            "compile error",
            "runtime error",
            "syntax error",
            "stack trace",
        ],
        negatives=[
            "joke",
            "dice",
            "coin",
            "mood",
            "weather",
        ],
        priority=1.1,
    ),

    # --------------------------------------------------------- vague problem
    "problem_report": IntentDef(
        keywords=[
            "broken",
            "stuck",
            "failing",
            "failed",
            "weird",
            "wrong",
        ],
        phrases=[
            "it's not working",
            "it is not working",
            "it broke",
            "it's broken",
            "it is broken",
            "this isn't working",
            "this is not working",
            "doesn't work",
            "does not work",
            "something is wrong",
            "something went wrong",
            "not working properly",
            "acting weird",
        ],
        related=[
            "acting weird",
            "misfiring",
            "something's wrong",
        ],
        negatives=[
            "python",
            "javascript",
            "typescript",
            "java",
            "code",
            "coding",
            "programming",
            "traceback",
            "exception",
        ],
        priority=0.4,
    ),

    # ------------------------------------------------------------- teach fact
    "teach_fact": IntentDef(
        keywords=[
            "remember",
            "fyi",
            "note",
            "memorize",
        ],
        phrases=[
            "remember that",
            "just so you know",
            "for future reference",
            "i want you to know",
            "keep in mind that",
            "when i say",
            "don't forget that",
            "do not forget that",
            "remember this",
        ],
        related=[
            "heads up",
            "side note",
            "keep this in mind",
        ],
        priority=0.9,
    ),

    # ------------------------------------------------------------- name share
    "name_share": IntentDef(
        weak_keywords=[
            "name",
        ],
        phrases=[
            "my name is",
            "call me",
            "you can call me",
            "i go by",
            "my name:",
        ],
        priority=1.0,
    ),

    # ---------------------------------------------------------- project share
    "project_share": IntentDef(
        weak_keywords=[
            "project",
        ],
        phrases=[
            "i'm building",
            "i am building",
            "i'm making",
            "i am making",
            "i'm working on a",
            "i am working on a",
            "i started a",
            "i started building",
            "my project is",
            "building a project",
            "working on a project",
            "i've been trying to build",
            "i have been trying to build",
            "i'm developing",
            "i am developing",
            "i've started building",
            "i have started building",
        ],
        related=[
            "building out",
            "putting together",
            "developing",
        ],
        negatives=[
            "fix",
            "bug",
            "error",
            "crash",
            "broken",
        ],
        priority=0.9,
    ),

    # --------------------------------------------------------- language share
    "language_share": IntentDef(
        keywords=[
            "python",
            "javascript",
            "typescript",
            "java",
            "rust",
            "golang",
            "go",
            "kotlin",
            "swift",
            "ruby",
            "php",
            "html",
            "css",
            "sql",
            "c++",
            "c#",
            "cpp",
            "csharp",
            "js",
            "ts",
            "py",
        ],
        phrases=[
            "i'm using",
            "i am using",
            "in python",
            "in javascript",
            "in typescript",
            "with python",
            "with javascript",
            "with typescript",
            "coding in",
            "programming in",
            "written in",
            "built with",
            "made with",
        ],
        related=[
            "language",
            "programming language",
            "tech stack",
            "stack",
        ],
        negatives=[
            "what is",
            "what's",
            "how does",
            "how do i",
            "tutorial",
        ],
        priority=0.75,
    ),

    # ----------------------------------------------------------- feature share
    "feature_share": IntentDef(
        weak_keywords=[
            "feature",
            "commands",
            "system",
            "module",
            "functionality",
            "component",
        ],
        phrases=[
            "i'm working on",
            "i am working on",
            "working on the",
            "working on a",
            "adding a",
            "adding an",
            "adding the",
            "building the",
            "implementing",
            "i'm implementing",
            "i am implementing",
        ],
        related=[
            "subsystem",
            "component",
            "function",
            "capability",
        ],
        negatives=[
            "what is",
            "how do i",
            "error",
            "bug",
            "crash",
            "broken",
        ],
        priority=0.65,
    ),

    # ----------------------------------------------------------- subject share
    "subject_share": IntentDef(
        keywords=[
            "studying",
            "learning",
        ],
        phrases=[
            "i'm studying",
            "i am studying",
            "i'm learning",
            "i am learning",
            "learning about",
            "trying to learn",
            "want to learn",
            "want to study",
        ],
        related=[
            "studying",
            "education",
            "course",
            "lesson",
        ],
        priority=0.65,
    ),

    # --------------------------------------------------------------- game
    "game_share": IntentDef(
        weak_keywords=[
            "game",
        ],
        phrases=[
            "i'm playing",
            "i am playing",
            "playing a game",
            "building a game",
            "making a game",
            "working on a game",
            "developing a game",
            "game called",
        ],
        related=[
            "gaming",
            "videogame",
            "video game",
        ],
        priority=0.8,
    ),

    # ---------------------------------------------------------------- goal
    "goal_share": IntentDef(
        weak_keywords=[
            "goal",
        ],
        phrases=[
            "i'm trying to",
            "i am trying to",
            "my goal is",
            "i want to",
            "trying to get",
            "hoping to",
            "planning to",
            "i'd like to",
            "i would like to",
            "i need to",
        ],
        related=[
            "objective",
            "aim",
            "plan",
        ],
        negatives=[
            "bug",
            "error",
            "crash",
            "broken",
            "doesn't work",
            "not working",
        ],
        priority=0.35,
    ),

    # --------------------------------------------------------------- ask
    "ask_about": IntentDef(
        weak_keywords=[
            "who",
            "what",
            "why",
            "how",
            "when",
            "where",
        ],
        phrases=[
            "who created",
            "who made",
            "who invented",
            "what is",
            "what's",
            "what are",
            "tell me about",
            "how does",
            "how do i",
            "how can i",
            "how can",
            "why does",
            "why is",
            "when did",
            "where is",
            "what did i say about",
            "what did i say regarding",
            "did i mention",
            "did i say anything about",
            "do you remember",
            "do you recall",
            "have i told you about",
            "have i mentioned",
            "what do you know about",
            "do you know about",
            "what did we talk about",
            "remind me about",
            "remind me what i said",
        ],
        related=[
            "explain",
            "tell me",
            "can you explain",
            "meaning",
            "definition",
        ],
        negatives=[
            "joke",
            "dice",
            "coin",
            "fix",
            "broken",
            "crash",
            "error",
            "bug",
            "working",
        ],
        priority=0.25,
    ),
}


# ============================================================================
# SPECIAL CASE SIGNALS
# ============================================================================

# Messages that look like mathematical expressions.
_MATH_EXPRESSION_RE = re.compile(
    r"""
    ^\s*
    [\d\s()+\-*/%.]+
    \s*$
    """,
    re.VERBOSE,
)

# Dice notation such as:
#   d20
#   2d6
#   3 d8
_DICE_NOTATION_RE = re.compile(
    r"\b\d{0,2}\s*d\s*(?:4|6|8|10|12|20|100)\b",
    re.IGNORECASE,
)

# Questions.
_QUESTION_START_RE = re.compile(
    r"^\s*(?:who|what|why|how|when|where|which|can|could|would|is|are|do|does|did|will|should)\b",
    re.IGNORECASE,
)


# ============================================================================
# SCORING
# ============================================================================

def score_intent(
    text: str,
    definition: IntentDef,
) -> tuple[float, list[str]]:
    """
    Score one intent against normalized text.

    Returns:
        (score, matched_terms)
    """
    text = normalize(text)

    if not text:
        return 0.0, []

    score = 0.0
    matched: list[str] = []

    # ---------------------------------------------------------------- keywords
    for keyword in definition.keywords:
        if _contains_word(text, keyword):
            score += KEYWORD_WEIGHT
            matched.append(keyword)

    # ----------------------------------------------------------- weak keywords
    for keyword in definition.weak_keywords:
        if _contains_word(text, keyword):
            score += WEAK_KEYWORD_WEIGHT
            matched.append(f"weak:{keyword}")

    # ---------------------------------------------------------------- phrases
    for phrase in definition.phrases:
        occurrences = _count_phrase_occurrences(text, phrase)

        if occurrences <= 0:
            continue

        # A phrase should be a strong signal, but repeating the same phrase
        # shouldn't let a message dominate purely through repetition.
        phrase_score = PHRASE_WEIGHT

        if len(phrase.split()) >= 3:
            phrase_score += SPECIFIC_PHRASE_BONUS

        score += phrase_score

        matched.append(phrase)

        # Small additional signal for repeated phrase use.
        if occurrences > 1:
            score += min(occurrences - 1, 2) * 0.15

    # ---------------------------------------------------------------- related
    for related in definition.related:
        if (
            _contains_phrase(text, related)
            or _contains_word(text, related)
        ):
            score += RELATED_WEIGHT
            matched.append(f"related:{related}")

    # ---------------------------------------------------------------- negative
    negative_hits = 0

    for negative in definition.negatives:
        if (
            _contains_phrase(text, negative)
            or _contains_word(text, negative)
        ):
            score -= NEGATIVE_WEIGHT
            negative_hits += 1
            matched.append(f"negative:{negative}")

    # ---------------------------------------------------------------- priority
    # Priority is intentionally tiny compared with real lexical evidence.
    # It is mostly a tie-breaker.
    if score > 0 and definition.priority:
        score += definition.priority * 0.12

    # Never return a negative score.
    score = max(score, 0.0)

    return round(score, 3), matched


# ============================================================================
# CONTEXTUAL BONUSES
# ============================================================================

def _apply_special_signals(
    text: str,
    board: dict[str, dict[str, Any]],
) -> None:
    """
    Apply small high-value signals that are easier to express procedurally
    than through static keyword lists.
    """
    normalized = normalize(text)

    # ---------------------------------------------------------------- math
    if _MATH_EXPRESSION_RE.fullmatch(normalized):
        board["math_request"]["score"] += 4.0
        board["math_request"]["matched"].append("math-expression")

    # -------------------------------------------------------------- dice
    if _DICE_NOTATION_RE.search(normalized):
        board["dice_request"]["score"] += 3.5
        board["dice_request"]["matched"].append("dice-notation")

    # ----------------------------------------------------------- question
    if _QUESTION_START_RE.search(normalized):
        board["ask_about"]["score"] += 0.35
        board["ask_about"]["matched"].append("question-start")

    # -------------------------------------------------------------- digits
    if _HAS_DIGIT_RE.search(normalized):
        for name, definition in INTENTS.items():
            if definition.digit_sensitive and board[name]["score"] > 0:
                board[name]["score"] += DIGIT_BONUS
                board[name]["matched"].append("digit-signal")


# ============================================================================
# INTENT TIE BREAKING
# ============================================================================

# When two intents are close, these are generally more specific.
INTENT_TIE_PRIORITY = {
    "dice_request": 1.00,
    "coinflip_request": 0.95,
    "name_share": 0.90,
    "teach_fact": 0.90,
    "mood_change": 0.85,
    "coding_help": 0.85,
    "project_share": 0.80,
    "language_share": 0.75,
    "game_share": 0.75,
    "subject_share": 0.70,
    "feature_share": 0.65,
    "joke_request": 0.60,
    "math_request": 0.55,
    "help_command": 0.50,
    "goodbye": 0.45,
    "greeting": 0.40,
    "thanks": 0.40,
    "ask_about": 0.30,
    "goal_share": 0.25,
    "problem_report": 0.20,
}


def _rank_board(
    board: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    """
    Rank intents by score, using specificity only as a small tie-breaker.
    """
    return sorted(
        board.items(),
        key=lambda item: (
            item[1]["score"],
            INTENT_TIE_PRIORITY.get(item[0], 0.0),
        ),
        reverse=True,
    )


# ============================================================================
# CONFIDENCE
# ============================================================================

def _calculate_confidence(
    top_score: float,
    second_score: float,
) -> float:
    """
    Calculate confidence from both absolute score and separation.

    A high score is good.

    A high score where the second-place intent is almost equally strong is
    less trustworthy.
    """
    if top_score <= 0:
        return 0.0

    # Soft saturation.
    absolute_confidence = top_score / (top_score + 2.0)

    # How clearly did the winner beat second place?
    gap = max(top_score - second_score, 0.0)

    if gap < AMBIGUITY_GAP:
        separation = 0.55
    elif gap < 1.0:
        separation = 0.72
    elif gap < 2.0:
        separation = 0.88
    else:
        separation = 1.0

    confidence = absolute_confidence * separation

    return round(
        min(max(confidence, 0.0), 0.99),
        2,
    )


# ============================================================================
# INTENT DETECTION
# ============================================================================

def detect_intent(raw_text: str) -> dict[str, Any]:
    """
    Detect the most likely intent.

    Returns a dictionary compatible with the existing SideEye pipeline.

    Example:
        {
            "intent": "coding_help",
            "confidence": 0.87,
            "accepted": True,
            "matched": ["bug", "help me fix"],
            "board": {...},
            "ranked": [...]
        }
    """
    text = normalize(raw_text)

    if not text:
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "accepted": False,
            "matched": [],
            "board": {},
            "ranked": [],
        }

    board: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------- base scores
    for name, definition in INTENTS.items():
        score, matched = score_intent(
            text,
            definition,
        )

        board[name] = {
            "score": score,
            "matched": matched,
        }

    # ------------------------------------------------------- special signals
    _apply_special_signals(
        text,
        board,
    )

    # Round scores after contextual bonuses.
    for data in board.values():
        data["score"] = round(
            max(data["score"], 0.0),
            3,
        )

    # --------------------------------------------------------------- ranking
    ranked = _rank_board(board)

    top_name, top_data = ranked[0]

    top_score = top_data["score"]

    second_score = (
        ranked[1][1]["score"]
        if len(ranked) > 1
        else 0.0
    )

    # ------------------------------------------------------------- confidence
    confidence = _calculate_confidence(
        top_score,
        second_score,
    )

    # --------------------------------------------------------------- unclear
    if top_score < FLOOR_THRESHOLD:
        return {
            "intent": "unclear",
            "confidence": 0.0,
            "accepted": False,
            "matched": [],
            "board": board,
            "ranked": [
                {
                    "intent": name,
                    "score": data["score"],
                    "matched": data["matched"],
                }
                for name, data in ranked
            ],
        }

    # ---------------------------------------------------------- ambiguity guard
    #
    # Example:
    #     "I'm working on Python"
    #
    # This could look like project_share, feature_share, and language_share.
    #
    # If the winner isn't clearly ahead, lower confidence rather than blindly
    # claiming certainty.
    gap = top_score - second_score

    accepted = (
        top_score >= ACCEPT_THRESHOLD
        and confidence >= 0.45
    )

    if gap < 0.25 and top_score < 4.0:
        accepted = False

    return {
        "intent": top_name,
        "confidence": confidence,
        "accepted": accepted,
        "matched": top_data["matched"],
        "board": board,
        "ranked": [
            {
                "intent": name,
                "score": data["score"],
                "matched": data["matched"],
            }
            for name, data in ranked
        ],
    }


# ============================================================================
# DEBUGGING
# ============================================================================

def explain_intent(raw_text: str) -> str:
    """
    Produce a human-readable explanation of SideEye's NLU decision.

    Useful while developing/testing the bot.

    Example:
        print(explain_intent("I'm coding in Python and my bot keeps crashing"))
    """
    result = detect_intent(raw_text)

    lines = [
        f"Input: {raw_text!r}",
        f"Intent: {result['intent']}",
        f"Confidence: {result['confidence']}",
        f"Accepted: {result['accepted']}",
        "",
        "Top matches:",
    ]

    ranked = result.get("ranked", [])

    for item in ranked[:5]:
        matched = ", ".join(item["matched"]) or "none"

        lines.append(
            f"  {item['intent']:<20} "
            f"{item['score']:<6} "
            f"[{matched}]"
        )

    return "\n".join(lines)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "IntentDef",
    "INTENTS",
    "LANGUAGE_ALIASES",
    "normalize",
    "tokenize",
    "score_intent",
    "detect_intent",
    "explain_intent",
]