"""
database.py
-----------
Persistent memory layer for SideEye.

All SQLite access lives in this module. The rest of the application should
use these helpers instead of writing raw SQL.

Database contents
-----------------
users
    The local SideEye user.

context
    Structured conversation slots such as project, language, feature, etc.

entities
    Recently mentioned things used by the reference-resolution system
    ("it", "them", "that project", etc.).

messages
    Full conversation history.

facts
    Things the user explicitly taught SideEye.

learned_patterns
    Custom "when I say X, reply Y" rules.

state
    Live conversation state such as mood, pending slot, last intent,
    and turn count.

The module intentionally uses only Python's standard library.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.getenv(
        "SIDEEYE_DATA_DIR",
        str(BASE_DIR / "data"),
    )
)

DB_PATH = Path(
    os.getenv(
        "SIDEEYE_DB_PATH",
        str(DATA_DIR / "sideeye.db"),
    )
)

# SideEye is currently designed as a single-user local application.
DEFAULT_USER_ID = 1


# ============================================================================
# VALID VALUES
# ============================================================================

CONTEXT_KEYS = [
    "project",
    "language",
    "feature",
    "problem",
    "subject",
    "game",
    "goal",
    "task",
    "topic",
]

VALID_CONTEXT_KEYS = set(CONTEXT_KEYS)

VALID_ROLES = {
    "user",
    "bot",
}


# ============================================================================
# SCHEMA
# ============================================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context (
    user_id     INTEGER NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    updated_at  TEXT NOT NULL,

    PRIMARY KEY (user_id, key),

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    entity_type TEXT NOT NULL,
    value       TEXT NOT NULL,
    is_plural   INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    intent      TEXT,
    confidence  REAL,
    mood        TEXT,
    created_at  TEXT NOT NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    subject     TEXT NOT NULL,
    fact_text   TEXT NOT NULL,
    created_at  TEXT NOT NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS learned_patterns (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    trigger_phrase  TEXT NOT NULL,
    response_text   TEXT NOT NULL,
    created_at      TEXT NOT NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS state (
    user_id         INTEGER PRIMARY KEY,
    mood            TEXT NOT NULL DEFAULT 'deadpan',
    pending_slot    TEXT,
    last_intent     TEXT,
    turn_count      INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
);
"""


# ============================================================================
# INDEXES
# ============================================================================

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_context_user
    ON context(user_id);

CREATE INDEX IF NOT EXISTS idx_entities_user_recent
    ON entities(user_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_messages_user_recent
    ON messages(user_id, id DESC);

CREATE INDEX IF NOT EXISTS idx_messages_user_role
    ON messages(user_id, role);

CREATE INDEX IF NOT EXISTS idx_facts_user_subject
    ON facts(user_id, subject);

CREATE INDEX IF NOT EXISTS idx_patterns_user
    ON learned_patterns(user_id);

CREATE INDEX IF NOT EXISTS idx_state_user
    ON state(user_id);
"""


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> str:
    """Convert a value to clean text."""
    if value is None:
        return ""

    return str(value).strip()


def _normalize_lookup(value: Any) -> str:
    """
    Normalize text used for lookups.

    This intentionally keeps normal internal values readable while making
    comparisons case-insensitive.
    """
    return _clean_text(value).casefold()


def _validate_user_id(user_id: int) -> int:
    """Validate and normalize a user ID."""
    try:
        user_id = int(user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("user_id must be an integer") from exc

    if user_id <= 0:
        raise ValueError("user_id must be greater than zero")

    return user_id


def _validate_context_key(key: str) -> str:
    """Validate a context slot name."""
    key = _normalize_lookup(key)

    if key not in VALID_CONTEXT_KEYS:
        raise ValueError(
            f"Unknown context key: {key!r}. "
            f"Expected one of: {', '.join(CONTEXT_KEYS)}"
        )

    return key


def _validate_role(role: str) -> str:
    """Validate a message role."""
    role = _normalize_lookup(role)

    if role not in VALID_ROLES:
        raise ValueError(
            f"Invalid message role: {role!r}. "
            f"Expected one of: user, bot"
        )

    return role


def _ensure_user(conn: sqlite3.Connection, user_id: int) -> None:
    """
    Ensure the user and state rows exist.

    This makes individual helper functions more robust: callers don't always
    have to remember to call init_db() first.
    """
    user_id = _validate_user_id(user_id)

    existing = conn.execute(
        "SELECT id FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()

    if existing is None:
        conn.execute(
            """
            INSERT INTO users (id, name, created_at)
            VALUES (?, ?, ?)
            """,
            (user_id, None, _now()),
        )

    state = conn.execute(
        "SELECT user_id FROM state WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    if state is None:
        conn.execute(
            """
            INSERT INTO state (
                user_id,
                mood,
                pending_slot,
                last_intent,
                turn_count,
                updated_at
            )
            VALUES (?, 'deadpan', NULL, NULL, 0, ?)
            """,
            (user_id, _now()),
        )


def _ensure_database_directory() -> None:
    """Create the database directory if necessary."""
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# CONNECTION
# ============================================================================

def get_connection() -> sqlite3.Connection:
    """
    Open a configured SQLite connection.

    SQLite is configured with:
        - Row objects
        - foreign keys
        - WAL journal mode
        - busy timeout
        - synchronous NORMAL
    """
    _ensure_database_directory()

    conn = sqlite3.connect(
        str(DB_PATH),
        timeout=10.0,
    )

    conn.row_factory = sqlite3.Row

    # Foreign-key enforcement is connection-local in SQLite.
    conn.execute("PRAGMA foreign_keys = ON;")

    # WAL improves reliability when reads and writes happen close together.
    conn.execute("PRAGMA journal_mode = WAL;")

    # Avoid immediately failing if the database is briefly busy.
    conn.execute("PRAGMA busy_timeout = 10000;")

    # Good balance for a local application using WAL.
    conn.execute("PRAGMA synchronous = NORMAL;")

    return conn


# ============================================================================
# INITIALIZATION
# ============================================================================

def init_db() -> None:
    """
    Create all tables/indexes and initialize the default user/state.

    Safe to call multiple times.
    """
    conn = get_connection()

    try:
        conn.executescript(SCHEMA)
        conn.executescript(INDEXES)

        _ensure_user(
            conn,
            DEFAULT_USER_ID,
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================================
# USERS
# ============================================================================

def set_user_name(
    user_id: int,
    name: str,
) -> None:
    """Set or replace the user's name."""
    user_id = _validate_user_id(user_id)
    name = _clean_text(name)

    if not name:
        raise ValueError("name cannot be empty")

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        conn.execute(
            """
            UPDATE users
            SET name = ?
            WHERE id = ?
            """,
            (name, user_id),
        )

        conn.commit()

    finally:
        conn.close()


def get_user_name(
    user_id: int,
) -> str | None:
    """Return the user's stored name, or None."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT name
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()

        return row["name"] if row else None

    finally:
        conn.close()


# ============================================================================
# CONTEXT
# ============================================================================

def set_context(
    user_id: int,
    key: str,
    value: str,
) -> None:
    """
    Store/update a structured context slot.

    Example:
        set_context(1, "project", "SideEye")
    """
    user_id = _validate_user_id(user_id)
    key = _validate_context_key(key)
    value = _clean_text(value)

    if not value:
        return

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        conn.execute(
            """
            INSERT INTO context (
                user_id,
                key,
                value,
                updated_at
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(user_id, key)
            DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (
                user_id,
                key,
                value,
                _now(),
            ),
        )

        conn.commit()

    finally:
        conn.close()


def get_context(
    user_id: int,
) -> dict[str, str]:
    """Return all context slots in a stable order."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT key, value
            FROM context
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

        unordered = {
            row["key"]: row["value"]
            for row in rows
        }

        ordered = {
            key: unordered[key]
            for key in CONTEXT_KEYS
            if key in unordered
        }

        # Preserve any future/custom keys that might exist.
        ordered.update(
            {
                key: value
                for key, value in unordered.items()
                if key not in ordered
            }
        )

        return ordered

    finally:
        conn.close()


def get_context_value(
    user_id: int,
    key: str,
) -> str | None:
    """Get one context value without loading the entire context."""
    user_id = _validate_user_id(user_id)
    key = _validate_context_key(key)

    conn = get_connection()

    try:
        row = conn.execute(
            """
            SELECT value
            FROM context
            WHERE user_id = ?
              AND key = ?
            """,
            (
                user_id,
                key,
            ),
        ).fetchone()

        return row["value"] if row else None

    finally:
        conn.close()


def delete_context(
    user_id: int,
    key: str,
) -> None:
    """Delete one context slot."""
    user_id = _validate_user_id(user_id)
    key = _validate_context_key(key)

    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM context
            WHERE user_id = ?
              AND key = ?
            """,
            (
                user_id,
                key,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def clear_context(
    user_id: int,
) -> None:
    """Delete all structured context for a user."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM context
            WHERE user_id = ?
            """,
            (user_id,),
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================================
# ENTITIES
# ============================================================================

def push_entity(
    user_id: int,
    entity_type: str,
    value: str,
    is_plural: bool = False,
) -> None:
    """
    Record an entity for later reference resolution.

    New rows are always inserted so the newest entity naturally appears
    first when queried by descending ID.
    """
    user_id = _validate_user_id(user_id)

    entity_type = _clean_text(entity_type).lower()
    value = _clean_text(value)

    if not entity_type:
        raise ValueError("entity_type cannot be empty")

    if not value:
        return

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        conn.execute(
            """
            INSERT INTO entities (
                user_id,
                entity_type,
                value,
                is_plural,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                entity_type,
                value,
                1 if is_plural else 0,
                _now(),
            ),
        )

        conn.commit()

    finally:
        conn.close()


def get_recent_entities(
    user_id: int,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Return recent entities, newest first."""
    user_id = _validate_user_id(user_id)
    limit = max(1, min(int(limit), 500))

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT entity_type, value, is_plural
            FROM entities
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                user_id,
                limit,
            ),
        ).fetchall()

        return [
            {
                "type": row["entity_type"],
                "value": row["value"],
                "plural": bool(row["is_plural"]),
            }
            for row in rows
        ]

    finally:
        conn.close()


def clear_entities(
    user_id: int,
) -> None:
    """Delete stored entities."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM entities
            WHERE user_id = ?
            """,
            (user_id,),
        )

        conn.commit()

    finally:
        conn.close()


def prune_entities(
    user_id: int,
    keep: int = 100,
) -> None:
    """
    Keep only the newest N entities.

    Useful for preventing the entity table from growing forever.
    """
    user_id = _validate_user_id(user_id)
    keep = max(1, int(keep))

    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM entities
            WHERE user_id = ?
              AND id NOT IN (
                  SELECT id
                  FROM entities
                  WHERE user_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (
                user_id,
                user_id,
                keep,
            ),
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================================
# MESSAGES
# ============================================================================

def log_message(
    user_id: int,
    role: str,
    content: str,
    intent: str | None = None,
    confidence: float | None = None,
    mood: str | None = None,
) -> None:
    """
    Add a message to conversation history.
    """
    user_id = _validate_user_id(user_id)
    role = _validate_role(role)
    content = _clean_text(content)

    if not content:
        return

    if confidence is not None:
        confidence = float(confidence)
        confidence = max(
            0.0,
            min(confidence, 1.0),
        )

    if intent is not None:
        intent = _clean_text(intent) or None

    if mood is not None:
        mood = _clean_text(mood) or None

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        conn.execute(
            """
            INSERT INTO messages (
                user_id,
                role,
                content,
                intent,
                confidence,
                mood,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                role,
                content,
                intent,
                confidence,
                mood,
                _now(),
            ),
        )

        conn.commit()

    finally:
        conn.close()


def get_recent_messages(
    user_id: int,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Return recent messages in chronological order.

    The SQL query gets the newest rows efficiently, then reverses them so
    callers receive oldest -> newest.
    """
    user_id = _validate_user_id(user_id)
    limit = max(1, min(int(limit), 1000))

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                role,
                content,
                intent,
                confidence,
                mood,
                created_at
            FROM messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                user_id,
                limit,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in reversed(rows)
        ]

    finally:
        conn.close()


def search_messages(
    user_id: int,
    term: str,
    limit: int = 5,
    role: str = "user",
) -> list[dict[str, Any]]:
    """
    Search conversation history using SQLite LIKE.

    Most recent matching message is returned first.

    ``role=None`` searches both user and bot messages.
    """
    user_id = _validate_user_id(user_id)

    term = _clean_text(term)

    if not term:
        return []

    limit = max(1, min(int(limit), 100))

    if role is not None:
        role = _validate_role(role)

    conn = get_connection()

    try:
        query = """
            SELECT
                role,
                content,
                intent,
                confidence,
                mood,
                created_at
            FROM messages
            WHERE user_id = ?
              AND content LIKE ?
        """

        params: list[Any] = [
            user_id,
            f"%{term}%",
        ]

        if role:
            query += " AND role = ? "
            params.append(role)

        query += """
            ORDER BY id DESC
            LIMIT ?
        """

        params.append(limit)

        rows = conn.execute(
            query,
            params,
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def count_messages(
    user_id: int,
    role: str | None = None,
) -> int:
    """Return the number of stored messages."""
    user_id = _validate_user_id(user_id)

    if role is not None:
        role = _validate_role(role)

    conn = get_connection()

    try:
        if role:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages
                WHERE user_id = ?
                  AND role = ?
                """,
                (
                    user_id,
                    role,
                ),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM messages
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

        return int(row["count"])

    finally:
        conn.close()


def clear_messages(
    user_id: int,
) -> None:
    """Delete the entire conversation history."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM messages
            WHERE user_id = ?
            """,
            (user_id,),
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================================
# FACTS
# ============================================================================

def add_fact(
    user_id: int,
    subject: str,
    fact_text: str,
) -> None:
    """
    Store a fact explicitly taught by the user.

    Duplicate facts for the same subject are ignored.
    """
    user_id = _validate_user_id(user_id)

    subject = _normalize_lookup(subject)
    fact_text = _clean_text(fact_text)

    if not subject:
        raise ValueError("subject cannot be empty")

    if not fact_text:
        return

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        existing = conn.execute(
            """
            SELECT id
            FROM facts
            WHERE user_id = ?
              AND subject = ?
              AND fact_text = ?
            LIMIT 1
            """,
            (
                user_id,
                subject,
                fact_text,
            ),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO facts (
                    user_id,
                    subject,
                    fact_text,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    subject,
                    fact_text,
                    _now(),
                ),
            )

            conn.commit()

    finally:
        conn.close()


def find_facts(
    user_id: int,
    subject: str,
) -> list[dict[str, Any]]:
    """Return all facts associated with a subject, newest first."""
    user_id = _validate_user_id(user_id)
    subject = _normalize_lookup(subject)

    if not subject:
        return []

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT fact_text, created_at
            FROM facts
            WHERE user_id = ?
              AND subject = ?
            ORDER BY id DESC
            """,
            (
                user_id,
                subject,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def get_recent_facts(
    user_id: int,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recently taught facts."""
    user_id = _validate_user_id(user_id)
    limit = max(1, min(int(limit), 500))

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                subject,
                fact_text,
                created_at
            FROM facts
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                user_id,
                limit,
            ),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def clear_facts(
    user_id: int,
) -> None:
    """Delete all taught facts."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM facts
            WHERE user_id = ?
            """,
            (user_id,),
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================================
# LEARNED PATTERNS
# ============================================================================

def add_learned_pattern(
    user_id: int,
    trigger_phrase: str,
    response_text: str,
) -> None:
    """
    Store a custom trigger -> response rule.

    Duplicate trigger/response combinations are ignored.
    """
    user_id = _validate_user_id(user_id)

    trigger_phrase = _normalize_lookup(trigger_phrase)
    response_text = _clean_text(response_text)

    if not trigger_phrase:
        raise ValueError("trigger_phrase cannot be empty")

    if not response_text:
        return

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        existing = conn.execute(
            """
            SELECT id
            FROM learned_patterns
            WHERE user_id = ?
              AND trigger_phrase = ?
              AND response_text = ?
            LIMIT 1
            """,
            (
                user_id,
                trigger_phrase,
                response_text,
            ),
        ).fetchone()

        if existing is None:
            conn.execute(
                """
                INSERT INTO learned_patterns (
                    user_id,
                    trigger_phrase,
                    response_text,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    trigger_phrase,
                    response_text,
                    _now(),
                ),
            )

            conn.commit()

    finally:
        conn.close()


def find_learned_pattern(
    user_id: int,
    text: str,
) -> str | None:
    """
    Find a learned response for the supplied text.

    More specific/longer triggers are checked first.

    Example:

        learned:
            "hello" -> "yo"
            "hello sideeye" -> "what's up?"

        input:
            "hello sideeye"

    The second rule wins because it is more specific.
    """
    user_id = _validate_user_id(user_id)

    text = _normalize_lookup(text)

    if not text:
        return None

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                trigger_phrase,
                response_text
            FROM learned_patterns
            WHERE user_id = ?
              AND trigger_phrase != ''
            """,
            (user_id,),
        ).fetchall()

        candidates: list[tuple[int, str, str]] = []

        for row in rows:
            trigger = _normalize_lookup(
                row["trigger_phrase"]
            )

            if trigger and trigger in text:
                candidates.append(
                    (
                        len(trigger),
                        trigger,
                        row["response_text"],
                    )
                )

        if not candidates:
            return None

        # Longest trigger wins.
        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return candidates[0][2]

    finally:
        conn.close()


def get_learned_patterns(
    user_id: int,
) -> list[dict[str, Any]]:
    """Return all custom learned patterns."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        rows = conn.execute(
            """
            SELECT
                trigger_phrase,
                response_text,
                created_at
            FROM learned_patterns
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    finally:
        conn.close()


def clear_learned_patterns(
    user_id: int,
) -> None:
    """Delete all custom learned patterns."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        conn.execute(
            """
            DELETE FROM learned_patterns
            WHERE user_id = ?
            """,
            (user_id,),
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================================
# STATE
# ============================================================================

def get_state(
    user_id: int,
) -> dict[str, Any]:
    """Return the current conversation state."""
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        row = conn.execute(
            """
            SELECT
                user_id,
                mood,
                pending_slot,
                last_intent,
                turn_count,
                updated_at
            FROM state
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        return dict(row) if row else {}

    finally:
        conn.close()


def update_state(
    user_id: int,
    **fields: Any,
) -> None:
    """
    Update one or more state fields.

    Example:
        update_state(
            1,
            mood="soft",
            last_intent="greeting",
        )

    Only known state columns may be updated.
    """
    user_id = _validate_user_id(user_id)

    if not fields:
        return

    allowed_fields = {
        "mood",
        "pending_slot",
        "last_intent",
        "turn_count",
    }

    unknown = set(fields) - allowed_fields

    if unknown:
        raise ValueError(
            "Unknown state field(s): "
            + ", ".join(sorted(unknown))
        )

    cleaned: dict[str, Any] = {}

    for key, value in fields.items():

        if key == "mood":
            value = _clean_text(value)

            if not value:
                raise ValueError("mood cannot be empty")

        elif key == "pending_slot":
            if value is not None:
                value = _validate_context_key(value)

        elif key == "last_intent":
            if value is not None:
                value = _clean_text(value) or None

        elif key == "turn_count":
            value = max(0, int(value))

        cleaned[key] = value

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        cleaned["updated_at"] = _now()

        columns = list(cleaned.keys())

        set_clause = ", ".join(
            f"{column} = ?"
            for column in columns
        )

        values = [
            cleaned[column]
            for column in columns
        ]

        values.append(user_id)

        conn.execute(
            f"""
            UPDATE state
            SET {set_clause}
            WHERE user_id = ?
            """,
            values,
        )

        conn.commit()

    finally:
        conn.close()


def increment_turn_count(
    user_id: int,
) -> int:
    """
    Increment and return the current turn count.

    Uses one SQL UPDATE instead of reading and then writing the count,
    reducing the chance of lost increments.
    """
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        now = _now()

        conn.execute(
            """
            UPDATE state
            SET
                turn_count = turn_count + 1,
                updated_at = ?
            WHERE user_id = ?
            """,
            (
                now,
                user_id,
            ),
        )

        row = conn.execute(
            """
            SELECT turn_count
            FROM state
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

        conn.commit()

        return int(
            row["turn_count"]
            if row
            else 0
        )

    finally:
        conn.close()


# ============================================================================
# RESET / MAINTENANCE
# ============================================================================

def reset_all(
    user_id: int,
) -> None:
    """
    Reset the active conversation state.

    This intentionally keeps:

        - chat history
        - taught facts
        - learned patterns
        - user name

    It removes:

        - structured context
        - entities
        - mood
        - pending slot
        - last intent
        - turn count
    """
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        conn.execute(
            """
            DELETE FROM context
            WHERE user_id = ?
            """,
            (user_id,),
        )

        conn.execute(
            """
            DELETE FROM entities
            WHERE user_id = ?
            """,
            (user_id,),
        )

        conn.execute(
            """
            UPDATE state
            SET
                mood = 'deadpan',
                pending_slot = NULL,
                last_intent = NULL,
                turn_count = 0,
                updated_at = ?
            WHERE user_id = ?
            """,
            (
                _now(),
                user_id,
            ),
        )

        conn.commit()

    finally:
        conn.close()


def delete_user_data(
    user_id: int,
    keep_messages: bool = False,
    keep_facts: bool = False,
    keep_patterns: bool = False,
) -> None:
    """
    Explicitly delete selected persistent user data.

    This is separate from reset_all() because reset_all() intentionally
    preserves long-term memory.
    """
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        _ensure_user(conn, user_id)

        conn.execute(
            "DELETE FROM context WHERE user_id = ?",
            (user_id,),
        )

        conn.execute(
            "DELETE FROM entities WHERE user_id = ?",
            (user_id,),
        )

        if not keep_messages:
            conn.execute(
                "DELETE FROM messages WHERE user_id = ?",
                (user_id,),
            )

        if not keep_facts:
            conn.execute(
                "DELETE FROM facts WHERE user_id = ?",
                (user_id,),
            )

        if not keep_patterns:
            conn.execute(
                "DELETE FROM learned_patterns WHERE user_id = ?",
                (user_id,),
            )

        conn.execute(
            """
            UPDATE state
            SET
                mood = 'deadpan',
                pending_slot = NULL,
                last_intent = NULL,
                turn_count = 0,
                updated_at = ?
            WHERE user_id = ?
            """,
            (
                _now(),
                user_id,
            ),
        )

        conn.commit()

    finally:
        conn.close()


# ============================================================================
# DATABASE STATS
# ============================================================================

def get_stats(
    user_id: int,
) -> dict[str, int]:
    """
    Return useful database statistics for debugging/UI.

    Example:
        {
            "messages": 42,
            "entities": 18,
            "facts": 5,
            "learned_patterns": 3,
            "context_slots": 4,
        }
    """
    user_id = _validate_user_id(user_id)

    conn = get_connection()

    try:
        stats: dict[str, int] = {}

        queries = {
            "messages": """
                SELECT COUNT(*)
                FROM messages
                WHERE user_id = ?
            """,
            "entities": """
                SELECT COUNT(*)
                FROM entities
                WHERE user_id = ?
            """,
            "facts": """
                SELECT COUNT(*)
                FROM facts
                WHERE user_id = ?
            """,
            "learned_patterns": """
                SELECT COUNT(*)
                FROM learned_patterns
                WHERE user_id = ?
            """,
            "context_slots": """
                SELECT COUNT(*)
                FROM context
                WHERE user_id = ?
            """,
        }

        for name, query in queries.items():
            row = conn.execute(
                query,
                (user_id,),
            ).fetchone()

            stats[name] = int(row[0])

        return stats

    finally:
        conn.close()


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Configuration
    "BASE_DIR",
    "DATA_DIR",
    "DB_PATH",
    "DEFAULT_USER_ID",
    "CONTEXT_KEYS",

    # Initialization
    "init_db",
    "get_connection",

    # Users
    "set_user_name",
    "get_user_name",

    # Context
    "set_context",
    "get_context",
    "get_context_value",
    "delete_context",
    "clear_context",

    # Entities
    "push_entity",
    "get_recent_entities",
    "clear_entities",
    "prune_entities",

    # Messages
    "log_message",
    "get_recent_messages",
    "search_messages",
    "count_messages",
    "clear_messages",

    # Facts
    "add_fact",
    "find_facts",
    "get_recent_facts",
    "clear_facts",

    # Learned patterns
    "add_learned_pattern",
    "find_learned_pattern",
    "get_learned_patterns",
    "clear_learned_patterns",

    # State
    "get_state",
    "update_state",
    "increment_turn_count",

    # Maintenance
    "reset_all",
    "delete_user_data",
    "get_stats",
]