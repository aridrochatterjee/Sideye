# SideEye 👀

A fully local, rule-based chatbot. No AI APIs, no LLMs, no chatbot
libraries, no third-party packages of any kind — just Python's standard
library, SQLite, and old-fashioned programming logic (scored keyword
matching, regexes, and a state machine).



SideEye tracks what you're building, remembers it across messages (and
across restarts, since it's all in a SQLite file), understands follow-up
words like "it" and "them", asks natural follow-up questions, and can
switch between three personalities. It also rolls dice, flips coins, does
quick arithmetic, tells jokes, and can be taught new facts and custom
canned replies.

## Running it

You need Python 3.9+ and nothing else.

```bash
python run.py
```

Then open **http://127.0.0.1:8765** in a browser. That's it — no `pip
install`, no build step. The first run creates `data/sideeye.db`
automatically.

Want a different port? `python run.py 9000`.

## How it's put together

The spec this was built from describes a pipeline, and the code follows
it fairly literally. Every incoming message goes through
`backend/pipeline.py`, which walks through these stages in order:

```
User Message
      │
      ▼
Normalize text            (nlu.normalize)
      │
      ▼
Load context memory        (database.get_context / get_recent_entities)
      │
      ▼
Resolve references          (references.resolve_references — turns
      │                       "it"/"them"/etc. into the actual thing
      │                       they point at, for understanding only)
      ▼
Detect intent               (nlu.detect_intent — scored, not exact-match)
      │
      ▼
Check conversation state    (conversation.py — are we mid-way through
      │                       asking "project → language → feature"?)
      ▼
Generate a response          (personality.py — same intent, three moods)
      │
      ▼
Save memory & context        (database.py — SQLite)
```

Each stage is its own file, so you can read them independently:

| File | Job |
|---|---|
| `backend/database.py` | Every SQL statement in the app lives here. Nothing else touches the `.db` file directly. |
| `backend/nlu.py` | The scored intent detector — see below for how the scoring works. |
| `backend/extraction.py` | Regex patterns that pull a clean value ("python") out of a sentence ("I'm using Python for this"). |
| `backend/references.py` | Pronoun resolution for it/this/that/they/them/these/those. |
| `backend/conversation.py` | The `project → language → feature` follow-up question flow. |
| `backend/personality.py` | Response templates for all three moods. |
| `backend/utils.py` | Dice, coins, safe arithmetic, jokes. |
| `backend/pipeline.py` | Wires all of the above together into one `process_message()` call. |
| `backend/server.py` | A `ThreadingHTTPServer` — the whole "backend framework" is this one file. |
| `frontend/` | Plain HTML/CSS/JS. No React, no build tools, no npm. |

### The NLU: scored intents, not exact matching

Instead of `if "project" in message`, every intent (greeting, coding
help, sharing a project, asking for a joke, etc.) has up to four lists in
`nlu.py`:

- **keywords** — single words, worth 1.0 point each
- **weak_keywords** — single words that are *ambiguous on their own*
  (like "project", "feature", "game" — see the note below), worth only
  0.4 points
- **phrases** — multi-word patterns, worth 2.5 points, because a whole
  phrase matching is a much stronger signal than one word
- **negatives** — words that make an intent *less* likely, to break ties
  with a similar intent

A message is scored against every intent at once, and the highest scorer
wins — but only if it clears a minimum bar (0.9 points). This is the bit
that stops a message like *"what's your favorite project?"* from being
misread as "the user is announcing a new project": the bare word
"project" alone only earns 0.4 points, which isn't enough on its own. But
*"I'm building a Discord bot"* matches the phrase `"i'm building"` (2.5
points) plus the weak keyword `"project"`-adjacent context, easily
clearing the bar.

### Reference resolution

`references.py` keeps a small recency-ordered stack of "things worth
pointing a pronoun at" (pushed every time a context slot gets filled).
When a message contains "it", "this", "that", "they", "them", "these", or
"those", the resolver swaps in the most recent matching entity — but only
in an internal copy of the text used for understanding. **The message
you actually see in the chat log is never rewritten.** If you want to see
this happening, tick "Show how SideEye understood each message" in the
sidebar — it'll show you things like `resolved: "it" -> "python"` under
the bot's reply.

One deliberate guard: "that" is genuinely ambiguous in English — it's a
pointing word in *"fix that"* but just a conjunction in *"remember that
Python is a language"*. SideEye checks the word right before "that" and
skips substitution after verbs like "remember", "know", "think", "said",
so it doesn't garble sentences that use "that" grammatically rather than
referentially.

### Multi-turn flow

`conversation.py` defines one simple flow: once a project is mentioned,
SideEye wants to know the language, then the feature, asking one question
at a time and tracking `pending_slot` in the database. When you answer, the
pipeline first tries the *specific* extractor for whatever slot is
pending (so "I'm using Python" correctly becomes "python", not the whole
sentence) — and only falls back to treating your raw message as the
answer verbatim if that extractor comes up empty and nothing else was
confidently recognized (so a bare answer like "Moderation commands" is
still understood, but a genuine interruption like "actually, tell me a
joke" is never swallowed as a slot answer).

### Personality

Three moods live in `personality.py` as response templates: **acid**
(sarcastic), **deadpan** (flat), and **soft** (encouraging). Switch with
the sidebar buttons, by typing something like "be more sarcastic", or
with the `/mood acid` command directly.

### Remembering the conversation

Every message — yours and SideEye's — is written to the `messages` table
as it happens, so the full transcript is on disk from the start; that
part needs no extra code. The part that needed building is *retrieval*:
being able to answer when you ask about something later.

When you ask a question SideEye recognizes as "asking about something"
(`ask_about` in `nlu.py` — this covers "what is X", "who created X", "do
you remember X", "what did I say about X", "have I mentioned X", and
similar phrasings), `pipeline.py` looks in three places, in order:

1. **Taught facts** — anything you explicitly told it to remember
   ("remember that X is Y").
2. **The current case file** — if you ask about something that's
   literally one of the tracked slots ("what's my language?", "do you
   remember my project?"), it answers straight from context memory.
3. **The chat log itself** — a plain SQL `LIKE` search over every past
   message you've sent, most recent match first. This is what makes
   "what did I say about it?" work even for something you mentioned in
   passing and never explicitly asked SideEye to remember.

Combined with reference resolution, this means a pronoun in a recall
question gets resolved first ("it" → "discord bot") and *then* searched
for — so "what did I say about it?" really does search for what you said
about the actual thing "it" refers to, not the literal word "it".

This is still a plain substring search, not real semantic search, so
very differently-worded questions about the same thing won't always
connect (asking about "the moderation system" won't find a message that
only said "commands" some other way) — but for the common phrasings above
it works well.

## Things SideEye can do

- Hold a running "case file" of your project, language, feature, problem,
  subject, game, and goal — visible in the sidebar, persisted in SQLite
- Understand "it"/"they"/"them"/etc. as references to whatever was
  mentioned most recently
- Ask natural follow-up questions instead of treating every message as
  independent
- Roll dice (`roll 2d6`, `roll a d20`, `roll 3 dice`)
- Flip a coin
- Do arithmetic (`what is 12 times 4`) safely — it parses the expression
  into an AST and only ever evaluates `+ - * / % **` on plain numbers,
  never a real `eval()`
- Tell a joke
- Learn facts you teach it ("remember that SideEye is a rule-based
  chatbot") and recall them later ("what is SideEye?")
- Recall things from earlier in the same conversation even if you never
  explicitly taught them — "what did I say about it?", "do you remember
  my project?", "have I mentioned Hades before?" all search the full chat
  log, not just the taught-facts list (see "Remembering the conversation"
  below)
- Learn custom canned replies ("when I say ping reply pong")
- Switch mood
- Slash commands: `/mood <acid|deadpan|soft>`, `/help`, `/reset`

## Known limitations

This is traditional pattern-matching, not real language understanding,
so a few rough edges are expected and are more "future work" than bugs:

- Reference resolution is a recency heuristic, not real coreference —
  if you mention three things in a row and then say "it", SideEye
  guesses based on what was mentioned most recently and its best plural
  guess, which won't always be what a human would infer.
- Slot extraction is regex-based, so phrasing far outside the patterns in
  `extraction.py` may not get parsed into a clean value (it'll usually
  still get *acknowledged*, just with a rougher value).
- The intent scorer is tuned by hand-picked weights, not trained on data,
  so unusual phrasing can occasionally tie between two similar intents.

## Extending it

Everything is designed to be extended by adding to a list or dict rather
than restructuring code:

- **New intent**: add an entry to `INTENTS` in `nlu.py`.
- **New context slot**: add the key to `CONTEXT_KEYS` in `database.py`,
  write an extractor function in `extraction.py`, and register it in
  `EXTRACTORS`.
- **New joke**: add a string to `JOKES` in `utils.py`.
- **New mood**: add it to `MOODS` in `personality.py` and give every
  intent in `TEMPLATES` a template for it.
