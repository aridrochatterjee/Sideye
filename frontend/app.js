/*
  app.js
  ------
  Plain JS, no build step, no framework -- fetch() against SideEye's own
  local API and render the result. On load it restores whatever is
  already in the database (chat history + case file + mood) so refreshing
  the page doesn't lose the conversation.
*/

const messagesEl = document.getElementById("messages");
const suggestionsEl = document.getElementById("suggestions");
const composerEl = document.getElementById("composer");
const inputEl = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const caseFieldsEl = document.getElementById("caseFields");
const caseEmptyNote = document.getElementById("caseEmptyNote");
const moodDot = document.getElementById("moodDot");
const moodLabel = document.getElementById("moodLabel");
const moodButtons = document.querySelectorAll("#moodButtons button");
const eyeGlyph = document.getElementById("eyeGlyph");
const dossierToggle = document.getElementById("dossierToggle");
const dossier = document.getElementById("dossier");
const reasoningToggle = document.getElementById("reasoningToggle");

const FIELD_LABELS = {
  project: "project",
  language: "language",
  feature: "feature",
  problem: "problem",
  subject: "subject",
  game: "game",
  goal: "goal",
  task: "task",
  topic: "topic",
};

let lastContextKeys = new Set();

// -------------------------------------------------------------- utils --

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function formatTime(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch (e) {
    return "";
  }
}

// ----------------------------------------------------------- rendering --

function appendMessage(role, content, meta) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  row.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "msg-meta";
    metaEl.textContent = meta;
    row.appendChild(metaEl);
  }

  messagesEl.appendChild(row);
  return row;
}

function appendReasoningLine(row, debug) {
  if (!debug || !reasoningToggle.checked) return;
  const line = document.createElement("div");
  line.className = "reasoning-line";
  const replacements = (debug.reference_replacements || [])
    .map(([pronoun, value]) => `"${pronoun}" -> "${value}"`)
    .join(", ");
  const top = (debug.top_scores || [])
    .filter(([, score]) => score > 0)
    .slice(0, 3)
    .map(([name, score]) => `${name}:${score}`)
    .join(" ");
  let text = "";
  if (replacements) text += `resolved: ${replacements}. `;
  if (top) text += `scores: ${top}`;
  if (text) {
    line.textContent = text;
    row.appendChild(line);
  }
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "msg-row bot typing";
  row.id = "typingRow";
  const bubble = document.createElement("div");
  bubble.className = "bubble typing-dots";
  bubble.textContent = "thinking";
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  scrollToBottom();
}

function hideTyping() {
  const row = document.getElementById("typingRow");
  if (row) row.remove();
}

function renderCaseFields(context) {
  const keys = Object.keys(context || {});
  caseFieldsEl.querySelectorAll(".case-field").forEach((el) => el.remove());

  if (keys.length === 0) {
    caseEmptyNote.style.display = "block";
  } else {
    caseEmptyNote.style.display = "none";
    keys.forEach((key) => {
      const dt = document.createElement("dt");
      dt.textContent = FIELD_LABELS[key] || key;
      const dd = document.createElement("dd");
      dd.textContent = context[key];
      const wrapper = document.createElement("div");
      wrapper.className = "case-field";
      wrapper.appendChild(dt);
      wrapper.appendChild(dd);
      caseFieldsEl.appendChild(wrapper);
    });
  }

  const newKeys = new Set(keys);
  let grew = false;
  newKeys.forEach((k) => {
    if (!lastContextKeys.has(k)) grew = true;
  });
  if (grew && lastContextKeys.size > 0) {
    triggerEyeBlink();
  }
  lastContextKeys = newKeys;
}

function triggerEyeBlink() {
  eyeGlyph.classList.remove("blink");
  // Force reflow so the animation can re-trigger on consecutive updates.
  void eyeGlyph.offsetWidth;
  eyeGlyph.classList.add("blink");
}

function renderMood(mood) {
  const safeMood = mood || "deadpan";
  moodDot.className = `mood-dot ${safeMood}`;
  moodLabel.textContent = safeMood;
  moodButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mood === safeMood);
  });
}

function renderSuggestions(suggestions) {
  suggestionsEl.innerHTML = "";
  (suggestions || []).forEach((text) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "suggestion-chip";
    chip.textContent = text;
    chip.addEventListener("click", () => sendMessage(text));
    suggestionsEl.appendChild(chip);
  });
}

// -------------------------------------------------------------- network --

async function sendMessage(text) {
  const trimmed = (text || "").trim();
  if (!trimmed) return;

  inputEl.value = "";
  sendButton.disabled = true;

  appendMessage("user", trimmed);
  scrollToBottom();
  showTyping();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: trimmed }),
    });
    const data = await res.json();
    hideTyping();

    if (data.error) {
      appendMessage("bot", `Something broke on the server: ${data.error}`);
    } else {
      const row = appendMessage("bot", data.reply);
      appendReasoningLine(row, data.debug);
      renderCaseFields(data.context);
      renderMood(data.mood);
      renderSuggestions(data.suggestions);
    }
  } catch (err) {
    hideTyping();
    appendMessage("bot", "Couldn't reach the SideEye server. Is it still running?");
  } finally {
    sendButton.disabled = false;
    scrollToBottom();
    inputEl.focus();
  }
}

async function loadInitialState() {
  try {
    const [contextRes, historyRes] = await Promise.all([
      fetch("/api/context"),
      fetch("/api/history"),
    ]);
    const context = await contextRes.json();
    const history = await historyRes.json();

    renderMood(context.mood);
    renderCaseFields(context.context);
    lastContextKeys = new Set(Object.keys(context.context || {}));

    if (history.history && history.history.length > 0) {
      history.history.forEach((msg) => {
        appendMessage(msg.role, msg.content, formatTime(msg.created_at));
      });
      scrollToBottom();
    } else {
      appendMessage(
        "bot",
        "Hey. I'm SideEye. Tell me what you're building, ask me for a joke, " +
        "or roll some dice -- I'm not picky. Try the quick actions on the left."
      );
    }
  } catch (err) {
    appendMessage(
      "bot",
      "Couldn't load saved state -- starting fresh. (Is the server running?)"
    );
  }
}

// --------------------------------------------------------------- events --

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(inputEl.value);
});

moodButtons.forEach((btn) => {
  btn.addEventListener("click", () => sendMessage(`/mood ${btn.dataset.mood}`));
});

document.querySelectorAll(".quick-actions button[data-quick]").forEach((btn) => {
  btn.addEventListener("click", () => sendMessage(btn.dataset.quick));
});

dossierToggle.addEventListener("click", () => {
  dossier.classList.toggle("open");
});

document.addEventListener("click", (e) => {
  const withinDossier = dossier.contains(e.target);
  const isToggle = dossierToggle.contains(e.target);
  if (!withinDossier && !isToggle && dossier.classList.contains("open")) {
    if (window.innerWidth <= 760) dossier.classList.remove("open");
  }
});

loadInitialState();
inputEl.focus();
