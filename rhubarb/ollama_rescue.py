"""Ollama-backed helpers (issue #114 originally; substantially retired by
issue #230, `gh issue view 230` for full context).

This module used to hold a local, offline LLM "rescue" fallback
(`rescue_grilling_response`/`rescue_qa_response`, gated by
`should_attempt_grilling_rescue`/`should_attempt_qa_rescue`) for turns that
`qa_parser.py`'s strict regex parser couldn't match. Issue #230 retired
`qa_parser.py` entirely and migrated every call site (grilling, QA-grilling,
implement) onto the parser-session/`/rhubarb:parse-interview` skill pipeline
(`rhubarb/parser_session.py`) as the one uniform extraction mechanism, with
no per-phase regex-then-Ollama-rescue fallback left anywhere -- so those
four functions, and the schema/prompt constants they alone used, were
deleted along with them.

What remains: `_is_valid_grilling_shape`/`_is_valid_qa_shape` (still used to
validate the parser-session pipeline's own JSON responses --
`parser_session.py`/`session_runner.py` import them directly), the shared
`_call_ollama`/`_default_http_post` HTTP-call machinery, and
`classify_turn_needs_input` below -- a separate, still-live concern (issue
#175) unrelated to question extraction.

## Needs-input classification (issue #175, child of PRD #174)

`classify_turn_needs_input` is a second, unrelated use of the same Ollama
call machinery: given a COMPLETED turn's already-rendered text and the
phase name that produced it, ask Ollama whether that text is waiting on a
human (a question, a choice to make) as opposed to being a self-contained
update or a turn that will keep going on its own. This replaces the
per-chunk PTY silence-timer stall mechanism (PRD #168, then #172/#173,
retired by this same change in `pty_engine.py`) with one semantic
classification call made ONCE per finished turn, instead of a purely
timing-based heuristic evaluated continuously while a turn is still in
flight.

Same failure philosophy as the rescue functions above: a timeout, a
connection failure, invalid JSON, or a response that doesn't validate
against the `{needs_input, reason}` shape all collapse to a plain `None` --
never a partially-trusted result. This module only builds the call itself;
it is not yet wired into any phase's turn-handling logic (see
`session_runner.classify_needs_input` for the small caller-side wrapper
that skips this entirely when the user has declined Ollama assistance, and
publishes a distinct "Ollama unavailable" notification when the call was
attempted but failed).
"""

import json
import urllib.request

from rhubarb.ollama_installer import OLLAMA_BASE_URL, OLLAMA_MODEL

RESCUE_TIMEOUT_SECONDS = 60


def _default_http_post(url: str, body: dict, *, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _call_ollama(prompt: str, schema: dict, *, http_post=None):
    """Call Ollama's `/api/generate` with JSON-schema-constrained output.
    Returns the parsed response body (whatever shape it turns out to be --
    validation is the caller's job), or `None` on any failure (timeout,
    connection error, invalid JSON). Never raises."""
    http_post = http_post or _default_http_post
    body = {"model": OLLAMA_MODEL, "prompt": prompt, "format": schema, "stream": False}
    print(prompt)
    try:
        result = http_post(f"{OLLAMA_BASE_URL}/api/generate", body, timeout=RESCUE_TIMEOUT_SECONDS)
        return json.loads(result["response"])
    except Exception:
        return None


def _is_valid_grilling_shape(data) -> bool:
    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("header"), str) or not isinstance(data.get("footer"), str):
        return False
    questions = data.get("questions")
    if not isinstance(questions, list):
        return False
    for q in questions:
        if not isinstance(q, dict):
            return False
        if not isinstance(q.get("id"), str) or not isinstance(q.get("text"), str):
            return False
        if not q["text"].strip():
            return False
        kind = q.get("kind")
        if kind not in ("single", "multi", "open"):
            return False
        options = q.get("options")
        if options is not None:
            if not isinstance(options, list) or not options:
                return False
        if kind == "open" and options is not None:
            return False
        recommended = q.get("recommended")
        if recommended is not None:
            if not isinstance(recommended, list):
                return False
            # An index can only be checked against options that actually
            # exist -- no options means no valid index at all.
            if not options:
                return False
            for idx in recommended:
                if not isinstance(idx, int) or isinstance(idx, bool) or not (1 <= idx <= len(options)):
                    return False
        if q.get("recommended_text") is not None and not isinstance(q.get("recommended_text"), str):
            return False
    return True


def _is_valid_qa_question_shape(q) -> bool:
    """One QA-grilling issue's question. Extended by PRD #227's follow-up
    (gap 2, `gh issue view 227`) to accept the SAME underlying per-question
    schema `_is_valid_grilling_shape` validates (`kind`/`options`/
    `recommended`, alongside the pre-existing `id`/`text`/
    `recommended_text`) -- the `parse-interview` skill's `qa_grilling_issues`
    phase now emits this fuller schema uniformly, reusing the exact same
    Recommended:-line-mapping/options rules the flat shape already has,
    rather than the old QA-only "always open, recommended_text only" inline
    prompt. `kind`/`options`/`recommended` stay OPTIONAL here (absent is
    still valid, meaning a plain open question with no options) since a real
    QA verification question is still overwhelmingly open-ended, and older
    extractions/fixtures never carried these fields at all."""
    if not isinstance(q, dict):
        return False
    if not isinstance(q.get("id"), str) or not isinstance(q.get("text"), str):
        return False
    if not q["text"].strip():
        return False
    kind = q.get("kind")
    if kind is not None and kind not in ("single", "multi", "open"):
        return False
    options = q.get("options")
    if options is not None:
        if not isinstance(options, list) or not options:
            return False
    recommended = q.get("recommended")
    if recommended is not None:
        if not isinstance(recommended, list):
            return False
        # Same as `_is_valid_grilling_shape`: an index can only be checked
        # against options that actually exist.
        if not options:
            return False
        for idx in recommended:
            if not isinstance(idx, int) or isinstance(idx, bool) or not (1 <= idx <= len(options)):
                return False
    if q.get("recommended_text") is not None and not isinstance(q.get("recommended_text"), str):
        return False
    return True


def _is_valid_qa_shape(data) -> bool:
    if not isinstance(data, dict):
        return False
    prd = data.get("prd")
    if prd is not None:
        if not isinstance(prd, dict) or not isinstance(prd.get("number"), int) or not isinstance(prd.get("title"), str):
            return False
    issues = data.get("issues")
    if not isinstance(issues, list):
        return False
    for issue in issues:
        if not isinstance(issue, dict):
            return False
        if not isinstance(issue.get("number"), int) or not isinstance(issue.get("title"), str):
            return False
        if not issue["title"].strip():
            return False
        questions = issue.get("questions")
        if not isinstance(questions, list):
            return False
        for q in questions:
            if not _is_valid_qa_question_shape(q):
                return False
    return True


# ---------------------------------------------------------------------------
# Needs-input classification (issue #175, child of PRD #174)
# ---------------------------------------------------------------------------

# Deliberately minimal: a bool plus an optional short human-readable reason
# (for logging/debugging, not shown anywhere yet) is enough to classify
# reliably without giving the model room to return something ambiguous.
_NEEDS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_input": {"type": "boolean"},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["needs_input", "reason"],
}

_NEEDS_INPUT_PROMPT_TEMPLATE = (
    "The following text is the complete, final reply a Claude Code assistant "
    'just gave for one turn during Rhubarb\'s "{phase}" phase. Decide '
    "whether a human needs to read this and respond before the assistant "
    "can usefully continue -- for example, it asked a question, presented "
    "choices, or is waiting for approval/clarification -- as opposed to "
    "being a self-contained update, a finished result, or a turn that will "
    "keep going on its own without any human input. Answer as JSON matching "
    "the given schema: `needs_input` (true or false), and `reason` (a short "
    "one-sentence explanation, or null if you have nothing to add).\n\n"
    "Text:\n{text}"
)


def _is_valid_needs_input_shape(data) -> bool:
    if not isinstance(data, dict):
        return False
    if not isinstance(data.get("needs_input"), bool):
        return False
    reason = data.get("reason")
    if reason is not None and not isinstance(reason, str):
        return False
    return True


def classify_turn_needs_input(text: str, phase: str, *, http_post=None) -> dict | None:
    """Ask Ollama whether a completed turn's rendered `text` (produced
    during `phase`) needs a human's input before the session can usefully
    continue. Returns `{"needs_input": bool, "reason": str | None}` on a
    valid response, or `None` on any failure (timeout, connection error,
    invalid JSON, or a response that doesn't validate against this shape) --
    never a partially-trusted result, same failure philosophy as
    `_call_ollama` itself.

    Does NOT itself check `db.get_ollama_declined` -- see
    `session_runner.classify_needs_input` for the small wrapper that skips
    this call entirely when the user has declined Ollama assistance, and
    turns a failure here (while NOT declined) into a distinct "Ollama
    unavailable" notification rather than silently doing nothing."""
    data = _call_ollama(
        _NEEDS_INPUT_PROMPT_TEMPLATE.format(phase=phase, text=text), _NEEDS_INPUT_SCHEMA, http_post=http_post
    )
    if not _is_valid_needs_input_shape(data):
        return None
    return data
