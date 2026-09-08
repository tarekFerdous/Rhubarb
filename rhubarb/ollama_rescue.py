"""Ollama rescue parser (issue #114): a local, offline LLM fallback for
turns that `qa_parser.py`'s strict regex parser can't match.

Only invoked when the regex parser comes back empty on text that looks like
it was trying to be in the structured format -- see
`should_attempt_grilling_rescue`/`should_attempt_qa_rescue` and
`session_runner.py`'s call sites. The HTTP call is injectable, mirroring
`pty_engine.PtyEngine`'s `pty_factory` pattern, so tests never require a
real Ollama install.

The rescue call is prompted, using Ollama's JSON-schema-constrained
generation mode, to directly produce the exact same shape
`qa_parser.parse_grilling_response`/`parse_qa_response` already return.
Anything that doesn't validate against that shape -- malformed JSON, wrong
types, missing keys, a timeout, or a connection failure -- is treated as a
rescue failure: the public functions here return `None`, and callers fall
back to the pre-rescue empty result. Nothing partially trusted is ever
returned.
"""

import json
import urllib.request

from rhubarb.ollama_installer import OLLAMA_BASE_URL, OLLAMA_MODEL

RESCUE_TIMEOUT_SECONDS = 60

_GRILLING_TRIGGER = "question "
_QA_TRIGGER = "qa session for prd"

_GRILLING_SCHEMA = {
    "type": "object",
    "properties": {
        "header": {"type": "string"},
        "footer": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": ["single", "multi", "open"]},
                    "options": {"type": ["array", "null"], "items": {"type": "string"}},
                    "recommended": {"type": ["array", "null"], "items": {"type": "integer"}},
                    "recommended_text": {"type": ["string", "null"]},
                },
                "required": ["id", "text", "kind", "options", "recommended", "recommended_text"],
            },
        },
    },
    "required": ["header", "footer", "questions"],
}

_QA_SCHEMA = {
    "type": "object",
    "properties": {
        "prd": {
            "type": ["object", "null"],
            "properties": {"number": {"type": "integer"}, "title": {"type": "string"}},
            "required": ["number", "title"],
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "recommended_text": {"type": ["string", "null"]},
                            },
                            "required": ["id", "text", "recommended_text"],
                        },
                    },
                },
                "required": ["number", "title", "questions"],
            },
        },
    },
    "required": ["prd", "issues"],
}

_GRILLING_PROMPT_TEMPLATE = (
    "The following text is a Claude Code assistant's reply during a "
    "requirements-grilling round. It was meant to contain one or more "
    'questions in a strict format (Question N: "..." / Options: / '
    'Option N: "..." / Recommended: [...] or Recommended text: "..."), '
    "but a parser failed to extract them -- likely due to formatting "
    "drift. Extract the header prose (before the first question), each "
    "question (with its kind -- single-select, multi-select, or "
    "open-ended -- its options if any, and its recommendation), and the "
    "footer prose (after the last question), as JSON matching the given "
    "schema. If truly no questions are present, return empty questions "
    "with the whole text as header.\n\nText:\n{text}"
)

_QA_PROMPT_TEMPLATE = (
    "The following text is a Claude Code assistant's reply during a QA "
    "verification round. It was meant to contain a PRD number/title "
    "followed by one or more issues, each with its own verification "
    'questions, in a strict format (QA session for PRD N: "..." / '
    'Issue N: "..." / Question N: "..." / optional Recommended text: '
    '"..."), but a parser failed to extract them -- likely due to '
    "formatting drift. Extract the PRD number/title, each issue (number, "
    "title), and each issue's questions (with any recommended text), as "
    "JSON matching the given schema.\n\nText:\n{text}"
)


def should_attempt_grilling_rescue(parsed: dict, raw_text: str) -> bool:
    """True when `parse_grilling_response` found nothing but the raw text
    looks like it was trying to contain questions -- never true for a
    genuine "grilling is done" wrap-up turn, which contains no
    `"Question "` substring at all."""
    return not parsed["questions"] and _GRILLING_TRIGGER in raw_text.lower()


def should_attempt_qa_rescue(parsed: dict, raw_text: str) -> bool:
    """True when `parse_qa_response` found nothing but the raw text looks
    like it was trying to be a QA session."""
    return not parsed["issues"] and _QA_TRIGGER in raw_text.lower()


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
            if not isinstance(q, dict):
                return False
            if not isinstance(q.get("id"), str) or not isinstance(q.get("text"), str):
                return False
            if not q["text"].strip():
                return False
            if q.get("recommended_text") is not None and not isinstance(q.get("recommended_text"), str):
                return False
    return True


def rescue_grilling_response(raw_text: str, *, http_post=None) -> dict | None:
    """Attempt to rescue a grilling turn's raw text into the
    `{header, questions, footer, source}` shape via Ollama, with
    `source: "ollama_rescue"` marking it as such (see `qa_parser.
    parse_grilling_response`'s `"regex"` counterpart) so the frontend can
    flag it for the user to double-check. Returns `None` on any failure --
    callers fall back to the pre-rescue empty result."""
    data = _call_ollama(_GRILLING_PROMPT_TEMPLATE.format(text=raw_text), _GRILLING_SCHEMA, http_post=http_post)
    if not _is_valid_grilling_shape(data):
        return None
    data["source"] = "ollama_rescue"
    return data


def rescue_qa_response(raw_text: str, *, http_post=None) -> dict | None:
    """Attempt to rescue a QA turn's raw text into the
    `{prd, issues, source}` shape via Ollama, with `source: "ollama_rescue"`
    marking it as such. Returns `None` on any failure -- callers fall back
    to the pre-rescue empty result."""
    data = _call_ollama(_QA_PROMPT_TEMPLATE.format(text=raw_text), _QA_SCHEMA, http_post=http_post)
    if not _is_valid_qa_shape(data):
        return None
    data["source"] = "ollama_rescue"
    return data
