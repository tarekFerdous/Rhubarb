"""Turn a grilling-style assistant reply into structured question blocks.

Pure local text parsing (regex) -- no extra Claude calls. Matches the real
format the `grilling` Claude Code skill produces:

    ❓ **Q1** - **Language/runtime**: Should this be Python or Node?
    Options: Python, Node

    ➡️ Python, since the rest of the codebase is Python.

    ---

    ❓ **Q2** - **Storage**: Where should results be persisted?

    ➡️ SQLite, matching the existing db module.

A question block starts with a line matching `❓ **Qn** - **<title>**: <body>`
(the `**<title>**:` part is optional -- if absent, `title` is None and the
whole remainder of the line is treated as the start of the body). The body
may continue across multiple lines/paragraphs until a `➡️ <recommendation>`
line, the next `❓` block, or a `---` separator on its own line. Blocks are
separated by a `---` line on its own.

Best-effort: any question the heuristics can't cleanly split into options
just falls back to a free-text answer box, so nothing is ever lost, only
sometimes rendered less richly.
"""

import re

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_Q_HEADER_RE = re.compile(r"^❓\s*\*\*Q(\d+)\*\*\s*-\s*(.*)$")
_TITLE_RE = re.compile(r"^\*\*(.+?)\*\*:\s*(.*)$")
_REC_RE = re.compile(r"^➡️\s*(.*)$")
_OPTIONS_RE = re.compile(r"^options:\s*(.*)$", re.IGNORECASE)


def _clean(text: str) -> str:
    return _BOLD_RE.sub(r"\1", text).strip()


def _detect_options(question: str) -> list[str] | None:
    if not question.endswith("?"):
        return None
    body = question[:-1]
    if " or " not in body:
        return None
    normalized = re.sub(r"\s+or\s+", ", ", body)
    parts = [p.strip(" '\"") for p in normalized.split(",") if p.strip()]
    if 2 <= len(parts) <= 4 and all(0 < len(p) <= 40 for p in parts):
        return parts
    return None


def _split_option_list(raw: str) -> list[str] | None:
    normalized = re.sub(r",?\s+or\s+", ", ", raw)
    parts = [p.strip(" '\"") for p in normalized.split(",") if p.strip()]
    return parts or None


def _parse_question_block(lines: list[str]) -> dict:
    """Parse one question block (header line + everything up to the next
    header or the end of the round) into a question dict."""
    header_match = _Q_HEADER_RE.match(lines[0].strip())
    qnum = header_match.group(1)
    rest = header_match.group(2).strip()

    title = None
    body_first = rest
    title_match = _TITLE_RE.match(rest)
    if title_match:
        title = _clean(title_match.group(1))
        body_first = title_match.group(2).strip()

    body_lines: list[str] = [body_first] if body_first else []
    rec_lines: list[str] = []
    options_raw: str | None = None
    in_recommendation = False

    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if stripped == "---":
            break
        if not stripped:
            continue

        rec_match = _REC_RE.match(stripped)
        if rec_match:
            in_recommendation = True
            if rec_match.group(1).strip():
                rec_lines.append(rec_match.group(1).strip())
            continue

        if in_recommendation:
            rec_lines.append(stripped)
            continue

        options_match = _OPTIONS_RE.match(stripped)
        if options_match:
            options_raw = options_match.group(1).strip()
            continue

        body_lines.append(stripped)

    text = _clean(" ".join(body_lines).strip())
    recommendation = _clean(" ".join(rec_lines).strip()) if rec_lines else None
    options = _split_option_list(options_raw) if options_raw else _detect_options(text)

    return {
        "id": f"q{qnum}",
        "title": title,
        "text": text,
        "recommendation": recommendation,
        "options": options,
    }


def parse_grilling_response(text: str) -> dict:
    """Return {"preamble": str, "sections": [{"heading": str|None, "questions": [...]}]}.

    Each question is
    {"id": str, "title": str | None, "text": str, "recommendation": str | None, "options": list[str] | None}.

    `sections` never groups under headings for this format -- it's always a
    single flat section (`heading: None`) containing every question in the
    round, or `[]` when the response has no `❓` blocks at all.
    """
    lines = text.splitlines()
    header_indices = [i for i, line in enumerate(lines) if _Q_HEADER_RE.match(line.strip())]

    if not header_indices:
        preamble_lines = [line.strip() for line in lines if line.strip()]
        return {"preamble": " ".join(preamble_lines).strip(), "sections": []}

    preamble_lines = [line.strip() for line in lines[: header_indices[0]] if line.strip()]

    questions = []
    for idx, start in enumerate(header_indices):
        end = header_indices[idx + 1] if idx + 1 < len(header_indices) else len(lines)
        questions.append(_parse_question_block(lines[start:end]))

    sections = [{"heading": None, "questions": questions}] if questions else []

    return {
        "preamble": " ".join(preamble_lines).strip(),
        "sections": sections,
    }
