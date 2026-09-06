"""Turn a structured assistant reply into question blocks for the frontend.

Pure local text parsing (regex) -- no extra Claude calls. Two independent,
strict formats are parsed here, each unambiguous by construction (unlike the
old heuristic-based `_detect_options`/`_OPTIONS_RE`/"❓ Qn" convention this
replaces), so nothing here needs to guess.

Grilling/do format (see `parse_grilling_response`), produced by the
`grilling`/`grill-with-docs` skills:

    Here is round 1 of questions.

    Question 1: "Should this be Python or Node?"
    Options:
    Option 1: "Python"
    Option 2: "Node"
    Recommended: [1]

    Question 2 (select multiple): "Which environments should this support?"
    Options:
    Option 1: "Dev"
    Option 2: "Staging"
    Option 3: "Prod"
    Recommended: [1, 3]

    Question 3: "Where should this run?"
    Recommended text: "On the existing droplet, matching current infra."

    Based on your answers, a second wave might be needed.

QA format (see `parse_qa_response`), produced by the `qa`/`qa-grilling`
skills -- every QA question is open-ended (no options), grouped by the
issue it verifies against:

    QA session for PRD 98: "Structured question format"

    Issue 99: "Fix textarea auto-grow bug"
    Question 1: "Does a pre-filled recommendation box resize immediately?"
    Recommended text: "Yes, confirmed in the browser."
    Question 2: "Does typing still auto-grow the box?"

    Issue 100: "Rewrite qa_parser.py"
    Question 1: "Do the new unit tests cover multi-select?"

`_parse_implement_blocked_block`/`_parse_qa_grilling_block` in
`session_runner.py` (the JSON handoff markers gating phase transitions) are a
separate, unrelated mechanism -- not touched here.
"""

import re

_GRILLING_Q_RE = re.compile(r'^Question\s+(\d+)\s*(\(select multiple\))?\s*:\s*"(.*)"\s*$', re.IGNORECASE)
_OPTIONS_LABEL_RE = re.compile(r"^Options:\s*$", re.IGNORECASE)
_OPTION_RE = re.compile(r'^Option\s+(\d+)\s*:\s*"(.*)"\s*$', re.IGNORECASE)
_RECOMMENDED_RE = re.compile(r"^Recommended:\s*\[\s*([\d,\s]+)\]\s*$", re.IGNORECASE)
_RECOMMENDED_TEXT_RE = re.compile(r'^Recommended text:\s*"(.*)"\s*$', re.IGNORECASE)

_QA_HEADER_RE = re.compile(r'^QA session for PRD\s+(\d+)\s*:\s*"(.*)"\s*$', re.IGNORECASE)
_ISSUE_HEADER_RE = re.compile(r'^Issue\s+(\d+)\s*:\s*"(.*)"\s*$', re.IGNORECASE)
_QA_Q_RE = re.compile(r'^Question\s+(\d+)\s*:\s*"(.*)"\s*$', re.IGNORECASE)


def _join(lines: list[str]) -> str:
    return " ".join(line.strip() for line in lines if line.strip()).strip()


def _is_structured_grilling_line(line: str) -> bool:
    return bool(
        _OPTIONS_LABEL_RE.match(line)
        or _OPTION_RE.match(line)
        or _RECOMMENDED_RE.match(line)
        or _RECOMMENDED_TEXT_RE.match(line)
    )


def _footer_lines(last_block_lines: list[str]) -> list[str]:
    """`last_block_lines` is the final question's block, header line
    included. Returns the free-text lines that follow it: everything after
    the last recognised structured line, or everything after the header line
    itself if the block has no structured content at all (a bare open
    question with no `Recommended text:` line)."""
    last_structured = 0
    for i, raw_line in enumerate(last_block_lines):
        if i == 0:
            continue
        if _is_structured_grilling_line(raw_line.strip()):
            last_structured = i
    return last_block_lines[last_structured + 1 :]


def _parse_grilling_question(lines: list[str]) -> dict:
    """Parse one `Question N: "..."` block (header line + everything up to
    the next Question header or the end of input) into a question dict."""
    header_match = _GRILLING_Q_RE.match(lines[0].strip())
    qnum = header_match.group(1)
    is_multi = header_match.group(2) is not None
    text = header_match.group(3)

    options: list[str] = []
    recommended: list[int] | None = None
    recommended_text: str | None = None

    for raw_line in lines[1:]:
        stripped = raw_line.strip()
        if not stripped:
            continue

        option_match = _OPTION_RE.match(stripped)
        if option_match:
            options.append(option_match.group(2))
            continue

        rec_match = _RECOMMENDED_RE.match(stripped)
        if rec_match:
            recommended = [int(n) for n in rec_match.group(1).split(",") if n.strip()]
            continue

        rec_text_match = _RECOMMENDED_TEXT_RE.match(stripped)
        if rec_text_match:
            recommended_text = rec_text_match.group(1)
            continue

        # `Options:` label lines and anything unrecognised (including a
        # trailing footer that follows the round's last question) are
        # ignored here -- they carry no information for this question.

    if options:
        kind = "multi" if is_multi else "single"
    else:
        kind = "open"

    return {
        "id": f"q{qnum}",
        "text": text,
        "kind": kind,
        "options": options or None,
        "recommended": recommended if options else None,
        "recommended_text": recommended_text if not options else None,
    }


def parse_grilling_response(text: str) -> dict:
    """Return {"header": str, "questions": [...], "footer": str}.

    Each question is
    {"id": str, "text": str, "kind": "single"|"multi"|"open",
     "options": list[str] | None, "recommended": list[int] | None,
     "recommended_text": str | None}.

    `header` is the free text before the first `Question N:` line; `footer`
    is the free text after the last question's own structured content. Both
    are `""` when absent. A response with no `Question N:` lines at all
    returns `{"header": <all text>, "questions": [], "footer": ""}`.
    """
    lines = text.splitlines()
    header_indices = [i for i, line in enumerate(lines) if _GRILLING_Q_RE.match(line.strip())]

    if not header_indices:
        return {"header": _join(lines), "questions": [], "footer": ""}

    header = _join(lines[: header_indices[0]])

    questions = []
    for idx, start in enumerate(header_indices):
        end = header_indices[idx + 1] if idx + 1 < len(header_indices) else len(lines)
        questions.append(_parse_grilling_question(lines[start:end]))

    footer = _join(_footer_lines(lines[header_indices[-1] :]))

    return {"header": header, "questions": questions, "footer": footer}


def parse_qa_response(text: str) -> dict:
    """Return {"prd": {"number": int, "title": str} | None, "issues": [...]}.

    Each issue is {"number": int, "title": str, "questions": [...]}, and each
    question is {"id": str, "text": str, "recommended_text": str | None} --
    QA questions are always open-ended, never carry options. `id` is scoped
    to the issue (`Question N:` numbering restarts per issue) but unique
    across the whole session, shaped `issue<issue_number>-q<n>`.

    Returns `{"prd": None, "issues": []}` when the text doesn't contain a
    `QA session for PRD N: "..."` header line.
    """
    lines = text.splitlines()
    prd_index = next((i for i, line in enumerate(lines) if _QA_HEADER_RE.match(line.strip())), None)
    if prd_index is None:
        return {"prd": None, "issues": []}

    prd_match = _QA_HEADER_RE.match(lines[prd_index].strip())
    prd = {"number": int(prd_match.group(1)), "title": prd_match.group(2)}

    issue_indices = [i for i in range(prd_index + 1, len(lines)) if _ISSUE_HEADER_RE.match(lines[i].strip())]

    issues = []
    for idx, start in enumerate(issue_indices):
        end = issue_indices[idx + 1] if idx + 1 < len(issue_indices) else len(lines)
        issue_match = _ISSUE_HEADER_RE.match(lines[start].strip())
        issue_number = int(issue_match.group(1))
        issue_title = issue_match.group(2)

        question_indices = [i for i in range(start + 1, end) if _QA_Q_RE.match(lines[i].strip())]
        questions = []
        for qidx, qstart in enumerate(question_indices):
            qend = question_indices[qidx + 1] if qidx + 1 < len(question_indices) else end
            q_match = _QA_Q_RE.match(lines[qstart].strip())
            qnum = q_match.group(1)
            recommended_text = None
            for raw_line in lines[qstart + 1 : qend]:
                rec_match = _RECOMMENDED_TEXT_RE.match(raw_line.strip())
                if rec_match:
                    recommended_text = rec_match.group(1)
                    break
            questions.append(
                {
                    "id": f"issue{issue_number}-q{qnum}",
                    "text": q_match.group(2),
                    "recommended_text": recommended_text,
                }
            )

        issues.append({"number": issue_number, "title": issue_title, "questions": questions})

    return {"prd": prd, "issues": issues}
