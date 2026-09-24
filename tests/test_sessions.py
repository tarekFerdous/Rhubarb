import asyncio
import json
import re
import subprocess
from pathlib import Path

import pytest

from rhubarb import db, error_log, live_stream, parser_session, session_runner, stream_json_engine
from rhubarb.cli_client import ClaudeCLIError
from rhubarb.stream_json_engine import StreamJsonEngineUnrecoverableError
from rhubarb.web import app as app_module

# Captured at collection time, before any test's `monkeypatch` fixture ever
# touches `session_runner.classify_needs_input` -- the genuine function, for
# the handful of tests below that restore it deliberately (see
# `_classify_needs_input_is_a_no_op_by_default`'s docstring).
_REAL_CLASSIFY_NEEDS_INPUT = session_runner.classify_needs_input


@pytest.fixture(autouse=True)
def _classify_needs_input_is_a_no_op_by_default(monkeypatch):
    """Issue #177: `classify_needs_input` is now wired into every
    `_run_chain_step` call (`creating_prd`/`creating_issues`), firing after
    EVERY resolved chain-step turn regardless of content -- unlike the
    pre-existing grilling/QA rescue calls, which only ever fire once a
    parser has already come back empty on suspicious-looking text. Left
    alone, every existing test in this file that drives that chain would
    suddenly make a REAL call to whatever Ollama install happens to be
    running on the machine that runs these tests -- slow, and
    non-deterministic (a genuinely different/differently-tuned local model
    could classify the exact same turn text differently on a different
    machine, and a positive result changes what the chain actually does).

    Defaults every test in this file to a `classify_needs_input` stub that
    always returns `None` -- the same "skip: declined or unavailable"
    outcome the real function already gives for plenty of legitimate
    reasons, so every existing test's chain behavior/event stream stays
    exactly what it always has been, deterministically, with zero network
    calls. The handful of tests that actually want to exercise real
    `classify_needs_input` behavior restore the genuine function first
    (`monkeypatch.setattr(session_runner, "classify_needs_input",
    _REAL_CLASSIFY_NEEDS_INPUT)` -- see the "Needs-input classification"
    section below); the issue #177 wiring tests instead override this stub
    with their own scenario-specific fake. Either way, that happens in the
    test's own body, using the same `monkeypatch` fixture instance this
    autouse fixture already used, so it simply wins over this default for
    the rest of that one test."""

    async def _stub(card_id, conn, text, phase, *, http_post=None):
        return None

    monkeypatch.setattr(session_runner, "classify_needs_input", _stub)


def _init_repo(path, remote_url):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=path, check=True)
    subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=path, check=True)


def _open_project(client, tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    _init_repo(root / "repo", f"https://github.com/x/{name}.git")
    client.post("/api/settings/root-dir", json={"root_dir": str(root), "confirm": True})
    project_id = client.get("/api/app-state").json()["projects"][0]["id"]
    client.post(f"/api/projects/{project_id}/open")
    # `open_project` fire-and-forgets its own pre-warm of this project's
    # standby `StreamJsonEngine` (issue #136/#184) -- a background task
    # whose completion timing relative to this synchronous helper returning
    # is not guaranteed either way (TestClient may or may not pump the
    # event loop far enough for it to finish first). No test in this file
    # is testing THAT ambient pre-warm itself (the tests that actually cover
    # it call `ensure_standby_stream_json_engine` explicitly, well after
    # this point) -- so clear it here for a deterministic, known-empty
    # starting registry, the same guarantee `_isolated_standby_stream_json_
    # engines` (conftest.py) already gives every test BEFORE `_open_project`
    # runs, now also given AFTER it.
    session_runner._standby_stream_json_engines.clear()
    return project_id


def _cwd_for(project_id):
    conn = db.get_connection()
    return db.get_project(conn, project_id)["path"]


def _result_event(text, session_id="s1", usage=None, model_usage=None):
    event = {"type": "result", "subtype": "success", "is_error": False, "result": text, "session_id": session_id}
    if usage is not None:
        event["usage"] = usage
    if model_usage is not None:
        event["modelUsage"] = model_usage
    return event


async def _run_and_drain(coro):
    """Await `coro`, then let any `asyncio.create_task(...)` it scheduled
    (e.g. `start_or_queue_implement`'s fire-and-forget implement job) run to
    completion too, before returning -- a resident `PtyEngine` tab's work no
    longer necessarily happens on a background OS thread the way the old
    subprocess-per-turn model's blocking I/O did, so a test asserting on a
    scheduled job's *end state* (pooled, implemented, etc.) drains it
    explicitly instead of relying on incidental event-loop-shutdown timing."""
    result = await coro
    pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)
    return result


def _make_fake_engine_class(handler, *, fresh_ids=None, reply_handler=None):
    """Build a fake stand-in for `session_runner.StreamJsonEngine` -- the
    sole engine transport since issue #225 removed `PtyEngine` (every phase
    dispatches to it now, not just grilling since issue #184). Public shape
    (`__init__` kwargs, `start`/`close`/`isalive`/`stream_turn`) matches the
    real class. `session_id` is exposed as an alias for `claude_session_id`
    (kept as this class's own internal attribute name for minimal diff
    against this file's long history) so a test exercising the standby-claim
    path (`register_stream_json_engine`, which reads `.session_id`) works
    against this same fake.

    `reply_handler()`, if given, stands in for a resumed-turn call driven
    directly against `_run_stream_json_turn` in tests that bypass the
    higher-level job functions -- returns an iterable of raw event dicts
    (same shape `handler` returns). A test that never expects it to be
    called at all can omit it -- the fake then raises `AssertionError` if
    it's ever actually called, the same "fail loudly on an unexpected call"
    shape `handler`'s own `raise AssertionError` fallback branches already
    use throughout this file.

    `handler(prompt, *, session_id, cwd, model, effort)` returns an iterable
    of raw event dicts -- the same shape the old `stream_prompt` fakes
    already produced, minus `card_id` (that concept is gone: every phase
    drives its turns through the same one-resident-tab-per-card_id
    mechanism).

    Since `session_runner` keeps ONE engine alive across every turn for a
    card (see `session_runner._get_or_create_stream_json_engine`), a fake
    engine forwards every turn over its own lifetime to this SAME `handler`
    -- a test needing different behavior across turns gives `handler` one
    prompt-keyed dispatcher covering the whole scenario (a resident engine
    wouldn't see a mock swapped out from under it either).

    `fresh_ids`, if given, is a queue of ids handed out (in order) to
    successive *fresh* (no `resume_session_id`) constructions -- standing in
    for what a genuinely fresh engine's own generated uuid would be, so a
    test can pin down the exact id a "clear"/"pool" produces.
    """
    fresh_queue = list(fresh_ids or [])

    class FakeEngine:
        instances: list["FakeEngine"] = []

        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, pty_factory=None):
            self.cwd = cwd
            self.model = model
            self.effort = effort
            self.resume_session_id = resume_session_id
            if resume_session_id is not None:
                self.claude_session_id = resume_session_id
            elif fresh_queue:
                self.claude_session_id = fresh_queue.pop(0)
            else:
                self.claude_session_id = f"generated-{len(FakeEngine.instances) + 1}"
            self.started = False
            self.closed = False
            FakeEngine.instances.append(self)

        def start(self):
            self.started = True
            return self

        def close(self):
            self.closed = True

        def isalive(self):
            return self.started and not self.closed

        async def stream_turn(self, prompt):
            yield {"type": "system", "subtype": "init", "session_id": self.claude_session_id}
            # `session_id` here mirrors the old `stream_prompt(prompt,
            # session_id=...)` kwarg tests already keyed off of: the
            # conversation being *resumed*, or None for a fresh one -- not
            # this engine's own assigned `claude_session_id` (which is
            # always some non-None value, fresh uuid included).
            for raw in handler(
                prompt, session_id=self.resume_session_id, cwd=self.cwd, model=self.model, effort=self.effort
            ):
                yield raw

        async def stream_reply(self):
            if reply_handler is None:
                raise AssertionError(
                    "stream_reply must not be called -- this test's FakeEngine was built with no reply_handler"
                )
            for raw in reply_handler():
                yield raw

        @property
        def session_id(self):
            # `StreamJsonEngine`'s own attribute name for the same concept
            # `claude_session_id` already tracks here -- see this class's
            # docstring above.
            return self.claude_session_id

    return FakeEngine


def _mock_engine(monkeypatch, handler, *, fresh_ids=None, reply_handler=None):
    """Monkeypatch `session_runner.StreamJsonEngine` -- the sole engine
    transport since issue #225 removed `PtyEngine` -- with a fake class
    driven by `handler` (see `_make_fake_engine_class`). Returns the fake
    class so a test can inspect `.instances` (e.g. to assert an engine
    was/wasn't reconstructed, or to check constructor args). `handler`
    doesn't need to know or care which phase's turn it's actually being
    called for; the fake responds identically regardless of phase."""
    fake_class = _make_fake_engine_class(handler, fresh_ids=fresh_ids, reply_handler=reply_handler)
    monkeypatch.setattr(session_runner, "StreamJsonEngine", fake_class)
    return fake_class


def _mock_parser_session_extraction(
    monkeypatch, project_id, cwd, *, header="", questions=None, footer="", completion="__default__"
):
    """Issue #221, simplified by issue #230: register a live, fake parser
    session for `project_id` whose every turn replies with the given
    `{header, questions, footer}` payload as JSON -- for tests exercising
    `_run_grilling_turn_stream_json`'s synchronous parser-session dispatch
    (now the ONLY extraction path -- issue #230 removed the regex-first
    attempt this used to be a fallback for) without spawning a real
    subprocess. Mirrors `tests/test_parser_session.py`'s own
    `_mock_parser_engine` fake-class shape, plus an explicit
    `ensure_parser_session` call so the session is registered and ready
    *before* the test drives any grilling turn -- `open_project`'s own
    pre-warm of this is fire-and-forget and racy, so tests that need this
    path deterministic can't rely on it.

    Issue #242 (child of PRD #241): when `questions` comes back empty,
    the payload also carries a `"completion"` verdict, since grilling's own
    empty-frontier auto-advance now requires one -- defaulting to a
    positive verdict (`completion="__default__"`, this function's own
    sentinel default) keeps every existing caller of this helper (none of
    which are testing the completion gate itself) behaving exactly as
    before. A caller exercising the gate directly passes its own
    `completion=` (a dict, or `None` to omit the field entirely and exercise
    the missing-verdict fail-safe path)."""
    payload_dict = {"header": header, "questions": questions or [], "footer": footer}
    if not payload_dict["questions"]:
        if completion == "__default__":
            payload_dict["completion"] = {"done": True, "reason": "Fake extraction: nothing left to ask."}
        elif completion is not None:
            payload_dict["completion"] = completion
    payload = json.dumps(payload_dict)

    class FakeParserEngine:
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.session_id = resume_session_id or "parser-session-fake"
            self._alive = False

        def start(self):
            self._alive = True
            return self

        def close(self):
            self._alive = False

        def isalive(self):
            return self._alive

        async def stream_turn(self, prompt):
            yield {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": payload,
                "session_id": self.session_id,
            }

    monkeypatch.setattr(parser_session, "StreamJsonEngine", FakeParserEngine)
    # `_open_project`'s own `/open` POST fire-and-forgets its own parser-
    # session warm (`open_project` -> `_warm_project_engines` ->
    # `ensure_parser_session`) -- a background task that may or may not have
    # already registered (and started) a DIFFERENT (default no-op) engine
    # for this project_id by the time this helper runs, racily. Drop any
    # such entry first so `ensure_parser_session` below is forced to spawn
    # fresh under `FakeParserEngine` rather than silently reusing whatever
    # the race already registered (double-checked locking treats "already
    # registered and alive" as nothing to do).
    parser_session._parser_sessions.pop(project_id, None)
    asyncio.run(parser_session.ensure_parser_session(project_id, cwd=cwd))


_QUESTION_QUOTE_RE = re.compile(r'Question\s+\d+[^\n:]*:\s*"([^"]*)"', re.IGNORECASE)
_OPTION_QUOTE_RE = re.compile(r'Option\s+\d+\s*:\s*"([^"]*)"', re.IGNORECASE)
_RECOMMENDED_IDX_RE = re.compile(r"Recommended:\s*\[\s*([\d,\s]+)\]", re.IGNORECASE)

# PRD #227 gap 2: `_extract_qa_issues_via_skill` now goes through the same
# `/rhubarb:parse-interview` skill invocation (`_build_extraction_prompt`)
# grilling/implement use, just with `phase: qa_grilling_issues`, asking for
# the nested `{prd, issues: [{number, title, questions}]}` shape -- these
# mirror the qa-grilling skill's own `QA session for PRD N: "..."` /
# `Issue N: "..."` / `Question N: "..."` / `Recommended text: "..."` format
# closely enough to fake that extraction from these tests' own fixtures.
_QA_PRD_HEADER_RE = re.compile(r'QA session for PRD\s+(\d+)\s*:\s*"([^"]*)"', re.IGNORECASE)
_QA_ISSUE_HEADER_RE = re.compile(r'Issue\s+(\d+)\s*:\s*"([^"]*)"', re.IGNORECASE)
_QA_QUESTION_RE = re.compile(r'Question\s+(\d+)\s*:\s*"([^"]*)"', re.IGNORECASE)
_QA_RECOMMENDED_TEXT_RE = re.compile(r'Recommended text:\s*"([^"]*)"', re.IGNORECASE)


def _fake_autoextract_grilling_shape(prompt: str) -> dict:
    """Pulls every `Question N: "..."` text appearing in a turn's own prompt
    (the raw reply `_extract_questions_via_parser_session` embeds via
    `parser_session._build_extraction_prompt`) back out as one question
    apiece, in order -- a minimal stand-in for what a real LLM extraction
    pass would produce from these simple fixtures. A question's own trailing
    `Option N: "..."` lines and `Recommended: [...]` line (up to the next
    `Question N:` header, or end of prompt) are folded in too, so tests that
    check the extracted `options`/`recommended` survive the round trip. A
    prompt with no quoted `Question N:` text (a genuine wrap-up turn) yields
    zero questions, exactly like a real extraction of prose with nothing
    left to ask would.

    Issue #242 (child of PRD #241): a `phase: grilling` prompt (see
    `_build_extraction_prompt`) whose frontier comes back empty also gets a
    fake positive completion verdict, mirroring the real `/rhubarb:parse-
    interview` skill's own grilling-specific instructions -- these fixtures
    are always genuine wrap-ups, never the "still not actually done" case a
    dedicated negative-verdict test exercises directly instead."""
    matches = list(_QUESTION_QUOTE_RE.finditer(prompt))
    questions = []
    for i, match in enumerate(matches):
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(prompt)
        block = prompt[match.end() : block_end]
        options = _OPTION_QUOTE_RE.findall(block)
        rec_match = _RECOMMENDED_IDX_RE.search(block)
        recommended = [int(n) for n in rec_match.group(1).split(",") if n.strip()] if rec_match else None
        questions.append(
            {
                "id": f"q{i + 1}",
                "text": match.group(1),
                "kind": "single" if options else "open",
                "options": options or None,
                "recommended": recommended if options else None,
                "recommended_text": None,
            }
        )
    data = {"header": "", "questions": questions, "footer": ""}
    if not questions and "phase: grilling" in prompt:
        data["completion"] = {"done": True, "reason": "Fake autoextract: nothing left to ask."}
    return data


def _fake_autoextract_qa_shape(prompt: str) -> dict:
    """The QA-grilling analogue of `_fake_autoextract_grilling_shape` --
    pulls a `QA session for PRD N: "..."` header and its `Issue N: "..."` /
    `Question N: "..."` / `Recommended text: "..."` groups back out of the
    prompt's own embedded raw text into the nested `{prd, issues}` shape
    `_extract_qa_issues_via_skill` expects."""
    prd_match = _QA_PRD_HEADER_RE.search(prompt)
    prd = {"number": int(prd_match.group(1)), "title": prd_match.group(2)} if prd_match else None

    issue_matches = list(_QA_ISSUE_HEADER_RE.finditer(prompt))
    issues = []
    for i, issue_match in enumerate(issue_matches):
        issue_end = issue_matches[i + 1].start() if i + 1 < len(issue_matches) else len(prompt)
        issue_block = prompt[issue_match.end() : issue_end]
        issue_number = int(issue_match.group(1))

        question_matches = list(_QA_QUESTION_RE.finditer(issue_block))
        questions = []
        for qi, q_match in enumerate(question_matches):
            q_end = question_matches[qi + 1].start() if qi + 1 < len(question_matches) else len(issue_block)
            q_block = issue_block[q_match.end() : q_end]
            rec_match = _QA_RECOMMENDED_TEXT_RE.search(q_block)
            questions.append(
                {
                    "id": f"issue{issue_number}-q{q_match.group(1)}",
                    "text": q_match.group(2),
                    "recommended_text": rec_match.group(1) if rec_match else None,
                }
            )

        issues.append({"number": issue_number, "title": issue_match.group(2), "questions": questions})

    return {"prd": prd, "issues": issues}


def _mock_parser_session_autoextract(monkeypatch, project_id, cwd):
    """Issue #230: register a live, fake parser session for `project_id`
    that fakes a real LLM extraction pass from whatever raw text a turn's
    own prompt embeds -- for tests whose real subject is unrelated to
    extraction accuracy (session pooling/reuse, engine lifecycle,
    concurrency, the QA-grilling corrective-retry/needs-input chain's own
    control flow) but that still need a real, distinguishable question (or
    QA issue) to come back, now that a real parser-session round trip
    (mocked or not) is the only way one survives, per issue #230's removal
    of the regex-first fast paths these tests used to rely on instead.

    Dispatches on the prompt's own `phase:` line -- every extraction call
    now goes through the same `/rhubarb:parse-interview` skill invocation
    (`_build_extraction_prompt`, PRD #227 gap 2's unification of the
    QA-grilling extraction call `_extract_qa_issues_via_skill` onto the
    skill too), so this fakes the skill's OWN phase-based branching instead
    of the prompt's outer shape: `phase: qa_grilling_issues` (the nested-
    shape phase `_extract_qa_issues_via_skill` sends) gets the nested
    `{prd, issues}` shape (`_fake_autoextract_qa_shape`); every other phase
    (grilling/implementing's flat `{header, questions, footer}` shape) gets
    `_fake_autoextract_grilling_shape`."""

    class FakeParserEngine:
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.session_id = resume_session_id or "parser-session-fake"
            self._alive = False

        def start(self):
            self._alive = True
            return self

        def close(self):
            self._alive = False

        def isalive(self):
            return self._alive

        async def stream_turn(self, prompt):
            if "phase: qa_grilling_issues" in prompt:
                data = _fake_autoextract_qa_shape(prompt)
            else:
                data = _fake_autoextract_grilling_shape(prompt)
            payload = json.dumps(data)
            yield {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": payload,
                "session_id": self.session_id,
            }

    monkeypatch.setattr(parser_session, "StreamJsonEngine", FakeParserEngine)
    parser_session._parser_sessions.pop(project_id, None)
    asyncio.run(parser_session.ensure_parser_session(project_id, cwd=cwd))


def _make_blocking_fake_engine_class(entered, release, enter_count, *, result_text, session_id=None):
    """Build a fake stand-in for `StreamJsonEngine` whose `stream_turn` genuinely
    suspends mid-turn -- used by the issue #144 lock tests to put a turn
    "in flight" on purpose and hold it there, so a second, overlapping call
    for the same card_id can be made while the first has not yet finished.

    `entered` (an `asyncio.Event`) is set the moment `stream_turn` is
    actually entered -- a test awaits it to know the first call has reached
    the engine before attempting the second, overlapping one. `enter_count`
    (a `dict` with an `"n"` key) is incremented on every such entry, so a
    test can assert how many of several concurrent calls actually reached
    the engine (must always be exactly one -- the lock's whole point).
    `release` (another `asyncio.Event`) gates the turn's completion -- the
    fake only yields its final result event, and stream_turn only returns,
    once the test sets it.
    """

    class BlockingFakeEngine:
        instances: list["BlockingFakeEngine"] = []

        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, pty_factory=None):
            self.cwd = cwd
            self.model = model
            self.effort = effort
            self.resume_session_id = resume_session_id
            self.claude_session_id = resume_session_id or f"fresh-{len(BlockingFakeEngine.instances) + 1}"
            self.started = False
            self.closed = False
            BlockingFakeEngine.instances.append(self)

        def start(self):
            self.started = True
            return self

        def close(self):
            self.closed = True

        def isalive(self):
            return self.started and not self.closed

        @property
        def session_id(self):
            # `StreamJsonEngine`'s own attribute name for the same concept
            # `claude_session_id` already tracks here.
            return self.claude_session_id

        async def stream_turn(self, prompt):
            enter_count["n"] += 1
            entered.set()
            yield {"type": "system", "subtype": "init", "session_id": self.claude_session_id}
            await release.wait()
            yield _result_event(result_text, session_id=session_id or self.claude_session_id)

    return BlockingFakeEngine


def test_start_session_job_publishes_usage_from_a_rate_limit_event(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    rate_limit_event = {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "unifiedWindows": {
                "five_hour": {"utilization": 12.5},
                "seven_day": {"utilization": 3.1},
            }
        },
    }
    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([rate_limit_event, _result_event("❓ **Q1** - **Scope**: Only question?")]),
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    events = live_stream._buffers.get(row_id, [])
    assert {"type": "usage", "five_hour_pct": 12.5, "seven_day_pct": 3.1} in events
    assert live_stream.last_usage() == {"type": "usage", "five_hour_pct": 12.5, "seven_day_pct": 3.1}


def test_count_resident_engines_tracks_tabs_as_sessions_open_and_close(client, tmp_path, monkeypatch):
    """Backs the web UI's tab-count indicator (issue #88):
    `session_runner.count_resident_engines()` must accurately reflect how
    many engine tabs are currently resident as sessions start (opening a
    tab, one per card_id) and close (dropping it)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    assert session_runner.count_resident_engines() == 0

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event(f'Question 1: "{prompt}?"', session_id=prompt)]),
    )

    conn = db.get_connection()
    row_a = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_a, "feature A", cwd=cwd))
    assert session_runner.count_resident_engines() == 1

    row_b = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_b, "feature B", cwd=cwd))
    assert session_runner.count_resident_engines() == 2

    session_runner._close_stream_json_engine(row_a)
    assert session_runner.count_resident_engines() == 1

    session_runner._close_stream_json_engine(row_b)
    assert session_runner.count_resident_engines() == 0


def test_close_session_terminates_a_live_resident_engine(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event('Question 1: "Only question?"')]),
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    assert session_runner.count_resident_engines() == 1

    session_runner.close_session(conn, row_id)

    assert fake_class.instances[0].closed is True
    assert session_runner.count_resident_engines() == 0


def test_close_session_on_a_card_with_no_resident_engine_does_not_raise(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    session_runner.close_session(conn, row_id)  # no PtyEngine was ever started for this card

    assert db.get_session(conn, row_id)["phase"] == "closed"


def test_close_session_marks_the_row_closed_and_publishes_a_terminal_event(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    session_runner.close_session(conn, row_id)

    assert db.get_session(conn, row_id)["phase"] == "closed"
    events = live_stream._buffers.get(row_id, [])
    assert {"type": "closed", "card_id": row_id} in events


def test_two_sessions_advance_concurrently_without_cross_contamination(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    row_a = db.create_session(conn, project_id)
    row_b = db.create_session(conn, project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling feature A":
            return iter([_result_event('Question 1: "Question A?"', session_id="sA")])
        if prompt == "/rhubarb:grilling feature B":
            return iter([_result_event('Question 1: "Question B?"', session_id="sB")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    async def run_both():
        await asyncio.gather(
            session_runner.start_session_job(row_a, "feature A", cwd=cwd),
            session_runner.start_session_job(row_b, "feature B", cwd=cwd),
        )

    asyncio.run(run_both())

    row_a_data = db.get_session(conn, row_a)
    row_b_data = db.get_session(conn, row_b)
    assert row_a_data["claude_session_id"] == "sA"
    assert row_b_data["claude_session_id"] == "sB"

    interview_a = json.loads(row_a_data["interview_json"])
    interview_b = json.loads(row_b_data["interview_json"])
    assert interview_a["questions"][0]["text"] == "Question A?"
    assert interview_b["questions"][0]["text"] == "Question B?"

    events_a = live_stream._buffers.get(row_a, [])
    events_b = live_stream._buffers.get(row_b, [])
    assert any(e["type"] == "turn" and e["interview"] == interview_a for e in events_a)
    assert any(e["type"] == "turn" and e["interview"] == interview_b for e in events_b)
    # Neither session's buffer leaked the other's content.
    assert not any("Question B" in json.dumps(e) for e in events_a)
    assert not any("Question A" in json.dumps(e) for e in events_b)


def test_no_cap_on_the_number_of_sessions_running_at_once(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    row_ids = [db.create_session(conn, project_id) for _ in range(8)]

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event(f'Question 1: "{prompt}?"', session_id=prompt)]),
    )
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    async def run_all():
        await asyncio.gather(
            *[session_runner.start_session_job(row_id, f"feature {i}", cwd=cwd) for i, row_id in enumerate(row_ids)]
        )

    asyncio.run(run_all())

    for i, row_id in enumerate(row_ids):
        row = db.get_session(conn, row_id)
        assert row["claude_session_id"] == f"/rhubarb:grilling feature {i}"
        interview = json.loads(row["interview_json"])
        assert interview["questions"][0]["text"] == f"/rhubarb:grilling feature {i}?"


def test_start_session_job_returns_card_with_grilling_questions(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter(
            [
                _result_event(
                    'Question 1: "What should it do?"\n'
                    "\n"
                    'Question 2: "Who is it for?"\n'
                )
            ]
        ),
    )
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert len(interview["questions"]) == 2
    assert row["claude_session_id"] == "s1"

    events = live_stream._buffers.get(row_id, [])
    assert {"type": "phase", "phase": "grilling"} in events
    assert any(e["type"] == "turn" and e["interview"] == interview for e in events)


def _write_question_file(cwd, filename, content):
    claude_dir = Path(cwd) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / filename).write_text(content, encoding="utf-8")


# Issue #184: the five tests that used to live here
# (`test_start_session_job_prefers_rhubarb_question_file_over_terminal_text`,
# `test_rhubarb_question_file_is_not_deleted_merely_by_being_read`,
# `test_continue_session_job_deletes_rhubarb_question_file_on_reply`,
# `test_confirm_advance_also_deletes_rhubarb_question_file`,
# `test_malformed_rhubarb_question_file_is_deleted_and_falls_back_to_terminal_text`)
# covered PRD #123's `.claude/rhubarb_question.md` file-preference mechanism
# for GRILLING specifically. That mechanism is now dispatched-around for
# grilling entirely (`_run_grilling_turn_stream_json` never reads or deletes
# this file -- see its docstring) since grilling no longer runs on
# `PtyEngine`, whose PTY-rendering/capture-timing unreliability this file
# existed to compensate for. Removed rather than kept red: the behavior they
# asserted no longer exists for this phase by design, not by regression. The
# mechanism itself is untouched and still covered by its `_QA_QUESTION_FILE`/
# `_IMPLEMENT_BLOCKED_FILE` equivalents elsewhere in this file; a new test
# below (`test_grilling_turn_under_new_engine_never_touches_the_question_file`)
# covers the new engine's "must not touch this file" requirement directly.


def test_start_session_job_publishes_interview_even_with_no_structured_questions(client, tmp_path, monkeypatch):
    """A real /rhubarb:grilling turn can reply with plain prose (no
    bullet/heading questions qa_parser recognizes as structured). The left
    card must still render that turn -- it must not look like nothing
    happened -- even though zero questions auto-advances the chain right
    after (issue #223)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a new feature":
            return iter([_result_event("Sure, tell me more about what you have in mind.")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #221: the regex parser finds no structured questions in this
    # plain-prose reply, so `_run_grilling_turn_stream_json` now falls
    # through to a synchronous parser-session extraction pass -- scripted
    # here to also find none (a genuine wrap-up), same as this test always
    # expected.
    _mock_parser_session_extraction(
        monkeypatch, project_id, cwd, header="Sure, tell me more about what you have in mind."
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    grilling_turns = [e for e in turn_events if e["phase"] == "grilling"]
    assert len(grilling_turns) == 1
    assert grilling_turns[0]["interview"]["questions"] == []
    assert grilling_turns[0]["interview"]["header"]

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"


def test_continue_session_job_with_remaining_questions_does_not_auto_advance(client, tmp_path, monkeypatch):
    """A reply that still has follow-up questions must never auto-advance --
    only a genuinely empty frontier triggers the chain (issue #223)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt in ("/rhubarb:to-prd", "/rhubarb:to-issues", "/rhubarb:publish-to-github"):
            raise AssertionError(f"chain must not run while questions remain, got {prompt!r}")
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event('Question 1: "First question?"')])
        return iter([_result_event('Question 1: "A follow-up question?"')])

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "all good", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    interview = json.loads(row["interview_json"])
    assert interview["questions"][0]["text"] == "A follow-up question?"

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    assert turn_events[-1]["interview"]["questions"]


def test_continue_session_job_with_no_more_questions_auto_advances(client, tmp_path, monkeypatch):
    """Issue #223 (child of PRD #222): a reply that comes back with zero
    remaining questions now auto-advances straight into
    /rhubarb:to-prd -> /rhubarb:to-issues -> /rhubarb:publish-to-github ->
    details, with no confirmation click required."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event('Question 1: "First question?"')])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        if prompt == "all good":
            return iter([_result_event("Thanks, that's everything I need.")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #239: a fixed-payload parser-session mock would return the SAME
    # (empty) question list for every extraction call, including the FIRST
    # grilling turn -- masking this test's actual scenario (first turn has
    # a genuine open question; only the reply comes back empty). Use the
    # content-aware autoextract fake instead so each turn's own text drives
    # its own extraction result, same as a real parser-session pass would.
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "all good", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["error_text"] is None
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [{"number": 6, "title": "Child one"}]

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    assert any(e["phase"] == "grilling" and e["interview"]["questions"] == [] for e in turn_events)
    assert any(e["phase"] == "details" for e in turn_events)
    assert any(e == {"type": "phase", "phase": "creating_prd"} for e in events)


def test_stream_json_grilling_turn_uses_parser_session_for_emoji_question_format(client, tmp_path, monkeypatch):
    """Issue #221 (child of PRD #187/#220): the grilling skill's real
    `❓ **Q1** - **title**: body` question format isn't recognized by the
    regex parser (`parse_grilling_response`), so `_run_grilling_turn_stream_
    json` must fall through to a synchronous parser-session extraction pass
    and publish ITS structured result as the turn's `interview` payload."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    turn_text = "❓ **Q1** - **Scope**: Should this cover mobile too?"
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(turn_text)]))
    _mock_parser_session_extraction(
        monkeypatch,
        project_id,
        cwd,
        header="",
        questions=[
            {
                "id": "q1",
                "text": "Should this cover mobile too?",
                "kind": "single",
                "options": ["Yes", "No"],
                "recommended": [1],
                "recommended_text": None,
            }
        ],
        footer="",
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    interview = json.loads(row["interview_json"])
    assert interview["source"] == "parser_session"
    assert interview["questions"] == [
        {
            "id": "q1",
            "text": "Should this cover mobile too?",
            "kind": "single",
            "options": ["Yes", "No"],
            "recommended": [1],
            "recommended_text": None,
        }
    ]

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    assert len(turn_events) == 1
    assert turn_events[0]["phase"] == "grilling"
    assert turn_events[0]["interview"] == interview


def _mock_parser_session_sequence(monkeypatch, project_id, cwd, payloads):
    """Like `_mock_parser_session_extraction`, but hands out a DIFFERENT
    scripted `{header, questions, footer}` payload per successive turn sent
    to this project's parser session, instead of one fixed payload for
    every turn -- for tests that need to script a first (broken) primary
    response followed by a corrected (or still-broken) retry response
    (issue #229's mismatch-detection/single-retry path, `gh issue view
    229`), which `_mock_parser_session_extraction`'s single fixed payload
    can't represent. The LAST payload in `payloads` is reused for any
    further turn beyond the scripted ones, mirroring a real parser session
    that keeps answering rather than raising once the script runs out."""
    remaining = list(payloads)

    class FakeParserEngine:
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.session_id = resume_session_id or "parser-session-fake"
            self._alive = False

        def start(self):
            self._alive = True
            return self

        def close(self):
            self._alive = False

        def isalive(self):
            return self._alive

        async def stream_turn(self, prompt):
            payload = remaining.pop(0) if len(remaining) > 1 else remaining[0]
            yield {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": json.dumps(payload),
                "session_id": self.session_id,
            }

    monkeypatch.setattr(parser_session, "StreamJsonEngine", FakeParserEngine)
    parser_session._parser_sessions.pop(project_id, None)
    asyncio.run(parser_session.ensure_parser_session(project_id, cwd=cwd))


def test_stream_json_grilling_turn_retries_mismatched_extraction_and_flags_incomplete(
    client, tmp_path, monkeypatch
):
    """PRD #227 follow-up, gap 1 (`gh issue view 227`, discovered during
    manual testing/design review after #228/#229/#230 were already closed
    out in code): before this fix, the LIVE grilling-turn extraction path
    (`_run_grilling_turn_stream_json` -> `_extract_grilling_questions_via_
    parser_session` -> `_extract_questions_via_parser_session`) called
    `parser_session.stream_turn`/`_extract_json_object`/
    `_is_valid_grilling_shape` directly, with NO call into issue #229's
    post-extraction mismatch-detection/single-retry/`extraction_incomplete`
    logic at all -- so the ORIGINAL PRD #227 bug (a question with a
    paragraph of prior context and a transition phrase before it, whose
    bulleted options and `Recommended:` line get silently dropped) was not
    actually protected against on the exact live-turn path that produced
    it; only the async needs-input queue's own consumer
    (`parser_session._process_one_queued_item`, covered by issue #229's own
    tests) had that protection.

    This reproduces that bug scenario end to end through the REAL
    `start_session_job` -> `_run_grilling_turn_stream_json` ->
    `_extract_grilling_questions_via_parser_session` chain (not
    `_process_one_queued_item` directly) and asserts the shared
    `extract_with_validation` retry now fires on this path too, and a
    still-broken retry result is flagged `extraction_incomplete`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    turn_text = (
        "We've settled the schema and the API shape already. The last open "
        "branch is how retries should behave under load, since that changes "
        "how aggressively the client backs off.\n\n"
        "One more branch to close:\n\n"
        "❓ **Q5** - **Retry backoff**: How should the client back off "
        "between retries?\n"
        "- Fixed 1s delay\n"
        "- Exponential backoff\n"
        "Recommended: Exponential backoff\n"
    )
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(turn_text)]))

    header_text = (
        "We've settled the schema and the API shape already. The last open branch is how retries should "
        "behave under load, since that changes how aggressively the client backs off.\n\nOne more branch to close:"
    )
    # First attempt silently drops both the options and the recommendation --
    # exactly the PRD #227 bug scenario.
    broken_payload = {
        "header": header_text,
        "footer": "",
        "questions": [
            {
                "id": "q5",
                "text": "How should the client back off between retries?",
                "kind": "open",
                "options": None,
                "recommended": None,
                "recommended_text": None,
            }
        ],
    }
    # The retry recovers the options but still drops the recommendation --
    # still fails the same check, so it must come back flagged.
    still_broken_retry_payload = {
        "header": header_text,
        "footer": "",
        "questions": [
            {
                "id": "q5",
                "text": "How should the client back off between retries?",
                "kind": "single",
                "options": ["Fixed 1s delay", "Exponential backoff"],
                "recommended": None,
                "recommended_text": None,
            }
        ],
    }
    _mock_parser_session_sequence(monkeypatch, project_id, cwd, [broken_payload, still_broken_retry_payload])

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview["source"] == "parser_session"
    question = interview["questions"][0]
    # The retry actually fired and its (still imperfect) result won, tagged.
    assert question["kind"] == "single"
    assert question["options"] == ["Fixed 1s delay", "Exponential backoff"]
    assert question["extraction_incomplete"] is True

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn" and e["phase"] == "grilling"]
    assert len(turn_events) == 1
    assert turn_events[0]["interview"]["questions"][0]["extraction_incomplete"] is True


def test_stream_json_grilling_turn_mentioning_prd_number_is_parsed_like_any_other_turn(
    client, tmp_path, monkeypatch
):
    """Issue #239 (child of PRD #237): the old issue #219 fast path used to
    treat a turn whose text merely *mentioned* a PRD/issue number (no
    structured `❓ **Qn**` questions) as "the chain already completed in one
    turn" and route straight to `_finish_chain`, skipping question
    extraction entirely -- turning the grilling model self-answering and
    free-running the whole /do chain into a silently-accepted outcome. That
    shortcut is removed: such a turn is now parsed for questions exactly
    like any other turn (via the parser-session extraction pass), finds
    none, and proceeds through the normal empty-frontier chain
    (`advance_past_grilling` -> to-prd -> to-issues -> publish-to-github),
    landing on phase=details only once that real chain actually completes --
    not via the removed shortcut."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    seen_prompts = []

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            # Natural-language format the real /do skill might emit if it
            # (wrongly) mentions a PRD number without asking anything --
            # must not be mistaken for a completed chain any more.
            return iter([_result_event("Thinking out loud about PRD #10 for a moment.")])
        seen_prompts.append(prompt)
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="Thinking out loud about PRD #10 for a moment.")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    # The grilling turn's text was parsed for questions (found none) and the
    # chain ran for real -- not skipped via the removed shortcut.
    assert seen_prompts == ["/rhubarb:to-prd", "/rhubarb:to-issues", "/rhubarb:publish-to-github"]

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["session_type"] == "do"
    assert row["available_for_reuse"] == 0

    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [{"number": 6, "title": "Child one"}]

    events = live_stream._buffers.get(row_id, [])
    assert any(e == {"type": "phase", "phase": "creating_prd"} for e in events), "must run the real chain"
    assert any(e.get("type") == "turn" and e.get("phase") == "details" for e in events)


# ---------------------------------------------------------------------------
# Issue #184: this whole block (Ollama rescue-path wiring for issue #114,
# the PRD #158 corrective-retry mechanism, and the issue #178 needs-input
# classification wired into grilling) used to test roughly 20 scenarios of
# PtyEngine-specific grilling behavior: the regex parser finding nothing,
# handing malformed/suspicious terminal text to `rescue_grilling_response`,
# a one-shot corrective follow-up turn when that also came back empty, and
# a last-resort Ollama needs-input classification before giving up. All of
# it existed to compensate for PtyEngine's PTY-rendering/capture-timing
# unreliability (word-wrap, ANSI cursor-movement redraws) -- see
# `_run_grilling_turn`'s docstring in session_runner.py.
#
# A grilling card now dispatches to `StreamJsonEngine`
# (`_run_grilling_turn_stream_json`), whose `result` event text comes
# straight from the CLI's own structured stream-json output, not a
# scraped/rendered terminal buffer -- there is nothing left for that whole
# chain to compensate for, and the issue's acceptance criteria explicitly
# requires grilling under this engine to skip it entirely (no file read, no
# Ollama rescue call). These tests are removed rather than kept red: the
# behavior they asserted no longer exists for this phase by design. At the
# time, `rescue_grilling_response`/`should_attempt_grilling_rescue` were
# still untouched and exercised by QA/implement's own equivalents elsewhere
# in this file -- issue #230 has since migrated those onto the
# parser-session pipeline too and deleted both functions entirely (see
# `test_qa_needs_input_classification_rescues_after_corrective_retry_also_
# failed`/`test_qa_corrective_retry_fires_once_and_uses_reformatted_result`
# for their current, parser-session-based equivalents; `classify_needs_input`
# itself is untouched). New tests below
# (`test_grilling_turn_under_new_engine_never_touches_the_question_file`,
# `test_grilling_turn_under_new_engine_skips_ollama_rescue_entirely`) cover
# the new engine's "must not touch that mechanism" requirement directly.
# ---------------------------------------------------------------------------


def test_start_session_job_with_empty_frontier_auto_advances_through_chain(client, tmp_path, monkeypatch):
    """Issue #223 (child of PRD #222): the moment a grilling turn's frontier
    comes back empty, the chain must go straight to /rhubarb:to-prd ->
    /rhubarb:to-issues -> /rhubarb:publish-to-github -> details on its own,
    with no confirmation click and no `confirm_advance` flag anymore. Issue
    #196 (child of PRD #195): the chain still pauses at `details` instead of
    auto-continuing into /rhubarb:implement."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    seen_prompts = []

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        seen_prompts.append(prompt)
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        raise AssertionError(f"unexpected prompt {prompt!r} -- do chain must not auto-implement")

    _mock_engine(monkeypatch, handler)
    # Issue #221: "❓ **Q1**" isn't a format the regex parser recognizes
    # either, so the first turn now falls through to the parser-session
    # extraction pass -- scripted here to find nothing, since this test
    # doesn't care about the parsed interview, only that the empty frontier
    # advances straight into the chain afterward.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    # Only the three chain prompts -- no /rhubarb:implement, no grilling reply.
    assert seen_prompts == ["/rhubarb:to-prd", "/rhubarb:to-issues", "/rhubarb:publish-to-github"]

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["session_type"] == "do"
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [{"number": 6, "title": "Child one"}]
    assert row["available_for_reuse"] == 0


# ---------------------------------------------------------------------------
# Issue #242 (child of PRD #241): an empty frontier alone no longer
# auto-advances grilling into PRD drafting -- the parser session's own
# completion verdict (folded into the same extraction call) must also say
# "done". A negative or missing/malformed verdict stalls instead, reusing
# the generic stall mechanism `_run_chain_step` already uses later in the
# chain.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parsed, expected_reason_substring",
    [
        ({"questions": []}, "didn't return a verdict"),
        ({"questions": [], "completion": "not a dict"}, "didn't return a verdict"),
        ({"questions": [], "completion": {"reason": "Missing done."}}, "malformed"),
        ({"questions": [], "completion": {"done": "yes", "reason": "Not a bool."}}, "malformed"),
        ({"questions": [], "completion": {"done": True}}, "malformed"),
        ({"questions": [], "completion": {"done": True, "reason": "   "}}, "malformed"),
    ],
)
def test_grilling_completion_verdict_fails_safe_on_malformed_or_missing_input(parsed, expected_reason_substring):
    """Issue #242 acceptance criterion: a malformed or missing completion-
    verdict field is treated as "not done" -- never silently treated as
    complete, regardless of which part of the expected shape is wrong."""
    done, reason = session_runner._grilling_completion_verdict(parsed)
    assert done is False
    assert expected_reason_substring in reason


def test_grilling_completion_verdict_true_only_for_a_wellformed_positive_verdict():
    done, reason = session_runner._grilling_completion_verdict(
        {"questions": [], "completion": {"done": True, "reason": "Every open branch was resolved."}}
    )
    assert done is True
    assert reason == "Every open branch was resolved."

    done, reason = session_runner._grilling_completion_verdict(
        {"questions": [], "completion": {"done": False, "reason": "Retry backoff is still undecided."}}
    )
    assert done is False
    assert reason == "Retry backoff is still undecided."


def test_start_session_job_negative_completion_verdict_stalls_instead_of_advancing(client, tmp_path, monkeypatch):
    """A turn whose frontier comes back empty but whose completion verdict
    says `"done": false` must NOT advance into `/rhubarb:to-prd` -- it
    stalls in `grilling` instead, using the same generic stall mechanism
    (`stalled_json`, `stalled`/`stalled_context` on the `turn` event) the
    chain phases already use, exactly the acceptance criteria this
    completion gate exists for."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("Sounds like we've covered the basics.")])
        raise AssertionError(f"the chain must not run while grilling is still unresolved, got {prompt!r}")

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_extraction(
        monkeypatch,
        project_id,
        cwd,
        header="Sounds like we've covered the basics.",
        completion={"done": False, "reason": "The retry-backoff strategy is still undecided."},
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    assert row["stalled_json"] is not None
    assert json.loads(row["stalled_json"]) == {
        "phase": "grilling",
        "context": "The retry-backoff strategy is still undecided.",
    }

    events = live_stream._buffers.get(row_id, [])
    stalled_events = [e for e in events if e.get("type") == "turn" and e.get("stalled")]
    assert len(stalled_events) == 1
    assert stalled_events[0]["phase"] == "grilling"
    assert stalled_events[0]["stalled_context"] == "The retry-backoff strategy is still undecided."
    assert not any(e == {"type": "phase", "phase": "creating_prd"} for e in events)


def test_grilling_stall_reply_resumes_grilling_and_can_then_advance(client, tmp_path, monkeypatch):
    """A reply to a grilling-phase stall resumes grilling itself via
    `continue_session_job` (see `test_stall_reply_endpoint_resumes_a_
    grilling_session_via_a_new_turn` in tests/test_app.py for the HTTP
    endpoint's own dispatch to that function) -- not the chain-phase resume
    path used by creating_prd/creating_issues/publishing. The resumed turn
    goes through the exact same extraction + completion-verdict check as
    any fresh turn, so once it comes back with a positive verdict, the
    chain advances exactly like a fresh run would."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("Sounds like we've covered the basics.")])
        if prompt == "Actually, it only needs to cover web for now.":
            return iter([_result_event("Got it, that resolves the last open branch.")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)

    class FakeParserEngine:
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.session_id = resume_session_id or "parser-session-fake"
            self._alive = False

        def start(self):
            self._alive = True
            return self

        def close(self):
            self._alive = False

        def isalive(self):
            return self._alive

        async def stream_turn(self, prompt):
            if "Got it, that resolves the last open branch." in prompt:
                data = {
                    "header": "",
                    "questions": [],
                    "footer": "",
                    "completion": {"done": True, "reason": "The last open branch was resolved."},
                }
            else:
                data = {
                    "header": "",
                    "questions": [],
                    "footer": "",
                    "completion": {"done": False, "reason": "Scope isn't settled yet."},
                }
            yield {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": json.dumps(data),
                "session_id": self.session_id,
            }

    monkeypatch.setattr(parser_session, "StreamJsonEngine", FakeParserEngine)
    parser_session._parser_sessions.pop(project_id, None)
    asyncio.run(parser_session.ensure_parser_session(project_id, cwd=cwd))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    assert row["stalled_json"] is not None

    # Resumes via `continue_session_job` -- the same function the
    # stall-reply endpoint's grilling branch dispatches to.
    asyncio.run(
        session_runner.continue_session_job(row_id, "Actually, it only needs to cover web for now.", cwd=cwd)
    )

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["stalled_json"] is None
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}

    events = live_stream._buffers.get(row_id, [])
    assert not any(e.get("type") == "done" for e in events)
    assert any(e == {"type": "phase", "phase": "creating_prd"} for e in events)
    assert any(e == {"type": "phase", "phase": "creating_issues"} for e in events)
    assert any(e.get("type") == "turn" and e.get("phase") == "details" for e in events)


def test_start_session_job_passes_the_configured_model_to_the_cli(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_model(conn, "claude-opus-4-8")

    seen_models = []

    def handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        seen_models.append(model)
        return iter([_result_event('Question 1: "Only question?"')])

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert seen_models == ["claude-opus-4-8"]
    # The model actually used is persisted onto the row for later turns.
    assert db.get_session(conn, row_id)["model"] == "claude-opus-4-8"


def test_start_implement_job_passes_the_model_the_session_was_launched_with(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_model(conn, "claude-opus-4-8")

    seen_models = []

    def handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        seen_models.append(model)
        return iter([_result_event("Implemented PRD #5", session_id="impl1")])

    _mock_engine(monkeypatch, handler, fresh_ids=["pooled-1"])

    started = asyncio.run(_run_and_drain(session_runner.start_or_queue_implement(project_id, 5, "My PRD", cwd)))
    card_id = started["card_id"]

    assert seen_models == ["claude-opus-4-8"]
    assert db.get_session(conn, card_id)["model"] == "claude-opus-4-8"


def test_chain_steps_use_the_model_the_session_was_created_with(client, tmp_path, monkeypatch):
    """/rhubarb:to-prd, /rhubarb:to-issues, and /rhubarb:publish-to-github
    (all run via advance_past_grilling) must be invoked with the same model
    the session's grilling turn used, not whatever `settings.model` currently
    is."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_model(conn, "claude-opus-4-8")

    seen_models = []

    def handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event('Question 1: "Only question?"')])
        seen_models.append((prompt, model))
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    # Issue #223: this reply's unstructured wrap-up text goes through the
    # parser-session extraction pass, finds nothing (no quoted `Question N:`
    # text in this prose), and auto-advances straight into the chain -- no
    # confirm_advance flag anymore.
    asyncio.run(session_runner.continue_session_job(row_id, "all good", cwd=cwd))

    assert seen_models == [
        ("all good", "claude-opus-4-8"),
        ("/rhubarb:to-prd", "claude-opus-4-8"),
        ("/rhubarb:to-issues", "claude-opus-4-8"),
        ("/rhubarb:publish-to-github", "claude-opus-4-8"),
    ]


def test_model_selector_change_respawns_only_the_open_cards_engine(client, tmp_path, monkeypatch):
    """Issue #141: changing the model selector for the currently-open card
    must tear down that card's live resident engine and replace it with a
    fresh, unresumed one under the new model -- a different card's engine,
    and the project's pre-warmed standby engine, must be completely
    untouched by that same change. (Before #141, an in-flight session simply
    kept its original model forever -- that behavior is superseded by this
    respawn; see git history for the old assertion.)"""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_model(conn, "claude-opus-4-8")

    def handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        return iter([_result_event("- First question?")])

    _mock_engine(monkeypatch, handler)

    # The card under test -- already has a live resident engine, seeded
    # directly via `_get_or_create_stream_json_engine` rather than through
    # `start_session_job` -- this test is about the respawn mechanism
    # itself, not about grilling.
    row_id = db.create_session(conn, project_id)
    original_engine = session_runner._get_or_create_stream_json_engine(
        row_id, cwd=cwd, model="claude-opus-4-8", effort="auto", resume_session_id=None
    )
    assert original_engine.model == "claude-opus-4-8"
    assert original_engine.effort == "auto"

    # A different card, also with a live resident engine -- must be left
    # completely untouched below.
    other_row_id = db.create_session(conn, project_id)
    other_engine = session_runner._get_or_create_stream_json_engine(
        other_row_id, cwd=cwd, model="claude-opus-4-8", effort="auto", resume_session_id=None
    )

    # The project's pre-warmed standby engine -- also must be left
    # completely untouched below.
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(
            project_id, cwd=cwd, model="claude-opus-4-8", effort=db.DEFAULT_EFFORT
        )
    )
    standby_engine, _, _ = session_runner._standby_stream_json_engines[project_id]

    resp = client.post("/api/settings/model", json={"model": "claude-sonnet-4-6", "card_id": row_id})
    assert resp.json() == {"model": "claude-sonnet-4-6", "respawned": True}

    # (a) The open card's engine was torn down and replaced with a fresh,
    # unresumed one under the new model (effort carried over unchanged).
    assert original_engine.closed is True
    new_engine = session_runner._stream_json_engines[row_id]
    assert new_engine is not original_engine
    assert new_engine.model == "claude-sonnet-4-6"
    assert new_engine.effort == "auto"
    assert new_engine.resume_session_id is None

    row = db.get_session(conn, row_id)
    assert row["model"] == "claude-sonnet-4-6"
    assert row["claude_session_id"] == new_engine.claude_session_id

    # The ground-truth indicator from #139 confirms the Live Terminal now
    # reflects the respawned engine's fresh conversation.
    state = client.get(f"/api/app-state?card_id={row_id}").json()
    assert state["model"] == "claude-sonnet-4-6"
    assert state["effort"] == "auto"
    assert state["session_model_effort_live"] is True

    # (b) A different card's engine, and the project's standby engine, are
    # provably untouched.
    assert session_runner._stream_json_engines[other_row_id] is other_engine
    assert other_engine.closed is False
    assert other_engine.model == "claude-opus-4-8"

    assert session_runner._standby_stream_json_engines[project_id][0] is standby_engine
    assert standby_engine.closed is False
    assert standby_engine.model == "claude-opus-4-8"


def test_effort_selector_change_respawns_only_the_open_cards_engine(client, tmp_path, monkeypatch):
    """Symmetric to `test_model_selector_change_respawns_only_the_open_cards_engine`
    above, for the effort equivalent (`POST /api/settings/effort`) -- both
    endpoints are independently capable of triggering a respawn."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_model(conn, "claude-opus-4-8")

    def handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        return iter([_result_event("- First question?")])

    _mock_engine(monkeypatch, handler)

    # See `test_model_selector_change_respawns_only_the_open_cards_engine`'s
    # docstring above for why this seeds the engine directly rather than
    # through `start_session_job`.
    row_id = db.create_session(conn, project_id, effort="low")
    original_engine = session_runner._get_or_create_stream_json_engine(
        row_id, cwd=cwd, model="claude-opus-4-8", effort="low", resume_session_id=None
    )
    assert original_engine.effort == "low"

    other_row_id = db.create_session(conn, project_id, effort="low")
    other_engine = session_runner._get_or_create_stream_json_engine(
        other_row_id, cwd=cwd, model="claude-opus-4-8", effort="low", resume_session_id=None
    )

    resp = client.post("/api/settings/effort", json={"effort": "high", "card_id": row_id})
    assert resp.json() == {"effort": "high", "respawned": True}

    new_engine = session_runner._stream_json_engines[row_id]
    assert new_engine is not original_engine
    assert original_engine.closed is True
    assert new_engine.model == "claude-opus-4-8"
    assert new_engine.effort == "high"
    assert new_engine.resume_session_id is None

    assert session_runner._stream_json_engines[other_row_id] is other_engine
    assert other_engine.closed is False
    assert other_engine.effort == "low"


def test_settings_change_with_no_live_engine_for_the_card_only_updates_the_global_setting(
    client, tmp_path, monkeypatch
):
    """A `card_id` naming a card with no live resident engine (never
    started, already finished/pooled) must behave exactly like passing no
    `card_id` at all -- global setting update only, no engine spawned."""
    project_id = _open_project(client, tmp_path, "proj")
    conn = db.get_connection()

    # A row that exists but was never given a live engine.
    row_id = db.create_session(conn, project_id)
    assert row_id not in session_runner._stream_json_engines

    resp = client.post("/api/settings/model", json={"model": "claude-opus-4-8", "card_id": row_id})
    assert resp.json() == {"model": "claude-opus-4-8", "respawned": False}
    assert row_id not in session_runner._stream_json_engines
    assert db.get_model(conn) == "claude-opus-4-8"


def test_continue_session_job_advances_through_prd_and_issues_to_details(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event('Question 1: "First question?"')])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one\nIssue Draft S2: Child two")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one\nIssue #7: Child two")])
        # the grilling reply itself: no more bullet/heading questions -> grilling is done
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"

    # Issue #223 (child of PRD #222): the reply's wrap-up (no more questions)
    # now auto-advances straight through to-prd -> to-issues -> details, with
    # no confirmation click, and no auto-implement either (issue #196, child
    # of PRD #195).
    asyncio.run(session_runner.continue_session_job(row_id, "all good", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["session_type"] == "do"
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [
        {"number": 6, "title": "Child one"},
        {"number": 7, "title": "Child two"},
    ]
    assert row["available_for_reuse"] == 0

    events = live_stream._buffers.get(row_id, [])
    assert not any(e.get("type") == "done" for e in events)
    assert any(e.get("type") == "turn" and e.get("phase") == "details" for e in events)
    assert any(e == {"type": "phase", "phase": "creating_prd"} for e in events)
    assert "$" not in json.dumps(events)


def test_chain_publishes_directly_and_details_reflect_issue_numbers(client, tmp_path, monkeypatch):
    """Issue #225 (child of PRD #222): `/rhubarb:to-prd`/`/rhubarb:to-issues`
    are pure drafting steps with no GitHub side effects of their own --
    `/rhubarb:publish-to-github` is the one place that calls `gh issue create`
    and prints the `PRD #N: <title>` / `Issue #N: <title>` lines the chain
    parses. The chain must reach `phase=details` with correct issue numbers
    parsed from those lines."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one\nIssue Draft S2: Child two")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one\nIssue #7: Child two")])
        return iter([_result_event("Implemented.")])

    _mock_engine(monkeypatch, handler)
    # Issue #223: the first turn's frontier already comes back empty (the
    # emoji-format text isn't regex-parseable). Issue #242: the empty
    # frontier alone isn't enough anymore -- a positive completion verdict
    # is also required before the chain auto-advances, so this test's fake
    # extraction supplies one (this test is about publish-to-github issue
    # number parsing, not the completion gate itself).
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [
        {"number": 6, "title": "Child one"},
        {"number": 7, "title": "Child two"},
    ]


@pytest.mark.parametrize(
    "message, expected",
    [
        ("gh: not logged in, run `gh auth login`", True),
        ("You are not logged into any GitHub hosts. To log in, run: gh auth login", True),
        ("authentication failed: invalid token", False),
        ("permission denied (401)", False),
        ("not logged in", False),
        ("Claude CLI exited with code 1: 403 Forbidden", False),
        ("boom", False),
    ],
)
def test_is_gh_auth_failure_classifier(message, expected):
    assert session_runner._is_gh_auth_failure(message) is expected


def test_grilling_claude_cli_error_never_sets_needs_github_login(client, tmp_path, monkeypatch):
    """Grilling's Claude turn never calls `gh` (see its skill instructions),
    so a ClaudeCLIError here -- even one containing gh-shaped auth text --
    must never be misclassified as a GitHub login need."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def failing(prompt, **kw):
        raise ClaudeCLIError("gh: not logged in, run `gh auth login`")

    _mock_engine(monkeypatch, failing)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert bool(row["needs_github_login"]) is False
    assert "not logged in" in row["error_text"]

    events = live_stream._buffers.get(row_id, [])
    error_turns = [e for e in events if e.get("type") == "turn" and e.get("phase") == "grilling"]
    assert error_turns[0]["needs_github_login"] is False


def test_publishing_gh_auth_failure_sets_needs_github_login(client, tmp_path, monkeypatch):
    """Issue #225 (child of PRD #222): `/rhubarb:publish-to-github` is now the
    only phase that calls `gh issue create` (`/rhubarb:to-prd`/`to-issues` are
    pure drafting steps with no GitHub side effects), so a `ClaudeCLIError`
    whose message matches `_is_gh_auth_failure` during `publishing` must set
    `needs_github_login=1` and publish a `turn` event with
    `needs_github_login: true` -- so the frontend shows the GitHub login
    prompt."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            raise ClaudeCLIError("gh: not logged in, run `gh auth login`")
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about publishing's own
    # gh-auth-failure handling, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    # Issue #223: the first turn's empty frontier auto-advances straight
    # through to-prd -> to-issues -> publishing within this same call -- no
    # confirmation click needed.
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "publishing"
    assert bool(row["needs_github_login"]) is True
    assert "not logged in" in row["error_text"]

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}
    error_turns = [e for e in events if e.get("type") == "turn" and e.get("phase") == "publishing"]
    assert error_turns[0]["needs_github_login"] is True


def test_retry_after_login_completes_the_failed_phase(client, tmp_path, monkeypatch):
    """Issue #225 (child of PRD #222): a `publishing`-phase failure (the only
    phase that calls `gh issue create` now) retries straight into another
    `/rhubarb:publish-to-github` turn via `retry_session_job`'s new
    `publishing` branch."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    attempt = {"n": 0}

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: Retried PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Only child")])
        if prompt == "/rhubarb:publish-to-github":
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ClaudeCLIError("not logged in")
            return iter([_result_event("PRD #9: Retried PRD\nIssue #10: Only child")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about publishing retry
    # mechanics, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "publishing"

    asyncio.run(session_runner.retry_session_job(row_id, cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["error_text"] is None
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 9, "title": "Retried PRD"}


def test_retry_on_errored_implement_session_creates_a_new_row_and_completes(client, tmp_path, monkeypatch):
    """Retrying an errored implement session must not resume the original
    row's (possibly dead) claude_session_id -- it must go through
    start_or_queue_implement, the same entry point a fresh PRD-list click
    uses, creating a brand-new session row that then progresses normally."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 12, "title": "Errored PRD"}},
    )
    db.update_session(conn, row_id, error_text="agent crashed mid-turn")

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Implemented PRD #12", session_id="impl-retry")]),
        fresh_ids=["pooled-retry"],
    )
    # Issue #179: this test asserts on the retried job's *end state*
    # immediately after `client.post` returns, with nothing draining the
    # `asyncio.create_task`-scheduled job the retry endpoint kicks off (see
    # `_run_and_drain`'s own docstring for why other tests need that drain
    # at all) -- unlike every other step in this synchronous fake-engine
    # chain, `classify_needs_input`'s real Ollama call genuinely hops
    # through a thread-pool executor (`asyncio.to_thread`), which isn't
    # guaranteed to resolve by the time this synchronous test client call
    # returns. Neutralize it here (this test is about retry mechanics, not
    # needs-input classification -- that has its own dedicated tests) the
    # same way every other unrelated implement test already can rely on
    # a real, unmocked call finishing fast enough not to matter.
    monkeypatch.setattr(session_runner, "classify_needs_input", _fake_classify_needs_input(None))

    response = client.post(f"/api/sessions/{row_id}/retry")
    assert response.status_code == 200

    sessions = db.list_sessions_for_project(conn, project_id)
    implement_sessions = [s for s in sessions if s["session_type"] == "implement"]
    assert len(implement_sessions) == 2

    new_row = next(s for s in implement_sessions if s["id"] != row_id)
    assert new_row["phase"] == "implemented"
    assert json.loads(new_row["details_json"])["prd"] == {"number": 12, "title": "Errored PRD"}

    original_row = db.get_session(conn, row_id)
    assert original_row["phase"] == "implementing"
    assert original_row["error_text"] == "agent crashed mid-turn"


def test_retry_on_errored_implement_session_respects_serial_mode_queueing(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_parallel_implementation(conn, False)

    # An already-live implement session for this project (serial mode: a
    # retry while this is running must queue rather than start immediately).
    db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 20, "title": "Currently Running"}},
    )

    errored_row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 21, "title": "Errored PRD"}},
    )
    db.update_session(conn, errored_row_id, error_text="agent crashed mid-turn")

    async def _noop_job(card_id, prd_number, *, cwd):
        return None

    monkeypatch.setattr(session_runner, "start_implement_job", _noop_job)

    asyncio.run(session_runner.retry_session_job(errored_row_id, cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    implement_sessions = [s for s in sessions if s["session_type"] == "implement"]
    # No new row was created -- the retry was enqueued instead.
    assert len(implement_sessions) == 2
    assert session_runner._implement_queues.get(project_id) == [{"number": 21, "title": "Errored PRD"}]

    original_row = db.get_session(conn, errored_row_id)
    assert original_row["phase"] == "implementing"
    assert original_row["error_text"] == "agent crashed mid-turn"


def test_retry_on_creating_prd_phase_is_unaffected_by_implement_branch(client, tmp_path, monkeypatch):
    """Guard against regressing the existing creating_prd/creating_issues
    retry path when adding the implement-session branch."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    attempt = {"n": 0}

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ClaudeCLIError("not logged in")
            return iter([_result_event("PRD Draft: Regression PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Only child")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #30: Regression PRD\nIssue #31: Only child")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about the creating_prd retry
    # path, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    asyncio.run(session_runner.retry_session_job(row_id, cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["error_text"] is None
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 30, "title": "Regression PRD"}


def test_start_implement_job_reaches_implemented_and_pools_falling_back_to_seeded_details(client, tmp_path, monkeypatch):
    """No `.claude/implement-tracker.json` written -- the seeded `{"prd": ...}`
    details from creation must survive into the terminal `implemented` row."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Implemented PRD #5", session_id="impl1")]),
        # Slot 0 is consumed by the turn's own engine (the row starts with no
        # claude_session_id, so that construction is "fresh" too); slot 1 is
        # the fresh engine minted when pooling.
        fresh_ids=["turn-engine", "pooled-1"],
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert row["available_for_reuse"] == 1
    assert row["claude_session_id"] == "pooled-1"
    assert json.loads(row["details_json"]) == {"prd": {"number": 5, "title": "My PRD"}}

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}
    assert {"type": "phase", "phase": "implementing"} in events
    assert any(e.get("type") == "turn" and e.get("phase") == "implemented" for e in events)


def test_start_implement_job_populates_details_from_tracker_file_when_present(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    tracker = {
        "prd": {"number": 7, "title": "Tracked PRD"},
        "issues": [{"number": 8, "title": "Child", "summary": "does the thing", "acceptance_criteria": ["works"]}],
        "qa_changes": [],
        "status": "implemented",
    }
    claude_dir = Path(cwd) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "implement-tracker.json").write_text(json.dumps(tracker), encoding="utf-8")

    _mock_engine(
        monkeypatch, lambda prompt, **kw: iter([_result_event("done", session_id="impl2")]), fresh_ids=["pooled-2"]
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert json.loads(row["details_json"]) == tracker

    events = live_stream._buffers.get(row_id, [])
    assert any(e.get("type") == "turn" and e.get("phase") == "implemented" and e.get("details") == tracker for e in events)


def test_start_implement_job_error_leaves_session_in_implementing_with_error_text(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def failing(prompt, **kw):
        raise ClaudeCLIError("gh: not logged in, run `gh auth login`")

    _mock_engine(monkeypatch, failing)

    conn = db.get_connection()
    row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 11, "title": "Errorable"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 11, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implementing"
    assert bool(row["needs_github_login"]) is True
    assert "not logged in" in row["error_text"]

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}


def test_start_implement_job_non_gh_error_does_not_set_needs_github_login(client, tmp_path, monkeypatch):
    """A ClaudeCLIError during implementing that isn't actually a `gh` auth
    failure (no `gh auth login` remediation text) must not trigger the
    GitHub login button, even though it mentions a generic auth-ish word."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def failing(prompt, **kw):
        raise ClaudeCLIError("Claude CLI exited with code 1: permission denied")

    _mock_engine(monkeypatch, failing)

    conn = db.get_connection()
    row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 11, "title": "Errorable"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 11, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implementing"
    assert bool(row["needs_github_login"]) is False
    assert "permission denied" in row["error_text"]

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}


def test_start_implement_job_wraps_unexpected_worker_exceptions_as_session_errors(client, tmp_path, monkeypatch):
    """A non-ClaudeCLIError exception out of the streaming worker (e.g. the
    CLI producing non-JSON output) must still resolve into a recorded
    session error, not an unhandled exception on the fire-and-forget
    asyncio task that would leave the card stuck in `implementing` forever."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def crashing(prompt, **kw):
        def gen():
            raise ValueError("boom")
            yield  # pragma: no cover

        return gen()

    _mock_engine(monkeypatch, crashing)

    conn = db.get_connection()
    row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 11, "title": "Errorable"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 11, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implementing"
    assert "boom" in row["error_text"]

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}


def test_parallel_mode_starts_multiple_prds_immediately_without_queueing(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    async def _noop_job(card_id, prd_number, *, cwd):
        return None

    monkeypatch.setattr(session_runner, "start_implement_job", _noop_job)

    conn = db.get_connection()
    db.set_parallel_implementation(conn, True)

    # parallel_implementation is on -- two different PRDs both start
    # immediately, no queueing involved.
    first = asyncio.run(session_runner.start_or_queue_implement(project_id, 5, "PRD Five", cwd))
    assert "card_id" in first

    second = asyncio.run(session_runner.start_or_queue_implement(project_id, 6, "PRD Six", cwd))
    assert "card_id" in second

    conn = db.get_connection()
    sessions = db.list_sessions_for_project(conn, project_id)
    implement_sessions = [s for s in sessions if s["session_type"] == "implement"]
    assert len(implement_sessions) == 2
    assert session_runner._implement_queues.get(project_id, []) == []


def test_parallel_mode_gives_each_prd_its_own_independent_engine(client, tmp_path, monkeypatch):
    """With parallel_implementation on, two concurrently-running PRDs must
    each get their own resident PtyEngine tab -- not share one."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_parallel_implementation(conn, True)

    fake_class = _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Implemented.", session_id="whichever")]),
        fresh_ids=["pooled-a", "pooled-b"],
    )

    first = asyncio.run(_run_and_drain(session_runner.start_or_queue_implement(project_id, 5, "PRD Five", cwd)))
    second = asyncio.run(_run_and_drain(session_runner.start_or_queue_implement(project_id, 6, "PRD Six", cwd)))

    assert first["card_id"] != second["card_id"]
    # Two independent tabs were constructed for the two turns (plus the two
    # pooling-time fresh engines) -- never one shared instance.
    assert len(fake_class.instances) >= 2
    assert session_runner._stream_json_engines == {}


def test_serial_mode_queues_second_prd_and_drains_it_when_first_finishes(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_parallel_implementation(conn, False)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:implement prd: 6":
            return iter([_result_event("Implemented PRD #6", session_id="impl-b")])
        return iter([_result_event("Implemented PRD #5", session_id="impl-a")])

    _mock_engine(monkeypatch, handler, fresh_ids=["pooled-a", "pooled-b"])

    async def start_both():
        # Both calls run inside one `asyncio.run` with no `await` in between
        # that would let A's fire-and-forget `asyncio.create_task` job make
        # any progress -- neither `start_or_queue_implement` call itself
        # awaits anything (every DB call it makes is synchronous), so A's
        # session row is still exactly as `_launch_implement` left it
        # ("implementing", job merely scheduled) by the time B's queueing
        # check runs, deterministically -- not by timing coincidence.
        started = await session_runner.start_or_queue_implement(project_id, 5, "PRD Five", cwd)
        row_a = started["card_id"]
        assert row_a is not None
        assert db.get_session(conn, row_a)["phase"] == "implementing"

        queued = await session_runner.start_or_queue_implement(project_id, 6, "PRD Six", cwd)
        assert queued == {"queued": True}
        return row_a

    row_a = asyncio.run(start_both())

    sessions_before_drain = db.list_sessions_for_project(conn, project_id)
    implement_sessions_before = [s for s in sessions_before_drain if s["session_type"] == "implement"]
    assert len(implement_sessions_before) == 1

    # Now run A's job for real, and let any follow-on drain task it
    # schedules run to completion too.
    asyncio.run(_run_and_drain(session_runner.start_implement_job(row_a, 5, cwd=cwd)))

    assert session_runner._implement_queues.get(project_id, []) == []

    sessions_after_drain = db.list_sessions_for_project(conn, project_id)
    implement_sessions_after = [s for s in sessions_after_drain if s["session_type"] == "implement"]
    assert len(implement_sessions_after) == 2

    row_b = next(s for s in implement_sessions_after if s["id"] != row_a)
    assert row_b["phase"] == "implemented"
    assert json.loads(row_b["details_json"])["prd"] == {"number": 6, "title": "PRD Six"}


def test_session_reuse_pool_is_scoped_per_project(client, tmp_path, monkeypatch):
    """Pool entries are scoped per project: a session marked available in
    project A must never be claimed for project B. Tests the DB-level scoping
    contract directly, independent of how a session ends up in the pool."""
    project_a = _open_project(client, tmp_path, "proj_a")
    cwd_a = _cwd_for(project_a)

    # Seed the pool for project A directly via the DB (PRD #195: /do sessions
    # no longer auto-pool themselves at details -- they pause for user input
    # instead. The pool is still used by /implement sessions, so we test the
    # scoping contract here without going through the /do flow.)
    conn = db.get_connection()
    pool_row_id = db.create_session(conn, project_a)
    db.update_session(conn, pool_row_id, phase="done", available_for_reuse=1, claude_session_id="pooled-session")

    seen_session_ids = []

    def recording_handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        seen_session_ids.append(session_id)
        return iter([_result_event('Question 1: "Another question?"', session_id="new")])

    _mock_engine(monkeypatch, recording_handler)
    _mock_parser_session_autoextract(monkeypatch, project_a, cwd_a)

    # Same project: should resume the pooled session.
    reused = db.claim_available_session(conn, project_a)
    resume_id = reused["claude_session_id"] if reused is not None else None
    new_row_id = db.create_session(conn, project_a, claude_session_id=resume_id)
    asyncio.run(session_runner.start_session_job(new_row_id, "another feature", cwd=cwd_a))
    assert seen_session_ids[-1] == "pooled-session"

    # A different project must never be handed project A's pooled session.
    project_b = _open_project(client, tmp_path, "proj_b")
    cwd_b = _cwd_for(project_b)
    _mock_parser_session_autoextract(monkeypatch, project_b, cwd_b)
    reused_b = db.claim_available_session(conn, project_b)
    resume_id_b = reused_b["claude_session_id"] if reused_b is not None else None
    row_id_b = db.create_session(conn, project_b, claude_session_id=resume_id_b)
    asyncio.run(session_runner.start_session_job(row_id_b, "unrelated feature", cwd=cwd_b))
    assert seen_session_ids[-1] is None


# ---------------------------------------------------------------------------
# `_extract_qa_issues_via_skill` (PRD #227 follow-up, gap 2, `gh issue view
# 227`): a prior agent, needing a nested `{prd, issues: [{questions}]}` shape
# for QA-grilling the flat parse-interview skill couldn't represent, built a
# completely separate mechanism (`_QA_EXTRACTION_PROMPT_TEMPLATE` sent
# directly via `parser_session.stream_turn`, bypassing the skill) instead of
# extending it. This was flagged as a real inconsistency and unified: the
# skill's own `qa_grilling_issues`-phase section now emits the nested shape,
# and `_extract_qa_issues_via_skill` invokes it the same way grilling/
# implement's own extraction does.
# ---------------------------------------------------------------------------


def test_extract_qa_issues_via_skill_invokes_the_parse_interview_skill(client, tmp_path, monkeypatch):
    """The prompt actually sent to the parser session must reference the
    `/rhubarb:parse-interview` skill with `phase: qa_grilling_issues` (a
    distinct phase from the `qa_grilling` phase the needs-input queue's own
    flat-shape extraction already uses for this same raw text via a
    different mechanism -- see `_extract_qa_issues_via_skill`'s own
    docstring for why that distinction matters), and the returned data must
    be the nested `{prd, issues: [...]}` shape with 2+ issues, each with
    their own question(s), for a scripted multi-issue response."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    scripted_response = {
        "prd": {"number": 12, "title": "Add retry backoff"},
        "issues": [
            {
                "number": 13,
                "title": "Client retry loop",
                "questions": [
                    {
                        "id": "issue13-q1",
                        "text": "Does the client actually stop retrying after 3 attempts?",
                        "recommended_text": "Yes, confirmed in the logs.",
                    }
                ],
            },
            {
                "number": 14,
                "title": "Backoff timing",
                "questions": [
                    {"id": "issue14-q1", "text": "Is the backoff delay actually exponential?", "recommended_text": None},
                    {"id": "issue14-q2", "text": "Is jitter applied between attempts?", "recommended_text": None},
                ],
            },
        ],
    }
    seen_prompts = []

    class FakeParserEngine:
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.session_id = resume_session_id or "parser-session-fake"
            self._alive = False

        def start(self):
            self._alive = True
            return self

        def close(self):
            self._alive = False

        def isalive(self):
            return self._alive

        async def stream_turn(self, prompt):
            seen_prompts.append(prompt)
            yield {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": json.dumps(scripted_response),
                "session_id": self.session_id,
            }

    monkeypatch.setattr(parser_session, "StreamJsonEngine", FakeParserEngine)
    parser_session._parser_sessions.pop(project_id, None)
    asyncio.run(parser_session.ensure_parser_session(project_id, cwd=cwd))

    raw_text = (
        'QA session for PRD 12: "Add retry backoff"\n\n'
        'Issue 13: "Client retry loop"\n'
        'Question 1: "Does the client actually stop retrying after 3 attempts?"\n'
        'Recommended text: "Yes, confirmed in the logs."\n\n'
        'Issue 14: "Backoff timing"\n'
        'Question 1: "Is the backoff delay actually exponential?"\n'
        'Question 2: "Is jitter applied between attempts?"\n'
    )
    result = asyncio.run(session_runner._extract_qa_issues_via_skill(project_id, raw_text))

    # The prompt sent invokes the skill uniformly, same convention as the
    # existing grilling/implement extraction calls.
    assert len(seen_prompts) == 1
    assert "/rhubarb:parse-interview" in seen_prompts[0]
    assert "phase: qa_grilling_issues" in seen_prompts[0]
    assert raw_text in seen_prompts[0]

    assert result["source"] == "parser_session"
    assert result["prd"] == {"number": 12, "title": "Add retry backoff"}
    assert len(result["issues"]) == 2
    assert result["issues"][0]["number"] == 13
    assert len(result["issues"][0]["questions"]) == 1
    assert result["issues"][0]["questions"][0]["recommended_text"] == "Yes, confirmed in the logs."
    assert result["issues"][1]["number"] == 14
    assert len(result["issues"][1]["questions"]) == 2
    assert result["issues"][1]["questions"][1]["text"] == "Is jitter applied between attempts?"


def test_extract_qa_issues_via_skill_returns_none_on_schema_invalid_response(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    class FakeParserEngine:
        def __init__(self, *, cwd=None, model=None, effort=None, resume_session_id=None, process_factory=None):
            self.session_id = "parser-session-fake"
            self._alive = False

        def start(self):
            self._alive = True
            return self

        def close(self):
            self._alive = False

        def isalive(self):
            return self._alive

        async def stream_turn(self, prompt):
            yield {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": json.dumps({"not": "the right shape"}),
                "session_id": self.session_id,
            }

    monkeypatch.setattr(parser_session, "StreamJsonEngine", FakeParserEngine)
    parser_session._parser_sessions.pop(project_id, None)
    asyncio.run(parser_session.ensure_parser_session(project_id, cwd=cwd))

    result = asyncio.run(session_runner._extract_qa_issues_via_skill(project_id, "some text"))

    assert result is None


# ---------------------------------------------------------------------------
# QA auto-handoff tests (issue #52)
# ---------------------------------------------------------------------------

_QA_BLOCK = """\
```json
{
  "phase": "qa_grilling",
  "prd": {"number": 7, "title": "Tracked PRD"},
  "checklist": [
    {
      "issue_number": 8,
      "issue_title": "Child",
      "items": [
        {"id": "8-0", "text": "works"}
      ]
    }
  ]
}
```"""

_IMPLEMENT_BLOCKED_BLOCK = """\
Some preamble text.

```json
{
  "phase": "implement_blocked",
  "issue": 8,
  "question": "Which auth provider should the login button use?",
  "context": "The issue body doesn't specify Google vs GitHub OAuth."
}
```"""


def test_finish_implement_turn_never_auto_spawns_qa_even_with_qa_grilling_block(client, tmp_path, monkeypatch):
    """Issue #250 (child of PRD #244): an implement turn that completes
    cleanly always takes the plain "implemented" path now -- no automatic
    QA handoff occurs even when the turn's text happens to contain a
    qa_grilling-shaped block. QA only ever starts via the explicit
    "Move to QA phase" action (`start_move_to_qa_job`)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    tracker = {
        "prd": {"number": 7, "title": "Tracked PRD"},
        "issues": [{"number": 8, "title": "Child", "summary": "does the thing", "acceptance_criteria": ["works"]}],
        "qa_changes": [],
        "status": "implemented",
    }
    claude_dir = Path(cwd) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "implement-tracker.json").write_text(json.dumps(tracker), encoding="utf-8")

    qa_turn_text = (
        _QA_BLOCK + "\n\n"
        'QA session for PRD 7: "Tracked PRD"\n\n'
        'Issue 8: "Child"\n'
        'Question 1: "Does it work?"\n'
        'Recommended text: "Yes."\n'
    )
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(qa_turn_text, session_id="impl-session-id")]))
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

    # Implement session ends implemented and pooled -- exactly like a turn
    # with no qa_grilling-shaped text at all.
    impl_row = db.get_session(conn, row_id)
    assert impl_row["phase"] == "implemented"
    assert impl_row["available_for_reuse"] == 1

    # No QA session was ever created.
    sessions = db.list_sessions_for_project(conn, project_id)
    assert not [s for s in sessions if s["session_type"] == "qa"]

    impl_events = live_stream._buffers.get(row_id, [])
    assert not [e for e in impl_events if e.get("type") == "qa_started"]
    assert impl_events[-1] == {"type": "done"}


def test_start_move_to_qa_job_creates_fresh_session_and_runs_qa_as_first_turn(client, tmp_path, monkeypatch):
    """Issue #250 (child of PRD #244): the explicit "Move to QA phase"
    action closes the finished implement session's card outright (no
    --resume) and runs /rhubarb:qa as a brand-new session's very first
    turn, publishing the parsed qa_grilling round on the new row exactly
    like the old auto-handoff used to."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    qa_turn_text = (
        _QA_BLOCK + "\n\n"
        'QA session for PRD 7: "Tracked PRD"\n\n'
        'Issue 8: "Child"\n'
        'Question 1: "Does it work?"\n'
        'Recommended text: "Yes."\n'
    )
    seen = []

    def handler(prompt, **kw):
        seen.append((prompt, kw["session_id"]))
        return iter([_result_event(qa_turn_text, session_id="qa-session-id")])

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
        claude_session_id="old-implement-session-id",
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    # The first (and only) turn was exactly "/rhubarb:qa", with no
    # --resume -- a genuinely fresh conversation.
    assert seen == [("/rhubarb:qa", None)]

    # The old implement card is closed outright.
    impl_row = db.get_session(conn, row_id)
    assert impl_row["phase"] == "closed"
    impl_events = live_stream._buffers.get(row_id, [])
    assert {"type": "closed", "card_id": row_id} in impl_events
    assert row_id not in session_runner._stream_json_engines

    # A brand-new QA session row was created.
    sessions = db.list_sessions_for_project(conn, project_id)
    qa_sessions = [s for s in sessions if s["session_type"] == "qa"]
    assert len(qa_sessions) == 1
    qa_row = qa_sessions[0]
    assert qa_row["phase"] == "qa_grilling"
    assert qa_row["claude_session_id"] == "qa-session-id"
    assert json.loads(qa_row["details_json"])["prd"] == {"number": 7, "title": "Tracked PRD"}

    # The nested issues/questions structure was extracted from the turn's
    # free text via the parser-session pipeline
    # (`_extract_qa_issues_via_skill`) and published on the new QA
    # session's stream -- the qa_grilling JSON block itself only carried the
    # {phase, prd} signal.
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert len(qa_turn_events) == 1
    assert qa_turn_events[0]["issues"] == [
        {
            "number": 8,
            "title": "Child",
            "questions": [{"id": "issue8-q1", "text": "Does it work?", "recommended_text": "Yes."}],
        }
    ]


def test_start_move_to_qa_job_prefers_rhubarb_qa_file_over_terminal_text(client, tmp_path, monkeypatch):
    """PRD #123: a QA handoff turn whose terminal text has no recognizable
    QA session block still produces a full issues/questions structure when
    `.claude/rhubarb_qa.md` is present with valid content."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)
    _write_question_file(
        cwd, "rhubarb_qa.md",
        'QA session for PRD 7: "Tracked PRD"\n\nIssue 8: "Child"\nQuestion 1: "Does it work?"\nRecommended text: "Yes."\n',
    )

    # Terminal text carries the handoff signal block but no recognizable QA session text.
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(_QA_BLOCK, session_id="qa-session-id")]))
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )
    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == [
        {
            "number": 8,
            "title": "Child",
            "questions": [{"id": "issue8-q1", "text": "Does it work?", "recommended_text": "Yes."}],
        }
    ]


def test_start_move_to_qa_job_rhubarb_qa_file_is_not_deleted_merely_by_being_read(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)
    _write_question_file(cwd, "rhubarb_qa.md", 'QA session for PRD 7: "Tracked PRD"\n\nIssue 8: "Child"\nQuestion 1: "Does it work?"\n')

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(_QA_BLOCK, session_id="qa-session-id")]))
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )
    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    assert (Path(cwd) / ".claude" / "rhubarb_qa.md").exists()


def test_continue_qa_job_deletes_rhubarb_qa_file(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id, session_type="qa", phase="qa_grilling",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )
    _write_question_file(cwd, "rhubarb_qa.md", 'QA session for PRD 7: "Tracked PRD"\n\nIssue 8: "Child"\nQuestion 1: "Does it work?"\n')

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("All good, PRD closed.")]))
    asyncio.run(session_runner.continue_qa_job(qa_row_id, {}, "", cwd=cwd))

    assert not (Path(cwd) / ".claude" / "rhubarb_qa.md").exists()


def test_start_move_to_qa_job_malformed_rhubarb_qa_file_falls_back_to_terminal_text(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)
    _write_question_file(cwd, "rhubarb_qa.md", "not a recognizable QA session format at all")

    qa_turn_text = (
        _QA_BLOCK + "\n\n"
        'QA session for PRD 7: "Tracked PRD"\n\n'
        'Issue 8: "Child"\n'
        'Question 1: "From terminal text"\n'
    )
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(qa_turn_text, session_id="qa-session-id")]))
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )
    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"][0]["questions"][0]["text"] == "From terminal text"
    assert not (Path(cwd) / ".claude" / "rhubarb_qa.md").exists()


def test_start_move_to_qa_job_qa_extraction_ignores_ollama_declined_setting(client, tmp_path, monkeypatch):
    """Issue #230: the QA-grilling extraction chain no longer has any
    Ollama-rescue step at all, so `ollama_declined` (issue #119's opt-out,
    which used to gate that rescue call) has no effect on it any more -- a
    well-formed QA round extracts identically via the parser-session
    pipeline whether or not the user has declined Ollama assistance.
    Supersedes `test_start_implement_job_skips_ollama_rescue_when_declined`/
    `test_start_implement_job_uses_ollama_rescue_when_not_declined` (removed
    rather than kept red: the Ollama-rescue mechanism they exercised no
    longer exists)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    _qa_tracker(cwd)

    qa_turn_text = (
        _QA_BLOCK + "\n\n"
        'QA session for PRD 7: "Tracked PRD"\n\n'
        'Issue 8: "Child"\n'
        'Question 1: "Does it work?"\n'
    )
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(qa_turn_text, session_id="qa-session-id")]))

    conn = db.get_connection()
    db.set_ollama_declined(conn, True)
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    assert qa_row["claude_session_id"] == "qa-session-id"
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"][0]["questions"][0]["text"] == "Does it work?"


# ---------------------------------------------------------------------------
# QA-grilling corrective-retry tests (issue #159 -- mirrors #158's grilling
# corrective-retry tests, applied to the QA-grilling parse path instead)
# ---------------------------------------------------------------------------


def test_move_to_qa_corrective_retry_fires_once_and_uses_reformatted_result(client, tmp_path, monkeypatch):
    """A QA handoff turn whose result contains recognizable QA-question-
    attempt content (the "QA session for PRD" trigger) but doesn't extract
    into any issues must trigger exactly one corrective follow-up turn --
    handing the model its own unparseable output back -- rather than
    silently handing off a QA session with zero issues."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    _qa_tracker(cwd)

    malformed_qa_text = _QA_BLOCK + "\n\nQA session for PRD 7: this is malformed, no quoted title at all\n"
    well_formed_retry_text = (
        'QA session for PRD 7: "Tracked PRD"\n\n'
        'Issue 8: "Child"\n'
        'Question 1: "Does the reformatted round parse now?"\n'
    )
    seen_prompts = []

    def handler(prompt, **kw):
        seen_prompts.append(prompt)
        if len(seen_prompts) == 1:
            return iter([_result_event(malformed_qa_text, session_id="qa-session-id")])
        return iter([_result_event(well_formed_retry_text, session_id="qa-session-id-retry")])

    _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    # Exactly one corrective retry -- the initial handoff turn, plus one
    # follow-up, and no more.
    assert len(seen_prompts) == 2
    # The retry prompt hands the model its own broken output back.
    assert malformed_qa_text in seen_prompts[1]

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    assert qa_row["claude_session_id"] == "qa-session-id-retry"
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"][0]["questions"][0]["text"] == "Does the reformatted round parse now?"


def test_move_to_qa_no_retry_when_result_has_no_recognizable_qa_content(client, tmp_path, monkeypatch):
    """A genuine 'no QA questions' handoff -- the qa_grilling JSON marker is
    present, but neither the (nonexistent) question file nor the terminal
    text contains anything resembling a QA session round -- must NOT trigger
    any corrective retry. This is treated as done, same as today: a QA
    session is still created with an empty issues list."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    seen_prompts = []

    def handler(prompt, **kw):
        seen_prompts.append(prompt)
        # Only the qa_grilling JSON marker -- no "QA session for PRD" prose
        # at all, in either the terminal text or (absent) file.
        return iter([_result_event(_QA_BLOCK, session_id="qa-session-id")])

    _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    # No corrective retry -- only the original handoff turn.
    assert len(seen_prompts) == 1

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    assert qa_row["claude_session_id"] == "qa-session-id"
    assert qa_row["error_text"] is None
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == []


def test_move_to_qa_corrective_retry_publishes_explicit_error_when_still_unparseable(client, tmp_path, monkeypatch):
    """If the corrective retry's own result also fails to parse, an explicit
    error must be published/persisted on the new QA row (and land in the
    app-wide error log) instead of silently handing off with empty issues.
    The old implement card is already closed by this point, so the QA row
    -- not the implement row -- is where this error now lands."""
    log_path = tmp_path / "err-home" / "logs" / "errors.log"
    monkeypatch.setattr(error_log, "DEFAULT_LOG_PATH", log_path)

    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    malformed_qa_text = _QA_BLOCK + "\n\nQA session for PRD 7: this is malformed, no quoted title at all\n"
    still_malformed_retry_text = "QA session for PRD 7: still no quoted title, still broken\n"
    seen_prompts = []

    def handler(prompt, **kw):
        seen_prompts.append(prompt)
        if len(seen_prompts) == 1:
            return iter([_result_event(malformed_qa_text, session_id="qa-session-id")])
        return iter([_result_event(still_malformed_retry_text, session_id="qa-session-id-retry")])

    _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    # Exactly one corrective retry -- not an unbounded loop.
    assert len(seen_prompts) == 2

    # The QA row still exists (created before the /rhubarb:qa turn ran),
    # but carries the explicit error instead of a parsed qa_grilling round.
    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    assert qa_row["error_text"]

    qa_events = live_stream._buffers.get(qa_row["id"], [])
    error_turn_events = [
        e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling" and e.get("error")
    ]
    assert len(error_turn_events) == 1
    assert qa_events[-1] == {"type": "done"}

    errors = error_log.query_errors(project_id, log_path=log_path)
    assert any(e["phase"] == "qa_grilling" and e["card_id"] == qa_row["id"] for e in errors)


# ---------------------------------------------------------------------------
# Ollama needs-input classification wired into QA-grilling (issue #178, child
# of PRD #174) -- mirrors the grilling tests above, applied to the
# QA-grilling handoff's own fallback chain instead.
# ---------------------------------------------------------------------------


def _qa_tracker(cwd):
    tracker = {
        "prd": {"number": 7, "title": "Tracked PRD"},
        "issues": [{"number": 8, "title": "Child", "summary": "does the thing", "acceptance_criteria": ["works"]}],
        "qa_changes": [],
        "status": "implemented",
    }
    claude_dir = Path(cwd) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "implement-tracker.json").write_text(json.dumps(tracker), encoding="utf-8")


def test_move_to_qa_needs_input_classification_renders_rich_ui_on_genuine_wrapup(client, tmp_path, monkeypatch):
    """A QA handoff round that looks like a genuine "no more QA questions"
    wrap-up (no "QA session for PRD" trigger at all, so the existing chain
    never even attempts a corrective retry) must still be handed to the
    Ollama needs-input classifier before the QA session is created with an
    empty issues list. When it says a human's input is still needed, and a
    second parser-session extraction attempt on the same text actually
    produces issue/question content, that rich content is what the new QA
    session is started with instead."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    wrapup_text = _QA_BLOCK + "\n\nEverything checks out, no further QA questions."
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(wrapup_text, session_id="qa-session-id")]))

    rescued = {
        "prd": None,
        "issues": [
            {
                "number": 8,
                "title": "Child",
                "questions": [{"id": "issue8-q1", "text": "Actually, one more check needed", "recommended_text": None}],
            }
        ],
    }
    seen = {}
    extract_calls = {"n": 0}

    async def fake_classify(card_id, conn, text, phase, **kw):
        seen["phase"] = phase
        return {"needs_input": True, "reason": "Still worth a follow-up check."}

    async def fake_extract(project_id, raw_text):
        # First call is the primary extraction attempt in
        # `start_move_to_qa_job` itself (finds nothing, matching this
        # genuine wrap-up text); the second is
        # `_maybe_extract_needs_input_qa`'s own last-resort attempt, once
        # the classifier has said a human's input is still needed -- that
        # second call is the one this test actually cares about.
        extract_calls["n"] += 1
        if extract_calls["n"] == 1:
            return None
        seen["rescue_raw_text"] = raw_text
        return rescued

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)
    monkeypatch.setattr(session_runner, "_extract_qa_issues_via_skill", fake_extract)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    assert seen["phase"] == "qa_grilling"
    assert seen["rescue_raw_text"] == wrapup_text

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == rescued["issues"]


def test_move_to_qa_needs_input_classification_false_falls_through_to_wrapup(client, tmp_path, monkeypatch):
    """A negative classification must fall through to today's unchanged QA
    handoff -- a QA session still created, but with an empty issues list --
    and must never trigger `_maybe_extract_needs_input_qa`'s own last-resort
    extraction attempt (a SECOND `_extract_qa_issues_via_skill`
    call, beyond `start_move_to_qa_job`'s own unconditional primary one)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event(_QA_BLOCK + "\n\nAll good, nothing further.", session_id="qa-session-id")]),
    )

    async def fake_classify(card_id, conn, text, phase, **kw):
        return {"needs_input": False, "reason": "Genuinely done."}

    extract_calls = {"n": 0}

    async def fake_extract(project_id, raw_text):
        extract_calls["n"] += 1
        return None

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)
    monkeypatch.setattr(session_runner, "_extract_qa_issues_via_skill", fake_extract)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    # Exactly one extraction attempt -- the primary one in
    # `start_move_to_qa_job` itself; the classifier's own last-resort
    # extraction must never fire when `needs_input` is false.
    assert extract_calls["n"] == 1

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == []


def test_move_to_qa_needs_input_true_but_extraction_fails_falls_through_to_wrapup(client, tmp_path, monkeypatch):
    """A positive classification whose extraction attempt still comes up
    empty must fall through to today's unchanged QA handoff wrap-up."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event(_QA_BLOCK + "\n\nAll good, nothing further.", session_id="qa-session-id")]),
    )

    async def fake_classify(card_id, conn, text, phase, **kw):
        return {"needs_input": True, "reason": "Looks unfinished."}

    async def fake_extract(project_id, raw_text):
        return None

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)
    monkeypatch.setattr(session_runner, "_extract_qa_issues_via_skill", fake_extract)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == []


def test_move_to_qa_needs_input_classification_rescues_after_corrective_retry_also_failed(client, tmp_path, monkeypatch):
    """Once the QA-grilling corrective retry (issue #159) has already run
    and its own extraction attempt *also* came up empty, the needs-input
    classifier gets one last look at that retry's own text before the
    session gives up with an explicit error. A positive classification whose
    OWN follow-up extraction attempt actually succeeds must create the QA
    session with that content instead of publishing that explicit error --
    modeled here as the retry's own `_extract_qa_issues_via_skill`
    call coming up empty the first time it's tried against the retry's text,
    then recovering on the classifier's own follow-up call against that same
    text (issue #230: both stages now share the exact same extraction
    mechanism, unlike the old regex-then-Ollama-rescue chain where they were
    two genuinely different attempts)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    malformed_qa_text = _QA_BLOCK + "\n\nQA session for PRD 7: this is malformed, no quoted title at all\n"
    still_malformed_retry_text = "QA session for PRD 7: still no quoted title, still broken\n"
    seen_prompts = []

    def handler(prompt, **kw):
        seen_prompts.append(prompt)
        if len(seen_prompts) == 1:
            return iter([_result_event(malformed_qa_text, session_id="qa-session-id")])
        return iter([_result_event(still_malformed_retry_text, session_id="qa-session-id-retry")])

    _mock_engine(monkeypatch, handler)

    rescued = {
        "prd": None,
        "issues": [
            {
                "number": 8,
                "title": "Child",
                "questions": [{"id": "issue8-q1", "text": "Extracted after the retry failed too", "recommended_text": None}],
            }
        ],
    }
    retry_text_extract_calls = {"n": 0}

    async def fake_extract(project_id, raw_text):
        if raw_text != still_malformed_retry_text:
            return None
        retry_text_extract_calls["n"] += 1
        if retry_text_extract_calls["n"] == 1:
            return None  # the retry turn's own extraction attempt still fails
        return rescued  # the needs-input classifier's own follow-up attempt succeeds

    async def fake_classify(card_id, conn, text, phase, **kw):
        assert text == still_malformed_retry_text
        assert phase == "qa_grilling"
        return {"needs_input": True, "reason": "The retry still looks unfinished."}

    monkeypatch.setattr(session_runner, "_extract_qa_issues_via_skill", fake_extract)
    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    assert len(seen_prompts) == 2, "expected exactly one corrective retry turn, in addition to the original"

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    assert qa_row["claude_session_id"] == "qa-session-id-retry"
    assert qa_row["error_text"] is None

    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == rescued["issues"]


def test_start_implement_job_pools_session_normally_when_no_qa_block(client, tmp_path, monkeypatch):
    """When the implement result has no qa_grilling block the session must be
    pooled as before (no QA session created)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Implemented PRD #5", session_id="impl1")]),
        fresh_ids=["turn-engine", "pooled-1"],
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["available_for_reuse"] == 1
    assert row["claude_session_id"] == "pooled-1"

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_sessions = [s for s in sessions if s["session_type"] == "qa"]
    assert len(qa_sessions) == 0

    # Pooling closed this card's tab.
    assert row_id not in session_runner._stream_json_engines


def test_start_qa_job_publishes_qa_grilling_turn_without_done(client, tmp_path, monkeypatch):
    """start_qa_job emits the qa_grilling phase + turn events and does NOT fire done."""
    project_id = _open_project(client, tmp_path, "proj")

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id,
        session_type="qa", phase="qa_grilling",
        details={"prd": {"number": 7, "title": "Test PRD"}},
    )

    prd = {"number": 7, "title": "Test PRD"}
    issues = [
        {
            "number": 8,
            "title": "Child",
            "questions": [{"id": "issue8-q1", "text": "Does it work?", "recommended_text": None}],
        }
    ]

    asyncio.run(session_runner.start_qa_job(qa_row_id, prd, issues, cwd=None))

    events = live_stream._buffers.get(qa_row_id, [])
    assert {"type": "phase", "phase": "qa_grilling"} in events
    turn_events = [e for e in events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert len(turn_events) == 1
    assert turn_events[0]["prd"] == prd
    assert turn_events[0]["issues"] == issues
    # No done event — session is suspended awaiting Perfect
    assert {"type": "done"} not in events


def test_continue_qa_job_runs_phase3_and_fires_done(client, tmp_path, monkeypatch):
    """continue_qa_job must run a CLI turn with the notes, then fire done."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    seen_prompts = []

    def recording(prompt, **kw):
        seen_prompts.append(prompt)
        return iter([_result_event("Closed all issues.", session_id="qa-done")])

    _mock_engine(monkeypatch, recording)

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id,
        session_type="qa", phase="qa_grilling",
        claude_session_id="qa-session-1",
        details={"prd": {"number": 7, "title": "Test PRD"}},
    )

    asyncio.run(
        session_runner.continue_qa_job(qa_row_id, {"issue8-q1": "Looks great"}, "", cwd=cwd)
    )

    assert len(seen_prompts) == 1
    assert "issue8-q1: Looks great" in seen_prompts[0]

    row = db.get_session(conn, qa_row_id)
    assert row["phase"] == "qa_closing"
    assert row["claude_session_id"] == "qa-done"

    events = live_stream._buffers.get(qa_row_id, [])
    assert {"type": "phase", "phase": "qa_closing"} in events
    assert events[-1] == {"type": "done"}
    # This card's tab is done either way (recycled or not).
    assert qa_row_id not in session_runner._stream_json_engines


def _usage_event(input_tokens, cache_creation, cache_read, context_window, model="claude-sonnet-4-6"):
    return _result_event(
        "done",
        usage={
            "input_tokens": input_tokens,
            "cache_creation_input_tokens": cache_creation,
            "cache_read_input_tokens": cache_read,
        },
        model_usage={model: {"contextWindow": context_window}},
    )


def test_continue_qa_job_recycles_a_low_usage_finished_session(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter(
            [_usage_event(input_tokens=0, cache_creation=0, cache_read=300, context_window=1000, model="claude-sonnet-4-6")]
        ),
    )

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id,
        session_type="qa", phase="qa_grilling",
        claude_session_id="qa-session-1",
        details={"prd": {"number": 7, "title": "Test PRD"}},
    )

    asyncio.run(session_runner.continue_qa_job(qa_row_id, {}, "Looks great", cwd=cwd))

    row = db.get_session(conn, qa_row_id)
    assert row["available_for_reuse"] == 1
    assert row["context_pct"] == pytest.approx(0.3)

    reused = db.claim_available_session(conn, project_id)
    assert reused is not None
    assert reused["id"] == qa_row_id


def test_continue_qa_job_does_not_recycle_a_high_usage_finished_session(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter(
            [_usage_event(input_tokens=0, cache_creation=0, cache_read=800, context_window=1000, model="claude-sonnet-4-6")]
        ),
    )

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id,
        session_type="qa", phase="qa_grilling",
        claude_session_id="qa-session-1",
        details={"prd": {"number": 7, "title": "Test PRD"}},
    )

    asyncio.run(session_runner.continue_qa_job(qa_row_id, {}, "Looks great", cwd=cwd))

    row = db.get_session(conn, qa_row_id)
    assert row["available_for_reuse"] == 0
    assert db.claim_available_session(conn, project_id) is None


def _word_wrap_at(text, width):
    """Simulate a terminal re-wrapping already-rendered text at `width`
    columns (issue #163): each physical line is independently re-flowed to
    `width` at word (space) boundaries -- mirroring how a PTY wraps a
    stream of already-authored text -- without collapsing the JSON
    pretty-printer's own original line breaks into one paragraph the way
    `textwrap.wrap` would if applied to the whole blob at once. A no-op for
    any line already within `width`."""
    import textwrap

    out_lines = []
    for line in text.split("\n"):
        if not line:
            out_lines.append("")
            continue
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
        out_lines.extend(wrapped or [""])
    return "\n".join(out_lines)


_QA_BLOCK_LONG_TEXT = """\
```json
{
  "phase": "qa_grilling",
  "prd": {"number": 7, "title": "Tracked PRD"},
  "checklist": [
    {
      "issue_number": 8,
      "issue_title": "Child",
      "items": [
        {"id": "8-0", "text": "works correctly even when the terminal wraps this long item text at a narrower column than it was authored at"}
      ]
    }
  ]
}
```"""

_IMPLEMENT_BLOCKED_BLOCK_LONG_TEXT = """\
Some preamble text.

```json
{
  "phase": "implement_blocked",
  "issue": 8,
  "question": "Which auth provider should the login button use, given the issue body never specifies Google vs GitHub OAuth for this particular flow?",
  "context": "The issue body doesn't specify Google vs GitHub OAuth."
}
```"""


def test_parse_qa_grilling_block_tolerates_word_wrap_at_a_different_column(client, tmp_path):
    """A fenced qa_grilling block that's been word-wrapped at an arbitrary
    column (simulating a narrower terminal than it was authored at, issue
    #163) must still parse -- the wrap can inject a raw newline in the
    middle of a long string value, which `_parse_qa_grilling_block` must
    rejoin before handing the block to `json.loads`."""
    from rhubarb.session_runner import _parse_qa_grilling_block

    wrapped = _word_wrap_at(_QA_BLOCK_LONG_TEXT, width=28)
    assert wrapped != _QA_BLOCK_LONG_TEXT  # sanity: this actually rewrapped something

    result = _parse_qa_grilling_block(wrapped)
    assert result is not None
    assert result["phase"] == "qa_grilling"
    assert result["prd"]["number"] == 7
    assert result["checklist"][0]["items"][0]["text"] == (
        "works correctly even when the terminal wraps this long item "
        "text at a narrower column than it was authored at"
    )


def test_parse_implement_blocked_block_tolerates_word_wrap_at_a_different_column():
    """Same word-wrap tolerance as the qa_grilling case above, for the
    implement_blocked block (issue #163)."""
    from rhubarb.session_runner import _parse_implement_blocked_block

    wrapped = _word_wrap_at(_IMPLEMENT_BLOCKED_BLOCK_LONG_TEXT, width=28)
    assert wrapped != _IMPLEMENT_BLOCKED_BLOCK_LONG_TEXT

    result = _parse_implement_blocked_block(wrapped)
    assert result is not None
    assert result["phase"] == "implement_blocked"
    assert result["issue"] == 8
    assert result["question"] == (
        "Which auth provider should the login button use, given the "
        "issue body never specifies Google vs GitHub OAuth for this "
        "particular flow?"
    )


def test_parse_qa_grilling_block_returns_none_for_malformed_json_in_fence(client, tmp_path):
    """A fenced block that matches `_FENCED_JSON_BLOCK_RE` but isn't valid
    JSON (a trailing comma, here) is a genuine parse failure -- not a
    word-wrap artifact -- and must still return None, unaffected by the
    word-wrap rejoin step."""
    from rhubarb.session_runner import _parse_qa_grilling_block

    malformed = '```json\n{"phase": "qa_grilling", "prd": {"number": 7,}}\n```'
    assert _parse_qa_grilling_block(malformed) is None


def test_parse_implement_blocked_block_returns_none_for_malformed_json_in_fence():
    """Same genuinely-malformed-JSON regression check as the qa_grilling
    case above, for the implement_blocked parser."""
    from rhubarb.session_runner import _parse_implement_blocked_block

    malformed = '```json\n{"phase": "implement_blocked", "issue": 8, "question": "Which?",}\n```'
    assert _parse_implement_blocked_block(malformed) is None


def test_parse_qa_grilling_block_extracts_json_from_code_fence(client, tmp_path):
    """_parse_qa_grilling_block must find and return the qa_grilling JSON block."""
    from rhubarb.session_runner import _parse_qa_grilling_block

    text = "Some preamble\n\n" + _QA_BLOCK + "\n\nSome trailing text"
    result = _parse_qa_grilling_block(text)
    assert result is not None
    assert result["phase"] == "qa_grilling"
    assert result["prd"]["number"] == 7
    assert len(result["checklist"]) == 1


def test_parse_qa_grilling_block_returns_none_for_plain_text(client, tmp_path):
    """_parse_qa_grilling_block must return None when no qa_grilling block is present."""
    from rhubarb.session_runner import _parse_qa_grilling_block

    assert _parse_qa_grilling_block("Implemented PRD #5") is None
    assert _parse_qa_grilling_block("") is None
    assert _parse_qa_grilling_block('```json\n{"phase": "other"}\n```') is None


def test_parse_implement_blocked_block_extracts_json_from_code_fence():
    """_parse_implement_blocked_block must find and return the implement_blocked JSON block."""
    from rhubarb.session_runner import _parse_implement_blocked_block

    result = _parse_implement_blocked_block(_IMPLEMENT_BLOCKED_BLOCK)
    assert result is not None
    assert result["phase"] == "implement_blocked"
    assert result["issue"] == 8
    assert "auth provider" in result["question"]


def test_parse_implement_blocked_block_extracts_bare_json():
    from rhubarb.session_runner import _parse_implement_blocked_block

    bare = json.dumps({"phase": "implement_blocked", "issue": None, "question": "Which one?", "context": "ambiguous"})
    result = _parse_implement_blocked_block(bare)
    assert result is not None
    assert result["question"] == "Which one?"


def test_parse_implement_blocked_block_returns_none_for_plain_text():
    from rhubarb.session_runner import _parse_implement_blocked_block

    assert _parse_implement_blocked_block("Implemented PRD #5") is None
    assert _parse_implement_blocked_block("") is None
    assert _parse_implement_blocked_block(_QA_BLOCK) is None


def test_start_implement_job_suspends_as_blocked_when_result_contains_blocked_marker(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch, lambda prompt, **kw: iter([_result_event(_IMPLEMENT_BLOCKED_BLOCK, session_id="impl-blocked")])
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    assert row["claude_session_id"] == "impl-blocked"
    blocked = json.loads(row["blocked_json"])
    assert blocked["question"] == "Which auth provider should the login button use?"
    assert row["available_for_reuse"] == 0

    events = live_stream._buffers.get(row_id, [])
    assert any(e.get("type") == "turn" and e.get("phase") == "blocked" for e in events)
    # Suspended, not finished -- no `done` yet, same shape as qa_grilling awaiting Perfect.
    assert {"type": "done"} not in events

    # Suspended, not finished -- the tab stays open for continue_implement_job.
    assert row_id in session_runner._stream_json_engines


def test_start_implement_job_prefers_rhubarb_blocked_file_over_terminal_text(client, tmp_path, monkeypatch):
    """PRD #123: a turn whose terminal text has no recognizable
    `implement_blocked` block still suspends the session as blocked, with
    the right question/context, when `.claude/rhubarb_blocked.json` is
    present with valid content."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _write_question_file(
        cwd, "rhubarb_blocked.json",
        json.dumps({
            "phase": "implement_blocked", "issue": 8,
            "question": "Which auth provider should the login button use?",
            "context": "The issue body doesn't specify Google vs GitHub OAuth.",
        }),
    )

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("Just plain prose, no marker.", session_id="impl-blocked")]))

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    blocked = json.loads(row["blocked_json"])
    assert blocked["question"] == "Which auth provider should the login button use?"
    assert blocked["issue"] == 8


def test_rhubarb_blocked_file_is_not_deleted_merely_by_being_read(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _write_question_file(
        cwd, "rhubarb_blocked.json",
        json.dumps({"phase": "implement_blocked", "issue": None, "question": "Which one?", "context": "..."}),
    )

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("plain prose", session_id="impl-blocked")]))

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    assert (Path(cwd) / ".claude" / "rhubarb_blocked.json").exists()


def test_continue_implement_job_deletes_rhubarb_blocked_file(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="blocked",
        claude_session_id="impl-blocked",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    db.update_session(conn, row_id, blocked_json=json.dumps({"phase": "implement_blocked", "question": "Which?"}))
    _write_question_file(
        cwd, "rhubarb_blocked.json",
        json.dumps({"phase": "implement_blocked", "issue": None, "question": "Which?", "context": "..."}),
    )

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("Implemented, using GitHub OAuth.", session_id="impl-done")]))
    asyncio.run(session_runner.continue_implement_job(row_id, "Use GitHub OAuth", cwd=cwd))

    assert not (Path(cwd) / ".claude" / "rhubarb_blocked.json").exists()


def test_malformed_rhubarb_blocked_file_is_deleted_and_falls_back_to_terminal_text(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _write_question_file(cwd, "rhubarb_blocked.json", "not valid json at all")

    _mock_engine(
        monkeypatch, lambda prompt, **kw: iter([_result_event(_IMPLEMENT_BLOCKED_BLOCK, session_id="impl-blocked")])
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    blocked = json.loads(row["blocked_json"])
    assert blocked["question"] == "Which auth provider should the login button use?"
    assert not (Path(cwd) / ".claude" / "rhubarb_blocked.json").exists()


def test_continue_implement_job_resolves_a_blocked_session(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="blocked",
        claude_session_id="impl-blocked",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    db.update_session(conn, row_id, blocked_json=json.dumps({"phase": "implement_blocked", "question": "Which?"}))

    seen_prompts = []

    def handler(prompt, **kw):
        seen_prompts.append(prompt)
        return iter([_result_event("Implemented, using GitHub OAuth as you said.", session_id="impl-done")])

    _mock_engine(monkeypatch, handler, fresh_ids=["pooled-1"])

    asyncio.run(session_runner.continue_implement_job(row_id, "Use GitHub OAuth", cwd=cwd))

    assert seen_prompts == ["Use GitHub OAuth"]
    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert row["blocked_json"] is None
    assert row["available_for_reuse"] == 1

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}


def test_continue_implement_job_can_re_block_on_a_second_question(client, tmp_path, monkeypatch):
    """A reply that doesn't fully unblock it must re-suspend as blocked
    again, not error or silently complete."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="blocked",
        claude_session_id="impl-blocked",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )

    second_block = _IMPLEMENT_BLOCKED_BLOCK.replace("auth provider", "callback URL")
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(second_block, session_id="impl-blocked-2")]))

    asyncio.run(session_runner.continue_implement_job(row_id, "GitHub OAuth", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    blocked = json.loads(row["blocked_json"])
    assert "callback URL" in blocked["question"]

    events = live_stream._buffers.get(row_id, [])
    assert {"type": "done"} not in events


# ---------------------------------------------------------------------------
# Resident-tab persistence (issue #87): one PtyEngine per card_id, kept
# alive across every turn for that card, not recreated per turn.
# ---------------------------------------------------------------------------


def test_engine_is_constructed_once_and_reused_across_turns_in_the_same_phase(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event('Question 1: "First question?"')])
        return iter([_result_event('Question 1: "Follow-up?"')])

    fake_class = _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "tell me more", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "and more", cwd=cwd))

    # Exactly one engine was ever constructed for this card_id, reused
    # across all three turns. Issue #184: this card stayed in grilling the
    # whole time, so its resident tab is a `StreamJsonEngine`
    # (`_stream_json_engines`), not a `PtyEngine` -- the resident-tab
    # persistence guarantee itself is unchanged, just backed by a different
    # registry for this phase.
    assert len(fake_class.instances) == 1
    assert fake_class.instances[0].started is True
    assert row_id in session_runner._stream_json_engines
    assert session_runner._stream_json_engines[row_id] is fake_class.instances[0]


def test_engine_reattaches_via_resume_when_continuing_an_existing_session_id(client, tmp_path, monkeypatch):
    """A brand-new card_id whose row already carries a claude_session_id
    (e.g. a reused pooled session) must construct its engine with
    `resume_session_id=`, not a fresh one."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event('Question 1: "Another question?"', session_id="pooled-session")]),
    )
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id, claude_session_id="pooled-session")
    asyncio.run(session_runner.start_session_job(row_id, "another feature", cwd=cwd))

    assert len(fake_class.instances) == 1
    assert fake_class.instances[0].resume_session_id == "pooled-session"


def test_implement_tab_closes_and_standby_is_kept_alive_when_pooled(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Implemented.", session_id="impl1")]),
        fresh_ids=["pooled-1"],
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    assert db.get_session(conn, row_id)["available_for_reuse"] == 1
    assert row_id not in session_runner._stream_json_engines
    # The turn's own engine is closed; the fresh one minted for pooling
    # (issue #136) is kept alive as this project's standby instead of also
    # being closed immediately -- it's what the next /do claims directly.
    assert fake_class.instances[0].closed is True
    assert fake_class.instances[1].closed is False
    assert session_runner._standby_stream_json_engines[project_id][0] is fake_class.instances[1]


# ---------------------------------------------------------------------------
# Pre-warmed standby StreamJsonEngine (issue #136): a project keeps one
# unclaimed, already-running engine ready so a new /do claims it directly
# instead of paying spawn latency inline. (Additional coverage --
# spawn-when-none-exists, claim-on-match, claim-on-model-mismatch,
# close-and-discard, register-then-reuse -- lives in the `_stream_json_engine`
# suffixed tests further down this file.)
# ---------------------------------------------------------------------------


def test_ensure_standby_stream_json_engine_is_a_no_op_when_a_live_matching_one_exists(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )

    assert len(fake_class.instances) == 1  # no second spawn


def test_ensure_standby_stream_json_engine_replaces_a_dead_one(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )
    fake_class.instances[0].closed = True  # simulate a crash while unclaimed

    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )

    assert len(fake_class.instances) == 2
    assert session_runner._standby_stream_json_engines[project_id][0] is fake_class.instances[1]


def test_claim_standby_stream_json_engine_returns_none_and_discards_on_effort_mismatch(
    client, tmp_path, monkeypatch
):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )

    claimed = session_runner.claim_standby_stream_json_engine(project_id, model="claude-sonnet-5", effort="high")

    assert claimed is None
    assert fake_class.instances[0].closed is True


def test_claim_standby_stream_json_engine_returns_none_when_dead(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )
    fake_class.instances[0].closed = True

    claimed = session_runner.claim_standby_stream_json_engine(project_id, model="claude-sonnet-5", effort="auto")

    assert claimed is None
    assert project_id not in session_runner._standby_stream_json_engines


def test_claim_standby_stream_json_engine_returns_none_when_none_exists(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")
    assert session_runner.claim_standby_stream_json_engine(project_id, model="claude-sonnet-5", effort="auto") is None


def test_close_standby_stream_json_engine_does_not_raise_when_none_exists(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")
    session_runner.close_standby_stream_json_engine(project_id)  # no-op, must not raise


def test_count_resident_engines_includes_standby_engines(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    assert session_runner.count_resident_engines() == 0
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )
    assert session_runner.count_resident_engines() == 1


# ---------------------------------------------------------------------------
# Argv-level end-to-end coverage (issue #138): every test above (and every
# other test in this file) mocks `session_runner.StreamJsonEngine` wholesale
# with a fake that never runs `_build_args()` at all -- it only ever proves
# the right Python kwarg (`model=`/`effort=`) reached the constructor. These
# tests instead leave the REAL `StreamJsonEngine` class in place and fake out
# only the OS-level subprocess spawn, so the actual argv handed to "the
# subprocess" is captured and can be asserted on -- all the way through
# `/api/session/start`'s real endpoint logic (app.py) and `start_session_job`
# (session_runner.py). This is what catches a discrepancy the Python-kwarg-
# level mock structurally cannot.
# ---------------------------------------------------------------------------


class _ArgvCapturingStreamJsonBackend:
    """Stream-json analogue of `_ArgvCapturingBackend` above: a minimal
    real-shaped `StreamJsonBackend` (see `stream_json_engine.py`) whose
    `read_line()` hands back one canned `result` NDJSON line on its first
    call (ending `stream_turn` in one round trip), then raises `EOFError`."""

    def __init__(self, result_text='Question 1: "Only question?"', session_id="s1"):
        self._line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": result_text,
                "session_id": session_id,
            }
        )
        self._served = False

    def write_line(self, line):
        pass

    def read_line(self):
        if not self._served:
            self._served = True
            return self._line
        raise EOFError

    def is_alive(self):
        return not self._served

    def terminate(self, force=False):
        pass


def _capture_real_stream_json_spawns(monkeypatch):
    """Stream-json analogue of `_capture_real_pty_spawns` above: replace
    `stream_json_engine._spawn_subprocess` (the seam every real, non-test
    `StreamJsonEngine()` construction resolves its `process_factory`
    through, when none is injected) with one that records every spawn's
    argv and hands back an `_ArgvCapturingStreamJsonBackend` instead of a
    real OS process. Returns the list of captured argvs, appended to in
    spawn order."""
    spawns = []

    def fake_factory(argv, *, cwd, env):
        spawns.append(argv)
        return _ArgvCapturingStreamJsonBackend()

    monkeypatch.setattr(stream_json_engine, "_spawn_subprocess", fake_factory)
    return spawns


def test_brand_new_session_after_changing_model_spawns_with_the_new_model_via_standby_claim(
    client, tmp_path, monkeypatch
):
    """Issue #138's reported repro (adapted for issue #184: a brand-new
    session's engine is now `StreamJsonEngine`, not `PtyEngine`): pick a
    model in the UI, then start what looks like a brand-new session -- it
    must actually spawn `claude` with THAT model, not a stale one left over
    from a pre-warmed standby engine warmed under the model that was
    configured before the switch. Goes through the real `/api/session/start`
    endpoint logic (`app_module.start_session`) and `start_session_job`,
    with only the OS-level subprocess spawn faked -- so this exercises the
    real `claim_standby_stream_json_engine`/`register_stream_json_engine`
    reuse path and the real `StreamJsonEngine._build_args()`, not a
    Python-kwarg-level mock."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    conn = db.get_connection()
    spawns = _capture_real_stream_json_spawns(monkeypatch)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    # A standby was pre-warmed (e.g. by `open_project`) under the model that
    # was configured at the time.
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(
            project_id, cwd=cwd, model="claude-sonnet-4-6", effort="auto"
        )
    )
    assert len(spawns) == 1
    idx = spawns[0].index("--model")
    assert spawns[0][idx + 1] == "claude-sonnet-4-6"

    # The user now picks a different model in the UI.
    client.post("/api/settings/model", json={"model": "claude-opus-4-8"})

    # A brand-new session, started right after.
    result = asyncio.run(
        _run_and_drain(app_module.start_session({"prompt": "a feature", "effort": "auto"}))
    )
    card_id = result["card_id"]

    # The stale standby must be discarded (model mismatch), not adopted --
    # a genuinely fresh engine is spawned for this card instead.
    assert len(spawns) == 2
    new_argv = spawns[-1]
    assert "--model" in new_argv
    idx = new_argv.index("--model")
    assert new_argv[idx + 1] == "claude-opus-4-8"
    assert session_runner._stream_json_engines[card_id].model == "claude-opus-4-8"


def test_brand_new_session_after_changing_model_spawns_with_the_new_model_via_standby_match(
    client, tmp_path, monkeypatch
):
    """The mirror-image case: the standby's model still matches what's
    currently configured (no change happened, or the user picked the SAME
    model again) -- it must be adopted (`register_stream_json_engine`)
    rather than discarded, and the argv it was ALREADY spawned with
    (captured back when the standby was warmed) must carry that same model.
    Confirms `claim_standby_stream_json_engine`'s reuse path itself is
    argv-correct, not just its discard path."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    spawns = _capture_real_stream_json_spawns(monkeypatch)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-opus-4-8", effort="auto")
    )
    assert len(spawns) == 1
    standby_argv = spawns[0]
    idx = standby_argv.index("--model")
    assert standby_argv[idx + 1] == "claude-opus-4-8"

    client.post("/api/settings/model", json={"model": "claude-opus-4-8"})  # same model, re-selected

    result = asyncio.run(
        _run_and_drain(app_module.start_session({"prompt": "a feature", "effort": "auto"}))
    )
    card_id = result["card_id"]

    # No second real spawn -- the standby (already carrying the right
    # --model) was adopted directly.
    assert len(spawns) == 1
    assert session_runner._stream_json_engines[card_id].model == "claude-opus-4-8"


def test_brand_new_session_after_changing_model_spawns_with_the_new_model_via_pooled_resume(
    client, tmp_path, monkeypatch
):
    """No standby is involved here at all -- a previously pooled session
    (from an earlier project's finished do->implement->qa chain) is reused
    via `db.claim_available_session`, carrying its own old
    `claude_session_id` to `--resume`. The model actually spawned with must
    still be the one just selected in the UI, fresh off `db.get_model`, not
    whatever model that old pooled conversation last ran under."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    conn = db.get_connection()
    spawns = _capture_real_stream_json_spawns(monkeypatch)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    pooled_row_id = db.create_session(
        conn, project_id, claude_session_id="pooled-conversation-1", model="claude-sonnet-4-6", effort="auto"
    )
    db.mark_session_available(conn, pooled_row_id, "pooled-conversation-1")

    client.post("/api/settings/model", json={"model": "claude-opus-4-8"})

    result = asyncio.run(
        _run_and_drain(app_module.start_session({"prompt": "a feature", "effort": "auto"}))
    )
    card_id = result["card_id"]

    assert len(spawns) == 1
    new_argv = spawns[-1]
    assert "--resume" in new_argv
    idx = new_argv.index("--resume")
    assert new_argv[idx + 1] == "pooled-conversation-1"
    assert "--model" in new_argv
    idx = new_argv.index("--model")
    assert new_argv[idx + 1] == "claude-opus-4-8"
    assert session_runner._stream_json_engines[card_id].model == "claude-opus-4-8"


def test_standby_is_still_claimed_when_the_request_omits_effort_entirely(client, tmp_path, monkeypatch):
    """Targeted check for a `effort=None` (omitted from the request body) vs.
    `db.DEFAULT_EFFORT` ("auto", what every standby is always pre-warmed
    with -- see `open_project`/`_clear_for_reuse`/`continue_qa_job`)
    semantically-equal-but-not-identical mismatch: `/api/session/start`
    resolves a missing `effort` via `effort or db.DEFAULT_EFFORT` before
    comparing against the standby's stored effort, so `None` must still
    match a standby stored as `"auto"` -- not be treated as a mismatch and
    wastefully discard a perfectly matching standby (or, worse, the other
    way around: adopt one it shouldn't)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    spawns = _capture_real_stream_json_spawns(monkeypatch)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(
            project_id, cwd=cwd, model="claude-sonnet-4-6", effort="auto"
        )
    )
    assert len(spawns) == 1

    # Body omits "effort" entirely, exactly like a request built without the
    # dropdown's value ever being set.
    result = asyncio.run(_run_and_drain(app_module.start_session({"prompt": "a feature"})))
    card_id = result["card_id"]

    # The standby matched (no mismatch-triggered discard-and-respawn).
    assert len(spawns) == 1
    assert session_runner._stream_json_engines[card_id] is not None
    assert session_runner._stream_json_engines[card_id].effort == "auto"


# ---------------------------------------------------------------------------
# Crash routing (issue #87, migrated to StreamJsonEngineUnrecoverableError by
# issue #225): the exception -> the same blocked-card flow a genuine
# implement_blocked marker already uses.
# ---------------------------------------------------------------------------


def test_implement_turn_crash_routes_into_the_blocked_flow(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def crashing(prompt, **kw):
        raise StreamJsonEngineUnrecoverableError("claude process died twice in a row", session_id="crashed-1")

    _mock_engine(monkeypatch, crashing)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    assert row["claude_session_id"] == "crashed-1"
    blocked = json.loads(row["blocked_json"])
    assert blocked["phase"] == "implement_blocked"
    assert "died twice in a row" in blocked["context"]

    events = live_stream._buffers.get(row_id, [])
    assert any(e.get("type") == "turn" and e.get("phase") == "blocked" for e in events)
    # Suspended, not finished -- no done, no error/needs_github_login path.
    assert {"type": "done"} not in events
    # The dead tab was dropped so a reply reattaches a fresh one via --resume.
    assert row_id not in session_runner._stream_json_engines


def test_implement_crash_recovers_via_the_same_reply_endpoint_as_a_real_block(client, tmp_path, monkeypatch):
    """After a crash-induced block, continue_implement_job (the same
    endpoint that resumes a genuine implement_blocked session) must resume
    this session too, reattaching via --resume at the crash's session id."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def crashing(prompt, **kw):
        raise StreamJsonEngineUnrecoverableError("died twice", session_id="crashed-1")

    _mock_engine(monkeypatch, crashing)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))
    assert db.get_session(conn, row_id)["phase"] == "blocked"

    fake_class = _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Recovered and implemented.", session_id="recovered-1")]),
        fresh_ids=["pooled-1"],
    )

    asyncio.run(session_runner.continue_implement_job(row_id, "still there?", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert row["blocked_json"] is None
    # The reply's engine reattached via --resume at the crash's session id.
    assert fake_class.instances[0].resume_session_id == "crashed-1"


def test_grilling_turn_crash_routes_into_the_blocked_flow_too(client, tmp_path, monkeypatch):
    """Issue #225: once every phase (including grilling) drives its turns
    through `StreamJsonEngine`, a `StreamJsonEngineUnrecoverableError` (this
    engine's own internal crash-retry-once already gave up) routes into the
    same generic blocked-card flow every other phase uses
    (`_route_crash_to_blocked`, `phase: blocked`) -- removing the asymmetry
    that used to exist only because grilling folded this into a plain error
    (there was no blocked-recovery UI for it before this migration)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def crashing(prompt, **kw):
        raise StreamJsonEngineUnrecoverableError("died twice", session_id="crashed-grilling")

    _mock_engine(monkeypatch, crashing)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    assert row["claude_session_id"] == "crashed-grilling"
    blocked = json.loads(row["blocked_json"])
    assert blocked["phase"] == "implement_blocked"
    assert "died twice" in blocked["context"]
    assert row_id not in session_runner._stream_json_engines

    events = live_stream._buffers.get(row_id, [])
    assert any(e.get("type") == "turn" and e.get("phase") == "blocked" for e in events)
    assert {"type": "done"} not in events


def test_implement_error_in_background_raises_a_notification(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def failing(prompt, **kw):
        raise ClaudeCLIError("boom")

    _mock_engine(monkeypatch, failing)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    notifications = session_runner.get_error_notifications(project_id)
    assert len(notifications) == 1
    assert notifications[0]["card_id"] == row_id
    assert notifications[0]["phase"] == "implementing"
    assert "boom" in notifications[0]["message"]


def test_qa_closing_error_in_background_raises_a_notification(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def failing(prompt, **kw):
        raise ClaudeCLIError("qa boom")

    _mock_engine(monkeypatch, failing)

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id, session_type="qa", phase="qa_grilling",
        claude_session_id="qa-session-1", details={"prd": {"number": 7, "title": "Test PRD"}},
    )
    asyncio.run(session_runner.continue_qa_job(qa_row_id, {}, "notes", cwd=cwd))

    notifications = session_runner.get_error_notifications(project_id)
    assert len(notifications) == 1
    assert notifications[0]["card_id"] == qa_row_id
    assert notifications[0]["phase"] == "qa_closing"
    assert "qa boom" in notifications[0]["message"]


def test_qa_closing_gh_auth_error_sets_needs_github_login(client, tmp_path, monkeypatch):
    """/qa's closing step calls `gh issue close`/`gh issue edit` directly, so
    a genuine gh auth failure must still surface the GitHub login button."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def failing(prompt, **kw):
        raise ClaudeCLIError("gh: not logged in, run `gh auth login`")

    _mock_engine(monkeypatch, failing)

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id, session_type="qa", phase="qa_grilling",
        claude_session_id="qa-session-1", details={"prd": {"number": 7, "title": "Test PRD"}},
    )
    asyncio.run(session_runner.continue_qa_job(qa_row_id, {}, "notes", cwd=cwd))

    row = db.get_session(conn, qa_row_id)
    assert bool(row["needs_github_login"]) is True
    assert "not logged in" in row["error_text"]


def test_qa_closing_non_gh_error_does_not_set_needs_github_login(client, tmp_path, monkeypatch):
    """A ClaudeCLIError during qa closing that isn't actually a `gh` auth
    failure must not trigger the GitHub login button."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def failing(prompt, **kw):
        raise ClaudeCLIError("Claude CLI exited with code 1: authentication error")

    _mock_engine(monkeypatch, failing)

    conn = db.get_connection()
    qa_row_id = db.create_session(
        conn, project_id, session_type="qa", phase="qa_grilling",
        claude_session_id="qa-session-1", details={"prd": {"number": 7, "title": "Test PRD"}},
    )
    asyncio.run(session_runner.continue_qa_job(qa_row_id, {}, "notes", cwd=cwd))

    row = db.get_session(conn, qa_row_id)
    assert bool(row["needs_github_login"]) is False
    assert "authentication error" in row["error_text"]


def test_context_window_pct_computes_documented_formula():
    event = _usage_event(input_tokens=10, cache_creation=200, cache_read=790, context_window=1000)
    assert session_runner._context_window_pct(event) == pytest.approx(1.0)

    event = _usage_event(input_tokens=0, cache_creation=0, cache_read=400, context_window=1000)
    assert session_runner._context_window_pct(event) == pytest.approx(0.4)


def test_context_window_pct_returns_none_without_context_window():
    assert session_runner._context_window_pct({"type": "result"}) is None
    assert session_runner._context_window_pct({"usage": {"input_tokens": 5}}) is None
    assert session_runner._context_window_pct({"usage": {}, "modelUsage": {"m": {}}}) is None


def test_start_session_job_persists_context_pct_from_the_turn(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter(
            [_usage_event(input_tokens=0, cache_creation=0, cache_read=500, context_window=1000)]
        ),
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["context_pct"] == pytest.approx(0.5)


def test_maybe_clear_for_next_phase_continues_same_session_at_or_under_cutoff(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id, claude_session_id="s1")
    db.update_session(conn, row_id, context_pct=0.40)
    row = db.get_session(conn, row_id)

    result = asyncio.run(
        session_runner._maybe_clear_for_next_phase(row_id, conn, row, cwd=cwd, cutoff=0.40)
    )
    assert result == "s1"
    assert db.get_session(conn, row_id)["claude_session_id"] == "s1"
    # No fresh engine was constructed -- under the cutoff, nothing clears.
    assert fake_class.instances == []


def test_maybe_clear_for_next_phase_treats_unknown_context_pct_as_safe(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id, claude_session_id="s1")
    row = db.get_session(conn, row_id)
    assert row["context_pct"] is None

    result = asyncio.run(
        session_runner._maybe_clear_for_next_phase(row_id, conn, row, cwd=cwd, cutoff=0.40)
    )
    assert result == "s1"
    assert fake_class.instances == []


def test_maybe_clear_for_next_phase_clears_when_over_cutoff(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]), fresh_ids=["cleared-1"])

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id, claude_session_id="s1")
    db.update_session(conn, row_id, context_pct=0.75)
    row = db.get_session(conn, row_id)

    result = asyncio.run(
        session_runner._maybe_clear_for_next_phase(row_id, conn, row, cwd=cwd, cutoff=0.68)
    )
    assert result == "cleared-1"

    updated = db.get_session(conn, row_id)
    assert updated["claude_session_id"] == "cleared-1"
    assert updated["context_pct"] is None

    # A genuinely fresh engine was constructed (no resume_session_id), and
    # it's now this card's resident tab going forward.
    assert len(fake_class.instances) == 1
    assert fake_class.instances[0].resume_session_id is None
    assert session_runner._stream_json_engines[row_id] is fake_class.instances[0]


def test_finish_chain_pauses_at_details_instead_of_starting_implementing(client, tmp_path, monkeypatch):
    """Issue #196 (child of PRD #195): a /do session that reaches `details`
    must stop there and stay there -- no automatic transition to `implementing`,
    no `done` event, no `minimize` event. The resident engine stays alive and
    idle. Manually starting implementation from "To be implemented" still works
    (that path is unchanged -- see start_or_queue_implement)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: p")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: i")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #1: p\nIssue #2: i")])
        raise AssertionError(f"unexpected prompt {prompt!r} -- must not auto-implement")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about the details pause
    # point, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["session_type"] == "do"
    assert row["available_for_reuse"] == 0

    events = live_stream._buffers.get(row_id, [])
    event_types = [e["type"] for e in events]
    assert "minimize" not in event_types
    assert "done" not in event_types
    assert any(e.get("type") == "turn" and e.get("phase") == "details" for e in events)


def test_finish_chain_pauses_at_details_even_when_no_prd_was_parsed(client, tmp_path, monkeypatch):
    """Issue #196 (child of PRD #195): when publish succeeds but parse_details
    finds no PRD number, the session still pauses at `details` and stays there
    -- it is NOT pooled for reuse, NOT given a done event, and the resident
    engine is NOT torn down. Both cases (PRD found vs. not found) now share
    one code path: stop at details and wait."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: p")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: i")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("no PRD or issue numbers in here at all")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about the details pause
    # point, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["session_type"] == "do"
    assert row["available_for_reuse"] == 0

    events = live_stream._buffers.get(row_id, [])
    assert not any(e.get("type") == "minimize" for e in events)
    assert not any(e.get("type") == "done" for e in events)
    assert any(e.get("type") == "turn" and e.get("phase") == "details" for e in events)


def test_dismiss_error_notifications_clears_the_project_queue(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    session_runner.add_error_notification(project_id, 1, "implementing", "boom")
    assert len(session_runner.get_error_notifications(project_id)) == 1

    session_runner.dismiss_error_notifications(project_id)
    assert session_runner.get_error_notifications(project_id) == []


# ---------------------------------------------------------------------------
# start_do_continue_job (issue #198, child of PRD #195): the Continue button
# on the do-finished banner starts a fresh grilling round on the same card.
# ---------------------------------------------------------------------------


def test_start_do_continue_job_reuses_existing_engine_under_cutoff(client, tmp_path, monkeypatch):
    """When context_pct is at or under the cutoff, the existing StreamJsonEngine
    is reused -- no new engine is spawned and the claude_session_id is
    unchanged. The new grilling round uses the same model/effort already on
    the row."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    do_prompts = []

    def handler(prompt, **kw):
        do_prompts.append(prompt)
        if "/rhubarb:grilling" in prompt:
            return iter([_result_event('Question 1: "a question?"', session_id="sess-1")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler, fresh_ids=["sess-1"])
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    # Land the session at details with a low context_pct.
    db.update_session(conn, row_id, phase="details", session_type="do", claude_session_id="sess-1", context_pct=0.20)

    asyncio.run(session_runner.start_do_continue_job(row_id, "next idea", cwd=cwd))

    # Same session id -- engine was reused, not replaced.
    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    assert row["claude_session_id"] == "sess-1"

    # The grilling prompt was sent.
    assert any("/rhubarb:grilling next idea" in p for p in do_prompts)


def test_start_do_continue_job_spawns_fresh_engine_over_cutoff(client, tmp_path, monkeypatch):
    """When context_pct exceeds the cutoff, the old StreamJsonEngine is torn
    down and a fresh one is spawned. The new claude_session_id is persisted
    and context_pct is reset to None before the grilling turn runs."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if "/rhubarb:grilling" in prompt:
            return iter([_result_event('Question 1: "a question?"', session_id="fresh-id")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler, fresh_ids=["old-id", "fresh-id"])
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    # Land at details with context_pct well over the cutoff.
    db.update_session(conn, row_id, phase="details", session_type="do", claude_session_id="old-id", context_pct=0.85)

    asyncio.run(session_runner.start_do_continue_job(row_id, "another idea", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    # Old session is gone; the new engine's id was persisted.
    assert row["claude_session_id"] != "old-id"


def test_start_do_continue_job_keeps_model_and_effort_from_row(client, tmp_path, monkeypatch):
    """Continue uses the model/effort already on the row, not the global setting."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_model(conn, "claude-opus-5")

    seen_models = []
    seen_efforts = []

    def handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        seen_models.append(model)
        seen_efforts.append(effort)
        # Issue #223: a real regex-parseable question, so the turn's frontier
        # stays non-empty and this test doesn't trigger an auto-advance into
        # the chain (which would record more than one model/effort call).
        return iter([_result_event('Question 1: "a question?"')])

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    row_id = db.create_session(conn, project_id)
    db.update_session(conn, row_id, phase="details", session_type="do", model="claude-haiku-4-5-20251001", effort="low", context_pct=0.10)

    asyncio.run(session_runner.start_do_continue_job(row_id, "prompt", cwd=cwd))

    # Must have used the row's model/effort, not the global setting.
    assert seen_models == ["claude-haiku-4-5-20251001"]
    assert seen_efforts == ["low"]


# ---------------------------------------------------------------------------
# Per-card_id turn lock (issue #144): a race condition where two overlapping
# calls for the same card_id could both write to and read from the same
# resident engine's stream_turn at once. `_run_stream_json_turn` now checks a
# per-card_id `asyncio.Lock` before doing anything else; a call made while
# the lock is already held returns `None` immediately, touching neither the
# engine nor anything else.
# ---------------------------------------------------------------------------


def test_run_turn_lock_rejects_a_concurrent_call_for_the_same_card_id(client, tmp_path, monkeypatch):
    """The core invariant: of two overlapping `_run_stream_json_turn` calls
    for the same card_id, only the first ever reaches the engine's
    `stream_turn` -- the second, made while the first is still mid-turn
    (blocked inside `stream_turn` via a controlled `asyncio.Event`), returns
    `None` immediately without constructing or touching the engine at all.
    Issue #149: that second, rejected call must also publish an explicit
    error `turn` event on this card's stream, tagged with the caller's own
    `phase`, instead of leaving no trace at all."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    entered = asyncio.Event()
    release = asyncio.Event()
    enter_count = {"n": 0}
    fake_class = _make_blocking_fake_engine_class(
        entered, release, enter_count, result_text="❓ **Q1** - **Scope**: Only question?"
    )
    monkeypatch.setattr(session_runner, "StreamJsonEngine", fake_class)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    async def scenario():
        first_task = asyncio.create_task(
            session_runner._run_stream_json_turn(
                row_id, "first prompt", session_id=None, cwd=cwd, model=None, effort=None, phase="grilling"
            )
        )
        await entered.wait()  # first call is now mid-turn, blocked inside stream_turn

        # A second, overlapping call for the SAME card_id while the first
        # is still in flight -- must be rejected outright.
        second_result = await session_runner._run_stream_json_turn(
            row_id, "second prompt", session_id=None, cwd=cwd, model=None, effort=None, phase="grilling"
        )
        assert second_result is None

        release.set()
        return await first_task

    first_result = asyncio.run(scenario())

    # Only the first call's turn ever actually reached the engine.
    assert enter_count["n"] == 1
    assert len(fake_class.instances) == 1
    assert first_result["session_id"] == fake_class.instances[0].claude_session_id

    # The rejected duplicate published an explicit error `turn` event
    # (issue #149) -- not silence -- tagged with the phase it was called
    # with, matching the shape every other turn failure publishes via
    # `_turn_event`.
    events = live_stream._buffers.get(row_id, [])
    error_turn_events = [e for e in events if e["type"] == "turn" and e.get("error")]
    assert len(error_turn_events) == 1
    assert error_turn_events[0]["phase"] == "grilling"
    assert error_turn_events[0]["needs_github_login"] is False


def test_concurrent_start_implement_job_calls_only_one_reaches_the_engine(client, tmp_path, monkeypatch):
    """End-to-end through a real turn-initiating function: two overlapping
    `start_implement_job` calls for the same card_id (e.g. a double-clicked
    "implement" action) must not both drive the same resident PtyEngine.
    The second, made while the first is still mid-turn, never reaches the
    engine and never publishes the turn's own completion events (`done`
    included) -- but per issue #149, it must no longer be silent either: it
    publishes an explicit error `turn` event on the busy lock, so exactly
    one *successful* `turn` event, one *error* `turn` event, and one `done`
    event reach the session's live buffer, and the row still lands in its
    normal single-turn end state."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    entered = asyncio.Event()
    release = asyncio.Event()
    enter_count = {"n": 0}
    fake_class = _make_blocking_fake_engine_class(entered, release, enter_count, result_text="Implemented PRD #5")
    monkeypatch.setattr(session_runner, "StreamJsonEngine", fake_class)

    conn = db.get_connection()
    row_id = db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )

    async def scenario():
        first_task = asyncio.create_task(session_runner.start_implement_job(row_id, 5, cwd=cwd))
        await entered.wait()  # first call's turn is now mid-flight

        # A second, overlapping call for the same card_id -- rejected on the
        # busy lock, but no longer silently (issue #149).
        await session_runner.start_implement_job(row_id, 5, cwd=cwd)

        release.set()
        await first_task

    asyncio.run(scenario())

    # Only one turn ever actually reached the engine's stream_turn.
    assert enter_count["n"] == 1

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert row["available_for_reuse"] == 1

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    done_events = [e for e in events if e["type"] == "done"]
    # The real turn's own completion events, plus the rejected duplicate's
    # explicit error `turn` event (issue #149) -- no `done` from the
    # duplicate, since it returns before ever reaching that point.
    assert len(turn_events) == 2
    assert len(done_events) == 1

    error_turn_events = [e for e in turn_events if e.get("error")]
    assert len(error_turn_events) == 1
    assert error_turn_events[0]["phase"] == "implementing"

    # The engine the legitimate call used was closed exactly once (normal
    # end-of-turn pooling), not disturbed or double-closed by the rejected
    # duplicate.
    turn_engine = fake_class.instances[0]
    assert turn_engine.closed is True
    assert row_id not in session_runner._stream_json_engines


def test_turn_lock_is_released_after_the_turn_completes(client, tmp_path, monkeypatch):
    """The lock must never be left stuck held after a turn finishes -- a
    call for the same card_id made strictly after the first has completed
    (not concurrently) must proceed normally, not be rejected."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(f"result for {prompt}", session_id="s1")]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    async def scenario():
        first = await session_runner._run_stream_json_turn(
            row_id, "first", session_id=None, cwd=cwd, model=None, effort=None, phase="grilling"
        )
        second = await session_runner._run_stream_json_turn(
            row_id, "second", session_id="s1", cwd=cwd, model=None, effort=None, phase="grilling"
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert first is not None
    assert second is not None
    assert not session_runner._get_turn_lock(row_id).locked()


def test_close_engine_removes_the_turn_lock(client, tmp_path, monkeypatch):
    """`_close_stream_json_engine` must pop the card's entry out of
    `_turn_locks` too, alongside `_stream_json_engines` -- otherwise the
    lock registry grows unboundedly over a long-running instance."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event('Question 1: "Only question?"')]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert row_id in session_runner._turn_locks

    session_runner._close_stream_json_engine(row_id)

    assert row_id not in session_runner._turn_locks


# ---------------------------------------------------------------------------
# Needs-input classification wrapper (issue #175, child of PRD #174 --
# replaces the per-chunk PTY stall mechanism with a single per-turn Ollama
# classification, not yet wired into any phase's own turn-handling flow).
# ---------------------------------------------------------------------------


def test_classify_needs_input_is_a_no_op_when_ollama_declined(client):
    """Skipped entirely -- no HTTP call attempted, nothing published -- when
    the user has declined Ollama assistance, same precedent as the existing
    grilling/QA rescue call sites."""
    conn = db.get_connection()
    db.set_ollama_declined(conn, True)

    def fake_post(url, body, *, timeout):
        raise AssertionError("Ollama must not be called when ollama_declined is true")

    result = asyncio.run(_REAL_CLASSIFY_NEEDS_INPUT(1, conn, "some turn text", "grilling", http_post=fake_post))

    assert result is None
    assert live_stream._buffers.get(1, []) == []


def test_classify_needs_input_publishes_unavailable_notification_on_failure_when_not_declined(client):
    """A call failure/timeout while Ollama assistance is enabled is a
    genuine, unexpected failure -- surfaced as a distinct `ollama_unavailable`
    event, NOT folded into the existing turn `error` field."""
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)

    def timing_out_post(url, body, *, timeout):
        raise TimeoutError("Ollama took too long")

    result = asyncio.run(_REAL_CLASSIFY_NEEDS_INPUT(1, conn, "some turn text", "grilling", http_post=timing_out_post))

    assert result is None
    events = live_stream._buffers.get(1, [])
    assert events == [{"type": "ollama_unavailable"}]


def test_classify_needs_input_returns_result_and_publishes_nothing_on_success(client):
    """A successful classification is returned as-is and does not publish
    any notification -- the unavailable notification is only for a call
    that was attempted and failed."""
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)

    payload = {"needs_input": True, "reason": "Asked which database to use."}

    def fake_post(url, body, *, timeout):
        import json as _json

        return {"response": _json.dumps(payload)}

    result = asyncio.run(_REAL_CLASSIFY_NEEDS_INPUT(1, conn, "Which database?", "grilling", http_post=fake_post))

    assert result == payload
    assert live_stream._buffers.get(1, []) == []


def test_classify_needs_input_publishes_unavailable_notification_when_response_is_malformed(client):
    """A malformed/invalid Ollama response is a classification failure just
    like a timeout or connection error -- never partially trusted -- and
    the call WAS attempted (not declined), so this still fires the
    unavailable notification exactly like a timeout would."""
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)

    def fake_post(url, body, *, timeout):
        return {"response": "not valid json at all {{{"}

    result = asyncio.run(_REAL_CLASSIFY_NEEDS_INPUT(1, conn, "text", "grilling", http_post=fake_post))

    assert result is None
    assert live_stream._buffers.get(1, []) == [{"type": "ollama_unavailable"}]


# ---------------------------------------------------------------------------
# Wiring classify_needs_input into the /to-prd, /to-issues chain (issue #177,
# child of PRD #174 -- these phases have no structured question format of
# their own, so a positive classification must show the existing generic
# stall-reply panel, never a rich question/options UI).
# ---------------------------------------------------------------------------


def test_creating_prd_turn_is_classified_before_creating_issues_starts(client, tmp_path, monkeypatch):
    """Issue #177: `_run_chain_step` must call `classify_needs_input` with
    this turn's own rendered text right after `creating_prd`'s turn
    resolves, and that call must happen BEFORE `/rhubarb:to-issues` is ever
    sent -- i.e. before the phase's normal automatic continuation. A `None`
    (declined/unavailable) or negative result must leave the chain running
    exactly as it always has -- no behavior change."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    seen_prompts = []

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        seen_prompts.append(prompt)
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        if prompt == "/rhubarb:implement prd: 5":
            return iter([_result_event("Implemented.")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about the creating_prd/
    # creating_issues classify-before-continuing ordering, not the
    # completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    # Opt back into (fake) Ollama assistance for this test -- the module's
    # own autouse fixture defaults every other test here to declined so
    # this wiring doesn't attempt a real HTTP call unexpectedly.
    db.set_ollama_declined(conn, False)

    calls = []
    seen_prompts_at_creating_prd_call = None

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        calls.append((card_id, phase, text))
        if phase == "creating_prd":
            nonlocal seen_prompts_at_creating_prd_call
            seen_prompts_at_creating_prd_call = list(seen_prompts)
        return None

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    # Issue #223: the first grilling turn's empty frontier auto-advances
    # straight through the chain within this single call -- classify_needs_input
    # is already mocked above so this exercises the exact same ordering the
    # old explicit confirm_advance call used to.
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert calls[0] == (row_id, "creating_prd", "PRD Draft: My PRD")
    # /rhubarb:to-issues had NOT been sent yet at the moment creating_prd's
    # turn was classified -- classification runs before the automatic
    # continuation, not after.
    assert seen_prompts_at_creating_prd_call == ["/rhubarb:to-prd"]
    assert calls[1] == (row_id, "creating_issues", "Issue Draft S1: Child one")
    # Issue #225 (child of PRD #222): the new publishing step is classified
    # too, via the exact same `_run_chain_step` hook.
    assert calls[2] == (row_id, "publishing", "PRD #5: My PRD\nIssue #6: Child one")

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"


def test_creating_prd_positive_classification_shows_generic_stall_panel_not_rich_ui(
    client, tmp_path, monkeypatch
):
    """Issue #177: `creating_prd` has no structured question format of its
    own, so a positive `classify_needs_input` result must resurrect the
    generic PRD #168/#172 stall-reply panel (a `turn` event carrying
    `stalled=True` and `stalled_context` set to the turn's own rendered
    text) -- never the rich `interview`/`blocked` question UI grilling and
    implementing use for their own positive results (sibling issues
    #178/#179).

    Issue #225 (child of PRD #222): the pause is a real return, not an
    inline await -- `start_session_job` returns with the row parked at
    `phase="creating_prd"`/`stalled_json` set, and no `done` published. The
    human's reply resumes via `continue_stalled_chain_step_job`, sent as a
    genuinely new turn (reattached via the row's existing
    `claude_session_id`), which then cascades through the rest of the chain
    exactly like a fresh run would."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft, but scope is unclear.")])
        if prompt == "Understood, using Postgres for scope.":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about creating_prd's own
    # generic stall panel, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    db.set_ollama_declined(conn, False)

    classify_calls = {"n": 0}

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        # Only the FIRST creating_prd turn ("Wrote PRD draft, but scope is
        # unclear.") needs input -- the resumed turn's own draft resolves
        # cleanly, letting the chain cascade forward instead of re-stalling.
        if phase == "creating_prd" and classify_calls["n"] == 0:
            classify_calls["n"] += 1
            return {"needs_input": True, "reason": "Scope is ambiguous."}
        return None

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    # Issue #223: the first grilling turn's empty frontier auto-advances into
    # the chain within this single call.
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    events = live_stream._buffers.get(row_id, [])
    stalled_events = [e for e in events if e.get("type") == "turn" and e.get("stalled")]
    assert len(stalled_events) == 1
    stalled = stalled_events[0]
    assert stalled["phase"] == "creating_prd"
    assert stalled["stalled_context"] == "Wrote PRD draft, but scope is unclear."
    # Never the rich question/options UI -- that's sibling issues #178/#179's
    # own phases, not this generic panel.
    assert stalled["interview"] is None
    assert stalled["blocked"] is None

    row = db.get_session(conn, row_id)
    # Genuinely paused -- not resolved inline -- and no `done` published.
    assert row["phase"] == "creating_prd"
    assert row["stalled_json"] is not None
    assert not any(e.get("type") == "done" for e in events)

    asyncio.run(session_runner.continue_stalled_chain_step_job(row_id, "Understood, using Postgres for scope.", cwd=cwd))

    row = db.get_session(conn, row_id)
    # The reply resolved successfully, so the chain's normal continuation
    # cascaded forward exactly like any other completed turn.
    assert row["phase"] == "details"
    assert row["stalled_json"] is None


def test_creating_prd_negative_classification_does_not_pause_the_chain(client, tmp_path, monkeypatch):
    """Issue #177: `needs_input: False` must leave `creating_prd` continuing
    exactly as it does today -- no stalled event, straight into
    `creating_issues`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        if prompt == "/rhubarb:implement prd: 5":
            return iter([_result_event("Implemented.")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about creating_prd's negative
    # classification path, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    db.set_ollama_declined(conn, False)

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        return {"needs_input": False, "reason": None}

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    # Issue #223: the first grilling turn's empty frontier auto-advances into
    # the chain within this single call.
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    events = live_stream._buffers.get(row_id, [])
    assert not any(e.get("type") == "turn" and e.get("stalled") for e in events)

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"


# ---------------------------------------------------------------------------
# Issue #179 (child of PRD #174): wiring classify_needs_input into
# implementing's own turn handling, right after the pre-existing
# implement_blocked marker check (_parse_implement_blocked_block) concludes
# a turn's result carries no blocked marker at all -- a pure addition, never
# touching that check or its own marker-parsing logic.
# ---------------------------------------------------------------------------


def _fake_classify_needs_input(result):
    """Build a fake stand-in for `session_runner.classify_needs_input` (an
    async function) that always returns `result` regardless of its
    arguments. The real call's own behavior (declined/unavailable/malformed
    handling) is already covered by the dedicated tests above; these tests
    only care about how `_finish_implement_turn` reacts to each possible
    result."""

    async def fake(card_id, conn, text, phase, *, http_post=None):
        return result

    return fake


def test_start_implement_job_renders_rich_question_ui_when_classification_finds_a_question(
    client, tmp_path, monkeypatch
):
    """A positive classification (`needs_input: True`) whose turn text still
    parses into a real grilling-shaped question must render the same rich
    question UI grilling's own rounds use (`renderInterview` on the frontend)
    -- `interview_json` persisted and a `turn` event published for phase
    `implementing` carrying that `interview` -- rather than falling through
    to automatic continuation or the generic reply panel. The session stays
    suspended (no `done`), same "waiting on a human" shape as `blocked`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    question_text = (
        'Question 1: "Python or Node?"\nOptions:\nOption 1: "Python"\nOption 2: "Node"\nRecommended: [1]\n'
    )
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(question_text, session_id="impl-q1")]))
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    monkeypatch.setattr(
        session_runner,
        "classify_needs_input",
        _fake_classify_needs_input({"needs_input": True, "reason": "Asked which language to use."}),
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["claude_session_id"] == "impl-q1"
    assert row["stalled_json"] is None
    interview = json.loads(row["interview_json"])
    assert interview["questions"][0]["text"] == "Python or Node?"
    assert interview["questions"][0]["options"] == ["Python", "Node"]

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e.get("type") == "turn" and e.get("phase") == "implementing"]
    assert turn_events
    assert turn_events[-1]["interview"]["questions"][0]["text"] == "Python or Node?"
    # Suspended, not finished -- no `done` yet, same shape as implement_blocked.
    assert {"type": "done"} not in events
    assert row_id in session_runner._stream_json_engines


def test_start_implement_job_falls_back_to_generic_panel_when_no_question_can_be_extracted(
    client, tmp_path, monkeypatch
):
    """A positive classification whose turn text does NOT fit the
    structured question/option shape must fall back to the pre-existing
    generic reply panel (`renderStalledSession`/`sendStallReply` in
    prompt.html) instead: `stalled_json` persisted and a `turn` event
    published with `stalled: True`/`stalled_context` set to the
    classifier's own reason, for phase `implementing`. Session stays
    suspended (no `done`)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    # No parser session mocked for this project -- `_extract_implementing_
    # question`'s parser-session extraction pass fails closed (the default
    # no-op fake engine's `stream_turn` raises, caught as "nothing
    # extracted"), matching a real turn with no recognizable structured
    # content at all and no live session to try.
    prose = "I went ahead with the OAuth approach, but I'd like you to confirm before I touch the schema."
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(prose, session_id="impl-stall1")]))
    monkeypatch.setattr(
        session_runner,
        "classify_needs_input",
        _fake_classify_needs_input({"needs_input": True, "reason": "Asked for confirmation before a schema change."}),
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["claude_session_id"] == "impl-stall1"
    assert row["interview_json"] is None
    stalled = json.loads(row["stalled_json"])
    assert stalled == {"phase": "implementing", "context": "Asked for confirmation before a schema change."}

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e.get("type") == "turn" and e.get("phase") == "implementing"]
    assert turn_events
    assert turn_events[-1]["stalled"] is True
    assert turn_events[-1]["stalled_context"] == "Asked for confirmation before a schema change."
    # Suspended, not finished -- no `done` yet, same shape as implement_blocked.
    assert {"type": "done"} not in events
    assert row_id in session_runner._stream_json_engines


def test_start_implement_job_continues_automatically_when_classification_says_no_input_needed(
    client, tmp_path, monkeypatch
):
    """A negative classification (`needs_input: False`) must fall straight
    through to today's unchanged automatic continuation -- the turn
    completes normally (`phase: implemented`, pooled), with no `interview`/
    `stalled` state at all, exactly as if `classify_needs_input` had never
    been added."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Implemented PRD #5", session_id="impl-ok")]),
        fresh_ids=["turn-engine", "pooled-1"],
    )
    monkeypatch.setattr(
        session_runner,
        "classify_needs_input",
        _fake_classify_needs_input({"needs_input": False, "reason": None}),
    )

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert row["available_for_reuse"] == 1
    assert row["interview_json"] is None
    assert row["stalled_json"] is None

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}
    assert any(e.get("type") == "turn" and e.get("phase") == "implemented" for e in events)


def test_start_implement_job_continues_automatically_when_ollama_declines_or_is_unavailable(
    client, tmp_path, monkeypatch
):
    """`classify_needs_input` itself already no-ops (returns `None`) when the
    user has declined Ollama assistance, or when the call was attempted but
    failed -- either way, `_finish_implement_turn` must treat that exactly
    like a negative classification: fall straight through to automatic
    continuation, no behavior change from before this wiring existed."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Implemented PRD #5", session_id="impl-ok2")]),
        fresh_ids=["turn-engine", "pooled-2"],
    )
    monkeypatch.setattr(session_runner, "classify_needs_input", _fake_classify_needs_input(None))

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert row["available_for_reuse"] == 1
    assert row["interview_json"] is None
    assert row["stalled_json"] is None

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}


def test_start_implement_job_blocked_marker_takes_priority_over_classification(client, tmp_path, monkeypatch):
    """When the turn's result DOES carry a genuine `implement_blocked`
    marker, `classify_needs_input` must never even be called -- the
    pre-existing blocked-marker check is untouched and still short-circuits
    everything after it, exactly as before this issue's wiring existed."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch, lambda prompt, **kw: iter([_result_event(_IMPLEMENT_BLOCKED_BLOCK, session_id="impl-blocked2")])
    )

    def must_not_be_called(card_id, conn, text, phase, *, http_post=None):
        raise AssertionError("classify_needs_input must not be called when a blocked marker is present")

    monkeypatch.setattr(session_runner, "classify_needs_input", must_not_be_called)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    blocked = json.loads(row["blocked_json"])
    assert blocked["question"] == "Which auth provider should the login button use?"


# ---------------------------------------------------------------------------
# Issue #184 (step 2 of PRD #182): dispatch, standby pre-warming, the shared
# turn lock, and question-parsing coverage for a grilling card driven by the
# new `StreamJsonEngine` (issue #183) instead of `PtyEngine`.
# ---------------------------------------------------------------------------


def test_grilling_turn_dispatches_to_stream_json_engine(client, tmp_path, monkeypatch):
    """A grilling card's turn must be driven by `StreamJsonEngine`, and the
    turn's translated events (`action`/`text`/`turn`) must reach this card's
    live SSE-visible buffer (`live_stream._buffers`) via the
    `publish()`/`stream_translate.translate_event()` pipeline (acceptance
    criterion: "Turn/text/tool-use events ... appear on the existing ... SSE
    endpoint, translated via stream_translate.translate_event()")."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        return iter(
            [
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "echo hi"}}
                        ]
                    },
                },
                {
                    "type": "stream_event",
                    "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
                },
                {
                    "type": "stream_event",
                    "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "\n\n"}},
                },
                # Mirrors real streaming: the CLI's final `result` text was
                # already streamed as its own delta(s) before the boundary
                # `result` event arrives -- see the `full_text` accumulation
                # fix in `_run_stream_json_turn`.
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": 'Question 1: "What should it do?"'},
                    },
                },
                _result_event('Question 1: "What should it do?"'),
            ]
        )

    fake_class = _make_fake_engine_class(handler)
    monkeypatch.setattr(session_runner, "StreamJsonEngine", fake_class)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    # StreamJsonEngine (the fake) was actually used to drive this turn.
    assert len(fake_class.instances) == 1

    events = live_stream._buffers.get(row_id, [])
    assert {"type": "action", "summary": "$ echo hi"} in events
    assert {"type": "text", "text": "Hello"} in events
    turn_events = [e for e in events if e["type"] == "turn"]
    assert turn_events
    assert turn_events[-1]["interview"]["questions"][0]["text"] == "What should it do?"


def test_grilling_turn_recovers_questions_dropped_from_the_terse_result_text(client, tmp_path, monkeypatch):
    """Live-testing regression (PRD #222): the CLI's own `result` event text
    is only the LAST assistant text block of the turn -- if the model prints
    a full question round, then calls a tool (e.g. writing
    `.claude/rhubarb_question.md`), then closes with a short remark, `result`
    captures only that closing remark and silently drops the actual
    questions the Live Terminal panel already streamed to the user. Before
    the `full_text` accumulation fix, this made a grilling turn with real
    outstanding questions parse as empty and auto-advance straight into
    `creating_prd` (issue #223), discarding a whole round of unanswered
    questions with no way to recover."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        return iter(
            [
                # The real round of questions, streamed as text deltas --
                # exactly what the Live Terminal panel shows the user.
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": 'Question 1: "First question?"\n\n'},
                    },
                },
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": 'Question 2: "Second question?"\n\n'},
                    },
                },
                # A tool call in between (writing the question file) --
                # doesn't itself carry any `text`, but its presence is why
                # the CLI's own `result` field ends up being only the
                # trailing remark below, not the questions above.
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "t1",
                                "name": "Write",
                                "input": {"file_path": ".claude/rhubarb_question.md"},
                            }
                        ]
                    },
                },
                # The turn's terse closing remark -- this is ALL the CLI's
                # own `result` event carries, per the bug this test guards
                # against.
                _result_event("Waiting on your answers to Q1-Q2 before the next round."),
            ]
        )

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    # Must still be waiting in grilling, with the real questions recovered
    # from the accumulated stream -- NOT auto-advanced into creating_prd off
    # the terse `result` sentence alone.
    assert row["phase"] == "grilling"
    interview = json.loads(row["interview_json"])
    assert [q["text"] for q in interview["questions"]] == ["First question?", "Second question?"]

    events = live_stream._buffers.get(row_id, [])
    assert not any(e == {"type": "phase", "phase": "creating_prd"} for e in events)


# Issue #230 removed `test_grilling_questions_parsed_directly_via_qa_parser_
# match_existing_parser_output`, which covered issue #184's original
# acceptance criterion that a grilling turn's persisted/published interview
# must be byte-for-byte what `qa_parser.parse_grilling_response` produces
# from the same text -- true back when that regex was `_run_grilling_turn_
# stream_json`'s first-attempt fast path. Issue #230 confirmed that regex
# format is dead for the grilling skill's real output and removed the
# regex-first attempt entirely (see that function's docstring): a grilling
# turn's interview now always comes from the parser-session extraction pass
# instead, covered by `test_stream_json_grilling_turn_uses_parser_session_
# for_emoji_question_format` above. Removed rather than kept red: the
# byte-for-byte-matches-the-regex-parser behavior no longer exists for this
# call site by design, not by regression.


def test_ensure_standby_stream_json_engine_spawns_when_none_exists(client, tmp_path, monkeypatch):
    """Stream-json mirror of `test_ensure_standby_engine_spawns_when_none_exists`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )

    assert len(fake_class.instances) == 1
    engine, model, effort = session_runner._standby_stream_json_engines[project_id]
    assert engine is fake_class.instances[0]
    assert (model, effort) == ("claude-sonnet-5", "auto")


def test_claim_standby_stream_json_engine_returns_it_on_a_match_and_removes_it(client, tmp_path, monkeypatch):
    """Stream-json mirror of `test_claim_standby_engine_returns_it_on_a_match_and_removes_it`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )

    claimed = session_runner.claim_standby_stream_json_engine(project_id, model="claude-sonnet-5", effort="auto")

    assert claimed is fake_class.instances[0]
    assert claimed.closed is False
    assert project_id not in session_runner._standby_stream_json_engines


def test_claim_standby_stream_json_engine_returns_none_and_discards_on_model_mismatch(client, tmp_path, monkeypatch):
    """Stream-json mirror of `test_claim_standby_engine_returns_none_and_discards_on_model_mismatch`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )

    claimed = session_runner.claim_standby_stream_json_engine(project_id, model="claude-opus-5", effort="auto")

    assert claimed is None
    assert fake_class.instances[0].closed is True  # discarded, not left dangling
    assert project_id not in session_runner._standby_stream_json_engines


def test_close_standby_stream_json_engine_closes_and_discards_it(client, tmp_path, monkeypatch):
    """Stream-json mirror of `test_close_standby_engine_closes_and_discards_it`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )

    session_runner.close_standby_stream_json_engine(project_id)

    assert fake_class.instances[0].closed is True
    assert project_id not in session_runner._standby_stream_json_engines


def test_register_stream_json_engine_makes_get_or_create_reuse_it_without_spawning(client, tmp_path, monkeypatch):
    """Stream-json mirror of `test_register_engine_makes_get_or_create_engine_reuse_it_without_spawning`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("hi")]))

    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto")
    )
    claimed = session_runner.claim_standby_stream_json_engine(project_id, model="claude-sonnet-5", effort="auto")
    session_runner.register_stream_json_engine(999, claimed)

    engine = session_runner._get_or_create_stream_json_engine(
        999, cwd=cwd, model="claude-sonnet-5", effort="auto", resume_session_id=None
    )

    assert engine is claimed
    assert len(fake_class.instances) == 1  # no second spawn triggered by _get_or_create_stream_json_engine


def test_a_grilling_sessions_first_turn_claims_the_pre_warmed_standby_via_the_start_endpoint(
    client, tmp_path, monkeypatch
):
    """Issue #184 acceptance criterion: "A grilling card's first turn
    benefits from standby pre-warming, matching PtyEngine's existing latency
    characteristics for a first turn." Mirrors the real production path:
    `open_project` warms a matching standby, then `/api/session/start`
    (`start_session`) claims it directly -- no second engine spawn pays the
    inline "wait for claude to open" cost on this session's actual first
    turn."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    conn = db.get_connection()

    # A real, regex-parseable question -- the frontier must stay non-empty
    # so this test's single grilling turn doesn't auto-advance into
    # creating_prd (issue #223), which would spawn a second, unrelated
    # engine for that phase and defeat this test's own "no second spawn"
    # assertion below.
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event('Question 1: "Only question?"')]))
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)
    asyncio.run(
        session_runner.ensure_standby_stream_json_engine(
            project_id, cwd=cwd, model=db.get_model(conn), effort=db.DEFAULT_EFFORT
        )
    )
    assert len(fake_class.instances) == 1
    standby_instance = fake_class.instances[0]

    result = asyncio.run(_run_and_drain(app_module.start_session({"prompt": "a feature", "effort": "auto"})))
    card_id = result["card_id"]

    # No second engine spawned -- the pre-warmed standby was claimed and
    # drove the first turn directly.
    assert len(fake_class.instances) == 1
    assert session_runner._stream_json_engines[card_id] is standby_instance


def test_grilling_turn_under_new_engine_never_touches_the_question_file(client, tmp_path, monkeypatch):
    """Issue #184 acceptance criterion: "A grilling card under this engine
    does NOT read or write .claude/rhubarb_question.md." Terminal text alone
    carries a DIFFERENT, unambiguous question than the file -- if the new
    engine's path ever preferred the file the way the old PtyEngine path
    does, the persisted interview would come from the file instead."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    original_file_content = (
        'Question 1: "Python or Node?"\nOptions:\nOption 1: "Python"\nOption 2: "Node"\nRecommended: [1]\n'
    )
    _write_question_file(cwd, "rhubarb_question.md", original_file_content)

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event('Question 1: "From the turn text"')]))
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview["questions"][0]["text"] == "From the turn text"

    # The file was never read (so it's also never deleted the way a
    # malformed/consumed file would be) -- completely untouched.
    question_file = Path(cwd) / ".claude" / "rhubarb_question.md"
    assert question_file.exists()
    assert question_file.read_text(encoding="utf-8") == original_file_content


def test_continue_session_job_under_new_engine_does_not_delete_the_question_file(client, tmp_path, monkeypatch):
    """Companion to the test above, for the reply path: unlike the
    `PtyEngine` path (`test_continue_session_job_deletes_rhubarb_question_file_on_reply`,
    removed above since it no longer applies to grilling), `continue_session_job`
    must skip its `delete_question_file` call entirely for a grilling card
    dispatched to the new engine."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        return iter([_result_event('Question 1: "A follow-up?"')])

    _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    _write_question_file(cwd, "rhubarb_question.md", 'Question 1: "Leftover from somewhere else?"')
    asyncio.run(session_runner.continue_session_job(row_id, "my answer", cwd=cwd))

    assert (Path(cwd) / ".claude" / "rhubarb_question.md").exists()


def test_grilling_turn_under_new_engine_skips_ollama_rescue_entirely(client, tmp_path, monkeypatch):
    """Issue #184 acceptance criterion: "... does not invoke the Ollama
    rescue classifier." Issue #230 deleted `rescue_grilling_response`
    entirely (it's no longer importable, let alone callable) -- there is no
    code path left in `_run_grilling_turn_stream_json` that could ever call
    it (the old regex-first attempt this test originally targeted is gone;
    extraction now goes straight to the parser session, with no Ollama
    rescue fallback at this call site at all). This test now just confirms
    a malformed-for-extraction turn under the new engine still resolves
    cleanly (empty questions, no crash) with Ollama assistance NOT declined
    -- proving there's nothing left that WOULD have called a rescue function
    even if one still existed.

    Issue #242 (child of PRD #241): a missing parser session means a
    missing completion verdict too, which fails safe as "not done" -- this
    turn's empty frontier no longer auto-advances the chain on its own; it
    stalls in `grilling` instead, waiting on a human reply, exactly like any
    other malformed/missing completion verdict would."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    malformed_text = "Some prose that doesn't fit any structured question format at all.\n"
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(malformed_text)]))
    # No parser session mocked -- `_extract_grilling_questions_via_parser_
    # session` fails closed (`LookupError`, no live session registered) and
    # this turn's frontier comes back genuinely empty, with no completion
    # verdict at all.

    conn = db.get_connection()
    db.set_ollama_declined(conn, False)
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    # No rescue was attempted, and no questions survived, but the missing
    # completion verdict fails safe: the session stalls in grilling rather
    # than auto-advancing into a PRD it never confirmed was ready.
    assert interview["questions"] == []
    assert row["phase"] == "grilling"
    assert row["stalled_json"] is not None
    events = live_stream._buffers.get(row_id, [])
    stalled_events = [e for e in events if e.get("type") == "turn" and e.get("stalled")]
    assert len(stalled_events) == 1
    assert stalled_events[0]["phase"] == "grilling"


def test_grilling_multiline_composed_reply_under_new_engine_completes_as_a_single_turn(client, tmp_path, monkeypatch):
    """Direct regression test for the PRD #180/#181 bug class this whole
    engine exists to fix (see `stream_json_engine.py`'s own module docstring
    and its `test_multiline_prompt_transmits_and_completes_as_a_single_turn`):
    a multi-line composed reply answering 2+ grilling questions, joined with
    embedded newlines, must reach the CLI and complete as ONE turn on the
    first attempt -- no splitting, no retry, no special handling anywhere in
    `continue_session_job`'s dispatch to the new engine."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    composed_reply = "1. Python\n2. For internal tooling"
    seen_prompts = []

    def handler(prompt, **kw):
        seen_prompts.append(prompt)
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event('Question 1: "Language?"\n\nQuestion 2: "Who is it for?"')])
        if prompt == composed_reply:
            return iter([_result_event("Thanks, that's everything I need.")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD Draft: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue Draft S1: Child one")])
        if prompt == "/rhubarb:publish-to-github":
            return iter([_result_event("PRD #5: My PRD\nIssue #6: Child one")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    _mock_parser_session_autoextract(monkeypatch, project_id, cwd)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, composed_reply, cwd=cwd))

    # Exactly one write per turn -- the composed multi-line reply was sent
    # whole, as a single prompt, on the first (and only) attempt -- before
    # the chain auto-advances into to-prd/to-issues/publish-to-github (issue
    # #223, since this reply's wrap-up leaves zero remaining questions).
    assert seen_prompts == [
        "/rhubarb:grilling a feature",
        composed_reply,
        "/rhubarb:to-prd",
        "/rhubarb:to-issues",
        "/rhubarb:publish-to-github",
    ]

    row = db.get_session(conn, row_id)
    assert row["error_text"] is None
    assert row["phase"] == "details"

    events = live_stream._buffers.get(row_id, [])
    grilling_turns = [e for e in events if e["type"] == "turn" and e["phase"] == "grilling"]
    assert grilling_turns[-1]["interview"]["questions"] == []


# ---------------------------------------------------------------------------
# `handle_turn_completed` -- the shared "on turn complete" hook (issue #191,
# child of PRD #187) every phase's turn-handling code now calls instead of
# calling `classify_needs_input` directly, so a positive classification also
# enqueues the turn onto its project's needs-input queue
# (`parser_session.enqueue_needs_input_turn`/`get_needs_input_queue`, the
# registry issue #189 built) -- consolidating what used to be separate,
# duplicated per-phase call sites (grilling, qa-grilling, implementing,
# creating_prd/creating_issues) behind one place. `classify_needs_input`'s
# own behavior (declined/unavailable/malformed handling, the
# `ollama_unavailable` notification) is already fully covered by the
# dedicated tests above and is untouched here -- these tests cover only the
# ADDITIONAL enqueue behavior this hook layers on top.
# ---------------------------------------------------------------------------


def test_handle_turn_completed_enqueues_when_classifier_flags_needs_input(client, monkeypatch):
    # This whole file's own autouse `_classify_needs_input_is_a_no_op_by_default`
    # fixture stubs `session_runner.classify_needs_input` to always return
    # `None` -- restore the genuine function so these tests actually drive
    # `handle_turn_completed`'s real classification call via `http_post`,
    # same precedent as the dedicated `classify_needs_input` tests above.
    monkeypatch.setattr(session_runner, "classify_needs_input", _REAL_CLASSIFY_NEEDS_INPUT)
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)
    row = {"project_id": 7}
    payload = {"needs_input": True, "reason": "Asked which database to use."}

    def fake_post(url, body, *, timeout):
        return {"response": json.dumps(payload)}

    result = asyncio.run(
        session_runner.handle_turn_completed(3, conn, row, "Which database?", "grilling", http_post=fake_post)
    )

    assert result == payload
    assert parser_session.get_needs_input_queue(7) == [
        {"project_id": 7, "card_id": 3, "phase": "grilling", "text": "Which database?"}
    ]


def test_handle_turn_completed_does_not_enqueue_when_classification_is_negative(client, monkeypatch):
    monkeypatch.setattr(session_runner, "classify_needs_input", _REAL_CLASSIFY_NEEDS_INPUT)
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)
    row = {"project_id": 7}

    def fake_post(url, body, *, timeout):
        return {"response": json.dumps({"needs_input": False, "reason": None})}

    result = asyncio.run(
        session_runner.handle_turn_completed(3, conn, row, "All done.", "implementing", http_post=fake_post)
    )

    assert result == {"needs_input": False, "reason": None}
    assert parser_session.get_needs_input_queue(7) == []


def test_handle_turn_completed_does_not_enqueue_when_ollama_declined(client, monkeypatch):
    monkeypatch.setattr(session_runner, "classify_needs_input", _REAL_CLASSIFY_NEEDS_INPUT)
    conn = db.get_connection()
    db.set_ollama_declined(conn, True)
    row = {"project_id": 7}

    def fake_post(url, body, *, timeout):
        raise AssertionError("Ollama must not be called when ollama_declined is true")

    result = asyncio.run(
        session_runner.handle_turn_completed(3, conn, row, "text", "creating_prd", http_post=fake_post)
    )

    assert result is None
    assert parser_session.get_needs_input_queue(7) == []


def test_handle_turn_completed_does_not_enqueue_when_classification_call_fails(client, monkeypatch):
    """A genuinely unexpected classification failure (timeout, connection
    error, malformed response) must not enqueue anything -- there is no
    trustworthy classification to act on, same as a negative one. The
    pre-existing `ollama_unavailable` notification behavior is completely
    unchanged by this hook."""
    monkeypatch.setattr(session_runner, "classify_needs_input", _REAL_CLASSIFY_NEEDS_INPUT)
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)
    row = {"project_id": 7}

    def timing_out_post(url, body, *, timeout):
        raise TimeoutError("Ollama took too long")

    result = asyncio.run(
        session_runner.handle_turn_completed(3, conn, row, "text", "creating_issues", http_post=timing_out_post)
    )

    assert result is None
    assert parser_session.get_needs_input_queue(7) == []
    assert live_stream._buffers.get(3, []) == [{"type": "ollama_unavailable"}]


def test_handle_turn_completed_fifo_order_across_concurrent_sessions_same_project(client, monkeypatch):
    """Multiple concurrent sessions under the SAME project must enqueue in
    the order their turns actually completed -- no loss, no reordering."""
    monkeypatch.setattr(session_runner, "classify_needs_input", _REAL_CLASSIFY_NEEDS_INPUT)
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)
    row = {"project_id": 9}

    def fake_post(url, body, *, timeout):
        return {"response": json.dumps({"needs_input": True, "reason": None})}

    async def _drive_three():
        await session_runner.handle_turn_completed(101, conn, row, "first turn", "grilling", http_post=fake_post)
        await session_runner.handle_turn_completed(
            102, conn, row, "second turn", "implementing", http_post=fake_post
        )
        await session_runner.handle_turn_completed(101, conn, row, "third turn", "grilling", http_post=fake_post)

    asyncio.run(_drive_three())

    queue = parser_session.get_needs_input_queue(9)
    assert [item["card_id"] for item in queue] == [101, 102, 101]
    assert [item["text"] for item in queue] == ["first turn", "second turn", "third turn"]
    assert [item["phase"] for item in queue] == ["grilling", "implementing", "grilling"]


def test_handle_turn_completed_keeps_different_projects_queues_separate(client, monkeypatch):
    """Turns from sessions under different projects must go to their own
    project's queue, never cross-mixed."""
    monkeypatch.setattr(session_runner, "classify_needs_input", _REAL_CLASSIFY_NEEDS_INPUT)
    conn = db.get_connection()
    db.set_ollama_declined(conn, False)

    def fake_post(url, body, *, timeout):
        return {"response": json.dumps({"needs_input": True, "reason": None})}

    async def _drive_both():
        await session_runner.handle_turn_completed(
            1, conn, {"project_id": 1}, "project one's turn", "grilling", http_post=fake_post
        )
        await session_runner.handle_turn_completed(
            2, conn, {"project_id": 2}, "project two's turn", "implementing", http_post=fake_post
        )

    asyncio.run(_drive_both())

    assert [item["text"] for item in parser_session.get_needs_input_queue(1)] == ["project one's turn"]
    assert [item["text"] for item in parser_session.get_needs_input_queue(2)] == ["project two's turn"]


# ---------------------------------------------------------------------------
# Real per-phase wiring: confirming grilling, qa-grilling, implementing
# (including two independent per-issue implement sessions under the same
# project, the shape `/implement`'s parallel mode runs), and the
# creating_prd/creating_issues chain all route through the shared
# `handle_turn_completed` hook and genuinely enqueue a positive
# classification end to end -- not just via a direct hook call above.
# ---------------------------------------------------------------------------


def test_grilling_turn_no_longer_enqueues_onto_needs_input_queue(client, tmp_path, monkeypatch):
    """Issue #221 (child of PRD #187/#220): `_run_grilling_turn_stream_json`
    no longer calls `handle_turn_completed` at all -- grilling always
    transitions to PRD next, so there is no "needs input" holding state for
    it to gate into, and the Ollama classifier gate that used to enqueue a
    positively-classified turn onto the project's needs-input queue for
    grilling specifically is dead code, removed.

    Supersedes `test_grilling_last_resort_needs_input_classification_
    enqueues_onto_project_queue` (removed rather than kept red: the behavior
    it asserted no longer exists for this phase by design, not by
    regression). QA/implement/creating_prd/creating_issues keep the real
    `handle_turn_completed` wiring untouched -- see their own equivalent
    tests elsewhere in this file (e.g.
    `test_qa_grilling_last_resort_needs_input_classification_enqueues_onto_project_queue`)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    prose = "Still thinking this through -- should I use REST or GraphQL here?"
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(prose, session_id="g1")]))
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header=prose)

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        # Issue #223 (child of PRD #222): this same empty-frontier turn now
        # auto-advances straight into creating_prd within this call, whose
        # own `_run_chain_step` legitimately calls `classify_needs_input`
        # (issue #177, unrelated to this test) -- only a `phase == "grilling"`
        # call would mean the removed mechanism came back.
        if phase == "grilling":
            raise AssertionError("classify_needs_input must not be called for a grilling turn (issue #221)")
        return None

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert parser_session.get_needs_input_queue(project_id) == []


def test_qa_grilling_last_resort_needs_input_classification_enqueues_onto_project_queue(
    client, tmp_path, monkeypatch
):
    """Issue #191: QA-grilling's own last-resort needs-input check
    (`_maybe_extract_needs_input_qa`, issue #178) now also enqueues a
    positively-classified turn, tagged with the NEW QA session's own card id
    (issue #250, child of PRD #244: the /rhubarb:qa turn now runs on a
    brand-new card started by `start_move_to_qa_job`, not on the originating
    implementing session's card) and phase `"qa_grilling"`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _qa_tracker(cwd)

    wrapup_text = _QA_BLOCK + "\n\nEverything checks out, no further QA questions."
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(wrapup_text, session_id="qa-session-id")]))

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        return {"needs_input": True, "reason": "Still worth a follow-up check."}

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implemented",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_move_to_qa_job(row_id, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")

    queue = parser_session.get_needs_input_queue(project_id)
    assert queue == [{"project_id": project_id, "card_id": qa_row["id"], "phase": "qa_grilling", "text": wrapup_text}]


def test_implementing_needs_input_classification_enqueues_onto_project_queue(client, tmp_path, monkeypatch):
    """Issue #191: implementing's own needs-input classification
    (`_finish_implement_turn`, issue #179) now also enqueues a
    positively-classified turn, tagged with this session's own card id and
    phase `"implementing"`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    prose = "I went ahead with the OAuth approach, but I'd like you to confirm before I touch the schema."
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(prose, session_id="impl-stall1")]))

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        return {"needs_input": True, "reason": "Asked for confirmation before a schema change."}

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "My PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 5, cwd=cwd))

    queue = parser_session.get_needs_input_queue(project_id)
    assert queue == [{"project_id": project_id, "card_id": row_id, "phase": "implementing", "text": prose}]


def test_creating_prd_needs_input_classification_enqueues_onto_project_queue(client, tmp_path, monkeypatch):
    """Issue #191: the /to-prd, /to-issues chain's own needs-input
    classification (`_run_chain_step`, issue #177) now also enqueues a
    positively-classified turn, tagged with this session's own card id and
    phase `"creating_prd"`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:grilling a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft, but scope is unclear.")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    def reply_handler():
        # Stands in for the live process's response once a human has
        # replied through the generic stall-reply panel -- see
        # `_await_stalled_reply`. This test only cares that the ORIGINAL
        # turn got enqueued; it doesn't drive the chain any further.
        return [_result_event("Understood, using Postgres for scope.")]

    _mock_engine(monkeypatch, handler, reply_handler=reply_handler)
    # Issue #242: a positive completion verdict is required for the empty
    # frontier to auto-advance -- this test is about creating_prd's own
    # needs-input enqueue behavior, not the completion gate.
    _mock_parser_session_extraction(monkeypatch, project_id, cwd, header="❓ **Q1** - **Scope**: Only question?")
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        if phase == "creating_prd":
            return {"needs_input": True, "reason": "Scope is ambiguous."}
        return None

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    # Issue #223: the first grilling turn's empty frontier auto-advances into
    # the chain within this single call.
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    queue = parser_session.get_needs_input_queue(project_id)
    assert queue == [
        {
            "project_id": project_id,
            "card_id": row_id,
            "phase": "creating_prd",
            "text": "Wrote PRD draft, but scope is unclear.",
        }
    ]


def test_parallel_per_issue_implement_sessions_enqueue_in_fifo_order_under_the_same_project(
    client, tmp_path, monkeypatch
):
    """`/implement`'s parallel mode runs one independent PtyEngine tab per
    PRD, all under the same project -- both must flow through the exact same
    `_finish_implement_turn` -> `handle_turn_completed` tail a single
    implement session does, enqueueing onto the SAME project's queue, in the
    order each turn actually finished, with no loss."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:implement prd: 5":
            return iter([_result_event("Confirm before touching schema for PRD 5.", session_id="impl-5")])
        if prompt == "/rhubarb:implement prd: 6":
            return iter([_result_event("Confirm before touching schema for PRD 6.", session_id="impl-6")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)

    async def fake_classify(card_id, conn_arg, text, phase, *, http_post=None):
        return {"needs_input": True, "reason": "Needs schema confirmation."}

    monkeypatch.setattr(session_runner, "classify_needs_input", fake_classify)

    conn = db.get_connection()
    row_5 = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 5, "title": "A"}},
    )
    row_6 = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 6, "title": "B"}},
    )

    async def _both():
        await session_runner.start_implement_job(row_5, 5, cwd=cwd)
        await session_runner.start_implement_job(row_6, 6, cwd=cwd)

    asyncio.run(_both())

    queue = parser_session.get_needs_input_queue(project_id)
    assert [item["card_id"] for item in queue] == [row_5, row_6]
    assert [item["phase"] for item in queue] == ["implementing", "implementing"]
    assert queue[0]["text"] == "Confirm before touching schema for PRD 5."
    assert queue[1]["text"] == "Confirm before touching schema for PRD 6."
