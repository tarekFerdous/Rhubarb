import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from rhubarb import db, live_stream, pty_engine, session_runner
from rhubarb.cli_client import ClaudeCLIError
from rhubarb.github_publisher import GithubPublishError
from rhubarb.pty_engine import PtyEngineUnrecoverableError
from rhubarb.web import app as app_module


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


def _make_fake_engine_class(handler, *, fresh_ids=None):
    """Build a fake stand-in for the `PtyEngine` class, for monkeypatching
    `session_runner.PtyEngine` in these tests -- see `tests/test_pty_engine.py`
    for the equivalent fake-backend style one level down (the real PTY
    backend, rather than the engine built on top of it).

    `handler(prompt, *, session_id, cwd, model, effort)` returns an iterable
    of raw event dicts -- the same shape the old `stream_prompt` fakes
    already produced, minus `card_id` (that concept is gone: every phase
    now drives its turns through the same one-resident-tab-per-card_id
    mechanism, not just the /do chain).

    Since `session_runner` now keeps ONE engine alive across every turn for
    a card (see `session_runner._get_or_create_engine`), a fake engine
    forwards every turn over its own lifetime to this SAME `handler` --
    tests that used to swap `stream_prompt` mid-test to change behavior
    between calls instead give `handler` one prompt-keyed dispatcher
    covering the whole scenario (a resident engine wouldn't see a mock
    swapped out from under it either).

    `fresh_ids`, if given, is a queue of ids handed out (in order) to
    successive *fresh* (no `resume_session_id`) constructions -- standing
    in for what a genuinely fresh `PtyEngine`'s own generated uuid would be,
    so a test can pin down the exact id a "clear"/"pool" produces, the way
    tests used to control `cli_client.clear_session`'s return value
    directly.
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

    return FakeEngine


def _mock_engine(monkeypatch, handler, *, fresh_ids=None):
    """Monkeypatch `session_runner.PtyEngine` with a fake driven by
    `handler` -- see `_make_fake_engine_class`. Returns the fake class so a
    test can inspect `.instances` (e.g. to assert an engine was/wasn't
    reconstructed, or to check constructor args)."""
    fake_class = _make_fake_engine_class(handler, fresh_ids=fresh_ids)
    monkeypatch.setattr(session_runner, "PtyEngine", fake_class)
    return fake_class


def _make_blocking_fake_engine_class(entered, release, enter_count, *, result_text, session_id=None):
    """Build a fake stand-in for `PtyEngine` whose `stream_turn` genuinely
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


def test_start_session_job_publishes_terminal_output_events_from_the_pty(client, tmp_path, monkeypatch):
    """Issue #88: a `terminal_output` raw event yielded by (a faked)
    `PtyEngine.stream_turn` must reach the session's live-stream buffer
    verbatim -- this is what the SSE endpoint hands the frontend's terminal
    view."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    raw_chunk = "\x1b[2K\rWorking on it...\r\n"
    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter(
            [
                {"type": "terminal_output", "data": raw_chunk},
                _result_event("❓ **Q1** - **Scope**: Only question?"),
            ]
        ),
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    events = live_stream._buffers.get(row_id, [])
    assert {"type": "terminal_output", "data": raw_chunk} in events


def test_open_pty_tab_count_tracks_tabs_as_sessions_open_and_close(client, tmp_path, monkeypatch):
    """Backs the web UI's tab-count indicator (issue #88):
    `session_runner.open_pty_tab_count()` must accurately reflect how many
    `PtyEngine` tabs are currently resident as sessions start (opening a
    tab, one per card_id) and close (dropping it)."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    assert session_runner.open_pty_tab_count() == 0

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event(f'Question 1: "{prompt}?"', session_id=prompt)]),
    )

    conn = db.get_connection()
    row_a = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_a, "feature A", cwd=cwd))
    assert session_runner.open_pty_tab_count() == 1

    row_b = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_b, "feature B", cwd=cwd))
    assert session_runner.open_pty_tab_count() == 2

    session_runner._close_engine(row_a)
    assert session_runner.open_pty_tab_count() == 1

    session_runner._close_engine(row_b)
    assert session_runner.open_pty_tab_count() == 0


def test_close_session_terminates_a_live_resident_engine(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("❓ **Q1** - **Scope**: Only question?")]),
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    assert session_runner.open_pty_tab_count() == 1

    session_runner.close_session(conn, row_id)

    assert fake_class.instances[0].closed is True
    assert session_runner.open_pty_tab_count() == 0


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
        if prompt == "/rhubarb:do feature A":
            return iter([_result_event('Question 1: "Question A?"', session_id="sA")])
        if prompt == "/rhubarb:do feature B":
            return iter([_result_event('Question 1: "Question B?"', session_id="sB")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)

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

    async def run_all():
        await asyncio.gather(
            *[session_runner.start_session_job(row_id, f"feature {i}", cwd=cwd) for i, row_id in enumerate(row_ids)]
        )

    asyncio.run(run_all())

    for i, row_id in enumerate(row_ids):
        row = db.get_session(conn, row_id)
        assert row["claude_session_id"] == f"/rhubarb:do feature {i}"
        interview = json.loads(row["interview_json"])
        assert interview["questions"][0]["text"] == f"/rhubarb:do feature {i}?"


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


def test_start_session_job_prefers_rhubarb_question_file_over_terminal_text(client, tmp_path, monkeypatch):
    """PRD #123: a turn whose rendered terminal text would parse to nothing
    still produces a full interview when `.claude/rhubarb_question.md` is
    present with valid content -- the file-based path this PRD adds."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _write_question_file(
        cwd, "rhubarb_question.md",
        'Question 1: "Python or Node?"\nOptions:\nOption 1: "Python"\nOption 2: "Node"\nRecommended: [1]\n',
    )

    # Terminal text alone has no recognizable Question block -- would parse empty.
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("Working on it, one moment.")]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview["source"] == "regex"
    assert len(interview["questions"]) == 1
    assert interview["questions"][0]["options"] == ["Python", "Node"]
    assert interview["questions"][0]["recommended"] == [1]


def test_rhubarb_question_file_is_not_deleted_merely_by_being_read(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _write_question_file(cwd, "rhubarb_question.md", 'Question 1: "Python or Node?"')

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("irrelevant terminal text")]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    assert (Path(cwd) / ".claude" / "rhubarb_question.md").exists()


def test_continue_session_job_deletes_rhubarb_question_file_on_reply(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        return iter([_result_event('Question 1: "A follow-up?"')])

    _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    _write_question_file(cwd, "rhubarb_question.md", 'Question 1: "First question?"')
    asyncio.run(session_runner.continue_session_job(row_id, "my answer", cwd=cwd))

    assert not (Path(cwd) / ".claude" / "rhubarb_question.md").exists()


def test_confirm_advance_also_deletes_rhubarb_question_file(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt in ("/rhubarb:to-prd", "/rhubarb:to-issues"):
            return iter([_result_event("done")])
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    _write_question_file(cwd, "rhubarb_question.md", 'Question 1: "First question?"')
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    assert not (Path(cwd) / ".claude" / "rhubarb_question.md").exists()


def test_malformed_rhubarb_question_file_is_deleted_and_falls_back_to_terminal_text(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _write_question_file(cwd, "rhubarb_question.md", "not a recognizable question format at all")

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event('Question 1: "From terminal text"')]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview["questions"][0]["text"] == "From terminal text"
    assert not (Path(cwd) / ".claude" / "rhubarb_question.md").exists()


def test_start_session_job_publishes_interview_even_with_no_structured_questions(client, tmp_path, monkeypatch):
    """A real /rhubarb:do turn can reply with plain prose (no bullet/heading
    questions qa_parser recognizes as structured). The left card must still
    render that turn -- it must not look like nothing happened."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(
        monkeypatch,
        lambda prompt, **kw: iter([_result_event("Sure, tell me more about what you have in mind.")]),
    )

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a new feature", cwd=cwd))

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    assert len(turn_events) == 1
    assert turn_events[0]["interview"]["questions"] == []
    assert turn_events[0]["interview"]["header"]


def test_continue_session_job_with_remaining_questions_does_not_auto_advance(client, tmp_path, monkeypatch):
    """A reply that still has follow-up questions must never auto-advance --
    this was already true before #33, but this test locks it in explicitly
    and confirms it needs no `confirm_advance` flag."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt in ("/rhubarb:to-prd", "/rhubarb:to-issues"):
            raise AssertionError(f"chain must not run without confirm_advance, got {prompt!r}")
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event('Question 1: "First question?"')])
        return iter([_result_event('Question 1: "A follow-up question?"')])

    _mock_engine(monkeypatch, handler)

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


def test_continue_session_job_with_no_more_questions_does_not_auto_advance(client, tmp_path, monkeypatch):
    """Issue #33: a reply that comes back with zero remaining questions must
    stay in grilling and publish the wrap-up turn -- it must NOT silently
    fire /rhubarb:to-prd on its own anymore."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt in ("/rhubarb:to-prd", "/rhubarb:to-issues"):
            raise AssertionError(f"chain must not run without confirm_advance, got {prompt!r}")
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event('Question 1: "First question?"')])
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "all good", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"
    assert row["error_text"] is None

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    # One turn from start_session_job's first question, one from this reply.
    assert len(turn_events) == 2
    assert turn_events[-1]["phase"] == "grilling"
    assert turn_events[-1]["interview"]["questions"] == []
    assert turn_events[-1]["interview"]["header"]
    assert not any(e == {"type": "phase", "phase": "creating_prd"} for e in events)


# ---------------------------------------------------------------------------
# Ollama rescue-path wiring (issue #114)
# ---------------------------------------------------------------------------


def test_grilling_turn_uses_ollama_rescue_when_regex_parser_finds_nothing(client, tmp_path, monkeypatch):
    """A turn whose text looks like it was trying to contain a question
    (contains "Question ") but doesn't match the strict regex format must
    be handed to the Ollama rescue path, and a valid rescue result must be
    used as the published/persisted interview."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    malformed_text = 'Question 1: The quote never closes, so the regex parser finds nothing here.\n'
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(malformed_text)]))

    rescued = {
        "header": "",
        "footer": "",
        "questions": [
            {
                "id": "q1",
                "text": "Rescued question text",
                "kind": "open",
                "options": None,
                "recommended": None,
                "recommended_text": None,
            }
        ],
    }
    seen = {}

    def fake_rescue(raw_text):
        seen["raw_text"] = raw_text
        return rescued

    monkeypatch.setattr(session_runner, "rescue_grilling_response", fake_rescue)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert seen["raw_text"] == malformed_text

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview == rescued

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    assert turn_events[-1]["interview"] == rescued


def test_grilling_turn_does_not_use_ollama_rescue_when_regex_parser_already_finds_questions(client, tmp_path, monkeypatch):
    """The rescue path must never fire when the regex parser already found
    questions -- it's the fast, free, primary path for the common case."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event('Question 1: "A clean, well-formed question?"')]))

    def fake_rescue(raw_text):
        raise AssertionError("rescue must not be called when the regex parser already found questions")

    monkeypatch.setattr(session_runner, "rescue_grilling_response", fake_rescue)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview["questions"][0]["text"] == "A clean, well-formed question?"


def test_grilling_turn_does_not_use_ollama_rescue_on_a_genuine_wrap_up(client, tmp_path, monkeypatch):
    """A genuine "grilling is done" wrap-up message (no "Question " substring
    at all) must never trigger a rescue attempt."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("Thanks, that's everything I need.")]))

    def fake_rescue(raw_text):
        raise AssertionError("rescue must not be called on a genuine wrap-up message")

    monkeypatch.setattr(session_runner, "rescue_grilling_response", fake_rescue)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview["questions"] == []


def test_grilling_turn_falls_back_to_empty_result_when_ollama_rescue_returns_none(client, tmp_path, monkeypatch):
    """When Ollama is unavailable/times out/returns something invalid
    (modeled here as rescue_grilling_response returning None), the turn must
    not crash. Issue #158: since the raw text still looks like it was
    trying to contain questions, this no longer just silently falls back to
    an empty parsed result -- a corrective retry runs, and since (in this
    test) it comes back equally unparseable, an explicit error is
    published/persisted instead of a bare empty interview."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    malformed_text = 'Question 1: unterminated, no closing quote\n'
    calls = []

    def handler(prompt, **kw):
        calls.append(prompt)
        return iter([_result_event(malformed_text, session_id=f"s{len(calls)}")])

    _mock_engine(monkeypatch, handler)
    monkeypatch.setattr(session_runner, "rescue_grilling_response", lambda raw_text: None)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert len(calls) == 2

    row = db.get_session(conn, row_id)
    assert row["error_text"] is not None


def test_grilling_turn_skips_ollama_rescue_when_declined(client, tmp_path, monkeypatch):
    """Issue #119: with `ollama_declined` true, a turn whose regex parse is
    empty and whose raw text contains the trigger substring must NOT invoke
    the rescue function -- the toggle must actually gate the rescue call,
    not just the install-gate modal's own display logic. Issue #158: the
    decline toggle only gates the Ollama rescue call, not the separate
    corrective-retry mechanism -- that still fires, and since (in this test)
    its result is equally unparseable, ends in an explicit error rather
    than a silently empty interview."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    malformed_text = 'Question 1: The quote never closes, so the regex parser finds nothing here.\n'
    calls = []

    def handler(prompt, **kw):
        calls.append(prompt)
        return iter([_result_event(malformed_text, session_id=f"s{len(calls)}")])

    _mock_engine(monkeypatch, handler)

    def fake_rescue(raw_text):
        raise AssertionError("rescue must not be called when ollama_declined is true")

    monkeypatch.setattr(session_runner, "rescue_grilling_response", fake_rescue)

    conn = db.get_connection()
    db.set_ollama_declined(conn, True)
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert len(calls) == 2

    row = db.get_session(conn, row_id)
    assert row["error_text"] is not None


def test_grilling_turn_uses_ollama_rescue_when_not_declined(client, tmp_path, monkeypatch):
    """Issue #119: with `ollama_declined` explicitly false, the rescue call
    still fires, unchanged from current behavior."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    malformed_text = 'Question 1: The quote never closes, so the regex parser finds nothing here.\n'
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(malformed_text)]))

    rescued = {
        "header": "",
        "footer": "",
        "questions": [
            {
                "id": "q1",
                "text": "Rescued question text",
                "kind": "open",
                "options": None,
                "recommended": None,
                "recommended_text": None,
            }
        ],
    }
    monkeypatch.setattr(session_runner, "rescue_grilling_response", lambda raw_text: rescued)

    conn = db.get_connection()
    db.set_ollama_declined(conn, False)
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    interview = json.loads(row["interview_json"])
    assert interview == rescued


# ---------------------------------------------------------------------------
# Corrective retry for a grilling round that failed to parse (issue #158)
# ---------------------------------------------------------------------------


def test_grilling_turn_corrective_retry_fires_on_suspicious_unparseable_result(client, tmp_path, monkeypatch):
    """A round whose terminal text looks like it was trying to contain
    questions (contains "Question "), but which the whole existing fallback
    chain (file -> terminal text -> Ollama rescue) still fails to parse,
    must trigger exactly one corrective follow-up turn: the model is handed
    its own broken output back and asked to reformat it. When that
    corrective turn comes back parseable, its questions -- not the generic
    "ready to proceed?" wrap-up -- are what gets published/persisted."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    malformed_text = 'Question 1: the quote never closes, so the regex parser finds nothing here.\n'
    calls = []

    def handler(prompt, **kw):
        calls.append(prompt)
        if "did not parse" in prompt:
            assert malformed_text in prompt
            return iter([_result_event('Question 1: "Reformatted question?"', session_id="s2")])
        return iter([_result_event(malformed_text, session_id="s1")])

    _mock_engine(monkeypatch, handler)
    monkeypatch.setattr(session_runner, "rescue_grilling_response", lambda raw_text: None)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert len(calls) == 2, "expected exactly one corrective retry turn, in addition to the original"

    row = db.get_session(conn, row_id)
    assert row["error_text"] is None
    assert row["claude_session_id"] == "s2"
    interview = json.loads(row["interview_json"])
    assert interview["questions"][0]["text"] == "Reformatted question?"

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    assert len(turn_events) == 1
    assert turn_events[0]["interview"]["questions"][0]["text"] == "Reformatted question?"
    assert turn_events[0]["error"] is None


def test_grilling_turn_no_corrective_retry_on_a_genuine_wrap_up(client, tmp_path, monkeypatch):
    """A genuine "grilling is done" wrap-up (no "Question " substring at all,
    in either the file or the terminal text) must NOT trigger any corrective
    retry -- it's treated as done, same as before issue #158."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    calls = []

    def handler(prompt, **kw):
        calls.append(prompt)
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)

    def fail_retry(*args, **kwargs):
        raise AssertionError("corrective retry must not run on a genuine wrap-up")

    monkeypatch.setattr(session_runner, "_run_grilling_corrective_retry", fail_retry)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert len(calls) == 1

    row = db.get_session(conn, row_id)
    assert row["error_text"] is None
    interview = json.loads(row["interview_json"])
    assert interview["questions"] == []


def test_grilling_turn_corrective_retry_failure_publishes_explicit_error(client, tmp_path, monkeypatch):
    """When the corrective retry's own result also fails to parse, the
    session must stop automatically with an explicit, persisted/published
    error -- never the generic "ready to proceed?" gate a genuine empty
    wrap-up would show."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    malformed_text = 'Question 1: still no closing quote here either\n'
    calls = []

    def handler(prompt, **kw):
        calls.append(prompt)
        # Every turn -- original and corrective retry alike -- comes back
        # equally unparseable.
        return iter([_result_event(malformed_text, session_id=f"s{len(calls)}")])

    _mock_engine(monkeypatch, handler)
    monkeypatch.setattr(session_runner, "rescue_grilling_response", lambda raw_text: None)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert len(calls) == 2, "must not retry more than once"

    row = db.get_session(conn, row_id)
    assert row["error_text"] is not None
    assert "did not" in row["error_text"] or "failed" in row["error_text"]
    assert row["claude_session_id"] == "s2"

    events = live_stream._buffers.get(row_id, [])
    turn_events = [e for e in events if e["type"] == "turn"]
    assert turn_events, "an explicit error turn event must be published"
    assert turn_events[-1]["error"] == row["error_text"]
    assert turn_events[-1]["phase"] == "grilling"


def test_confirm_advance_skips_grilling_turn_and_advances_through_chain(client, tmp_path, monkeypatch):
    """Issue #33: the explicit "Yes, proceed" path (confirm_advance=True)
    must go straight to /rhubarb:to-prd -> /rhubarb:to-issues -> details, resuming the
    session's existing claude_session_id, WITHOUT sending another grilling
    CLI turn first. Since #75, `details` auto-continues straight into
    /rhubarb:implement -- this test's handler covers that turn too and asserts
    the chain lands on `implemented`, not `details`."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    seen_prompts = []

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        seen_prompts.append(prompt)
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Wrote issues draft.")])
        if prompt == "/rhubarb:implement prd: 5":
            return iter([_result_event("Implemented.")])
        raise AssertionError(f"unexpected grilling-style prompt {prompt!r} during confirm_advance")

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    monkeypatch.setattr(
        session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "PRD #5: My PRD\nIssue #6: Child one"
    )

    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    # The two chain prompts ran, followed by the auto-continued implement
    # turn -- no grilling reply was ever sent.
    assert seen_prompts == ["/rhubarb:to-prd", "/rhubarb:to-issues", "/rhubarb:implement prd: 5"]

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [{"number": 6, "title": "Child one"}]
    assert row["available_for_reuse"] == 1

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}
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
        return iter([_result_event("- Only question?")])

    _mock_engine(monkeypatch, handler)

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
    """/rhubarb:to-prd and /rhubarb:to-issues (run via advance_past_grilling) must be
    invoked with the same model the session's grilling turn used, not
    whatever `settings.model` currently is."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_model(conn, "claude-opus-4-8")

    seen_models = []

    def handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("- Only question?")])
        seen_models.append((prompt, model))
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("PRD #5: My PRD")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Issue #6: Child one")])
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    asyncio.run(session_runner.continue_session_job(row_id, "all good", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    assert seen_models == [
        ("all good", "claude-opus-4-8"),
        ("/rhubarb:to-prd", "claude-opus-4-8"),
        ("/rhubarb:to-issues", "claude-opus-4-8"),
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

    # The card under test -- already has a live resident engine (its first
    # turn already ran).
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    original_engine = session_runner._pty_engines[row_id]
    assert original_engine.model == "claude-opus-4-8"
    assert original_engine.effort == "auto"

    # A different card, also with a live resident engine -- must be left
    # completely untouched below.
    other_row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(other_row_id, "another feature", cwd=cwd))
    other_engine = session_runner._pty_engines[other_row_id]

    # The project's pre-warmed standby engine -- also must be left
    # completely untouched below.
    asyncio.run(
        session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-opus-4-8", effort=db.DEFAULT_EFFORT)
    )
    standby_engine, _, _ = session_runner._standby_engines[project_id]

    resp = client.post("/api/settings/model", json={"model": "claude-sonnet-4-6", "card_id": row_id})
    assert resp.json() == {"model": "claude-sonnet-4-6", "respawned": True}

    # (a) The open card's engine was torn down and replaced with a fresh,
    # unresumed one under the new model (effort carried over unchanged).
    assert original_engine.closed is True
    new_engine = session_runner._pty_engines[row_id]
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
    assert session_runner._pty_engines[other_row_id] is other_engine
    assert other_engine.closed is False
    assert other_engine.model == "claude-opus-4-8"

    assert session_runner._standby_engines[project_id][0] is standby_engine
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

    row_id = db.create_session(conn, project_id, effort="low")
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    original_engine = session_runner._pty_engines[row_id]
    assert original_engine.effort == "low"

    other_row_id = db.create_session(conn, project_id, effort="low")
    asyncio.run(session_runner.start_session_job(other_row_id, "another feature", cwd=cwd))
    other_engine = session_runner._pty_engines[other_row_id]

    resp = client.post("/api/settings/effort", json={"effort": "high", "card_id": row_id})
    assert resp.json() == {"effort": "high", "respawned": True}

    new_engine = session_runner._pty_engines[row_id]
    assert new_engine is not original_engine
    assert original_engine.closed is True
    assert new_engine.model == "claude-opus-4-8"
    assert new_engine.effort == "high"
    assert new_engine.resume_session_id is None

    assert session_runner._pty_engines[other_row_id] is other_engine
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
    assert row_id not in session_runner._pty_engines

    resp = client.post("/api/settings/model", json={"model": "claude-opus-4-8", "card_id": row_id})
    assert resp.json() == {"model": "claude-opus-4-8", "respawned": False}
    assert row_id not in session_runner._pty_engines
    assert db.get_model(conn) == "claude-opus-4-8"


def test_continue_session_job_advances_through_prd_and_issues_to_details(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: First question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Wrote issues draft.")])
        # the grilling reply itself: no more bullet/heading questions -> grilling is done
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    monkeypatch.setattr(
        session_runner,
        "publish_draft",
        lambda draft_path, cwd, on_progress=None: "PRD #5: My PRD\nIssue #6: Child one\nIssue #7: Child two",
    )

    # The grilling reply itself: no more questions -> stays in grilling and
    # publishes the wrap-up turn, no auto-advance.
    asyncio.run(session_runner.continue_session_job(row_id, "all good", cwd=cwd))
    row = db.get_session(conn, row_id)
    assert row["phase"] == "grilling"

    # Explicit "Yes, proceed" is what actually advances the chain -- since
    # #75, straight through into an auto-continued /rhubarb:implement turn too
    # (this test's fallback branch answers that prompt the same generic way).
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [
        {"number": 6, "title": "Child one"},
        {"number": 7, "title": "Child two"},
    ]
    assert row["available_for_reuse"] == 1

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}
    assert any(e.get("type") == "turn" and e.get("phase") == "details" for e in events)
    assert any(e == {"type": "phase", "phase": "creating_prd"} for e in events)
    assert "$" not in json.dumps(events)


def test_confirm_advance_runs_publishing_phase_with_no_extra_cli_calls(client, tmp_path, monkeypatch):
    """Issue #58: the chain must sequence phase:creating_prd -> phase:creating_issues
    -> phase:publishing -> turn(details) -> done, and github_publisher.publish_draft
    (not a Claude CLI turn) must be what actually creates the GitHub issues."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    seen_prompts = []

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        seen_prompts.append(prompt)
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Wrote issues draft.")])
        if prompt == "/rhubarb:implement prd: 5":
            return iter([_result_event("Implemented.")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    seen_publish_calls = []

    def fake_publish_draft(draft_path, cwd, on_progress=None):
        seen_publish_calls.append((draft_path, cwd))
        return "PRD #5: My PRD\nIssue #6: Child one"

    monkeypatch.setattr(session_runner, "publish_draft", fake_publish_draft)

    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    # /rhubarb:to-prd, /rhubarb:to-issues, and the auto-continued implement turn
    # went through the Claude CLI -- publishing did not.
    assert seen_prompts == ["/rhubarb:to-prd", "/rhubarb:to-issues", "/rhubarb:implement prd: 5"]
    assert len(seen_publish_calls) == 1

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 5, "title": "My PRD"}
    assert details["issues"] == [{"number": 6, "title": "Child one"}]
    assert row["available_for_reuse"] == 1

    events = live_stream._buffers.get(row_id, [])
    chain_phases = [
        e["phase"] for e in events if e.get("type") == "phase" and e["phase"] != "grilling"
    ]
    assert chain_phases == ["creating_prd", "creating_issues", "publishing", "implementing"]
    assert events[-1] == {"type": "done"}
    assert any(e.get("type") == "turn" and e.get("phase") == "details" for e in events)
    # Issue #146: the do-to-implement handoff no longer auto-minimizes the
    # left card -- the frontend shows a "Proceed" banner instead.
    assert not any(e.get("type") == "minimize" for e in events)


def test_publish_draft_failure_stops_chain_with_error_turn(client, tmp_path, monkeypatch):
    """Issue #58: a GithubPublishError from publish_draft() must be caught,
    logged as an error turn event (same shape as _run_chain_step failures),
    followed by done -- with no unhandled exception -- and the chain must
    stop before _finish_chain ever runs."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Wrote issues draft.")])
        raise AssertionError(f"unexpected prompt {prompt!r}")

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    def failing_publish_draft(draft_path, cwd, on_progress=None):
        raise GithubPublishError("gh: not logged in, run `gh auth login`")

    monkeypatch.setattr(session_runner, "publish_draft", failing_publish_draft)

    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "publishing"
    assert bool(row["needs_github_login"]) is True
    assert "not logged in" in row["error_text"]
    # The chain never reached _finish_chain.
    assert row["details_json"] is None
    assert row["available_for_reuse"] == 0

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}
    error_turns = [e for e in events if e.get("type") == "turn" and e.get("phase") == "publishing"]
    assert len(error_turns) == 1
    assert "not logged in" in error_turns[0]["error"]


def test_retry_on_publishing_phase_reruns_publisher_and_completes(client, tmp_path, monkeypatch):
    """Issue #58: retry_session_job must handle phase == 'publishing' -- a
    session stuck there (the publisher errored, or the app restarted
    mid-phase) resumes by re-running the publisher and completing the chain."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        return iter([_result_event("draft written")])

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    def failing_publish_draft(draft_path, cwd, on_progress=None):
        raise GithubPublishError("gh rate limited")

    monkeypatch.setattr(session_runner, "publish_draft", failing_publish_draft)
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "publishing"
    assert "gh rate limited" in row["error_text"]

    monkeypatch.setattr(
        session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "PRD #9: Retried PRD\nIssue #10: Only child"
    )

    asyncio.run(session_runner.retry_session_job(row_id, cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
    assert row["error_text"] is None
    details = json.loads(row["details_json"])
    assert details["prd"] == {"number": 9, "title": "Retried PRD"}
    assert details["issues"] == [{"number": 10, "title": "Only child"}]


def test_run_publish_step_publishes_intermediate_event_after_each_issue(client, tmp_path, monkeypatch):
    """Issue #154: `_run_publish_step` must publish a `turn` event carrying
    a `status_message` right after the PRD issue is created ("PRD published
    as #N") and right after each child issue is created ("Created issue
    #N") -- observable mid-call on the session's live stream, not only in
    the final aggregate result. Drives the real `publish_draft`/
    `create_issue` path (only the `gh` subprocess call itself is faked),
    exercising the actual production per-issue loop rather than mocking
    `publish_draft` away entirely."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    claude_dir = Path(cwd) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    draft = {
        "prd": {"title": "My PRD", "body": "PRD body", "labels": ["ready-for-agent"]},
        "issues": [
            {"title": "Child one", "body": "Body one", "labels": []},
            {"title": "Child two", "body": "Body two", "labels": []},
        ],
    }
    (claude_dir / "prd_draft.json").write_text(json.dumps(draft), encoding="utf-8")

    def fake_run(args, **kwargs):
        if "My PRD" in args:
            return type("Result", (), {"returncode": 0, "stdout": "https://github.com/x/y/issues/5\n", "stderr": ""})()
        if "Child one" in args:
            return type("Result", (), {"returncode": 0, "stdout": "https://github.com/x/y/issues/6\n", "stderr": ""})()
        if "Child two" in args:
            return type("Result", (), {"returncode": 0, "stdout": "https://github.com/x/y/issues/7\n", "stderr": ""})()
        raise AssertionError(f"unexpected args {args}")

    monkeypatch.setattr("rhubarb.github_publisher.subprocess.run", fake_run)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    row = db.get_session(conn, row_id)

    ok = asyncio.run(session_runner._run_publish_step(row_id, conn, row, cwd=cwd))
    assert ok is True

    events = live_stream._buffers.get(row_id, [])
    status_messages = [
        e["status_message"]
        for e in events
        if e.get("type") == "turn" and e.get("phase") == "publishing" and e.get("status_message")
    ]
    assert status_messages == [
        "PRD published as #5",
        "Created issue #6",
        "Created issue #7",
    ]


def test_run_chain_step_publishes_status_once_draft_files_are_written(client, tmp_path, monkeypatch):
    """Issue #154: while a session is in `creating_prd`/`creating_issues`,
    the user must see a status update once the corresponding draft content
    was actually written to `.claude/prd_draft.json` -- not just a generic
    spinner. `_run_chain_step` detects this by reading the draft file back
    after each turn completes."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    claude_dir = Path(cwd) / ".claude"

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            claude_dir.mkdir(parents=True, exist_ok=True)
            (claude_dir / "prd_draft.json").write_text(
                json.dumps({"prd": {"title": "My PRD", "body": "b", "labels": []}}), encoding="utf-8"
            )
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            (claude_dir / "prd_draft.json").write_text(
                json.dumps(
                    {
                        "prd": {"title": "My PRD", "body": "b", "labels": []},
                        "issues": [
                            {"title": "Child one", "body": "b1", "labels": []},
                            {"title": "Child two", "body": "b2", "labels": []},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return iter([_result_event("Wrote issues draft.")])
        return iter([_result_event("Implemented.")])

    _mock_engine(monkeypatch, handler)
    monkeypatch.setattr(session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "PRD #5: My PRD\nIssue #6: Child one\nIssue #7: Child two")

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    events = live_stream._buffers.get(row_id, [])
    status_by_phase = {
        e["phase"]: e["status_message"]
        for e in events
        if e.get("type") == "turn" and e.get("status_message")
    }
    assert status_by_phase["creating_prd"] == "PRD draft written."
    assert status_by_phase["creating_issues"] == "Issues draft written (2 issues)."


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


def test_to_prd_claude_cli_error_never_sets_needs_github_login(client, tmp_path, monkeypatch):
    """/to-prd is forbidden from calling `gh` (see its skill instructions), so
    a ClaudeCLIError here -- even one containing gh-shaped auth text -- must
    never be misclassified as a GitHub login need."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            raise ClaudeCLIError("gh: not logged in, run `gh auth login`")
        return iter([_result_event("Thanks, that's everything I need.")])

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "creating_prd"
    assert bool(row["needs_github_login"]) is False
    assert "not logged in" in row["error_text"]

    events = live_stream._buffers.get(row_id, [])
    assert events[-1] == {"type": "done"}
    error_turns = [e for e in events if e.get("type") == "turn" and e.get("phase") == "creating_prd"]
    assert error_turns[0]["needs_github_login"] is False


def test_retry_after_login_completes_the_failed_phase(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    attempt = {"n": 0}

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ClaudeCLIError("not logged in")
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Wrote issues draft.")])
        return iter([_result_event("done")])

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    monkeypatch.setattr(
        session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "PRD #9: Retried PRD\nIssue #10: Only child"
    )

    asyncio.run(session_runner.retry_session_job(row_id, cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
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
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            attempt["n"] += 1
            if attempt["n"] == 1:
                raise ClaudeCLIError("not logged in")
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Wrote issues draft.")])
        return iter([_result_event("done")])

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    monkeypatch.setattr(
        session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "PRD #30: Regression PRD\nIssue #31: Only child"
    )

    asyncio.run(session_runner.retry_session_job(row_id, cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "implemented"
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
    assert session_runner._pty_engines == {}


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
    project_a = _open_project(client, tmp_path, "proj_a")
    cwd_a = _cwd_for(project_a)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt in ("/rhubarb:to-prd", "/rhubarb:to-issues"):
            return iter([_result_event("wrote draft")])
        return iter([_result_event("done")])

    _mock_engine(monkeypatch, handler, fresh_ids=["turn-engine", "pooled-session"])
    conn = db.get_connection()
    row_id = db.create_session(conn, project_a)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd_a))

    monkeypatch.setattr(session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "PRD #1: p\nIssue #2: i")
    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd_a, confirm_advance=True))

    row = db.get_session(conn, row_id)
    assert row["available_for_reuse"] == 1
    assert row["claude_session_id"] == "pooled-session"

    seen_session_ids = []

    def recording_handler(prompt, *, session_id=None, cwd=None, model=None, effort=None):
        seen_session_ids.append(session_id)
        return iter([_result_event("❓ **Q1** - **Scope**: Another question?", session_id="new")])

    _mock_engine(monkeypatch, recording_handler)

    # Same project: should resume the pooled session.
    reused = db.claim_available_session(conn, project_a)
    resume_id = reused["claude_session_id"] if reused is not None else None
    new_row_id = db.create_session(conn, project_a, claude_session_id=resume_id)
    asyncio.run(session_runner.start_session_job(new_row_id, "another feature", cwd=cwd_a))
    assert seen_session_ids[-1] == "pooled-session"

    # A different project must never be handed project A's pooled session.
    project_b = _open_project(client, tmp_path, "proj_b")
    cwd_b = _cwd_for(project_b)
    reused_b = db.claim_available_session(conn, project_b)
    resume_id_b = reused_b["claude_session_id"] if reused_b is not None else None
    row_id_b = db.create_session(conn, project_b, claude_session_id=resume_id_b)
    asyncio.run(session_runner.start_session_job(row_id_b, "unrelated feature", cwd=cwd_b))
    assert seen_session_ids[-1] is None


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


def test_start_implement_job_triggers_qa_session_when_result_contains_qa_block(client, tmp_path, monkeypatch):
    """When /implement Phase 5 runs /qa and the result includes the qa_grilling
    JSON block, start_implement_job must create a new 'qa' session row, fire
    qa_started on the implement stream, and NOT pool the implement session."""
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
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(qa_turn_text, session_id="qa-session-id")]))

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

    # Implement session ends implemented, NOT pooled
    impl_row = db.get_session(conn, row_id)
    assert impl_row["phase"] == "implemented"
    assert impl_row["available_for_reuse"] == 0

    # A QA session row was created
    sessions = db.list_sessions_for_project(conn, project_id)
    qa_sessions = [s for s in sessions if s["session_type"] == "qa"]
    assert len(qa_sessions) == 1
    qa_row = qa_sessions[0]
    assert qa_row["phase"] == "qa_grilling"
    assert qa_row["claude_session_id"] == "qa-session-id"
    assert json.loads(qa_row["details_json"])["prd"] == {"number": 7, "title": "Tracked PRD"}

    # qa_started event published on implement stream before done
    impl_events = live_stream._buffers.get(row_id, [])
    qa_started_events = [e for e in impl_events if e.get("type") == "qa_started"]
    assert len(qa_started_events) == 1
    assert qa_started_events[0]["qa_card_id"] == qa_row["id"]
    assert impl_events[-1] == {"type": "done"}

    # The implement card's own tab is closed -- the new QA row starts fresh.
    assert row_id not in session_runner._pty_engines

    # The nested issues/questions structure was parsed from the turn's free
    # text (parse_qa_response) and published on the new QA session's stream
    # -- the qa_grilling JSON block itself only carried the {phase, prd} signal.
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


def test_start_implement_job_prefers_rhubarb_qa_file_over_terminal_text(client, tmp_path, monkeypatch):
    """PRD #123: a QA handoff turn whose terminal text has no recognizable
    QA session block still produces a full issues/questions structure when
    `.claude/rhubarb_qa.md` is present with valid content."""
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
    _write_question_file(
        cwd, "rhubarb_qa.md",
        'QA session for PRD 7: "Tracked PRD"\n\nIssue 8: "Child"\nQuestion 1: "Does it work?"\nRecommended text: "Yes."\n',
    )

    # Terminal text carries the handoff signal block but no recognizable QA session text.
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(_QA_BLOCK, session_id="qa-session-id")]))

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

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


def test_rhubarb_qa_file_is_not_deleted_merely_by_being_read(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    tracker = {
        "prd": {"number": 7, "title": "Tracked PRD"},
        "issues": [{"number": 8, "title": "Child", "summary": "does the thing", "acceptance_criteria": ["works"]}],
        "qa_changes": [], "status": "implemented",
    }
    claude_dir = Path(cwd) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "implement-tracker.json").write_text(json.dumps(tracker), encoding="utf-8")
    _write_question_file(cwd, "rhubarb_qa.md", 'QA session for PRD 7: "Tracked PRD"\n\nIssue 8: "Child"\nQuestion 1: "Does it work?"\n')

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(_QA_BLOCK, session_id="qa-session-id")]))

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

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


def test_malformed_rhubarb_qa_file_is_deleted_and_falls_back_to_terminal_text(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    tracker = {
        "prd": {"number": 7, "title": "Tracked PRD"},
        "issues": [{"number": 8, "title": "Child", "summary": "does the thing", "acceptance_criteria": ["works"]}],
        "qa_changes": [], "status": "implemented",
    }
    claude_dir = Path(cwd) / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "implement-tracker.json").write_text(json.dumps(tracker), encoding="utf-8")
    _write_question_file(cwd, "rhubarb_qa.md", "not a recognizable QA session format at all")

    qa_turn_text = (
        _QA_BLOCK + "\n\n"
        'QA session for PRD 7: "Tracked PRD"\n\n'
        'Issue 8: "Child"\n'
        'Question 1: "From terminal text"\n'
    )
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(qa_turn_text, session_id="qa-session-id")]))

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id, session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )
    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"][0]["questions"][0]["text"] == "From terminal text"
    assert not (Path(cwd) / ".claude" / "rhubarb_qa.md").exists()


def test_start_implement_job_uses_ollama_rescue_when_qa_regex_parser_finds_nothing(client, tmp_path, monkeypatch):
    """A QA handoff turn whose free text looks like it was trying to be a
    QA session (contains "QA session for PRD") but doesn't match the
    strict regex format must be rescued via Ollama, and a valid rescue
    result must be what gets published on the new QA session's stream."""
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

    malformed_qa_text = _QA_BLOCK + "\n\nQA session for PRD 7: this is malformed, no quoted title at all\n"
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(malformed_qa_text, session_id="qa-session-id")]))

    rescued = {
        "prd": {"number": 7, "title": "Tracked PRD"},
        "issues": [
            {"number": 8, "title": "Child", "questions": [{"id": "issue8-q1", "text": "Rescued question", "recommended_text": None}]}
        ],
    }
    seen = {}

    def fake_rescue(raw_text):
        seen["raw_text"] = raw_text
        return rescued

    monkeypatch.setattr(session_runner, "rescue_qa_response", fake_rescue)

    conn = db.get_connection()
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

    assert seen["raw_text"] == malformed_qa_text

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == rescued["issues"]


def test_start_implement_job_skips_ollama_rescue_when_declined(client, tmp_path, monkeypatch):
    """Issue #119: with `ollama_declined` true, a QA handoff turn whose free
    text looks like it was trying to be a QA session but doesn't match the
    strict regex format must NOT invoke the rescue function -- it falls back
    to the parser's original (empty) issues, same as Ollama being unavailable."""
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

    malformed_qa_text = _QA_BLOCK + "\n\nQA session for PRD 7: this is malformed, no quoted title at all\n"
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(malformed_qa_text, session_id="qa-session-id")]))

    def fake_rescue(raw_text):
        raise AssertionError("rescue must not be called when ollama_declined is true")

    monkeypatch.setattr(session_runner, "rescue_qa_response", fake_rescue)

    conn = db.get_connection()
    db.set_ollama_declined(conn, True)
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
    qa_events = live_stream._buffers.get(qa_row["id"], [])
    qa_turn_events = [e for e in qa_events if e.get("type") == "turn" and e.get("phase") == "qa_grilling"]
    assert qa_turn_events[0]["issues"] == []


def test_start_implement_job_uses_ollama_rescue_when_not_declined(client, tmp_path, monkeypatch):
    """Issue #119: with `ollama_declined` explicitly false, the QA rescue
    call still fires, unchanged from current behavior."""
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

    malformed_qa_text = _QA_BLOCK + "\n\nQA session for PRD 7: this is malformed, no quoted title at all\n"
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event(malformed_qa_text, session_id="qa-session-id")]))

    rescued = {
        "prd": {"number": 7, "title": "Tracked PRD"},
        "issues": [
            {"number": 8, "title": "Child", "questions": [{"id": "issue8-q1", "text": "Rescued question", "recommended_text": None}]}
        ],
    }
    monkeypatch.setattr(session_runner, "rescue_qa_response", lambda raw_text: rescued)

    conn = db.get_connection()
    db.set_ollama_declined(conn, False)
    row_id = db.create_session(
        conn, project_id,
        session_type="implement", phase="implementing",
        details={"prd": {"number": 7, "title": "Tracked PRD"}},
    )

    asyncio.run(session_runner.start_implement_job(row_id, 7, cwd=cwd))

    sessions = db.list_sessions_for_project(conn, project_id)
    qa_row = next(s for s in sessions if s["session_type"] == "qa")
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
    assert row_id not in session_runner._pty_engines


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
    assert qa_row_id not in session_runner._pty_engines


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
    assert row_id in session_runner._pty_engines


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
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: First question?")])
        return iter([_result_event("❓ **Q1** - **Scope**: Follow-up?")])

    fake_class = _mock_engine(monkeypatch, handler)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "tell me more", cwd=cwd))
    asyncio.run(session_runner.continue_session_job(row_id, "and more", cwd=cwd))

    # Exactly one PtyEngine was ever constructed for this card_id, reused
    # across all three turns.
    assert len(fake_class.instances) == 1
    assert fake_class.instances[0].started is True
    assert row_id in session_runner._pty_engines
    assert session_runner._pty_engines[row_id] is fake_class.instances[0]


def test_engine_reattaches_via_resume_when_continuing_an_existing_session_id(client, tmp_path, monkeypatch):
    """A brand-new card_id whose row already carries a claude_session_id
    (e.g. a reused pooled session) must construct its engine with
    `resume_session_id=`, not a fresh one."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    fake_class = _mock_engine(
        monkeypatch, lambda prompt, **kw: iter([_result_event("- Another question?", session_id="pooled-session")])
    )

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
    assert row_id not in session_runner._pty_engines
    # The turn's own engine is closed; the fresh one minted for pooling
    # (issue #136) is kept alive as this project's standby instead of also
    # being closed immediately -- it's what the next /do claims directly.
    assert fake_class.instances[0].closed is True
    assert fake_class.instances[1].closed is False
    assert session_runner._standby_engines[project_id][0] is fake_class.instances[1]


# ---------------------------------------------------------------------------
# Pre-warmed standby PtyEngine (issue #136): a project keeps one unclaimed,
# already-running engine ready so a new /do claims it directly instead of
# paying spawn latency inline.
# ---------------------------------------------------------------------------


def test_ensure_standby_engine_spawns_when_none_exists(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))

    assert len(fake_class.instances) == 1
    engine, model, effort = session_runner._standby_engines[project_id]
    assert engine is fake_class.instances[0]
    assert (model, effort) == ("claude-sonnet-5", "auto")


def test_ensure_standby_engine_is_a_no_op_when_a_live_matching_one_exists(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))

    assert len(fake_class.instances) == 1  # no second spawn


def test_ensure_standby_engine_replaces_a_dead_one(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))
    fake_class.instances[0].closed = True  # simulate a crash while unclaimed

    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))

    assert len(fake_class.instances) == 2
    assert session_runner._standby_engines[project_id][0] is fake_class.instances[1]


def test_claim_standby_engine_returns_it_on_a_match_and_removes_it(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))

    claimed = session_runner.claim_standby_engine(project_id, model="claude-sonnet-5", effort="auto")

    assert claimed is fake_class.instances[0]
    assert claimed.closed is False
    assert project_id not in session_runner._standby_engines


def test_claim_standby_engine_returns_none_and_discards_on_model_mismatch(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))

    claimed = session_runner.claim_standby_engine(project_id, model="claude-opus-5", effort="auto")

    assert claimed is None
    assert fake_class.instances[0].closed is True  # discarded, not left dangling
    assert project_id not in session_runner._standby_engines


def test_claim_standby_engine_returns_none_and_discards_on_effort_mismatch(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))

    claimed = session_runner.claim_standby_engine(project_id, model="claude-sonnet-5", effort="high")

    assert claimed is None
    assert fake_class.instances[0].closed is True


def test_claim_standby_engine_returns_none_when_dead(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))
    fake_class.instances[0].closed = True

    claimed = session_runner.claim_standby_engine(project_id, model="claude-sonnet-5", effort="auto")

    assert claimed is None
    assert project_id not in session_runner._standby_engines


def test_claim_standby_engine_returns_none_when_none_exists(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")
    assert session_runner.claim_standby_engine(project_id, model="claude-sonnet-5", effort="auto") is None


def test_close_standby_engine_closes_and_discards_it(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))

    session_runner.close_standby_engine(project_id)

    assert fake_class.instances[0].closed is True
    assert project_id not in session_runner._standby_engines


def test_close_standby_engine_does_not_raise_when_none_exists(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")
    session_runner.close_standby_engine(project_id)  # no-op, must not raise


def test_open_pty_tab_count_includes_standby_engines(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    _mock_engine(monkeypatch, lambda prompt, **kw: iter([]))

    assert session_runner.open_pty_tab_count() == 0
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))
    assert session_runner.open_pty_tab_count() == 1


def test_register_engine_makes_get_or_create_engine_reuse_it_without_spawning(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    fake_class = _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("hi")]))

    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-5", effort="auto"))
    claimed = session_runner.claim_standby_engine(project_id, model="claude-sonnet-5", effort="auto")
    session_runner.register_engine(999, claimed)

    engine = session_runner._get_or_create_engine(
        999, cwd=cwd, model="claude-sonnet-5", effort="auto", resume_session_id=None
    )

    assert engine is claimed
    assert len(fake_class.instances) == 1  # no second spawn triggered by _get_or_create_engine


# ---------------------------------------------------------------------------
# Argv-level end-to-end coverage (issue #138): every test above (and every
# other test in this file) mocks `session_runner.PtyEngine` wholesale with a
# fake that never runs `_build_args()` at all -- it only ever proves the
# right Python kwarg (`model=`/`effort=`) reached the constructor. These
# tests instead leave the REAL `PtyEngine` class in place and fake out only
# the OS-level pty spawn (`pty_engine._default_pty_factory`, the same
# injection seam `tests/test_pty_engine.py` uses directly), so the actual
# argv handed to "the subprocess" is captured and can be asserted on -- all
# the way through `/api/session/start`'s real endpoint logic (app.py) and
# `start_session_job` (session_runner.py). This is what catches a
# discrepancy the Python-kwarg-level mock structurally cannot.
# ---------------------------------------------------------------------------


class _ArgvCapturingBackend:
    """Minimal real-shaped `PtyBackend`: `read()` hands back a canned
    turn-complete response on its first call (ending `stream_turn` in one
    round trip) and never needs to be read again."""

    def __init__(self, result_text="- Only question?"):
        from rhubarb.pty_engine import TURN_COMPLETE_MARKER

        self._text = f"{result_text}\n{TURN_COMPLETE_MARKER}\n"
        self._served = False

    def write(self, data):
        return len(data)

    def read(self, size=4096):
        if not self._served:
            self._served = True
            return self._text
        raise EOFError

    def isalive(self):
        return not self._served

    def terminate(self, force=False):
        pass


def _capture_real_pty_spawns(monkeypatch):
    """Replace `pty_engine._default_pty_factory` (the seam every real,
    non-test `PtyEngine()` construction resolves its `pty_factory` through)
    with one that records every spawn's argv and hands back a
    `_ArgvCapturingBackend` instead of a real OS process. Returns the list of
    captured argvs, appended to in spawn order."""
    spawns = []

    def fake_factory(argv, *, cwd, env):
        spawns.append(argv)
        return _ArgvCapturingBackend()

    monkeypatch.setattr(pty_engine, "_default_pty_factory", lambda: fake_factory)
    return spawns


def test_brand_new_session_after_changing_model_spawns_with_the_new_model_via_standby_claim(
    client, tmp_path, monkeypatch
):
    """Issue #138's reported repro: pick a model in the UI, then start what
    looks like a brand-new session -- the Live Terminal must actually spawn
    `claude` with THAT model, not a stale one left over from a pre-warmed
    standby engine warmed under the model that was configured before the
    switch. Goes through the real `/api/session/start` endpoint logic
    (`app_module.start_session`) and `start_session_job`, with only the
    OS-level pty spawn faked -- so this exercises the real
    `claim_standby_engine`/`register_engine` reuse path and the real
    `PtyEngine._build_args()`, not a Python-kwarg-level mock."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    conn = db.get_connection()
    spawns = _capture_real_pty_spawns(monkeypatch)

    # A standby was pre-warmed (e.g. by `open_project`) under the model that
    # was configured at the time.
    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-4-6", effort="auto"))
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
    assert session_runner._pty_engines[card_id].model == "claude-opus-4-8"


def test_brand_new_session_after_changing_model_spawns_with_the_new_model_via_standby_match(
    client, tmp_path, monkeypatch
):
    """The mirror-image case: the standby's model still matches what's
    currently configured (no change happened, or the user picked the SAME
    model again) -- it must be adopted (`register_engine`) rather than
    discarded, and the argv it was ALREADY spawned with (captured back when
    the standby was warmed) must carry that same model. Confirms
    `claim_standby_engine`'s reuse path itself is argv-correct, not just its
    discard path."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)
    spawns = _capture_real_pty_spawns(monkeypatch)

    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-opus-4-8", effort="auto"))
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
    assert session_runner._pty_engines[card_id].model == "claude-opus-4-8"


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
    spawns = _capture_real_pty_spawns(monkeypatch)

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
    assert session_runner._pty_engines[card_id].model == "claude-opus-4-8"


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
    spawns = _capture_real_pty_spawns(monkeypatch)

    asyncio.run(session_runner.ensure_standby_engine(project_id, cwd=cwd, model="claude-sonnet-4-6", effort="auto"))
    assert len(spawns) == 1

    # Body omits "effort" entirely, exactly like a request built without the
    # dropdown's value ever being set.
    result = asyncio.run(_run_and_drain(app_module.start_session({"prompt": "a feature"})))
    card_id = result["card_id"]

    # The standby matched (no mismatch-triggered discard-and-respawn).
    assert len(spawns) == 1
    assert session_runner._pty_engines[card_id] is not None
    assert session_runner._pty_engines[card_id].effort == "auto"


# ---------------------------------------------------------------------------
# Crash routing (issue #87): PtyEngineUnrecoverableError -> the same
# blocked-card flow a genuine implement_blocked marker already uses.
# ---------------------------------------------------------------------------


def test_implement_turn_crash_routes_into_the_blocked_flow(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def crashing(prompt, **kw):
        raise PtyEngineUnrecoverableError("claude PTY process died twice in a row", claude_session_id="crashed-1")

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
    assert row_id not in session_runner._pty_engines


def test_implement_crash_recovers_via_the_same_reply_endpoint_as_a_real_block(client, tmp_path, monkeypatch):
    """After a crash-induced block, continue_implement_job (the same
    endpoint that resumes a genuine implement_blocked session) must resume
    this session too, reattaching via --resume at the crash's session id."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def crashing(prompt, **kw):
        raise PtyEngineUnrecoverableError("died twice", claude_session_id="crashed-1")

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


def test_grilling_turn_crash_routes_into_the_blocked_flow(client, tmp_path, monkeypatch):
    """The crash-routing mechanism is generic, not implement-specific --
    any phase's stream_turn call can raise PtyEngineUnrecoverableError."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def crashing(prompt, **kw):
        raise PtyEngineUnrecoverableError("died twice", claude_session_id="crashed-grilling")

    _mock_engine(monkeypatch, crashing)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "blocked"
    assert row["claude_session_id"] == "crashed-grilling"
    assert row_id not in session_runner._pty_engines


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
    assert session_runner._pty_engines[row_id] is fake_class.instances[0]


def test_finish_chain_does_not_publish_minimize_and_still_starts_implementing(client, tmp_path, monkeypatch):
    """Issue #146: the do-to-implement handoff no longer auto-minimizes the
    left card -- the frontend now shows a "Proceed" banner instead and only
    minimizes on an explicit click. `/rhubarb:implement` must still start
    immediately in the same tab/session regardless: the `implementing` phase
    (and, since this mock engine completes synchronously, the resulting
    `implement` session_type) must still show up, just with no `minimize`
    event anywhere in the stream."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("Wrote issues draft.")])
        return iter([_result_event("Implemented.")])

    _mock_engine(monkeypatch, handler)
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    monkeypatch.setattr(session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "PRD #1: p\nIssue #2: i")

    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    events = live_stream._buffers.get(row_id, [])
    event_types = [e["type"] for e in events]
    assert "minimize" not in event_types

    phases = [e["phase"] for e in events if e.get("type") == "phase"]
    assert "implementing" in phases

    row = db.get_session(conn, row_id)
    assert row["session_type"] == "implement"


def test_finish_chain_falls_back_to_pooling_when_no_prd_was_parsed(client, tmp_path, monkeypatch):
    """If parse_details comes up with no PRD number (e.g. an unexpected
    /rhubarb:to-issues result shape), the session must not get stuck --
    it falls back to the old clear-and-pool-immediately behavior."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    def handler(prompt, **kw):
        if prompt == "/rhubarb:do a feature":
            return iter([_result_event("❓ **Q1** - **Scope**: Only question?")])
        if prompt == "/rhubarb:to-prd":
            return iter([_result_event("Wrote PRD draft.")])
        if prompt == "/rhubarb:to-issues":
            return iter([_result_event("no PRD/issue numbers in here at all")])
        raise AssertionError(f"unexpected prompt {prompt!r} -- must not auto-continue without a PRD")

    _mock_engine(monkeypatch, handler, fresh_ids=["turn-engine", "s2"])
    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    monkeypatch.setattr(session_runner, "publish_draft", lambda draft_path, cwd, on_progress=None: "no PRD/issue numbers in here at all")

    asyncio.run(session_runner.continue_session_job(row_id, "", cwd=cwd, confirm_advance=True))

    row = db.get_session(conn, row_id)
    assert row["phase"] == "details"
    assert row["available_for_reuse"] == 1
    assert row["claude_session_id"] == "s2"

    events = live_stream._buffers.get(row_id, [])
    assert not any(e.get("type") == "minimize" for e in events)
    assert events[-1] == {"type": "done"}
    # No PRD to implement -- this card's tab was closed when pooled.
    assert row_id not in session_runner._pty_engines


def test_dismiss_error_notifications_clears_the_project_queue(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    session_runner.add_error_notification(project_id, 1, "implementing", "boom")
    assert len(session_runner.get_error_notifications(project_id)) == 1

    session_runner.dismiss_error_notifications(project_id)
    assert session_runner.get_error_notifications(project_id) == []


# ---------------------------------------------------------------------------
# Per-card_id turn lock (issue #144): a race condition where two overlapping
# calls for the same card_id could both write to and read from the same
# resident PtyEngine's stream_turn at once. `_run_turn` now checks a
# per-card_id `asyncio.Lock` before doing anything else; a call made while
# the lock is already held returns `None` immediately, touching neither the
# engine nor anything else.
# ---------------------------------------------------------------------------


def test_run_turn_lock_rejects_a_concurrent_call_for_the_same_card_id(client, tmp_path, monkeypatch):
    """The core invariant: of two overlapping `_run_turn` calls for the same
    card_id, only the first ever reaches the engine's `stream_turn` -- the
    second, made while the first is still mid-turn (blocked inside
    `stream_turn` via a controlled `asyncio.Event`), returns `None`
    immediately without constructing or touching the engine at all. Issue
    #149: that second, rejected call must also publish an explicit error
    `turn` event on this card's stream, tagged with the caller's own
    `phase`, instead of leaving no trace at all."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    entered = asyncio.Event()
    release = asyncio.Event()
    enter_count = {"n": 0}
    fake_class = _make_blocking_fake_engine_class(
        entered, release, enter_count, result_text="❓ **Q1** - **Scope**: Only question?"
    )
    monkeypatch.setattr(session_runner, "PtyEngine", fake_class)

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)

    async def scenario():
        first_task = asyncio.create_task(
            session_runner._run_turn(
                row_id, "first prompt", session_id=None, cwd=cwd, model=None, effort=None, phase="grilling"
            )
        )
        await entered.wait()  # first call is now mid-turn, blocked inside stream_turn

        # A second, overlapping call for the SAME card_id while the first
        # is still in flight -- must be rejected outright.
        second_result = await session_runner._run_turn(
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
    monkeypatch.setattr(session_runner, "PtyEngine", fake_class)

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
    assert row_id not in session_runner._pty_engines


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
        first = await session_runner._run_turn(
            row_id, "first", session_id=None, cwd=cwd, model=None, effort=None, phase="grilling"
        )
        second = await session_runner._run_turn(
            row_id, "second", session_id="s1", cwd=cwd, model=None, effort=None, phase="grilling"
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert first is not None
    assert second is not None
    assert not session_runner._get_turn_lock(row_id).locked()


def test_close_engine_removes_the_turn_lock(client, tmp_path, monkeypatch):
    """`_close_engine` must pop the card's entry out of `_turn_locks` too,
    alongside `_pty_engines` -- otherwise the lock registry grows
    unboundedly over a long-running instance."""
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    _mock_engine(monkeypatch, lambda prompt, **kw: iter([_result_event("❓ **Q1** - **Scope**: Only question?")]))

    conn = db.get_connection()
    row_id = db.create_session(conn, project_id)
    asyncio.run(session_runner.start_session_job(row_id, "a feature", cwd=cwd))

    assert row_id in session_runner._turn_locks

    session_runner._close_engine(row_id)

    assert row_id not in session_runner._turn_locks
