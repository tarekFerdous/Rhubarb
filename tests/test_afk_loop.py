import asyncio
import time

from rhubarb import afk_loop, db, session_runner

from tests.test_sessions import _cwd_for, _open_project


def _stub_prds(*entries):
    """`entries` are (number, title, blocked) tuples -- build the shape
    `compute_prd_list` returns."""
    prds = [{"number": n, "title": t, "blocked": b} for n, t, b in entries]
    return lambda cwd: prds


def test_fires_after_threshold_with_an_unblocked_prd(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    afk_loop._last_activity[project_id] = time.monotonic() - 3700  # just over 1 hour ago

    calls = []

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        calls.append((pid, number, title, job_cwd))
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    asyncio.run(
        afk_loop.check_once(project_id, cwd, _stub_prds((5, "Top PRD", False), (6, "Other PRD", True)))
    )

    assert calls == [(project_id, 5, "Top PRD", cwd)]


def test_does_not_fire_with_zero_unblocked_prds(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    afk_loop._last_activity[project_id] = time.monotonic() - 3700

    calls = []

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        calls.append((pid, number, title, job_cwd))
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    # Only blocked entries -- must not fire regardless of elapsed time.
    asyncio.run(afk_loop.check_once(project_id, cwd, _stub_prds((6, "Other PRD", True))))
    assert calls == []

    # Empty list -- same result.
    asyncio.run(afk_loop.check_once(project_id, cwd, lambda cwd: []))
    assert calls == []


def test_manual_click_resets_the_clock(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    afk_loop._last_activity[project_id] = time.monotonic() - 3700  # past threshold

    calls = []

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        calls.append((pid, number, title, job_cwd))
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    # Simulate the manual-click endpoint's reset.
    afk_loop.record_activity(project_id)

    asyncio.run(afk_loop.check_once(project_id, cwd, _stub_prds((5, "Top PRD", False))))

    assert calls == []


def test_self_implement_firing_resets_the_clock(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    afk_loop._last_activity[project_id] = time.monotonic() - 3700

    calls = []

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        calls.append((pid, number, title, job_cwd))
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    prd_list = _stub_prds((5, "Top PRD", False))

    asyncio.run(afk_loop.check_once(project_id, cwd, prd_list))
    assert len(calls) == 1

    # Immediately re-check with no time advance -- the internal reset from
    # the first fire must prevent a second fire.
    asyncio.run(afk_loop.check_once(project_id, cwd, prd_list))
    assert len(calls) == 1


def test_switching_projects_scopes_the_clock(client, tmp_path, monkeypatch):
    project_a = _open_project(client, tmp_path, "proj-a")
    cwd_a = _cwd_for(project_a)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    afk_loop._last_activity[project_a] = time.monotonic() - 3700  # well past threshold

    calls = []

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        calls.append((pid, number, title, job_cwd))
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    project_b = project_a + 1  # never-before-seen project id
    assert project_b not in afk_loop._last_activity

    prd_list = _stub_prds((5, "Top PRD", False))

    before = time.monotonic()
    asyncio.run(afk_loop.check_once(project_b, cwd_a, prd_list))
    after = time.monotonic()

    # B gets a fresh clock starting now, so it must not fire yet.
    assert calls == []
    assert before <= afk_loop._last_activity[project_b] <= after

    # A's clock is untouched and still eligible.
    asyncio.run(afk_loop.check_once(project_a, cwd_a, prd_list))
    assert calls == [(project_a, 5, "Top PRD", cwd_a)]


def test_a_fire_adds_exactly_one_notification(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    afk_loop._last_activity[project_id] = time.monotonic() - 3700

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    asyncio.run(
        afk_loop.check_once(project_id, cwd, _stub_prds((5, "Top PRD", False), (6, "Other PRD", True)))
    )

    assert afk_loop.get_notifications(project_id) == [{"number": 5, "title": "Top PRD"}]


def test_a_second_fire_before_dismissal_appends_a_second_notification(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    prd_list = _stub_prds((5, "Top PRD", False))

    afk_loop._last_activity[project_id] = time.monotonic() - 3700
    asyncio.run(afk_loop.check_once(project_id, cwd, prd_list))

    # Age the clock past threshold again for a second, independent fire.
    afk_loop._last_activity[project_id] = time.monotonic() - 3700
    asyncio.run(afk_loop.check_once(project_id, cwd, prd_list))

    assert afk_loop.get_notifications(project_id) == [
        {"number": 5, "title": "Top PRD"},
        {"number": 5, "title": "Top PRD"},
    ]


def test_dismiss_notifications_empties_the_queue(client, tmp_path):
    project_id = _open_project(client, tmp_path, "proj")

    afk_loop.add_notification(project_id, 5, "Top PRD")
    assert afk_loop.get_notifications(project_id) == [{"number": 5, "title": "Top PRD"}]

    afk_loop.dismiss_notifications(project_id)

    assert afk_loop.get_notifications(project_id) == []


def test_busy_project_is_skipped_without_queueing_or_notifying(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    db.set_parallel_implementation(conn, False)

    # Simulate serial mode already having an active implement session for
    # this project -- the real `start_or_queue_implement` (not stubbed) must
    # hit its busy branch and, with `allow_queue=False`, skip rather than
    # enqueue.
    db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 99, "title": "Running PRD"}},
    )

    afk_loop._last_activity[project_id] = time.monotonic() - 3700  # past threshold

    asyncio.run(
        afk_loop.check_once(project_id, cwd, _stub_prds((5, "Top PRD", False), (6, "Other PRD", True)))
    )

    # No queue entry, no new implement session started -- only the
    # already-running one exists.
    assert session_runner._implement_queues.get(project_id, []) == []
    sessions = db.list_sessions_for_project(conn, project_id)
    implement_sessions = [s for s in sessions if s["session_type"] == "implement"]
    assert len(implement_sessions) == 1

    assert afk_loop.get_notifications(project_id) == []


def test_busy_project_still_resets_the_activity_clock(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)
    db.set_parallel_implementation(conn, False)

    db.create_session(
        conn,
        project_id,
        session_type="implement",
        phase="implementing",
        details={"prd": {"number": 99, "title": "Running PRD"}},
    )

    afk_loop._last_activity[project_id] = time.monotonic() - 3700  # past threshold

    before = time.monotonic()
    asyncio.run(afk_loop.check_once(project_id, cwd, _stub_prds((5, "Top PRD", False))))
    after = time.monotonic()

    # The countdown restarts even though the fire was skipped as busy, so a
    # subsequent 60s tick doesn't immediately retry.
    assert before <= afk_loop._last_activity[project_id] <= after


def test_a_fire_after_dismissal_produces_a_fresh_single_entry_list(client, tmp_path, monkeypatch):
    project_id = _open_project(client, tmp_path, "proj")
    cwd = _cwd_for(project_id)

    conn = db.get_connection()
    db.set_afk_hours(conn, 1)

    async def fake_start_or_queue_implement(pid, number, title, job_cwd, **kwargs):
        return {"card_id": 1}

    monkeypatch.setattr(session_runner, "start_or_queue_implement", fake_start_or_queue_implement)

    prd_list = _stub_prds((5, "Top PRD", False))

    afk_loop._last_activity[project_id] = time.monotonic() - 3700
    asyncio.run(afk_loop.check_once(project_id, cwd, prd_list))
    assert len(afk_loop.get_notifications(project_id)) == 1

    afk_loop.dismiss_notifications(project_id)
    assert afk_loop.get_notifications(project_id) == []

    afk_loop._last_activity[project_id] = time.monotonic() - 3700
    asyncio.run(afk_loop.check_once(project_id, cwd, prd_list))

    assert afk_loop.get_notifications(project_id) == [{"number": 5, "title": "Top PRD"}]
