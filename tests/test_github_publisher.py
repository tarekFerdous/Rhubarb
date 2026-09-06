import json

import pytest

from rhubarb.github_publisher import GithubPublishError, publish_draft

_DRAFT = {
    "prd": {"title": "My PRD", "body": "PRD body", "labels": ["ready-for-agent"]},
    "issues": [
        {"title": "Child one", "body": "Body one", "labels": ["ready-for-agent"]},
        {"title": "Child two", "body": "Body two", "labels": []},
    ],
}


def _write_draft(tmp_path, draft=_DRAFT):
    path = tmp_path / "prd_draft.json"
    path.write_text(json.dumps(draft), encoding="utf-8")
    return path


def _ok(url):
    return type("Result", (), {"returncode": 0, "stdout": f"{url}\n", "stderr": ""})()


def _fail(stderr):
    return type("Result", (), {"returncode": 1, "stdout": "", "stderr": stderr})()


def test_publish_draft_happy_path(tmp_path, monkeypatch):
    draft_path = _write_draft(tmp_path)
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if "My PRD" in args:
            return _ok("https://github.com/x/y/issues/5")
        if "Child one" in args:
            return _ok("https://github.com/x/y/issues/6")
        if "Child two" in args:
            return _ok("https://github.com/x/y/issues/7")
        raise AssertionError(f"unexpected args {args}")

    monkeypatch.setattr("rhubarb.github_publisher.subprocess.run", fake_run)

    result = publish_draft(draft_path, cwd=str(tmp_path))

    assert result == "PRD #5: My PRD\nIssue #6: Child one\nIssue #7: Child two"
    assert len(calls) == 3
    assert calls[0][:3] == ["gh", "issue", "create"]
    assert "--title" in calls[0] and "My PRD" in calls[0]
    assert "--label" in calls[0] and "ready-for-agent" in calls[0]
    # Child two has no labels -- no --label flag for it.
    assert "--label" not in calls[2]
    assert not draft_path.exists()


def test_publish_draft_fails_on_prd_creation(tmp_path, monkeypatch):
    draft_path = _write_draft(tmp_path)

    def fake_run(args, **kwargs):
        return _fail("gh: not logged in")

    monkeypatch.setattr("rhubarb.github_publisher.subprocess.run", fake_run)

    with pytest.raises(GithubPublishError, match="not logged in"):
        publish_draft(draft_path, cwd=str(tmp_path))

    assert not draft_path.exists()


def test_publish_draft_fails_on_child_issue_after_prd_succeeds(tmp_path, monkeypatch):
    draft_path = _write_draft(tmp_path)

    def fake_run(args, **kwargs):
        if "My PRD" in args:
            return _ok("https://github.com/x/y/issues/5")
        return _fail("network error")

    monkeypatch.setattr("rhubarb.github_publisher.subprocess.run", fake_run)

    with pytest.raises(GithubPublishError, match="PRD #5") as exc_info:
        publish_draft(draft_path, cwd=str(tmp_path))

    assert "network error" in str(exc_info.value)
    assert not draft_path.exists()


def test_publish_draft_missing_file(tmp_path):
    draft_path = tmp_path / "does_not_exist.json"

    with pytest.raises(GithubPublishError):
        publish_draft(draft_path, cwd=str(tmp_path))

    assert not draft_path.exists()
