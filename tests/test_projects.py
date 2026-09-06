import subprocess

import pytest

from rhubarb.projects import get_current_branch, has_github_remote, scan_projects


def _init_repo(path, remote_url=None):
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "checkout", "-q", "-b", "main"], cwd=path, check=True)
    if remote_url:
        subprocess.run(["git", "remote", "add", "origin", remote_url], cwd=path, check=True)


@pytest.fixture
def root(tmp_path):
    (tmp_path / "plain_folder").mkdir()
    _init_repo(tmp_path / "repo_no_remote")
    _init_repo(tmp_path / "repo_github", "https://github.com/someone/repo.git")
    _init_repo(tmp_path / "repo_non_github", "https://gitlab.com/someone/repo.git")
    return tmp_path


def test_plain_folder_has_no_github_remote(root):
    assert has_github_remote(root / "plain_folder") is False


def test_repo_without_remote_is_excluded(root):
    assert has_github_remote(root / "repo_no_remote") is False


def test_repo_with_github_remote_is_included(root):
    assert has_github_remote(root / "repo_github") is True


def test_repo_with_non_github_remote_is_excluded(root):
    assert has_github_remote(root / "repo_non_github") is False


def test_scan_projects_only_returns_github_repos(root):
    found = {p["name"] for p in scan_projects(str(root))}
    assert found == {"repo_github"}


def test_scan_projects_reports_branch(root):
    [project] = scan_projects(str(root))
    assert project["branch"] == "main"
    assert get_current_branch(root / "repo_github") == "main"


def test_scan_projects_on_missing_root_returns_empty(tmp_path):
    assert scan_projects(str(tmp_path / "does-not-exist")) == []
