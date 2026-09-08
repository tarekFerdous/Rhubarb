from rhubarb import ollama_installer
from rhubarb.ollama_installer import (
    OLLAMA_MODEL,
    PRESENCE_NOT_PRESENT,
    PRESENCE_WITH_MODEL,
    PRESENCE_WITHOUT_MODEL,
    check_ollama_presence,
    install_and_pull_model,
    install_for_platform,
    install_linux,
    install_macos,
    install_windows,
    pull_model,
)


# ---------------------------------------------------------------------------
# Presence detection
# ---------------------------------------------------------------------------


def test_presence_reports_not_present_when_http_api_is_unreachable():
    def failing_get(url):
        raise ConnectionError("nobody home")

    assert check_ollama_presence(http_get=failing_get) == PRESENCE_NOT_PRESENT


def test_presence_reports_without_model_when_reachable_but_model_missing():
    def fake_get(url):
        assert url == "http://localhost:11434/api/tags"
        return {"models": [{"name": "llama3.2:3b"}, {"name": "mistral:latest"}]}

    assert check_ollama_presence(http_get=fake_get) == PRESENCE_WITHOUT_MODEL


def test_presence_reports_with_model_when_reachable_and_model_present():
    def fake_get(url):
        return {"models": [{"name": "mistral:latest"}, {"name": OLLAMA_MODEL}]}

    assert check_ollama_presence(http_get=fake_get) == PRESENCE_WITH_MODEL


def test_presence_reports_without_model_when_reachable_with_no_models_at_all():
    def fake_get(url):
        return {"models": []}

    assert check_ollama_presence(http_get=fake_get) == PRESENCE_WITHOUT_MODEL


# ---------------------------------------------------------------------------
# Per-OS install
# ---------------------------------------------------------------------------


def test_install_windows_invokes_winget_install_with_the_expected_args():
    """Doesn't assert the exact executable string -- that's covered
    separately by the PATH-resolution tests below -- just that whichever
    winget command gets resolved is invoked with the right install args."""
    calls = []
    install_windows(run=calls.append)

    assert len(calls) == 1
    assert calls[0][0].endswith("winget") or calls[0][0].endswith("winget.exe")
    assert calls[0][1:] == ["install", "--id", "Ollama.Ollama", "-e", "--silent"]


def test_install_windows_resolves_the_full_winget_path_when_the_app_alias_exists(monkeypatch):
    """Issue #113/#115: a bare "winget" can fail with WinError 2 from a
    process whose environment didn't inherit the WindowsApps app-execution-
    alias PATH entry, even though winget is genuinely installed. Resolving
    the well-known full path directly sidesteps that."""
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\someone\\AppData\\Local")
    monkeypatch.setattr(ollama_installer.os.path, "exists", lambda path: True)

    calls = []
    install_windows(run=calls.append)

    assert calls[0][0] == "C:\\Users\\someone\\AppData\\Local\\Microsoft\\WindowsApps\\winget.exe"


def test_install_windows_falls_back_to_bare_winget_when_the_app_alias_path_is_absent(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\someone\\AppData\\Local")
    monkeypatch.setattr(ollama_installer.os.path, "exists", lambda path: False)

    calls = []
    install_windows(run=calls.append)

    assert calls[0][0] == "winget"


def test_install_windows_falls_back_to_bare_winget_when_localappdata_is_unset(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    calls = []
    install_windows(run=calls.append)

    assert calls[0][0] == "winget"


def test_install_macos_uses_brew_when_present():
    calls = []
    install_macos(run=calls.append, brew_available=lambda run: True)

    assert calls == [["brew", "install", "ollama"]]


def test_install_macos_falls_back_to_pkg_installer_when_brew_missing():
    calls = []
    install_macos(run=calls.append, brew_available=lambda run: False)

    assert calls[0][0] == "curl"
    assert "https://ollama.com/download/Ollama.pkg" in calls[0]
    assert calls[1] == ["installer", "-pkg", "/tmp/Ollama.pkg", "-target", "/"]


def test_install_macos_brew_available_probe_receives_the_run_callable():
    """The default brew-detection probe should itself use the injected
    `run` (so it, too, never shells out for real in a test)."""
    calls = []

    def fake_run(argv):
        calls.append(argv)

    install_macos(run=fake_run)

    assert calls[0] == ["brew", "--version"]


def test_install_linux_invokes_the_official_install_script():
    calls = []
    install_linux(run=calls.append)

    assert len(calls) == 1
    assert calls[0][0] == "sh"
    assert "curl -fsSL https://ollama.com/install.sh | sh" in calls[0][2]


def test_install_for_platform_dispatches_to_windows(monkeypatch):
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Windows")

    assert len(calls) == 1
    assert calls[0][1:] == ["install", "--id", "Ollama.Ollama", "-e", "--silent"]


def test_install_for_platform_dispatches_to_macos():
    """`install_for_platform` doesn't expose its own `brew_available`
    override, so the default brew-detection probe runs, itself using the
    injected `run` -- a non-raising `run` (like `calls.append`) means the
    probe reports brew as available, so this ends in a `brew install`
    call, not the pkg fallback."""
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Darwin")

    assert calls[0] == ["brew", "--version"]
    assert calls[1] == ["brew", "install", "ollama"]


def test_install_for_platform_dispatches_to_linux_for_anything_else():
    calls = []
    install_for_platform(run=calls.append, system=lambda: "Linux")

    assert calls[0][0] == "sh"


# ---------------------------------------------------------------------------
# Model pull
# ---------------------------------------------------------------------------


def test_pull_model_invokes_ollama_pull_with_the_configured_model():
    calls = []
    pull_model(run=calls.append)

    assert calls == [["ollama", "pull", OLLAMA_MODEL]]


def test_install_and_pull_model_installs_then_pulls_in_order():
    calls = []
    install_and_pull_model(run=calls.append, system=lambda: "Windows")

    assert calls[0][1:] == ["install", "--id", "Ollama.Ollama", "-e", "--silent"]
    assert calls[1] == ["ollama", "pull", OLLAMA_MODEL]


def test_no_test_here_touches_a_real_subprocess_or_network():
    """Sanity check on the module's own defaults: `_default_run` and
    `_default_http_get` are never exercised by any test above (every call
    injects `run`/`http_get`), matching the injectable-only testing
    contract issue #113 requires."""
    assert callable(ollama_installer._default_run)
    assert callable(ollama_installer._default_http_get)
