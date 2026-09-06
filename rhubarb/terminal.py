"""Open a native terminal window running a given command, cross-platform."""

import platform
import shutil
import subprocess


class NoTerminalFoundError(RuntimeError):
    pass


def open_terminal_running(command: str) -> None:
    system = platform.system()

    if system == "Windows":
        subprocess.Popen(["cmd", "/c", "start", "", "powershell", "-NoExit", "-Command", command])
        return

    if system == "Darwin":
        script = f'tell application "Terminal" to do script "{command}"'
        subprocess.Popen(["osascript", "-e", script])
        return

    for terminal, build_args in [
        ("x-terminal-emulator", lambda cmd: ["-e", cmd]),
        ("gnome-terminal", lambda cmd: ["--", "bash", "-c", f"{cmd}; exec bash"]),
        ("konsole", lambda cmd: ["-e", "bash", "-c", f"{cmd}; exec bash"]),
        ("xterm", lambda cmd: ["-e", "bash", "-c", f"{cmd}; exec bash"]),
    ]:
        path = shutil.which(terminal)
        if path:
            subprocess.Popen([path, *build_args(command)])
            return

    raise NoTerminalFoundError("No supported terminal emulator found on this system")
