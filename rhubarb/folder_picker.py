"""Native OS folder picker.

A browser `<input webkitdirectory>` picker can't hand back an absolute
filesystem path (only a relative `webkitRelativePath`), but the server needs
a real path to scope `cwd` for subprocess calls. `tkinter` is stdlib, so this
avoids pulling in a new dependency just for a folder dialog.
"""

import tkinter
from tkinter import filedialog


def pick_folder() -> str | None:
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title="Select your local root directory")
    finally:
        root.destroy()
    return path or None
