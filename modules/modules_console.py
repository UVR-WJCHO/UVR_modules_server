from __future__ import annotations

import os
import select
import sys
from typing import Any, Optional, TextIO, Tuple

ConsoleState = Optional[Tuple[Any, int, Any]]


def enable_cbreak_stdin(stream: TextIO = sys.stdin) -> ConsoleState:
    """Enable single-key stdin reads on POSIX terminals when available."""
    try:
        import termios
        import tty

        fd = stream.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return termios, fd, old_settings
    except Exception:
        return None


def restore_stdin_cbreak(state: ConsoleState) -> None:
    if state is None:
        return
    termios, fd, old_settings = state
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        return


def read_key_nonblocking(stream: TextIO = sys.stdin, timeout: float = 0.01) -> str:
    if os.name == "nt":
        try:
            import msvcrt
        except Exception:
            return ""
        if not msvcrt.kbhit():
            return ""
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            if msvcrt.kbhit():
                msvcrt.getwch()
            return ""
        return key

    try:
        readable, _, _ = select.select([stream], [], [], float(timeout))
    except Exception:
        return ""
    if not readable:
        return ""
    try:
        return stream.read(1)
    except Exception:
        return ""
