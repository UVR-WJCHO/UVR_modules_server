from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "modules") not in sys.path:
    sys.path.insert(0, str(ROOT / "modules"))

import modules_console  # noqa: E402


def test_enable_cbreak_stdin_ignores_non_tty_stream():
    assert modules_console.enable_cbreak_stdin(io.StringIO()) is None


def test_restore_stdin_cbreak_accepts_missing_state():
    modules_console.restore_stdin_cbreak(None)


def test_read_key_nonblocking_uses_msvcrt_on_windows(monkeypatch):
    class FakeMsvcrt:
        @staticmethod
        def kbhit():
            return True

        @staticmethod
        def getwch():
            return " "

    monkeypatch.setattr(modules_console.os, "name", "nt", raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", FakeMsvcrt)

    assert modules_console.read_key_nonblocking() == " "


def test_read_key_nonblocking_uses_select_on_posix(monkeypatch):
    stream = io.StringIO("x")
    monkeypatch.setattr(modules_console.os, "name", "posix", raising=False)
    monkeypatch.setattr(modules_console.select, "select", lambda r, w, e, timeout: ([stream], [], []))

    assert modules_console.read_key_nonblocking(stream=stream, timeout=0.0) == "x"
