import os
import pty
import select
import sys
import threading
import time

import pytest

from graybox.cli import _decode_key, _utf8_sequence_len


class TestUtf8SequenceLen:
    def test_ascii_single_byte(self):
        assert _utf8_sequence_len(ord("a")) == 1

    def test_two_byte_lead(self):
        assert _utf8_sequence_len(0xC3) == 2

    def test_three_byte_lead(self):
        assert _utf8_sequence_len(0xE4) == 3

    def test_four_byte_lead(self):
        assert _utf8_sequence_len(0xF0) == 4

    def test_continuation_byte_falls_back_to_one(self):
        assert _utf8_sequence_len(0xA1) == 1


class TestDecodeKey:
    def test_ascii(self):
        assert _decode_key(b"a") == "a"

    def test_accented_two_byte_char(self):
        assert _decode_key("á".encode("utf-8")) == "á"

    def test_three_byte_char(self):
        assert _decode_key("中".encode("utf-8")) == "中"

    def test_four_byte_char(self):
        assert _decode_key("🪴".encode("utf-8")) == "🪴"

    def test_incomplete_sequence_returns_empty(self):
        assert _decode_key(b"\xc3") == ""

    def test_invalid_byte_returns_empty(self):
        assert _decode_key(b"\xff") == ""


class TestInteractiveInputUtf8:
    @pytest.mark.skipif(sys.platform == "win32", reason="requires termios/pty")
    def test_accented_input_preserved_through_getch(self):
        master, slave = pty.openpty()
        collected: list[bytes] = []
        stop = threading.Event()

        def drain() -> None:
            while not stop.is_set():
                r, _, _ = select.select([master], [], [], 0.05)
                if not r:
                    continue
                try:
                    d = os.read(master, 4096)
                except (OSError, ValueError):
                    return
                if not d:
                    return
                collected.append(d)

        drainer = threading.Thread(target=drain, daemon=True)
        drainer.start()

        def typer() -> None:
            deadline = time.time() + 10
            while time.time() < deadline:
                if b"prompt>" in b"".join(collected):
                    break
                time.sleep(0.05)
            os.write(master, "água para as plantas".encode("utf-8"))
            time.sleep(0.2)
            os.write(master, b"\r")

        typer_thread = threading.Thread(target=typer, daemon=True)
        typer_thread.start()

        holder: dict[str, str] = {}
        import io

        def run_input() -> None:
            import sys as _sys

            from graybox.cli import _interactive_input

            stdin_wrapper = os.fdopen(os.dup(slave), "rb", buffering=0)
            stdout_wrapper = os.fdopen(os.dup(slave), "w", buffering=1)
            saved_in, saved_out = _sys.stdin, _sys.stdout
            _sys.stdin, _sys.stdout = stdin_wrapper, stdout_wrapper
            try:
                holder["val"] = _interactive_input("prompt> ")
            except Exception as exc:  # pragma: no cover
                holder["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                _sys.stdin, _sys.stdout = saved_in, saved_out
                stdin_wrapper.close()
                stdout_wrapper.close()

        input_thread = threading.Thread(target=run_input)
        input_thread.start()
        input_thread.join(15)
        stop.set()

        try:
            os.close(master)
        finally:
            os.close(slave)

        assert not input_thread.is_alive(), "interactive input hung"
        assert "error" not in holder, holder.get("error")
        assert holder["val"] == "água para as plantas"
