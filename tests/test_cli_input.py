import os
import pty
import select
import sys
import threading
import time

import pytest

from graybox.cli import _decode_key, _utf8_sequence_len


class TestMenuDefaults:
    """Default menu behavior: capture pre-selected on open, append
    pre-selected after a capture."""

    def test_default_selection_is_capture_without_last_item(self):
        from graybox.cli import _default_menu_selection, _menu_options

        idx = _default_menu_selection(None)
        assert _menu_options(None)[idx][0] == "capture"

    def test_append_preselected_with_last_item(self):
        from graybox.cli import _default_menu_selection, _menu_options

        options = _menu_options("some-item-id")
        assert options[0][0] == "append"
        assert _default_menu_selection("some-item-id") == 0

    def test_import_is_a_plain_menu_item(self):
        from graybox.cli import _menu_options

        cmds = [c for c, _, _ in _menu_options(None)]
        assert "import" in cmds
        assert "capture" in cmds

    def test_every_menu_option_has_a_handler(self):
        """Every menu entry must be handled by _run_cli_command (no drift)."""
        import inspect

        from graybox.cli import _menu_options, _run_cli_command

        src = inspect.getsource(_run_cli_command)
        for name, _, _ in _menu_options():
            if name == "exit":
                continue
            assert f'cmd_name == "{name}"' in src, f"no handler for menu item {name}"


class TestInteractiveCaptureFlow:
    """The default capture flow must go straight to the note prompt —
    no "Press F to import a file" blocking notice before typing."""

    @pytest.mark.skipif(sys.platform == "win32", reason="requires termios/pty")
    def test_capture_goes_straight_to_note_prompt(self, temp_cfg):
        from graybox.cli import _capture_note_interactive

        master, slave = pty.openpty()
        collected: list[bytes] = []
        stop = threading.Event()

        def drain() -> None:
            while not stop.is_set():
                try:
                    r, _, _ = select.select([master], [], [], 0.05)
                except (OSError, ValueError):
                    return
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
                if b"Note text" in b"".join(collected):
                    break
                time.sleep(0.05)
            os.write(master, "água para as plantas".encode("utf-8"))
            time.sleep(0.2)
            os.write(master, b"\r")

        typer_thread = threading.Thread(target=typer, daemon=True)
        typer_thread.start()

        holder: dict = {}

        def run_capture() -> None:
            import sys as _sys

            stdin_wrapper = os.fdopen(os.dup(slave), "rb", buffering=0)
            stdout_wrapper = os.fdopen(os.dup(slave), "w", buffering=1)
            saved_in, saved_out = _sys.stdin, _sys.stdout
            _sys.stdin, _sys.stdout = stdin_wrapper, stdout_wrapper
            try:
                holder["item"] = _capture_note_interactive(temp_cfg)
            except Exception as exc:  # pragma: no cover
                holder["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                _sys.stdin, _sys.stdout = saved_in, saved_out
                stdin_wrapper.close()
                stdout_wrapper.close()

        input_thread = threading.Thread(target=run_capture)
        input_thread.start()
        input_thread.join(15)
        stop.set()

        try:
            os.close(master)
        finally:
            os.close(slave)

        assert not input_thread.is_alive(), "interactive capture hung"
        assert "error" not in holder, holder.get("error")
        assert b"Press F" not in b"".join(collected), "no F-import notice before typing"
        item = holder["item"]
        assert item is not None
        assert item.content == "água para as plantas"

    @pytest.mark.skipif(sys.platform == "win32", reason="requires termios/pty")
    def test_append_extends_last_note(self, temp_cfg):
        from graybox.cli import _append_note_interactive
        from graybox.capture import capture
        from graybox.storage import read_inbox_item

        item = capture(temp_cfg, "first thought")

        master, slave = pty.openpty()
        collected: list[bytes] = []
        stop = threading.Event()

        def drain() -> None:
            while not stop.is_set():
                try:
                    r, _, _ = select.select([master], [], [], 0.05)
                except (OSError, ValueError):
                    return
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
                if b"Add to last note" in b"".join(collected):
                    break
                time.sleep(0.05)
            os.write(master, "second thought".encode("utf-8"))
            time.sleep(0.2)
            os.write(master, b"\r")

        typer_thread = threading.Thread(target=typer, daemon=True)
        typer_thread.start()

        holder: dict = {}

        def run_append() -> None:
            import sys as _sys

            stdin_wrapper = os.fdopen(os.dup(slave), "rb", buffering=0)
            stdout_wrapper = os.fdopen(os.dup(slave), "w", buffering=1)
            saved_in, saved_out = _sys.stdin, _sys.stdout
            _sys.stdin, _sys.stdout = stdin_wrapper, stdout_wrapper
            try:
                holder["item"] = _append_note_interactive(temp_cfg, item.id)
            except Exception as exc:  # pragma: no cover
                holder["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                _sys.stdin, _sys.stdout = saved_in, saved_out
                stdin_wrapper.close()
                stdout_wrapper.close()

        input_thread = threading.Thread(target=run_append)
        input_thread.start()
        input_thread.join(15)
        stop.set()

        try:
            os.close(master)
        finally:
            os.close(slave)

        assert not input_thread.is_alive(), "interactive append hung"
        assert "error" not in holder, holder.get("error")
        loaded = read_inbox_item(temp_cfg, item.id)
        assert loaded is not None
        assert loaded.content.index("first thought") < loaded.content.index("second thought")


class TestTuiHomeOptions:
    """The Textual home menu exposes "append to last note" after a capture."""

    def test_capture_preselected_without_last_item(self):
        from graybox.tui_home import _home_options

        options = _home_options(None)
        cmds = [c for c, _ in options]
        assert "capture" in cmds
        assert "append" not in cmds

    def test_append_on_top_with_last_item(self):
        from graybox.tui_home import _home_options

        options = _home_options("some-item-id")
        assert options[0][0] == "append"

    def test_no_append_without_last_item(self):
        from graybox.tui_home import _home_options

        cmds = [c for c, _ in _home_options(None)]
        assert "append" not in cmds


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
                try:
                    r, _, _ = select.select([master], [], [], 0.05)
                except (OSError, ValueError):
                    return
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
