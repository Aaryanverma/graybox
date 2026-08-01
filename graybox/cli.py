from __future__ import annotations

import argparse
import os
import sys
from enum import Enum

from pyfiglet import Figlet

from graybox.ai import AIService
from graybox.capture import capture, capture_file
from graybox.config import load_config
from graybox.curate import (
    delete_page,
    edit_page,
    find_possible_duplicates,
    merge_pages,
)
from graybox.dashboard import write_dashboard
from graybox.summarizer import refresh_all_summaries
from graybox.embedding_index import ensure_indexed
from graybox.forget import forget_item
from graybox.organizer import organize_all
from graybox.retrieval import ask, ConversationTurn
from graybox.search import search_all
from graybox.storage import (
    ensure_workspace,
    list_inbox_items,
    list_pages,
    list_unprocessed,
    load_forgotten,
)
from graybox.workspace import Workspace
import logging

try:
    import readchar
except ImportError:  # pragma: no cover
    readchar = None

import itertools
import threading
import time

class ColorCodes:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    CYAN = "\x1b[36m"
    GREY = "\x1b[90m"


class Spinner:
    """Simple terminal spinner for long-running actions (LLM calls, etc.).

    Usage:
        with Spinner("Organizing inbox..."):
            do_the_slow_thing()

    Runs on a background thread so it doesn't block whatever it's wrapping.
    Safe to nest with the rest of the CLI's raw stdout writes since it always
    clears its own line before yielding control back.
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str, color: str = ColorCodes.CYAN):
        self.message = message
        self.color = color
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def _spin(self) -> None:
        for frame in itertools.cycle(self.FRAMES):
            if self._stop_event.is_set():
                break
            elapsed = time.time() - self._start_time
            sys.stdout.write(
                f"\r{self.color}{frame}{ColorCodes.RESET} {self.message} "
                f"{ColorCodes.DIM}({elapsed:.1f}s){ColorCodes.RESET}\x1b[K"
            )
            sys.stdout.flush()
            time.sleep(0.08)

    def __enter__(self) -> "Spinner":
        if not sys.stdout.isatty():
            # Non-interactive (piped/redirected) output: print once, no animation.
            print(f"{self.message}...")
            return self
        self._start_time = time.time()
        _hide_cursor()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        elapsed = time.time() - self._start_time
        sys.stdout.write("\r\x1b[K")
        if exc_type is None:
            sys.stdout.write(
                f"{ColorCodes.GREEN}✓{ColorCodes.RESET} {self.message} "
                f"{ColorCodes.DIM}({elapsed:.1f}s){ColorCodes.RESET}\n"
            )
        else:
            sys.stdout.write(
                f"{ColorCodes.RED}✗{ColorCodes.RESET} {self.message} "
                f"{ColorCodes.DIM}(failed after {elapsed:.1f}s){ColorCodes.RESET}\n"
            )
        sys.stdout.flush()
        _show_cursor()


class Key(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    ENTER = "ENTER"
    ESC = "ESC"
    BACKSPACE = "BACKSPACE"
    CTRL_C = "CTRL_C"


def _clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def _hide_cursor() -> None:
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()


def _show_cursor() -> None:
    sys.stdout.write("\x1b[?25h")
    sys.stdout.flush()


def supports_hyperlinks() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.environ.get("WT_SESSION"):
        return True
    if os.environ.get("TERM_PROGRAM") in {"vscode", "iTerm.app", "WezTerm", "ghostty"}:
        return True
    if os.environ.get("KITTY_WINDOW_ID"):
        return True
    if os.environ.get("VTE_VERSION"):
        return True
    return False


def hyperlink(text: str, url: str) -> str:
    if not supports_hyperlinks():
        return text
    OSC = "\x1b]"
    BEL = "\a"
    return f"{OSC}8;;{url}{BEL}{text}{OSC}8;;{BEL}"


def _pause(msg: str = "Press any key to return to menu...") -> None:
    print(f"\n{ColorCodes.DIM}{msg}{ColorCodes.RESET}")
    _getch()


def _normalize_readchar_key(k: str) -> Key | str:
    if readchar is not None:
        if k == getattr(readchar.key, "UP", object()):
            return Key.UP
        if k == getattr(readchar.key, "DOWN", object()):
            return Key.DOWN
        if k == getattr(readchar.key, "LEFT", object()):
            return Key.LEFT
        if k == getattr(readchar.key, "RIGHT", object()):
            return Key.RIGHT
        if k == getattr(readchar.key, "ENTER", object()) or k in ("\n", "\n"):
            return Key.ENTER
        if k == getattr(readchar.key, "ESC", object()) or k == "\x1b":
            return Key.ESC
        if k == getattr(readchar.key, "BACKSPACE", object()) or k in ("\b", "", "\b"):
            return Key.BACKSPACE
    return k


def _getch() -> Key | str:
    # Prefer the OS-native readers below over `readchar`: readchar.readkey()
    # does a *blocking* read to disambiguate a lone ESC from the start of an
    # arrow-key sequence (both start with \x1b), so a bare Esc keypress hangs
    # forever instead of cancelling. The termios/msvcrt paths use a short
    # select()/prefix-byte check instead, so ESC-alone resolves immediately.
    try:
        import msvcrt

        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            code = msvcrt.getch()
            mapping = {b"H": Key.UP, b"P": Key.DOWN, b"K": Key.LEFT, b"M": Key.RIGHT}
            return mapping.get(code, "")
        if ch == b"\n":
            return Key.ENTER
        if ch == b"\x1b":
            return Key.ESC
        if ch in (b"\b",):
            return Key.BACKSPACE
        return ch.decode("utf-8", "ignore")
    except ImportError:
        pass

    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = os.read(fd, 1)
            if ch != b"\x1b":
                if ch in (b"\n", b"\r"):
                    return Key.ENTER
                if ch in (b"\b", b""):
                    return Key.BACKSPACE
                return ch.decode("utf-8", "ignore")

            seq = b"\x1b"
            while True:
                r, _, _ = select.select([fd], [], [], 0.01)
                if not r:
                    break
                seq += os.read(fd, 1)

            if seq == b"\x1b":
                return Key.ESC
            decoded = seq.decode("utf-8", "ignore")
            if decoded == "\x1b[A":
                return Key.UP
            if decoded == "\x1b[B":
                return Key.DOWN
            if decoded == "\x1b[C":
                return Key.RIGHT
            if decoded == "\x1b[D":
                return Key.LEFT
            return decoded
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except ImportError:
        pass

    if readchar is not None:
        try:
            return _normalize_readchar_key(readchar.readkey())
        except Exception:
            pass
    return ""


class MockArgs:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _active_workspace_line(cfg) -> str:
    ws = cfg.workspace_manager.current()
    parts = [
        f"{ColorCodes.BOLD}Active workspace:{ColorCodes.RESET} {ColorCodes.CYAN}{ws.name}{ColorCodes.RESET}"
    ]
    if ws.description:
        parts.append(f"{ColorCodes.DIM}{ws.description}{ColorCodes.RESET}")
    parts.append(f"{ColorCodes.DIM}({ws.id}){ColorCodes.RESET}")
    return "  ".join(parts)


def _render_home_banner(cfg) -> None:
    linked = "https://www.linkedin.com/in/aaryanverma"
    fig = Figlet(font="slant", width=200)
    banner = fig.renderText("GRAY BOX").rstrip()
    print(f"{ColorCodes.GREY}{ColorCodes.BOLD}{banner}{ColorCodes.RESET}")
    author = hyperlink("Aaryan Verma", linked)
    print()
    print(
        f"{ColorCodes.DIM}Made with ♥ by {author}{ColorCodes.RESET}"
    )
    print(f"{ColorCodes.DIM}{'─' * 90}{ColorCodes.RESET}")
    print(f"{_active_workspace_line(cfg)}")
    print(f"{ColorCodes.DIM}{'─' * 90}{ColorCodes.RESET}\n")


def _move_cursor_up(lines: int) -> None:
    if lines > 0:
        sys.stdout.write(f"\x1b[{lines}A")
        sys.stdout.flush()


def _render_menu(selected: int, options: list[tuple[str, str, str]]) -> None:
    for i, (cmd, desc, icon) in enumerate(options):
        if i == selected:
            print(
                f"  {ColorCodes.BLUE}❯{ColorCodes.RESET} {icon}  "
                f"{ColorCodes.BOLD}{ColorCodes.BLUE}{cmd:<18}{ColorCodes.RESET}  "
                f"{ColorCodes.DIM}{desc}{ColorCodes.RESET}"
            )
        else:
            print(f"    {icon}  {cmd:<18}  {ColorCodes.DIM}{desc}{ColorCodes.RESET}")
    sys.stdout.flush()


def _interactive_input(prompt: str) -> str | None:
    print(prompt, end="", flush=True)
    buf: list[str] = []
    while True:
        ch = _getch()
        if ch == Key.ESC:
            print()
            return None
        if ch == Key.ENTER:
            print()
            return "".join(buf)
        if ch == Key.BACKSPACE:
            if buf:
                buf.pop()
                sys.stdout.write("\b \b")
                sys.stdout.flush()
            continue
        if ch == Key.CTRL_C:
            raise KeyboardInterrupt
        if isinstance(ch, str) and len(ch) == 1 and ch.isprintable():
            buf.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()


def _pick_workspace(cfg, prompt: str = "Select workspace") -> Workspace | None:
    workspaces = cfg.workspace_manager.list()
    if not workspaces:
        return None

    selected = 0
    active_id = cfg.workspace_manager.current().id
    for i, ws in enumerate(workspaces):
        if ws.id == active_id:
            selected = i
            break

    while True:
        _clear_screen()
        _render_home_banner(cfg)
        print(f"{ColorCodes.BOLD}{prompt}{ColorCodes.RESET}\n")
        for i, ws in enumerate(workspaces):
            active = "  "
            if i == selected:
                active = f"{ColorCodes.BLUE}❯{ColorCodes.RESET}"
            marker = (
                f"{ColorCodes.GREEN}•{ColorCodes.RESET}" if ws.id == active_id else " "
            )
            desc = ws.description or "No description"
            path = str(ws.root)
            print(
                f"{active} {marker} {ColorCodes.BOLD}{ws.name:<20}{ColorCodes.RESET} {ColorCodes.DIM}{desc}{ColorCodes.RESET}"
            )
            print(f"    {ColorCodes.DIM}{path}{ColorCodes.RESET}")
        print(
            f"\n{ColorCodes.DIM}Use arrow keys, Enter to select, Esc to cancel.{ColorCodes.RESET}"
        )
        ch = _getch()
        if ch in (Key.UP, "k"):
            selected = (selected - 1) % len(workspaces)
        elif ch in (Key.DOWN, "j"):
            selected = (selected + 1) % len(workspaces)
        elif ch == Key.ENTER:
            return workspaces[selected]
        elif ch in (Key.ESC, "q"):
            return None


def _run_cli_command(name: str, config_path: str | None) -> None:
    args = MockArgs(
        config=config_path,
        dry_run=False,
        type=None,
        top_k=10,
        threshold=None,
        file=None,
        text=None,
        question=None,
        query=None,
        all=False,
    )
    print(
        f"{ColorCodes.BLUE}◆ Gray Box{ColorCodes.RESET} {ColorCodes.DIM}› {name}{ColorCodes.RESET}\n"
    )
    if name == "status":
        cmd_status(args)
    elif name == "capture":
        print(
            f"{ColorCodes.DIM}Press {ColorCodes.RESET}{ColorCodes.BOLD}F{ColorCodes.RESET}"
            f"{ColorCodes.DIM} to import a file, or any other key to type a note directly "
            f"(Esc to cancel){ColorCodes.RESET}"
        )
        choice = _getch()
        if choice == Key.ESC:
            return
        if isinstance(choice, str) and choice.lower() == "f":
            path = _interactive_input(
                f"{ColorCodes.BOLD}File path {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
            )
            if path is not None and path.strip():
                args.file = path.strip()
                cmd_capture(args)
        else:
            text = _interactive_input(
                f"{ColorCodes.BOLD}Note text {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
            )
            if text is not None and text.strip():
                args.text = text.strip()
                cmd_capture(args)
    elif name == "organize":
        cmd_organize(args)
    elif name == "ask":
        q = _interactive_input(
            f"{ColorCodes.BOLD}Question {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
        )
        if q is not None and q.strip():
            args.question = q.strip()
            cmd_ask(args)
    elif name == "search":
        q = _interactive_input(
            f"{ColorCodes.BOLD}Search query {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
        )
        if q is not None and q.strip():
            args.query = q.strip()
            cmd_search(args)
    elif name == "pages":
        cmd_pages(args)
    elif name == "dupes":
        cmd_dupes(args)
    elif name == "dashboard":
        cmd_dashboard(args)
    elif name == "switch-workspace":
        cmd_workspace_switch(args)
    elif name == "create-workspace":
        cmd_workspace_create(args)


def cmd_capture(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    item = (
        capture_file(cfg, args.file)
        if args.file
        else capture(cfg, args.text or sys.stdin.read())
    )
    print(
        f"{ColorCodes.GREEN}✓ Captured{ColorCodes.RESET} {ColorCodes.DIM}→{ColorCodes.RESET} {ColorCodes.CYAN}inbox/{item.id}.md{ColorCodes.RESET}"
    )


def cmd_organize(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    llm = AIService(cfg)
    if args.dry_run:
        print(
            f"{ColorCodes.YELLOW}⚠️  [Dry-Run] No files will be written to disk; items stay unprocessed.{ColorCodes.RESET}\n"
        )
    with Spinner("Organizing inbox"):
        report = organize_all(cfg, llm, dry_run=args.dry_run)
    for entry in report["processed"]:
        pages = ", ".join(entry["pages"]) or "(no entities extracted)"
        print(
            f"{ColorCodes.GREEN}✓{ColorCodes.RESET} {entry['item']} {ColorCodes.DIM}→{ColorCodes.RESET} {pages}"
        )
    for entry in report["errors"]:
        print(
            f"{ColorCodes.RED}✗ Error on {entry['item']}:{ColorCodes.RESET} "
            f"{ColorCodes.DIM}{entry['error']}{ColorCodes.RESET}",
            file=sys.stderr,
        )
    verb = "Would process" if args.dry_run else "Processed"
    print(
        f"\n{ColorCodes.BOLD}✨ {verb}: {ColorCodes.CYAN}{len(report['processed'])}{ColorCodes.RESET}{ColorCodes.BOLD} items{ColorCodes.RESET} {ColorCodes.DIM}(Errors: {len(report['errors'])}){ColorCodes.RESET}"
    )

def cmd_ask(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    llm = AIService(cfg)
    with Spinner("Thinking"):
        answer = ask(cfg, llm, args.question, all_workspaces=args.all)
    print(f"\n{ColorCodes.BOLD}✨{ColorCodes.RESET} {answer.text}\n")
    if answer.sources:
        print(f"{ColorCodes.DIM}Sources: {', '.join(answer.sources)}{ColorCodes.RESET}")
    if answer.fallback:
        print(
            f"\n{ColorCodes.YELLOW}⚠️  Warning: This answer came from raw captures, not an organized wiki page.{ColorCodes.RESET}"
        )
        print(
            f"{ColorCodes.YELLOW}   Consider re-running 'organize' or reviewing the relevant page's extraction.{ColorCodes.RESET}"
        )

def cmd_chat(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    llm = AIService(cfg)
    history: list[ConversationTurn] = []

    print(
        f"{ColorCodes.BOLD}Chat mode{ColorCodes.RESET} "
        f"{ColorCodes.DIM}— ask follow-ups in context."
        f"Type 'exit' to leave.{ColorCodes.RESET}\n"
    )

    while True:
        try:
            q = _interactive_input(
                f"{ColorCodes.BOLD}You {ColorCodes.DIM}(Type exit to end chat){ColorCodes.RESET}: "
            )
        except (EOFError, KeyboardInterrupt):
            break
        if q is None:
            break
        q = q.strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        with Spinner("Thinking"):
            answer = ask(cfg, llm, q, all_workspaces=args.all, history=history)

        print(f"\n{ColorCodes.BOLD}✨{ColorCodes.RESET} {answer.text}\n")
        if answer.sources:
            print(f"{ColorCodes.DIM}Sources: {', '.join(answer.sources)}{ColorCodes.RESET}")
        if answer.fallback:
            print(
                f"{ColorCodes.YELLOW}⚠️  This answer came from raw captures, not an organized wiki page.{ColorCodes.RESET}"
            )
        print()

        history.append(ConversationTurn(question=q, answer=answer.text))

    print(f"{ColorCodes.DIM}Chat ended ({len(history)} exchange(s)).{ColorCodes.RESET}")


def cmd_search(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    wiki_hits, inbox_hits = search_all(
        cfg, args.query, top_k=args.top_k, all_workspaces=args.all
    )

    if wiki_hits:
        print(
            f"\n{ColorCodes.BOLD}Search results for '{args.query}':{ColorCodes.RESET}\n"
        )
        for h in wiki_hits:
            prefix = f"[{h.workspace_id}] " if h.workspace_id else ""
            print(
                f"{ColorCodes.DIM}[{h.score:>4.2f}]{ColorCodes.RESET} {ColorCodes.CYAN}{prefix + h.doc.search_id:<34}{ColorCodes.RESET} {h.doc.page.title}"
            )
        print()
        return

    if inbox_hits:
        print(
            f"\n{ColorCodes.BOLD}Search results for '{args.query}' (from raw captures):{ColorCodes.RESET}\n"
        )
        for h in inbox_hits:
            prefix = f"[{h.workspace_id}] " if h.workspace_id else ""
            excerpt = (
                (h.doc.item.content[:57] + "...")
                if len(h.doc.item.content) > 60
                else h.doc.item.content
            )
            print(
                f"{ColorCodes.DIM}[{h.score:>4.2f}]{ColorCodes.RESET} {ColorCodes.YELLOW}{prefix}inbox/{h.doc.item.id:<28}{ColorCodes.RESET} {excerpt}"
            )
        print(
            f"\n{ColorCodes.YELLOW}⚠️  These results came from raw captures, not organized wiki pages.{ColorCodes.RESET}"
        )
        print(
            f"{ColorCodes.YELLOW}   Consider running 'organize' to turn them into structured pages.{ColorCodes.RESET}\n"
        )
        return

    print(f"{ColorCodes.DIM}No matching pages or raw notes found.{ColorCodes.RESET}")


def cmd_pages(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    pages = list_pages(cfg, args.type)
    print()
    for p in pages:
        status = (
            f" {ColorCodes.DIM}·{ColorCodes.RESET} {ColorCodes.YELLOW}{p.status}{ColorCodes.RESET}"
            if p.status
            else ""
        )
        print(f"{ColorCodes.CYAN}{p.ref:<28}{ColorCodes.RESET} {p.title}{status}")
    print(
        f"\n{ColorCodes.BOLD}Total:{ColorCodes.RESET} {ColorCodes.CYAN}{len(pages)}{ColorCodes.RESET} {ColorCodes.DIM}page(s){ColorCodes.RESET}\n"
    )


def cmd_status(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    inbox = list_inbox_items(cfg)
    unprocessed = list_unprocessed(cfg)
    pages = list_pages(cfg)
    forgotten = load_forgotten(cfg)
    ws = cfg.workspace_manager.current()

    processed_count = len(inbox) - len(unprocessed)
    unprocessed_count = len(unprocessed)

    print(f"\n{ColorCodes.BOLD}Workspace Status{ColorCodes.RESET}\n")
    print(f"  {ColorCodes.BLUE}Root:{ColorCodes.RESET}    {cfg.root}")
    print(
        f"  {ColorCodes.BLUE}Active:{ColorCodes.RESET}  {ws.name} {ColorCodes.DIM}({ws.id}){ColorCodes.RESET}"
    )
    print(f"  {ColorCodes.BLUE}Path:{ColorCodes.RESET}    {cfg.workspace}")
    print(f"  {ColorCodes.BLUE}Workspace root:{ColorCodes.RESET} {ws.root}")
    print(
        f"  {ColorCodes.GREEN}Inbox:{ColorCodes.RESET}   {ColorCodes.BOLD}{unprocessed_count} unorganized{ColorCodes.RESET}, {processed_count} organized "
        f"{ColorCodes.DIM}({len(inbox)} total){ColorCodes.RESET}"
    )
    print(f"  {ColorCodes.CYAN}Pages:{ColorCodes.RESET}   {len(pages)} organized pages")
    print(f"  {ColorCodes.YELLOW}LLM:{ColorCodes.RESET}     {cfg.llm.model_name}")
    if forgotten:
        print(
            f"  {ColorCodes.RED}Forgotten:{ColorCodes.RESET} {len(forgotten)} item(s) {ColorCodes.DIM}(excluded from counts above){ColorCodes.RESET}"
        )
    print()


def cmd_forget(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    try:
        report = forget_item(
            cfg, args.item_id, purge=args.purge, scrub=args.scrub, reason=args.reason
        )
    except ValueError as e:
        print(f"{ColorCodes.RED}✗ {e}{ColorCodes.RESET}", file=sys.stderr)
        return
    print(f"{ColorCodes.GREEN}✓ Forgotten:{ColorCodes.RESET} inbox/{report['item_id']}")
    if report["purged"]:
        print(
            f"{ColorCodes.YELLOW}  Raw file deleted from disk (irreversible).{ColorCodes.RESET}"
        )
    if report["already_processed"]:
        pages = ", ".join(report["touched_pages"]) or "(none recorded)"
        print(f"{ColorCodes.DIM}  Already organized into: {pages}{ColorCodes.RESET}")
        if report["scrubbed_pages"]:
            print(
                f"{ColorCodes.GREEN}  Scrubbed its notes from:{ColorCodes.RESET} {', '.join(report['scrubbed_pages'])}"
            )
        elif not args.scrub:
            print(
                f"{ColorCodes.YELLOW}  Tip: re-run with --scrub to also strip its notes from those pages.{ColorCodes.RESET}"
            )


def cmd_dupes(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    threshold = (
        args.threshold if args.threshold is not None else cfg.retrieval.dedup_threshold
    )
    candidates = find_possible_duplicates(cfg, page_type=args.type, threshold=threshold)
    if not candidates:
        print(
            f"{ColorCodes.DIM}No likely duplicates found (threshold {threshold}).{ColorCodes.RESET}"
        )
        return
    print(f"\n{ColorCodes.BOLD}Possible duplicates:{ColorCodes.RESET}\n")
    for c in candidates:
        print(
            f"{ColorCodes.YELLOW}[{c.similarity:.2f}]{ColorCodes.RESET} "
            f'{ColorCodes.CYAN}{c.page_a.ref:<24}{ColorCodes.RESET} "{c.page_a.title}"  '
            f"{ColorCodes.DIM}~{ColorCodes.RESET}  "
            f'{ColorCodes.CYAN}{c.page_b.ref:<24}{ColorCodes.RESET} "{c.page_b.title}"  '
            f"{ColorCodes.DIM}({c.reason}){ColorCodes.RESET}"
        )
    print(
        f"\n{ColorCodes.DIM}This only flags candidates - nothing is merged automatically.{ColorCodes.RESET}"
    )
    print(
        f"{ColorCodes.DIM}Review, then fix with: graybox merge <keep-ref> <drop-ref>{ColorCodes.RESET}\n"
    )


def cmd_merge(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    try:
        report = merge_pages(
            cfg, args.primary_ref, args.secondary_ref, dry_run=args.dry_run
        )
    except ValueError as e:
        print(f"{ColorCodes.RED}✗ {e}{ColorCodes.RESET}", file=sys.stderr)
        return
    verb = "Would merge" if args.dry_run else "Merged"
    print(
        f"{ColorCodes.GREEN}✓ {verb}:{ColorCodes.RESET} {report['secondary']} "
        f"{ColorCodes.DIM}→{ColorCodes.RESET} {report['merged_into']} "
        f"{ColorCodes.DIM}({report['notes_before']}→{report['notes_after']} notes, "
        f"{report['sources_before']}→{report['sources_after']} sources){ColorCodes.RESET}"
    )
    if report["rewired_pages"]:
        print(
            f"{ColorCodes.DIM}  Rewired references in: {', '.join(report['rewired_pages'])}{ColorCodes.RESET}"
        )


def cmd_edit(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    try:
        report = edit_page(
            cfg,
            args.ref,
            new_title=args.title,
            new_type=args.new_type,
            new_status=args.status,
            add_aliases=args.alias,
            dry_run=args.dry_run,
        )
    except ValueError as e:
        print(f"{ColorCodes.RED}✗ {e}{ColorCodes.RESET}", file=sys.stderr)
        return
    verb = "Would update" if args.dry_run else "Updated"
    if report["moved"]:
        print(
            f"{ColorCodes.GREEN}✓ {verb}:{ColorCodes.RESET} {report['old_ref']} {ColorCodes.DIM}→{ColorCodes.RESET} {report['new_ref']}"
        )
    else:
        print(f"{ColorCodes.GREEN}✓ {verb}:{ColorCodes.RESET} {report['old_ref']}")
    if report["rewired_pages"]:
        print(
            f"{ColorCodes.DIM}  Rewired references in: {', '.join(report['rewired_pages'])}{ColorCodes.RESET}"
        )


def cmd_delete(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    try:
        report = delete_page(cfg, args.ref, dry_run=args.dry_run)
    except ValueError as e:
        print(f"{ColorCodes.RED}✗ {e}{ColorCodes.RESET}", file=sys.stderr)
        return
    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"{ColorCodes.GREEN}✓ {verb}:{ColorCodes.RESET} {report['ref']}")
    if report["sources"]:
        print(
            f"{ColorCodes.DIM}  Traced back to: {', '.join('inbox/' + s for s in report['sources'])}{ColorCodes.RESET}"
        )
    if report["rewired_pages"]:
        print(
            f"{ColorCodes.DIM}  Removed dangling links from: {', '.join(report['rewired_pages'])}{ColorCodes.RESET}"
        )

def cmd_rebuild_index(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    if not getattr(cfg.embeddings, "enabled", False):
        print(
            f"{ColorCodes.YELLOW}⚠️  Embeddings are disabled in config.{ColorCodes.RESET} "
            f"{ColorCodes.DIM}Set embeddings.enabled: true to use semantic search.{ColorCodes.RESET}"
        )
        return
    llm = AIService(cfg)
    pages = list_pages(cfg, args.type)
    indexed = 0
    errors = 0
    for p in pages:
        try:
            with Spinner(f"Indexing {p.ref}"):
                did_index = ensure_indexed(cfg, p, llm)
            if did_index:
                indexed += 1
            else:
                print(f"{ColorCodes.DIM}  {p.ref} (skipped){ColorCodes.RESET}")
        except Exception as e:
            errors += 1
            print(f"{ColorCodes.RED}✗ {p.ref}:{ColorCodes.RESET} {ColorCodes.DIM}{e}{ColorCodes.RESET}", file=sys.stderr)
    print(
        f"{ColorCodes.BOLD}✨ Indexed:{ColorCodes.RESET} {ColorCodes.CYAN}{indexed}{ColorCodes.RESET} "
        f"{ColorCodes.DIM}(Errors: {errors}, Total: {len(pages)}){ColorCodes.RESET}"
    )

def cmd_refresh(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    llm = AIService(cfg)
    if args.dry_run:
        print(
            f"{ColorCodes.YELLOW}⚠️  [Dry-Run] No files will be written.{ColorCodes.RESET}\n"
        )
    with Spinner("Refreshing summaries"):
        report = refresh_all_summaries(
            cfg, llm, page_type=args.type, dry_run=args.dry_run, min_notes=args.min_notes
        )
    for r in report["refreshed"]:
        print(
            f"{ColorCodes.GREEN}✓{ColorCodes.RESET} {ColorCodes.CYAN}{r.ref}{ColorCodes.RESET} "
            f"{ColorCodes.DIM}(cost: ${r.cost:.4f}){ColorCodes.RESET}"
        )
        if args.verbose:
            print(f"   {ColorCodes.DIM}Old:{ColorCodes.RESET} {r.old_summary}")
            print(f"   {ColorCodes.DIM}New:{ColorCodes.RESET} {r.new_summary}")
    if report["errors"]:
        for e in report["errors"]:
            print(
                f"{ColorCodes.RED}✗ {e['ref']}:{ColorCodes.RESET} {ColorCodes.DIM}{e['error']}{ColorCodes.RESET}",
                file=sys.stderr,
            )
    verb = "Would refresh" if args.dry_run else "Refreshed"
    print(
        f"\n{ColorCodes.BOLD}✨ {verb}: {ColorCodes.CYAN}{len(report['refreshed'])}{ColorCodes.RESET}{ColorCodes.BOLD} page(s){ColorCodes.RESET} "
        f"{ColorCodes.DIM}(Skipped: {report['skipped']}, Errors: {len(report['errors'])}, Cost: ${report['total_cost']:.4f}){ColorCodes.RESET}"
    )

def cmd_dashboard(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    path = write_dashboard(cfg)
    print(
        f"{ColorCodes.GREEN}✓ Dashboard generated:{ColorCodes.RESET} {ColorCodes.CYAN}{path}{ColorCodes.RESET}"
    )


def cmd_workspace_list(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    current = cfg.workspace_manager.current().id
    print(f"\n{ColorCodes.BOLD}Workspaces{ColorCodes.RESET}\n")
    for ws in cfg.workspace_manager.list():
        marker = f"{ColorCodes.GREEN}●{ColorCodes.RESET}" if ws.id == current else " "
        desc = f" — {ws.description}" if ws.description else ""
        print(
            f" {marker} {ColorCodes.CYAN}{ws.name:<20}{ColorCodes.RESET} {ColorCodes.DIM}({ws.id}){ColorCodes.RESET}{desc}"
        )
        print(f"    {ColorCodes.DIM}{ws.root}{ColorCodes.RESET}")
    print()


def cmd_workspace_switch(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    target = args.name
    if not target:
        picked = _pick_workspace(cfg, "Switch workspace")
        if picked is None:
            print(f"{ColorCodes.DIM}Cancelled.{ColorCodes.RESET}")
            return
        target = picked.id
    ws = cfg.workspace_manager.switch(target)
    print(
        f"{ColorCodes.GREEN}✓ Switched to:{ColorCodes.RESET} "
        f"{ColorCodes.CYAN}{ws.name}{ColorCodes.RESET} {ColorCodes.DIM}({ws.id}){ColorCodes.RESET} "
        f"{ColorCodes.DIM}[{ws.root}]{ColorCodes.RESET}"
    )


def cmd_workspace_create(args):
    cfg = load_config(args.config)
    ensure_workspace(cfg)
    name = args.name
    description = args.description or ""
    path = args.path if hasattr(args, "path") else None
    if not name:
        _clear_screen()
        _render_home_banner(cfg)
        name = _interactive_input(
            f"{ColorCodes.BOLD}Workspace name {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
        )
        if not name or not name.strip():
            print(f"{ColorCodes.DIM}Cancelled.{ColorCodes.RESET}")
            return
        description = (
            _interactive_input(
                f"{ColorCodes.BOLD}Description {ColorCodes.DIM}(optional, Esc to skip){ColorCodes.RESET}: "
            )
            or ""
        )
        path = (
            _interactive_input(
                f"{ColorCodes.BOLD}Workspace path {ColorCodes.DIM}(optional, Enter for default, Esc to skip){ColorCodes.RESET}: "
            )
            or ""
        )
    if isinstance(path, str) and path.strip():
        ws = cfg.workspace_manager.create(
            name.strip(), description.strip(), path=path.strip()
        )
    else:
        ws = cfg.workspace_manager.create(name.strip(), description.strip(), path=None)
    ws = cfg.workspace_manager.switch(ws.id)
    print(
        f"{ColorCodes.GREEN}✓ Created and switched to:{ColorCodes.RESET} "
        f"{ColorCodes.CYAN}{ws.name}{ColorCodes.RESET} {ColorCodes.DIM}({ws.id}){ColorCodes.RESET} "
        f"{ColorCodes.DIM}[{ws.root}]{ColorCodes.RESET}"
    )


def _run_cli_command(cmd_name: str, config_path: str | None):
    args = MockArgs(
        config=config_path,
        dry_run=False,
        type=None,
        top_k=10,
        threshold=None,
        file=None,
        text=None,
        question=None,
        query=None,
        all=False,
        name=None,
        description=None,
        purge=False,
        scrub=False,
        reason="",
        item_id=None,
        primary_ref=None,
        secondary_ref=None,
        title=None,
        new_type=None,
        status=None,
        alias=None,
        path=None,
    )
    print(
        f"{ColorCodes.BLUE}◆ Gray Box{ColorCodes.RESET} {ColorCodes.DIM}› {cmd_name}{ColorCodes.RESET}\n"
    )
    if cmd_name == "status":
        cmd_status(args)
    elif cmd_name == "capture":
        print(
            f"{ColorCodes.DIM}Press {ColorCodes.RESET}{ColorCodes.BOLD}F{ColorCodes.RESET}"
            f"{ColorCodes.DIM} to import a file, or any other key to type a note directly "
            f"(Esc to cancel){ColorCodes.RESET}"
        )
        choice = _getch()
        if choice == Key.ESC:
            return
        if isinstance(choice, str) and choice.lower() == "f":
            path = _interactive_input(
                f"{ColorCodes.BOLD}File path {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
            )
            if path is not None and path.strip():
                args.file = path.strip()
                cmd_capture(args)
        else:
            text = _interactive_input(
                f"{ColorCodes.BOLD}Note text {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
            )
            if text is not None and text.strip():
                args.text = text.strip()
                cmd_capture(args)
    elif cmd_name == "organize":
        cmd_organize(args)
    elif cmd_name == "ask":
        q = _interactive_input(
            f"{ColorCodes.BOLD}Question {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
        )
        if q and q.strip():
            args.question = q.strip()
            cmd_ask(args)
    elif cmd_name == "chat":
        cmd_chat(args)
    elif cmd_name == "search":
        q = _interactive_input(
            f"{ColorCodes.BOLD}Search query {ColorCodes.DIM}(Esc to cancel){ColorCodes.RESET}: "
        )
        if q and q.strip():
            args.query = q.strip()
            cmd_search(args)
    elif cmd_name == "pages":
        cmd_pages(args)
    elif cmd_name == "dupes":
        cmd_dupes(args)
    elif cmd_name == "dashboard":
        cmd_dashboard(args)
    elif cmd_name == "switch-workspace":
        cmd_workspace_switch(args)
    elif cmd_name == "create-workspace":
        cmd_workspace_create(args)


def interactive_main(config_path=None):
    options = [
        ("status", "Workspace summary", "📊"),
        ("capture", "Capture a note or import a file", "📥"),
        ("organize", "Process inbox items", "✨"),
        ("ask", "Ask a single question", "🧠"),
        ("chat", "Multi-turn Q&A with history", "💬"),
        ("search", "Search knowledge base", "🔍"),
        ("pages", "List all pages", "📄"),
        ("dupes", "Find possible duplicate pages", "🧬"),
        ("dashboard", "Generate HTML dashboard", "🌐"),
        ("switch-workspace", "Switch workspace", "🪄"),
        ("create-workspace", "Create workspace", "➕"),
        ("exit", "Quit", "❌"),
    ]
    selected = 0
    cfg = load_config(config_path)
    _hide_cursor()
    try:
        _clear_screen()
        _render_home_banner(cfg)
        _render_menu(selected, options)
        while True:
            ch = _getch()
            if ch in (Key.UP, "k"):
                selected = (selected - 1) % len(options)
            elif ch in (Key.DOWN, "j"):
                selected = (selected + 1) % len(options)
            elif ch == Key.ENTER:
                cmd_name = options[selected][0]
                if cmd_name == "exit":
                    return
                _clear_screen()
                try:
                    _run_cli_command(cmd_name, config_path)
                except (EOFError, KeyboardInterrupt):
                    print(f"\n{ColorCodes.YELLOW}Cancelled.{ColorCodes.RESET}")
                except Exception as e:
                    print(f"\n{ColorCodes.RED}Error: {e}{ColorCodes.RESET}")
                _pause()
                cfg = load_config(config_path)
                _clear_screen()
                _render_home_banner(cfg)
                _render_menu(selected, options)
                continue
            elif ch in (Key.ESC, "q"):
                return
            else:
                continue

            _move_cursor_up(len(options))
            _render_menu(selected, options)
    finally:
        _show_cursor()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graybox", description="Gray Box. Your personal digital memory"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml (default: ~/.graybox/config.yaml with legacy fallback)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_capture = sub.add_parser(
        "capture", help="Capture a note into the current workspace inbox."
    )
    p_capture.add_argument("text", nargs="?", help="Note text. Reads stdin if omitted.")
    p_capture.add_argument(
        "--file", help="Import a text file into the inbox instead of raw text."
    )
    p_capture.set_defaults(func=cmd_capture)

    p_organize = sub.add_parser(
        "organize", help="Process unorganized inbox items into wiki pages."
    )
    p_organize.add_argument(
        "--dry-run", action="store_true", help="Show what would happen, write nothing."
    )
    p_organize.set_defaults(func=cmd_organize)

    p_ask = sub.add_parser(
        "ask", help="Ask a question, get a cited answer from the current workspace."
    )
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--all", action="store_true", help="Search across all workspaces."
    )
    p_ask.set_defaults(func=cmd_ask)

    p_chat = sub.add_parser(
        "chat", help="Multi-turn Q&A session — ask follow-ups with conversation history."
    )
    p_chat.add_argument("--all", action="store_true", help="Search across all workspaces.")
    p_chat.set_defaults(func=cmd_chat)

    p_search = sub.add_parser("search", help="Keyword search over wiki pages.")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument(
        "--all", action="store_true", help="Search across all workspaces."
    )
    p_search.set_defaults(func=cmd_search)

    p_rebuild = sub.add_parser(
        "rebuild-index",
        help="Rebuild the embedding index for semantic search.",
    )
    p_rebuild.add_argument(
        "--type", default=None, help="Restrict to one page type."
    )
    p_rebuild.set_defaults(func=cmd_rebuild_index)

    p_refresh = sub.add_parser(
        "refresh-summaries",
        help="Re-summarize wiki pages from their accumulated notes.",
    )
    p_refresh.add_argument(
        "--type", default=None, help="Restrict to one page type."
    )
    p_refresh.add_argument(
        "--dry-run", action="store_true", help="Show what would change, write nothing."
    )
    p_refresh.add_argument(
        "--min-notes", type=int, default=3, help="Only refresh pages with at least N notes."
    )
    p_refresh.add_argument(
        "--verbose", action="store_true", help="Show old vs new summary for each page."
    )
    p_refresh.set_defaults(func=cmd_refresh)

    p_dashboard = sub.add_parser("dashboard", help="Generate a static HTML dashboard.")
    p_dashboard.set_defaults(func=cmd_dashboard)

    p_pages = sub.add_parser("pages", help="List wiki pages.")
    p_pages.add_argument(
        "--type",
        default=None,
        help="Filter by type: project, person, meeting, technology, company, topic, task, action, decision",
    )
    p_pages.set_defaults(func=cmd_pages)

    p_status = sub.add_parser("status", help="Show workspace summary.")
    p_status.set_defaults(func=cmd_status)

    p_forget = sub.add_parser(
        "forget",
        help="Retract a bad capture so it's excluded from search and organize.",
    )
    p_forget.add_argument("item_id", help="Inbox item id, e.g. 20260725-071000-9f3a")
    p_forget.add_argument(
        "--purge",
        action="store_true",
        help="Also delete the raw inbox file (irreversible).",
    )
    p_forget.add_argument(
        "--scrub",
        action="store_true",
        help="Also strip any notes already extracted from this item out of the wiki pages they landed in.",
    )
    p_forget.add_argument(
        "--reason", default="", help="Optional note on why this was forgotten."
    )
    p_forget.set_defaults(func=cmd_forget)

    p_dupes = sub.add_parser(
        "dupes",
        help="Find pages that look like duplicates (suggestion only - never merges automatically).",
    )
    p_dupes.add_argument("--type", default=None, help="Restrict to one page type.")
    p_dupes.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity threshold, 0-1, closer to 1 = more similar "
             "(default: retrieval.dedup_threshold from config).",
    )
    p_dupes.set_defaults(func=cmd_dupes)

    p_merge = sub.add_parser(
        "merge",
        help="Merge a duplicate page into another. Refs look like person/aaryan.",
    )
    p_merge.add_argument("primary_ref", help="The page to KEEP.")
    p_merge.add_argument("secondary_ref", help="The page to merge in and delete.")
    p_merge.add_argument(
        "--dry-run", action="store_true", help="Show what would happen, write nothing."
    )
    p_merge.set_defaults(func=cmd_merge)

    p_edit = sub.add_parser(
        "edit", help="Fix a page's title, type, status, or aliases."
    )
    p_edit.add_argument("ref", help="Page ref to fix, e.g. topic/aaryan")
    p_edit.add_argument(
        "--title", default=None, help="Correct the title (also updates the slug)."
    )
    p_edit.add_argument(
        "--type",
        dest="new_type",
        default=None,
        help="Correct the page type, e.g. person.",
    )
    p_edit.add_argument(
        "--status", default=None, help="Correct the status, e.g. open/done."
    )
    p_edit.add_argument(
        "--alias", action="append", default=None, help="Add an alias. Repeatable."
    )
    p_edit.add_argument(
        "--dry-run", action="store_true", help="Show what would happen, write nothing."
    )
    p_edit.set_defaults(func=cmd_edit)

    p_delete = sub.add_parser("delete", help="Remove a wrongly-created page.")
    p_delete.add_argument(
        "ref", help="Page ref to delete, e.g. person/hallucinated-name"
    )
    p_delete.add_argument(
        "--dry-run", action="store_true", help="Show what would happen, write nothing."
    )
    p_delete.set_defaults(func=cmd_delete)

    p_ws_list = sub.add_parser("workspace-list", help="List all workspaces.")
    p_ws_list.set_defaults(func=cmd_workspace_list)

    p_ws_switch = sub.add_parser(
        "workspace-switch", help="Switch the active workspace."
    )
    p_ws_switch.add_argument(
        "name", nargs="?", help="Workspace id or name. Omit to open the picker."
    )
    p_ws_switch.set_defaults(func=cmd_workspace_switch)

    p_ws_create = sub.add_parser("workspace-create", help="Create a new workspace.")
    p_ws_create.add_argument(
        "name", nargs="?", help="Workspace name. Omit to prompt interactively."
    )
    p_ws_create.add_argument("--description", default="", help="Optional description.")
    p_ws_create.add_argument(
        "--path", default=None, help="Optional custom data path for this workspace."
    )
    p_ws_create.set_defaults(func=cmd_workspace_create)

    return parser


def main(argv: list[str] | None = None) -> None:
    if os.name == "nt":
        os.system("")

    logging.basicConfig(
        level=logging.WARNING,
        format=f"{ColorCodes.YELLOW}⚠{ColorCodes.RESET}  %(message)s",
        stream=sys.stderr,
    )

    parser = build_parser()
    is_empty = (argv is None and len(sys.argv) == 1) or (
        argv is not None and len(argv) == 0
    )
    if is_empty and sys.stdin.isatty():
        interactive_main()
        return
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()