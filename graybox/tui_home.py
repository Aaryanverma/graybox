from __future__ import annotations
import subprocess

"""
Minimal Textual-based home screen for Gray Box.

This replaces the Rich-based live TUI with a full event-driven Textual App.
It keeps the command flow intact, but makes the home screen calmer:
- Tighter hierarchy
- Lighter information density
- Native input handling (no need for manual termios/msvcrt hacks)

Wire it the same way as before from graybox/cli.py:
    from graybox.textual_tui import interactive_main as textual_interactive_main
"""

import sys
import os
import webbrowser
from dataclasses import dataclass
from typing import Callable, Sequence

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from graybox.config import load_config
from graybox.dashboard import build_dashboard_data

LUXURY_GOLD = "#D4AF37"          # metallic gold
LUXURY_GOLD_BRIGHT = "#E6C668"   # brighter companion for hover/active

def _clear_terminal() -> None:
    if os.name == "nt":
        subprocess.run("cls", shell=True)
    else:
        subprocess.run(["clear"])


def _print_feedback_screen(message: str) -> None:
    """Clear the screen and show a single, unmistakable confirmation message
    before returning control to the caller's 'press enter' prompt. Used for
    workspace switch/create feedback, which otherwise risked getting lost
    beneath whatever the Textual app had last drawn to the terminal.
    """
    _clear_terminal()
    print(f"\n{message}\n")
    print("\x1b[2mPress Enter to return to menu...\x1b[0m")
    input()


def _resize_terminal(cols: int = 100, rows: int = 32) -> None:
    """Best-effort terminal resize via XTerm window-ops escape sequence.
    Honored by most modern emulators (iTerm2, Windows Terminal, GNOME
    Terminal, xterm); silently ignored by terminals that don't support it.
    """
    try:
        sys.stdout.write(f"\x1b[8;{rows};{cols}t")
        sys.stdout.flush()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HomeSnapshot:
    generated_at: str
    workspace_name: str
    workspace_id: str
    workspace_description: str
    root: str
    page_total: int
    unprocessed: int
    recent_lines: list[str]
    focus_lines: list[str]


def build_home_snapshot(cfg) -> HomeSnapshot:
    data = build_dashboard_data(cfg)
    summary = data["summary"]
    meta = data["meta"]

    ws = cfg.workspace_manager.current()

    recent_lines: list[str] = []
    for item in summary["recent_activity"]:
        if item["kind"] == "page":
            recent_lines.append(item["title"])
        if len(recent_lines) >= 5:
            break

    if not recent_lines:
        recent_lines.append("No Recent Memories")

    focus_lines: list[str] = []
    for task in summary["focus_items"][:5]:
        title = task.get("title") or "(untitled task)"
        due = task.get("due") or ""
        if due:
            focus_lines.append(f"{title} · {due}")
        else:
            focus_lines.append(title)
            
    if not focus_lines:
        focus_lines.append("No urgent tasks")

    return HomeSnapshot(
        generated_at=meta["generated_at"],
        workspace_name=ws.name,
        workspace_id=ws.id,
        workspace_description=ws.description or "",
        root=str(ws.root),
        page_total=summary["total_pages"],
        unprocessed=summary["unprocessed_inbox"],
        recent_lines=recent_lines,
        focus_lines=focus_lines,
    )

# ---------------------------------------------------------------------------
# Modal Screens
# ---------------------------------------------------------------------------

class WorkspaceSwitchScreen(ModalScreen[str | None]):
    CSS = f"""
    WorkspaceSwitchScreen {{
        align: center middle;
    }}
    #switch-dialog {{
        background: #050403;
        border: round {LUXURY_GOLD};
        padding: 1 2;
        width: 60;
        height: auto;
    }}
    .modal-title {{
        color: #F5EFDA;
        text-style: bold;
        padding-bottom: 1;
        text-align: center;
        width: 100%;
    }}
    #ws-options {{
        height: auto;
        max-height: 12;
        border: none;
        padding: 0;
    }}
    .modal-hint {{
        color: #695F46;
        text-align: center;
        width: 100%;
        margin-top: 1;
    }}
    """

    BINDINGS = [Binding("escape", "dismiss('')", "Cancel")]

    def __init__(self, workspaces: list, current_id: str):
        super().__init__()
        self.workspaces = workspaces
        self.current_id = current_id

    def compose(self) -> ComposeResult:
        with Vertical(id="switch-dialog"):
            yield Static("Switch Workspace", classes="modal-title")
            options = []
            for ws in self.workspaces:
                marker = "● " if ws.id == self.current_id else "  "
                options.append(Option(f"{marker}{ws.name} ({ws.id})", id=ws.id))
            yield OptionList(*options, id="ws-options")
            yield Static("[dim]Enter to select, Esc to cancel.[/]", classes="modal-hint")

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # Without this, the event bubbles from this modal's OptionList up to
        # GrayBoxApp.on_option_list_option_selected, which treats the raw
        # workspace id as a menu command and calls self.exit() a second
        # time - overwriting the correct "__switched__:<name>" result with
        # just the workspace id, which matches no known command.
        event.stop()
        self.dismiss(event.option_id)


class WorkspaceCreateScreen(ModalScreen[dict | None]):
    CSS = f"""
    WorkspaceCreateScreen {{
        align: center middle;
    }}
    #create-dialog {{
        background: #050403;
        border: round {LUXURY_GOLD};
        padding: 1 2;
        width: 70;
        height: auto;
    }}
    .modal-title {{
        color: #F5EFDA;
        text-style: bold;
        padding-bottom: 1;
        text-align: center;
        width: 100%;
    }}
    .field-label {{
        color: #695F46;
        margin-top: 1;
    }}
    Input {{
        border: solid {LUXURY_GOLD};
        margin-bottom: 1;
    }}
    Input:focus {{
        border: solid {LUXURY_GOLD_BRIGHT};
    }}
    .button-row {{
        align-horizontal: right;
        height: auto;
        margin-top: 1;
    }}
    
    /* Apply to BOTH buttons first to create a uniform shape */
    Button {{
        min-width: 16;
        height: 3;
        margin-left: 2;
        border: none;
        text-align: center;
        padding: 0 2;
    }}
    
    /* Specific styling for the Create button */
    #create-btn {{
        background: {LUXURY_GOLD};      
        color: #050403;                 
    }}
    #create-btn:hover {{
        background: {LUXURY_GOLD_BRIGHT}; 
        text-style: bold;
    }}
    #create-btn:focus {{
        background: {LUXURY_GOLD_BRIGHT};
        text-style: bold;
    }}
    
    /* Specific styling for the Cancel button */
    #cancel-btn {{
        background: transparent;
        color: #695F46;                 
        border: solid #695F46;
    }}
    #cancel-btn:hover {{
        color: #F5EFDA;
        border: solid #F5EFDA;
        background: transparent;
    }}
    """

    BINDINGS = [Binding("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="create-dialog"):
            yield Static("Create New Workspace", classes="modal-title")
            yield Label("Name:", classes="field-label")
            yield Input(id="ws-name", placeholder="Workspace name (required)")
            yield Label("Description:", classes="field-label")
            yield Input(id="ws-desc", placeholder="Optional description")
            yield Label("Path:", classes="field-label")
            yield Input(id="ws-path", placeholder="Optional custom data path")
            with Horizontal(classes="button-row"):
                yield Button("Cancel", id="cancel-btn")
                # IMPORTANT: Change variant to "default" so it doesn't fight the CSS
                yield Button("Create", id="create-btn", variant="default") 

    def on_mount(self) -> None:
        self.query_one("#ws-name", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "create-btn":
            name = self.query_one("#ws-name", Input).value.strip()
            if not name:
                self.app.bell()
                return
            desc = self.query_one("#ws-desc", Input).value.strip()
            path = self.query_one("#ws-path", Input).value.strip()
            self.dismiss({"name": name, "description": desc, "path": path or None})
        else:
            self.dismiss(None)


# ---------------------------------------------------------------------------
# Rendering & App
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rendering & App
# ---------------------------------------------------------------------------

class GrayBoxApp(App):
    CSS = f"""
    $accent: {LUXURY_GOLD};
    $text: #F5EFDA;
    $text-muted: #695F46;

    Screen {{
        align: left top;
        background: #050403;
        overflow: hidden;
    }}

    #header-box {{
        width: 100%;
        border: round {LUXURY_GOLD};
        padding: 1;
        content-align: center middle;
        height: auto;
    }}

    #main-area {{
        width: 100%;
        height: 1fr;
        layout: horizontal;
        padding-top: 1;
    }}

    #actions-panel {{
        width: 1fr;
        height: 100%;
        min-height: 16;
        border: round {LUXURY_GOLD};
    }}

    #digest-panel {{
        width: 1fr;
        height: 100%;
        min-height: 16;
        border: round {LUXURY_GOLD};
        padding: 0 1;
        margin-left: 1;
    }}

    OptionList {{
        width: 100%;
        height: 1fr;
        background: transparent;
        border: none;
        padding: 1 0;
        color: #F5EFDA
    }}

    OptionList > .option-list--option-highlighted {{
        background: {LUXURY_GOLD};
        color: $text;
    }}

    Footer {{
        width: 100%;
        height: auto;
        background: {LUXURY_GOLD} 18%;
    }}

    .digest-title {{
        text-style: bold;
        color: #F5EFDA;
        padding-top: 1;
        padding-left: 1;
    }}

    .digest-item {{
        color: #F5EFDA;
        padding-left: 1;
    }}
    """

    BINDINGS = [
        Binding("q,escape", "quit", "Quit", show=True),
        Binding("enter", "select_action", "Open", show=True),
        Binding("up,k", "cursor_up", "Move Up", show=False),
        Binding("down,j", "cursor_down", "Move Down", show=False),
    ]

    def __init__(self, snapshot: HomeSnapshot, options: Sequence[tuple[str, str]], cfg, **kwargs):
        super().__init__(**kwargs)
        self.snapshot = snapshot
        self.menu_options = options
        self.cfg = cfg

    def compose(self) -> ComposeResult:
        with Vertical(id="header-box"):
            header_text = (
                f"[bold #555555]██████╗  ██████╗  █████╗ ██╗   ██╗[/]  [bold {LUXURY_GOLD}]██████╗  ██████╗ ██╗  ██╗[/]\n"
                f"[bold #555555]██╔═══╝  ██╔══██╗██╔══██╗╚██╗ ██╔╝[/]  [bold {LUXURY_GOLD}]██╔══██╗██╔═══██╗╚██╗██╔╝[/]\n"
                f"[bold #555555]██║  ███╗██████╔╝███████║ ╚████╔╝ [/]  [bold {LUXURY_GOLD}]██████╔╝██║   ██║ ╚███╔╝ [/]\n"
                f"[bold #555555]██║   ██║██╔══██╗██╔══██║  ╚██╔╝  [/]  [bold {LUXURY_GOLD}]██╔══██╗██║   ██║ ██╔██╗ [/]\n"
                f"[bold #555555]╚██████╔╝██║  ██║██║  ██║   ██║   [/]  [bold {LUXURY_GOLD}]██████╔╝╚██████╔╝██╔╝ ██╗[/]\n"
                f"[bold #555555] ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   [/]  [bold {LUXURY_GOLD}]╚═════╝  ╚═════╝ ╚═╝  ╚═╝[/]\n"
            )
            logo = Static(header_text)
            logo.styles.content_align = ("center", "middle")
            logo.styles.width = "100%"
            yield logo

            tagline = Static("[dim]Made with ❤︎ by [@click=app.open_link('https://linkedin.com/in/aaryanverma')]Aaryan Verma[/][/]\n")
            tagline.styles.content_align = ("center", "middle")
            tagline.styles.width = "100%"
            tagline.styles.margin_bottom = 1
            yield tagline
        
            workspace_text = Static(f"[dim]Active workspace: [/][bold #F5EFDA]{self.snapshot.workspace_name}[/]")
            workspace_text.styles.content_align = ("center", "middle")
            workspace_text.styles.width = "100%"
            yield workspace_text

        with Horizontal(id="main-area"):
            actions = Vertical(id="actions-panel")
            actions.border_title = "Actions"
            with actions:
                option_items = [
                    Option(f"{cmd.ljust(12)}            [dim]{desc}[/dim]", id=cmd) 
                    if cmd not in {"switch-workspace", "create-workspace"}
                    else Option(f"{cmd.ljust(12)}        [dim]{desc}[/dim]", id=cmd)
                    for cmd, desc in self.menu_options
                ]
                yield OptionList(*option_items, id="action-list")

            digest = Vertical(id="digest-panel")
            digest.border_title = "What's in the Box"
            with digest:
                yield Static("Recent", classes="digest-title")
                for item in self.snapshot.recent_lines[:4]:
                    prefix = "" if item == "No Recent Memories" else "• "
                    yield Static(f"{prefix}{item}", classes="digest-item")
                
                yield Static("\nFocus", classes="digest-title")
                for item in self.snapshot.focus_lines[:4]:
                    prefix = "" if item == "No urgent tasks" else "• "
                    yield Static(f"{prefix}{item}", classes="digest-item")

        yield Footer()

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def action_open_link(self, url: str) -> None:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    def _execute_command(self, cmd: str) -> None:
        # Intercept workspace commands so they don't go to stdin
        if cmd == "switch-workspace":
            ws_list = self.cfg.workspace_manager.list()
            current_id = self.cfg.workspace_manager.current().id
            self.push_screen(WorkspaceSwitchScreen(ws_list, current_id), self.handle_switch_result)
        elif cmd == "create-workspace":
            self.push_screen(WorkspaceCreateScreen(), self.handle_create_result)
        elif cmd:
            self.exit(result=cmd)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # Only the home screen's own action list should ever trigger a
        # command here. Modal screens (workspace switch/create) have their
        # own OptionLists and must stop() their events - this check is a
        # second line of defense in case a future modal forgets to.
        if event.option_list.id != "action-list":
            return
        if event.option_id:
            self._execute_command(event.option_id)

    def action_select_action(self) -> None:
        option_list = self.query_one(OptionList)
        if option_list.highlighted is not None:
            cmd = self.menu_options[option_list.highlighted][0]
            self._execute_command(cmd)

    def handle_switch_result(self, ws_id: str | None) -> None:
        if ws_id:
            ws = self.cfg.workspace_manager.switch(ws_id)
            self.exit(result=f"__switched__:{ws.name}")

    def handle_create_result(self, data: dict | None) -> None:
        if data and data.get("name"):
            ws = self.cfg.workspace_manager.create(
                data["name"], data.get("description", ""), path=data.get("path")
            )
            self.cfg.workspace_manager.switch(ws.id)
            self.exit(result=f"__created__:{ws.name}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _home_options(last_item_id: str | None = None) -> list[tuple[str, str]]:
    """Menu entries for the Textual home. When a note was just captured, an
    "append to last note" option is inserted at the top so a follow-up is one
    keystroke away."""
    options: list[tuple[str, str]] = [
        ("status", "Workspace summary"),
        ("capture", "Capture a note or import"),
        ("organize", "Process captured notes"),
        ("ask", "Ask one question"),
        ("chat", "Multi-turn Q&A"),
        ("search", "Search knowledge base"),
        ("pages", "List pages"),
        ("dupes", "Find duplicates"),
        ("dashboard", "Generate dashboard"),
        ("switch-workspace", "Switch workspace"),
        ("create-workspace", "Create workspace"),
        ("exit", "Quit"),
    ]
    if last_item_id:
        options.insert(0, ("append", "Append to last note"))
    return options


def interactive_main(
    config_path: str | None = None,
    *,
    run_command: Callable[[str, str | None, str | None], str | None],
) -> None:
    _resize_terminal(110, 35)

    last_item_id: str | None = None

    while True:
        options = _home_options(last_item_id)

        try:
            cfg = load_config(config_path)
            snapshot = build_home_snapshot(cfg)
        except Exception as e:
            print(f"Failed to load dashboard data: {e}")
            return
            
        app = GrayBoxApp(snapshot, options, cfg)
        selected_cmd = app.run()

        # Handle workspace switch feedback
        if selected_cmd and selected_cmd.startswith("__switched__:"):
            ws_name = selected_cmd.split(":", 1)[1]
            _print_feedback_screen(
                f"\033[38;2;166;218;149m✓ Workspace switched to:\033[0m "
                f"\033[38;2;230;198;104m{ws_name}\033[0m"
            )
            continue

        # Handle workspace create feedback
        if selected_cmd and selected_cmd.startswith("__created__:"):
            ws_name = selected_cmd.split(":", 1)[1]
            _print_feedback_screen(
                f"\033[38;2;166;218;149m✓ Workspace created and switched to:\033[0m "
                f"\033[38;2;230;198;104m{ws_name}\033[0m"
            )
            continue

        if selected_cmd == "__refresh__":
            continue
            
        if not selected_cmd or selected_cmd == "exit":
            break

        # Suspend UI to run the standard CLI command
        if os.name == 'nt':
            subprocess.run('cls', shell=True)
        else:
            if os.name == 'posix':
                subprocess.run(['clear'])
            else:
                subprocess.run(['cls'], shell=True)
        
        try:
            result = run_command(selected_cmd, config_path, last_item_id)
            if result:
                last_item_id = result
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
        except Exception as e:
            print(f"\nError: {e}")

        print("\n\x1b[2mPress Enter to return to menu...\x1b[0m")
        input()