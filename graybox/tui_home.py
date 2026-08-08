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
from textual.widgets import Footer, OptionList, Static
from textual.widgets.option_list import Option

from graybox.config import load_config
from graybox.dashboard import build_dashboard_data

LUXURY_GOLD = "#D4AF37"          # metallic gold
LUXURY_GOLD_BRIGHT = "#E6C668"   # brighter companion for hover/active

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
# Rendering & App
# ---------------------------------------------------------------------------

class GrayBoxApp(App):
    # Overriding the $accent design token inside CSS natively propagates it 
    # to the OptionList, Footer, and focus rings without crashing ColorSystem.
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

    def __init__(
        self, 
        snapshot: HomeSnapshot, 
        options: Sequence[tuple[str, str]], 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.snapshot = snapshot
        self.menu_options = options            

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

            # Center the tagline automatically
            tagline = Static("[dim]Made with ❤︎ by [@click=app.open_link('https://linkedin.com/in/aaryanverma')]Aaryan Verma[/][/]\n")
            tagline.styles.content_align = ("center", "middle")
            tagline.styles.width = "100%"
            tagline.styles.margin_bottom = 1
            yield tagline
        
            # Left-align the active workspace line
            workspace_text = Static(f"[dim]Active workspace: [/][bold #F5EFDA]{self.snapshot.workspace_name}[/]")
            workspace_text.styles.content_align = ("center", "middle")
            workspace_text.styles.width = "100%"
            yield workspace_text

        # Main Area
        with Horizontal(id="main-area"):
        
            # Actions Panel
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

            # Digest Panel
            digest = Vertical(id="digest-panel")
            digest.border_title = "What's in the Box"
            with digest:
                yield Static("Recent", classes="digest-title")
                for item in self.snapshot.recent_lines[:4]:
                    # Omit the bullet point for the empty state message
                    prefix = "" if item == "No Recent Memories" else "• "
                    yield Static(f"{prefix}{item}", classes="digest-item")
                
                yield Static("\nFocus", classes="digest-title")
                for item in self.snapshot.focus_lines[:4]:
                    # Omit the bullet point for the empty state message
                    prefix = "" if item == "No urgent tasks" else "• "
                    yield Static(f"{prefix}{item}", classes="digest-item")

        yield Footer()
    def on_mount(self) -> None:
        """Auto-focus the OptionList on startup so keyboard nav works immediately."""
        self.query_one(OptionList).focus()

    def action_open_link(self, url: str) -> None:
        """Opens the provided URL in the default web browser."""
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Fires when user presses Enter on an item."""
        if event.option_id:
            self.exit(result=event.option_id)

    def action_select_action(self) -> None:
        """Fallback action for the Enter key binding."""
        option_list = self.query_one(OptionList)
        if option_list.highlighted is not None:
            option_id = self.menu_options[option_list.highlighted][0]
            self.exit(result=option_id)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def interactive_main(
    config_path: str | None = None,
    *,
    run_command: Callable[[str, str | None], None],
) -> None:
    _resize_terminal(110, 35)

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

    while True:
        try:
            cfg = load_config(config_path)
            snapshot = build_home_snapshot(cfg)
        except Exception as e:
            print(f"Failed to load dashboard data: {e}")
            return
            
        # Run the Textual App
        app = GrayBoxApp(snapshot, options)
        selected_cmd = app.run()

        # Handle exit condition
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
            run_command(selected_cmd, config_path)
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
        except Exception as e:
            print(f"\nError: {e}")

        print("\n\x1b[2mPress Enter to return to menu...\x1b[0m")
        input()