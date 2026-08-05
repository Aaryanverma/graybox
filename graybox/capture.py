
"""
Capture Agent.

Takes free-form text and writes it, verbatim, to an immutable file in the
active workspace inbox/.
"""
from __future__ import annotations

from graybox.config import Config
from graybox.models import InboxItem
from graybox.storage import write_inbox_item


def capture(cfg: Config, text: str, extra: dict | None = None) -> InboxItem:
    if not text or not text.strip():
        raise ValueError("Cannot capture empty content.")
    return write_inbox_item(cfg, text, extra=extra or {})


def capture_file(cfg: Config, file_path: str) -> InboxItem:
    from pathlib import Path

    path = Path(file_path).expanduser()
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {file_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"'{file_path}' isn't valid UTF-8 text (likely a binary or "
            "non-text file, e.g. PDF/DOCX/PPTX). Document ingestion for "
            "those formats isn't supported yet."
        )

    header = f"(imported from file: {file_path})\n\n"
    return capture(cfg, header + content)
