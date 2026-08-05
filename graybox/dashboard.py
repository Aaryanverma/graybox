"""Gray Box Dashboard.

Pure read-side feature: reads existing wiki pages and inbox items and renders
one self-contained HTML file. No new storage format, no server, no JS build
step — just a static file you open in a browser. Never writes back to wiki/
or inbox/.
"""

from __future__ import annotations
from pathlib import Path
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any
import json
import re

from graybox.config import Config
from graybox.models import now_iso
from graybox.storage import list_inbox_items, list_pages

PAGE_TYPE_ORDER = [
    "project",
    "task",
    "meeting",
    "decision",
    "person",
    "technology",
    "company",
    "topic",
    "action",
    "journal",
]

STATUS_ORDER = ["open", "in-progress", "blocked", "done", ""]

TYPE_META = {
    "project": {"label": "Projects", "icon": "\U0001F9ED", "accent": "#4A5568"},
    "task": {"label": "Tasks", "icon": "\u2713", "accent": "#4C6B8A"},
    "meeting": {"label": "Meetings", "icon": "\u2615", "accent": "#718096"},
    "decision": {"label": "Decisions", "icon": "\u2691", "accent": "#A0AEC0"},
    "person": {"label": "People", "icon": "\U0001F464", "accent": "#4A5568"},
    "technology": {"label": "Technologies", "icon": "\u2699", "accent": "#718096"},
    "company": {"label": "Companies", "icon": "\U0001F3E2", "accent": "#A0AEC0"},
    "topic": {"label": "Topics", "icon": "\u25CC", "accent": "#4A5568"},
    "action": {"label": "Actions", "icon": "\u2192", "accent": "#4C6B8A"},
    "journal": {"label": "Journal", "icon": "\u270E", "accent": "#718096"},
}

STATUS_META = {
    "open": {"label": "Open", "color": "#4C6B8A", "icon": "\u25CB"},
    "in-progress": {"label": "In progress", "color": "#718096", "icon": "\u25D4"},
    "blocked": {"label": "Blocked", "color": "#2D3748", "icon": "\u26D4"},
    "done": {"label": "Done", "color": "#A0AEC0", "icon": "\u2713"},
    "": {"label": "No status", "color": "#CBD5E0", "icon": "\u00B7"},
}

def _safe_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    dt = _safe_dt(value)
    if dt:
        return dt.date()
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None

def _safe_dt(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",     # current now_iso() format (UTC, explicit Z)
        "%Y-%m-%d %H:%M:%S",      # legacy naive-local format (pre-UTC-ISO change)
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _rel_date(value: str, today: date) -> str:
    d = _safe_date(value)
    if d is None:
        return "Unspecified"
    if d == today:
        return "Today"
    if d < today:
        delta = (today - d).days
        return f"{delta} day{'s' if delta != 1 else ''} ago"
    delta = (d - today).days
    return "Tomorrow" if delta == 1 else f"In {delta} days"


def _truncate(text: str, limit: int = 220) -> str:
    text = " ".join((text or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "\u2026"


def _safe_json(data: object) -> str:
    return (
        json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _search_text_for_page(p) -> str:
    parts = [
        p.title, p.summary, " ".join(p.notes or []), " ".join(p.aliases or []),
        " ".join(p.tags or []), p.type, p.status, p.owner, p.due, p.date,
        " ".join(p.attendees or []), " ".join(p.related or []),
        " ".join(p.backlinks or []), " ".join(p.sources or []), p.id, p.ref,
    ]
    return " ".join(parts).lower()


def _type_label(page_type: str) -> str:
    return TYPE_META.get(page_type, {"label": page_type.title() or "Other"})["label"]


def _type_icon(page_type: str) -> str:
    return TYPE_META.get(page_type, {"icon": "\u2022"})["icon"]


def _status_label(status: str) -> str:
    return STATUS_META.get(status or "", STATUS_META[""])["label"]


def _status_color(status: str) -> str:
    return STATUS_META.get(status or "", STATUS_META[""])["color"]


def _due_bucket(due: str, today: date) -> str:
    d = _safe_date(due)
    if d is None:
        return "unscheduled"
    if d < today:
        return "overdue"
    if d == today:
        return "today"
    if d <= today + timedelta(days=7):
        return "week"
    return "later"


def _snippet(text: str, limit: int = 240) -> str:
    return _truncate(text, limit)


def _health_score(*, total_pages: int, connected_pages: int, stale: int,
                   missing: int, unprocessed: int) -> dict:
    if total_pages == 0:
        return {"score": 100, "label": "Still", "tone": "good"}

    connected_ratio = connected_pages / total_pages
    freshness_ratio = max(0.0, 1 - (stale + missing) / total_pages)
    backlog_ratio = 1.0 if unprocessed == 0 else max(0.0, 1 - unprocessed / 10)

    score = round(100 * (0.5 * connected_ratio + 0.3 * freshness_ratio + 0.2 * backlog_ratio))
    score = max(0, min(100, score))

    if score >= 85:
        label, tone = "Flowing", "good"
    elif score >= 65:
        label, tone = "Steady", "good"
    elif score >= 40:
        label, tone = "Settling", "warn"
    else:
        label, tone = "Clouded", "danger"

    return {"score": score, "label": label, "tone": tone}


def build_dashboard_data(cfg: Config) -> dict:
    today = date.today()
    generated_at = now_iso()
    pages = list_pages(cfg)
    inbox = list_inbox_items(cfg)

    records: list[dict[str, Any]] = []
    by_ref: dict[str, dict[str, Any]] = {}

    for p in pages:
        updated_dt = _safe_dt(p.updated) or _safe_dt(p.created) or datetime.min
        raw = ""
        if getattr(p, "path", ""):
            try:
                raw = Path(p.path).read_text(encoding="utf-8")
            except OSError:
                raw = ""

        summary_refreshed_dt = _safe_dt(p.summary_refreshed_at) if p.summary_refreshed_at else None
        updated_for_stale = _safe_dt(p.updated)

        rec = {
            "ref": p.ref,
            "id": p.id,
            "type": p.type,
            "type_label": _type_label(p.type),
            "type_icon": _type_icon(p.type),
            "title": p.title,
            "summary": p.summary or "",
            "status": p.status or "",
            "status_label": _status_label(p.status),
            "status_color": _status_color(p.status),
            "owner": p.owner or "",
            "due": p.due or "",
            "date": p.date or "",
            "due_human": _rel_date(p.due, today) if p.due else "Unspecified",
            "date_human": _rel_date(p.date, today) if p.date else "Unspecified",
            "updated": p.updated,
            "updated_human": updated_dt.strftime("%Y-%m-%d") if updated_dt != datetime.min else p.updated,
            "created": p.created,
            "aliases": list(dict.fromkeys(p.aliases or [])),
            "related": sorted(set(p.related or [])),
            "backlinks": sorted(set(p.backlinks or [])),
            "sources": sorted(set(p.sources or [])),
            "tags": list(dict.fromkeys(p.tags or [])),
            "attendees": list(dict.fromkeys(p.attendees or [])),
            "path": p.path,
            "link_count": len(set(p.related or [])),
            "backlink_count": len(set(p.backlinks or [])),
            "source_count": len(set(p.sources or [])),
            "note_count": len(p.notes or []),
            "raw_excerpt": _truncate(raw or p.summary or "\n".join(p.notes or []), 1400),
            "search_text": _search_text_for_page(p),
            "frontmatter": p.frontmatter(),
            "is_orphan": not (p.related or p.backlinks),
            "is_stale_summary": bool(
                (summary_refreshed_dt and updated_for_stale and summary_refreshed_dt < updated_for_stale)
                or (not p.summary and len(p.notes or []) >= 3)
            ),
            "is_summary_missing": not bool((p.summary or "").strip()),
            "due_bucket": _due_bucket(p.due, today),
            "activity_score": len(set(p.related or [])) + len(set(p.backlinks or [])) + len(set(p.sources or [])) + len(p.notes or []),
        }
        records.append(rec)
        by_ref[p.ref] = rec

    records.sort(key=lambda r: (_safe_dt(r["updated"]) or datetime.min, r["title"].lower()), reverse=True)
    tasks = [r for r in records if r["type"] == "task"]
    counts = Counter(r["type"] for r in records)
    status_counts = Counter(r["status"] or "" for r in tasks)

    def _daily(items: list[dict[str, Any]], field: str, days: int, forward: bool = False) -> list[dict[str, Any]]:
        if forward:
            rng = [today + timedelta(days=i) for i in range(days)]
        else:
            start = today - timedelta(days=days - 1)
            rng = [start + timedelta(days=i) for i in range(days)]
        c = Counter()
        for item in items:
            d = _safe_date(item.get(field, ""))
            if d and rng[0] <= d <= rng[-1]:
                c[d.isoformat()] += 1
        return [{"label": d.strftime("%m/%d"), "date": d.isoformat(), "value": c.get(d.isoformat(), 0)} for d in rng]

    page_types = []
    for t in PAGE_TYPE_ORDER:
        if counts.get(t, 0):
            meta = TYPE_META.get(t, {"label": t.title(), "icon": "\u2022", "accent": "#8E8E93"})
            page_types.append({"key": t, "label": meta["label"], "icon": meta["icon"], "count": counts[t], "accent": meta["accent"]})
    for t in sorted(k for k in counts if k not in PAGE_TYPE_ORDER):
        meta = TYPE_META.get(t, {"label": t.title(), "icon": "\u2022", "accent": "#8E8E93"})
        page_types.append({"key": t, "label": meta["label"], "icon": meta["icon"], "count": counts[t], "accent": meta["accent"]})

    status_order = [("open", "Open"), ("in-progress", "In progress"), ("blocked", "Blocked"), ("done", "Done"), ("", "No status")]
    status_list = [
        {"key": k, "label": lbl, "count": status_counts.get(k, 0), "color": STATUS_META.get(k or "", STATUS_META[""])["color"]}
        for k, lbl in status_order
        if status_counts.get(k, 0)
    ]

    pages_by_day = _daily([{"updated": r["updated"]} for r in records], "updated", 30)
    tasks_due = _daily([{"due": r["due"]} for r in tasks], "due", 14, forward=True)

    graph_nodes = []
    graph_edges = []
    edge_keys = set()
    for r in records:
        refs = sorted(set(r["related"] + r["backlinks"]))
        graph_nodes.append({
            "ref": r["ref"], "label": r["title"], "type": r["type"], "type_label": r["type_label"],
            "type_icon": r["type_icon"], "summary": r["summary"], "degree": 0,
            "related": r["related"], "backlinks": r["backlinks"], "link_count": r["link_count"],
            "backlink_count": r["backlink_count"], "source_count": r["source_count"],
            "updated": r["updated"], "status": r["status"], "status_label": r["status_label"],
            "owner": r["owner"], "due": r["due"],
        })
        for ref in refs:
            if ref not in by_ref or ref == r["ref"]:
                continue
            key = tuple(sorted((r["ref"], ref)))
            if key in edge_keys:
                continue
            edge_keys.add(key)
            graph_edges.append({"source": key[0], "target": key[1]})

    deg = Counter()
    for e in graph_edges:
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    for n in graph_nodes:
        n["degree"] = deg[n["ref"]]

    def _focus_rank(t):
        d = _safe_date(t["due"])
        return (d is None, d or date.max, t["title"].lower())

    focus = sorted(
        [t for t in tasks if t["status"] != "done" and _safe_date(t["due"]) and _safe_date(t["due"]) <= today + timedelta(days=7)],
        key=_focus_rank,
    )
    if not focus:
        focus = sorted([t for t in tasks if t["status"] != "done"], key=_focus_rank)[:6]

    recent: list[dict[str, Any]] = []
    for p in records[:10]:
        recent.append({
            "kind": "page", "title": p["title"], "type_label": p["type_label"],
            "summary": _snippet(p["summary"], 160), "updated_human": p["updated_human"],
            "ref": p["ref"], "type": p["type"], "status": p["status"],
        })
    for item in inbox[:8]:
        created = _safe_dt(item.created) or datetime.min
        recent.append({
            "kind": "inbox", "id": item.id,
            "created_human": created.strftime("%Y-%m-%d") if created != datetime.min else item.created,
            "content_excerpt": _truncate(item.content, 380), "title": f"Note",
            "ref": f"inbox/{item.id}", "type": "inbox",
        })
    recent.sort(key=lambda e: e.get("updated_human", e.get("created_human", "")), reverse=True)
    recent = recent[:8]

    connected_pages = sum(1 for r in records if r["link_count"] or r["backlink_count"])
    orphan_pages = [r for r in records if r["is_orphan"]]
    stale_summaries = [r for r in records if r["is_stale_summary"]]
    missing_summaries = [r for r in records if r["is_summary_missing"] and r["note_count"] > 0]

    page_count = len(records)
    task_total = len(tasks)
    inbox_total = len(inbox)

    try:
        from graybox.storage import list_unprocessed
        unprocessed_count = len(list_unprocessed(cfg))
        processed_count = inbox_total - unprocessed_count
    except Exception:
        processed_count = 0
        unprocessed_count = inbox_total

    active_tasks = sum(1 for t in tasks if t["status"] != "done")
    overdue_tasks = sum(1 for t in tasks if t["status"] != "done" and _safe_date(t["due"]) and _safe_date(t["due"]) < today)
    due_today = sum(1 for t in tasks if t["status"] != "done" and _safe_date(t["due"]) == today)
    due_week = sum(1 for t in tasks if t["status"] != "done" and _safe_date(t["due"]) and today < _safe_date(t["due"]) <= today + timedelta(days=7))

    top_connected = sorted(records, key=lambda r: (r["link_count"] + r["backlink_count"] + r["source_count"], r["note_count"]), reverse=True)[:8]
    top_orphans = orphan_pages[:8]
    top_stale = stale_summaries[:8]
    top_missing = missing_summaries[:8]

    insights: list[str] = []
    if overdue_tasks:
        insights.append(f"{overdue_tasks} overdue task{'s' if overdue_tasks != 1 else ''} await.")
    if unprocessed_count:
        insights.append(f"{unprocessed_count} unfiled note{'s' if unprocessed_count != 1 else ''} remain.")
    if orphan_pages:
        insights.append(f"{len(orphan_pages)} detached page{'s' if len(orphan_pages) != 1 else ''} found.")
    if stale_summaries:
        insights.append(f"{len(stale_summaries)} abstract{'s' if len(stale_summaries) != 1 else ''} resting unrevised.")
    if missing_summaries:
        insights.append(f"{len(missing_summaries)} page{'s' if len(missing_summaries) != 1 else ''} hold notes but lack abstracts.")
    if not insights:
        insights.append("A state of quiet order.")

    signals = [
        {"key": "overdue_tasks", "label": "Overdue", "count": overdue_tasks, "tone": "danger" if overdue_tasks else "good"},
        {"key": "inbox_unprocessed", "label": "Unprocessed", "count": unprocessed_count, "tone": "warn" if unprocessed_count else "good"},
        {"key": "orphan_pages", "label": "Detached", "count": len(orphan_pages), "tone": "warn" if orphan_pages else "good"},
        {"key": "stale_summaries", "label": "Stale", "count": len(stale_summaries), "tone": "warn" if stale_summaries else "good"},
        {"key": "missing_summaries", "label": "Unsummarized", "count": len(missing_summaries), "tone": "warn" if missing_summaries else "good"},
        {"key": "connected_pages", "label": "Connected", "count": connected_pages, "tone": "good"},
    ]

    health = _health_score(
        total_pages=page_count, connected_pages=connected_pages,
        stale=len(stale_summaries), missing=len(missing_summaries),
        unprocessed=unprocessed_count,
    )

    return {
        "meta": {
            "generated_at": generated_at, "workspace": str(cfg.workspace),
            "workspace_name": cfg.workspace_name, "workspace_id": cfg.workspace_id,
            "today": today.isoformat(), "type_meta": TYPE_META, "status_meta": STATUS_META,
            "page_types": page_types,
        },
        "summary": {
            "generated_at": generated_at, "workspace": str(cfg.workspace),
            "workspace_name": cfg.workspace_name, "workspace_id": cfg.workspace_id,
            "today": today.isoformat(), "total_pages": page_count, "task_total": task_total,
            "inbox_total": inbox_total, "open_tasks": status_counts.get("open", 0),
            "active_tasks": active_tasks, "done_tasks": status_counts.get("done", 0),
            "overdue_tasks": overdue_tasks, "due_today": due_today, "due_week": due_week,
            "project_total": counts.get("project", 0), "meeting_total": counts.get("meeting", 0),
            "decision_total": counts.get("decision", 0), "people_total": counts.get("person", 0),
            "topic_total": counts.get("topic", 0),
            "technology_total": counts.get("technology", 0) + counts.get("company", 0),
            "connected_pages": connected_pages,
            "pages_updated_30d": sum(x["value"] for x in pages_by_day),
            "focus_items": focus, "recent_activity": recent,
            "orphan_pages": len(orphan_pages), "stale_summaries": len(stale_summaries),
            "missing_summaries": len(missing_summaries), "unprocessed_inbox": unprocessed_count,
            "processed_inbox": processed_count, "insights": insights,
            "health_score": health["score"], "health_label": health["label"], "health_tone": health["tone"],
        },
        "pages": records,
        "inbox": [
            {
                "kind": "inbox", "id": i.id, "created": i.created,
                "created_human": (_safe_dt(i.created).strftime("%Y-%m-%d") if _safe_dt(i.created) else i.created),
                "content_excerpt": _truncate(i.content, 420), "title": f"inbox/{i.id}",
                "ref": f"inbox/{i.id}", "type": "inbox",
            }
            for i in inbox
        ],
        "analytics": {
            "total_pages": page_count, "task_total": task_total, "type_counts": page_types,
            "type_max": max([x["count"] for x in page_types], default=1),
            "status_counts": status_list, "status_max": max([x["count"] for x in status_list], default=1),
            "pages_by_day": pages_by_day, "tasks_due_by_day": tasks_due,
        },
        "graph": {"nodes": graph_nodes, "edges": graph_edges},
        "signals": signals,
        "insights": insights,
        "derived": {
            "top_connected": top_connected, "top_recent": records[:8],
            "top_orphans": top_orphans, "top_stale": top_stale, "top_missing": top_missing,
        },
    }

HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gray Box</title>
<style>
:root, :root[data-theme="light"] {
    --bg: #F5F6F8;
    --surface: #FFFFFF;
    --surface-2: #EAECEF;
    --border: #D8DBDF;
    --text: #1E2125;
    --muted: #5C626A;
    --faint: #8E959E;
    --accent: #4C6B8A;
    --accent-soft: rgba(76, 107, 138, 0.12);
    --warn: #718096;
    --warn-soft: rgba(113, 128, 150, 0.12);
    --danger: #2D3748;
    --danger-soft: rgba(45, 55, 72, 0.12);
    --good: #4C6B8A;
    --chip: #EEF0F2;
    --radius: 10px;
    --radius-lg: 14px;
    --shadow: 0 1px 2px rgba(30, 33, 37, 0.05), 0 8px 24px rgba(30, 33, 37, 0.06);
}
:root[data-theme="dark"] {
    --bg: #141517;
    --surface: #1C1E21;
    --surface-2: #25282C;
    --border: rgba(255, 255, 255, 0.08);
    --text: #E2E4E8;
    --muted: #8E959E;
    --faint: #5C626A;
    --accent: #6E94B8;
    --accent-soft: rgba(110, 148, 184, 0.14);
    --warn: #A0AEC0;
    --warn-soft: rgba(160, 174, 192, 0.14);
    --danger: #E2E4E8;
    --danger-soft: rgba(226, 228, 232, 0.14);
    --good: #6E94B8;
    --chip: #292C31;
    --shadow: 0 1px 2px rgba(0,0,0,0.2), 0 8px 24px rgba(0,0,0,0.28);
}
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --bg: #141517;
        --surface: #1C1E21;
        --surface-2: #25282C;
        --border: rgba(255, 255, 255, 0.08);
        --text: #E2E4E8;
        --muted: #8E959E;
        --faint: #5C626A;
        --accent: #6E94B8;
        --accent-soft: rgba(110, 148, 184, 0.14);
        --warn: #A0AEC0;
        --warn-soft: rgba(160, 174, 192, 0.14);
        --danger: #E2E4E8;
        --danger-soft: rgba(226, 228, 232, 0.14);
        --good: #6E94B8;
        --chip: #292C31;
        --shadow: 0 1px 2px rgba(0,0,0,0.2), 0 8px 24px rgba(0,0,0,0.28);
    }
}

* { box-sizing: border-box; -webkit-font-smoothing: antialiased; }
html.theme-transition, html.theme-transition * {
    transition: background-color .35s ease, border-color .35s ease, color .35s ease, fill .35s ease !important;
}
body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, Helvetica, Arial, sans-serif;
    padding-bottom: 80px;
}
a { color: inherit; }
button, input { font: inherit; }
::selection { background: var(--accent-soft); }

.shell { width: min(1180px, calc(100% - 48px)); margin: 0 auto; }

/* ---------- Top bar ---------- */
.topbar {
    position: sticky; top: 0; z-index: 50;
    display: flex; align-items: center; gap: 20px;
    padding: 16px 0; margin-bottom: 28px;
    background: color-mix(in srgb, var(--bg) 88%, transparent);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 10px; white-space: nowrap; }
.brand img { height: 48px; width: auto; flex-shrink: 0; border-radius: 4px; }
.brand span { font-size: 12.5px; color: var(--faint); }

.tabs { display: flex; gap: 2px; background: var(--chip); padding: 3px; border-radius: 10px; }
.tab {
    border: none; background: transparent; cursor: pointer;
    padding: 7px 13px; border-radius: 8px;
    font-size: 12.5px; font-weight: 500; color: var(--muted);
    transition: background .15s ease, color .15s ease;
}
.tab:hover { color: var(--text); }
.tab.active { background: var(--surface); color: var(--text); box-shadow: var(--shadow); }

.topbar-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.search { position: relative; }
.search svg { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); width: 14px; height: 14px; stroke: var(--faint); }
.search input {
    width: 200px; padding: 7px 10px 7px 30px; border-radius: 8px;
    border: 1px solid var(--border); background: var(--surface); color: var(--text);
    font-size: 13px; transition: width .2s ease, border-color .2s ease;
}
.search input::placeholder { color: var(--faint); }
.search input:focus { outline: none; border-color: var(--accent); width: 240px; }
.iconbtn {
    width: 30px; height: 30px; display: grid; place-items: center;
    border: 1px solid var(--border); background: var(--surface); border-radius: 8px;
    color: var(--muted); cursor: pointer;
}
.iconbtn:hover { color: var(--text); border-color: var(--faint); }
.iconbtn svg { width: 14px; height: 14px; }

@media (max-width: 860px) {
    .topbar { flex-wrap: wrap; row-gap: 12px; }
    .tabs { order: 3; width: 100%; overflow-x: auto; }
    .search input { width: 100%; }
    .search { flex: 1; }
}

main { padding-bottom: 40px; }
.panel { display: none; }
.panel.active { display: block; animation: fadeIn .25s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

/* ---------- Building blocks ---------- */
.card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 22px;
}
.section-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 14px; }
.section-head h2 { font-size: 14px; font-weight: 600; margin: 0; }
.section-head .kicker { font-size: 12px; color: var(--faint); margin-top: 2px; }
.section-head .see-all { font-size: 12px; color: var(--accent); cursor: pointer; font-weight: 500; white-space: nowrap; }
.section-head .see-all:hover { text-decoration: underline; }

.grid { display: grid; gap: 16px; }
.grid-6 { grid-template-columns: repeat(6, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.split { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
@media (max-width: 1000px) {
    .grid-6 { grid-template-columns: repeat(3, 1fr); }
    .grid-4, .grid-3 { grid-template-columns: repeat(2, 1fr); }
    .split { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
    .grid-6, .grid-4, .grid-3, .grid-2 { grid-template-columns: repeat(2, 1fr); }
}

.stack { margin-top: 24px; }
.stack:first-child { margin-top: 0; }

/* status banner */
.banner {
    display: flex; align-items: center; gap: 14px;
    padding: 16px 20px; border-radius: var(--radius-lg);
    background: var(--surface); border: 1px solid var(--border);
    margin-bottom: 20px;
}
.banner .dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.banner .msg { font-size: 13.5px; color: var(--text); }
.banner .msg b { font-weight: 600; }
.banner .score { margin-left: auto; font-size: 12px; color: var(--faint); white-space: nowrap; }

/* metric cards */
.metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 16px 18px;
}
.metric-card .label { font-size: 11.5px; color: var(--faint); font-weight: 500; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px; }
.metric-card .label .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--faint); }
.metric-card.warn .label .dot { background: var(--warn); }
.metric-card.danger .label .dot { background: var(--danger); }
.metric-card.good .label .dot { background: var(--good); }
.metric-card .value { font-size: 26px; font-weight: 600; margin-top: 8px; letter-spacing: -0.01em; }

/* rows */
.row {
    display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
    padding: 11px 0; border-top: 1px solid var(--border);
    cursor: pointer; border-radius: 6px;
    margin: 0 -8px; padding-left: 8px; padding-right: 8px;
}
.stack .row:first-of-type { border-top: none; }
.row:hover { background: var(--surface-2); }
.row-main { min-width: 0; }
.row-title { font-size: 13.5px; font-weight: 500; color: var(--text); }
.row-meta { font-size: 12px; color: var(--faint); margin-top: 3px; display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.row-right { font-size: 11.5px; color: var(--faint); white-space: nowrap; padding-top: 2px; }

.badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 500; color: var(--muted);
    background: var(--chip); border-radius: 6px; padding: 2px 8px;
}
.badge .dot { width: 6px; height: 6px; border-radius: 50%; }

.empty { padding: 28px 4px; color: var(--faint); font-size: 13px; text-align: center; }

/* type overview cards */
.type-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 14px 16px; display: flex; align-items: center; gap: 10px; cursor: pointer;
}
.type-card:hover { border-color: var(--faint); }
.type-card .swatch { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.type-card .icon { font-size: 15px; }
.type-card .n { font-size: 16px; font-weight: 600; margin-left: auto; }
.type-card .lbl { font-size: 12px; color: var(--muted); }

/* kanban */
.kanban { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; align-items: start; }
.lane { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 14px; }
.lane h3 {
    margin: 0 0 10px; font-size: 12px; font-weight: 600; color: var(--muted);
    display: flex; justify-content: space-between; align-items: center;
    text-transform: uppercase; letter-spacing: 0.03em;
}
.lane h3 .count { color: var(--faint); font-weight: 500; }
.lane .row { padding-left: 0; padding-right: 0; margin: 0; }

/* table */
.tablewrap { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); overflow: hidden; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
    text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--faint); font-weight: 600; padding: 12px 16px; border-bottom: 1px solid var(--border);
    background: var(--surface-2);
}
td { padding: 12px 16px; border-bottom: 1px solid var(--border); vertical-align: top; color: var(--muted); }
tr:last-child td { border-bottom: none; }
tbody tr { cursor: pointer; }
tbody tr:hover td { background: var(--surface-2); }
td strong { color: var(--text); font-weight: 500; display: block; }
td .sub { color: var(--faint); font-size: 12px; margin-top: 3px; }

/* graph */
.graphwrap { display: grid; grid-template-columns: 1fr 300px; gap: 14px; height: 560px; }
@media (max-width: 900px) { .graphwrap { grid-template-columns: 1fr; height: auto; } }
.graphstage {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    position: relative; overflow: hidden; min-height: 400px;
}
.graphpanel {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 20px; overflow-y: auto;
}
.node-circle { cursor: pointer; transition: r .15s ease, opacity .15s ease; }
.node-label { font-size: 10.5px; fill: var(--muted); pointer-events: none; opacity: 0; transition: opacity .15s ease; }
.graph-node:hover .node-label { opacity: 1; }
.graphcontrols { position: absolute; bottom: 14px; left: 14px; }
.graphcontrols button {
    padding: 6px 12px; border-radius: 7px; border: 1px solid var(--border);
    background: var(--surface); color: var(--muted); font-size: 12px; cursor: pointer;
}
.graphcontrols button:hover { color: var(--text); border-color: var(--faint); }

.drawer h3 { margin: 0; font-size: 15px; font-weight: 600; }
.drawer .sub { font-size: 11.5px; color: var(--faint); margin-top: 4px; font-family: ui-monospace, monospace; }
.kv { display: flex; justify-content: space-between; font-size: 12.5px; padding: 9px 0; border-top: 1px solid var(--border); }
.kv:first-of-type { border-top: none; margin-top: 16px; }
.kv .k { color: var(--faint); }
.kv .v { color: var(--text); font-weight: 500; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.chip {
    font-size: 11.5px; padding: 4px 9px; border-radius: 7px;
    background: var(--chip); color: var(--muted); border: none; cursor: pointer;
}
.chip:hover { color: var(--text); }
.summary-text { font-size: 12.5px; color: var(--muted); line-height: 1.6; margin-top: 12px; }
</style>
</head>
<body>
<div class="shell">
<div class="topbar">
    <div class="brand">
        <!-- Replace the src URL below with your raw GitHub image URL -->
        <img src="https://raw.githubusercontent.com/Aaryanverma/graybox/e261a479b683a2c3d996f1377deab44b7a575377/assets/brand_logo.svg" alt="Gray Box Logo">
        <span id="meta-workspace">Workspace</span>
    </div>
    
    <div class="tabs" id="tabs">
        <button class="tab active" data-tab="overview">Overview</button>
        <button class="tab" data-tab="focus">Tasks</button>
        <button class="tab" data-tab="knowledge">Pages</button>
        <button class="tab" data-tab="graph">Graph</button>
        <button class="tab" data-tab="data">All</button>
    </div>
    <div class="topbar-right">
        <div class="search">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            <input id="search" type="search" placeholder="Search (press /)">
        </div>
        <button class="iconbtn" id="theme-toggle" aria-label="Toggle theme"></button>
    </div>
</div>

<main>
    <section id="panel-overview" class="panel active"></section>
    <section id="panel-focus" class="panel"></section>
    <section id="panel-knowledge" class="panel"></section>
    <section id="panel-graph" class="panel"></section>
    <section id="panel-data" class="panel"></section>
</main>
</div>

<script id="dashboard-data" type="application/json">@@DATA@@</script>
<script>
const D = JSON.parse(document.getElementById('dashboard-data').textContent);
const P = D.pages || [];
const T = D.summary || {};
const G = D.graph || {nodes: [], edges: []};
const SIGNALS = D.signals || [];
const TYPE_META = D.meta?.type_meta || {};
const BY_REF = new Map(P.map(x => [x.ref, x]));

const STATE = { tab: 'overview', q: '', theme: localStorage.getItem('graybox-theme') || '', selected: null };

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const clip = (s, n) => { s = s || ''; return s.length > n ? s.slice(0, n - 1) + '\u2026' : s; };

function applyTheme(theme) {
    const root = document.documentElement;
    if (!theme) { root.removeAttribute('data-theme'); localStorage.removeItem('graybox-theme'); }
    else { root.setAttribute('data-theme', theme); localStorage.setItem('graybox-theme', theme); }
    updateThemeIcon();
}
function updateThemeIcon() {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark' ||
        (!document.documentElement.hasAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    $('theme-toggle').innerHTML = isDark
        ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`
        : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
}
$('theme-toggle').addEventListener('click', () => {
    document.documentElement.classList.add('theme-transition');
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark' ||
        (!document.documentElement.hasAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    applyTheme(isDark ? 'light' : 'dark');
    setTimeout(() => document.documentElement.classList.remove('theme-transition'), 350);
});

function setActiveTab(tab) {
    STATE.tab = tab;
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.panel').forEach(p => p.classList.toggle('active', p.id === `panel-${tab}`));
    if (tab === 'graph') setTimeout(renderGraph, 30);
    refresh();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function typeColor(type) { return (TYPE_META[type]?.accent) || 'var(--faint)'; }
function statusColor(item) { return item.status_color || 'var(--faint)'; }

function matchesSearch(item) {
    if (!STATE.q) return true;
    const q = STATE.q.trim().toLowerCase();
    return (item.search_text || '').includes(q) ||
           (item.title || '').toLowerCase().includes(q) ||
           (item.ref || '').toLowerCase().includes(q) ||
           (item.type_label || '').toLowerCase().includes(q) ||
           (item.status_label || '').toLowerCase().includes(q) ||
           (item.owner || '').toLowerCase().includes(q);
}
function filteredPages() { return P.filter(matchesSearch); }
function filteredTasks() { return P.filter(x => x.type === 'task' && matchesSearch(x)); }

function badge(label, color) {
    return `<span class="badge"><span class="dot" style="background:${color}"></span>${esc(label)}</span>`;
}

function pageRow(item, rightText) {
    const color = item.type ? typeColor(item.type) : 'var(--faint)';
    const right = rightText !== undefined ? rightText : (item.updated_human || item.created_human || '');
    return `
    <div class="row" data-ref="${esc(item.ref)}">
        <div class="row-main">
            <div class="row-title">${esc(item.title || item.id || '')}</div>
            <div class="row-meta">
                ${badge(item.type_label || item.type || '', color)}
                ${item.status ? badge(item.status_label, item.status_color) : ''}
            </div>
        </div>
        <div class="row-right">${esc(right)}</div>
    </div>`;
}

function healthColor(tone) { return tone === 'good' ? 'var(--good)' : tone === 'warn' ? 'var(--warn)' : 'var(--danger)'; }

function renderOverview() {
    if (STATE.q.trim()) { renderOverviewSearch(); return; }

    const insight = (T.insights || [])[0] || 'Everything looks tidy.';
    const metrics = [
        { label: 'Pages', value: T.total_pages || 0, tone: '' },
        { label: 'Active tasks', value: T.active_tasks || 0, tone: '' },
        { label: 'Overdue', value: T.overdue_tasks || 0, tone: T.overdue_tasks ? 'danger' : '' },
        { label: 'Due this week', value: T.due_week || 0, tone: T.due_week ? 'warn' : '' },
        { label: 'Unfiled notes', value: T.unprocessed_inbox || 0, tone: T.unprocessed_inbox ? 'warn' : '' },
        { label: 'Connected pages', value: T.connected_pages || 0, tone: 'good' },
    ];

    const focusPreview = (T.focus_items || []).slice(0, 5);
    const recentPreview = (T.recent_activity || []).slice(0, 6);

    $('panel-overview').innerHTML = `
    <div class="banner">
        <span class="dot" style="background:${healthColor(T.health_tone)}"></span>
        <span class="msg"><b>${esc(T.health_label || 'Steady')}</b> &middot; ${esc(insight)}</span>
        <span class="score">${T.health_score ?? 100}/100</span>
    </div>

    <div class="grid grid-6" style="margin-bottom:20px">
        ${metrics.map(m => `
            <div class="metric-card ${m.tone}">
                <div class="label"><span class="dot"></span>${esc(m.label)}</div>
                <div class="value">${m.value}</div>
            </div>
        `).join('')}
    </div>

    <div class="split">
        <div class="card">
            <div class="section-head">
                <div><h2>Needs attention</h2><div class="kicker">Overdue and upcoming tasks</div></div>
                <span class="see-all" data-jump-tab="focus">View all</span>
            </div>
            <div class="stack">
                ${focusPreview.map(t => pageRow(t, t.due_human || 'No date')).join('') || '<div class="empty">No tasks need attention right now.</div>'}
            </div>
        </div>
        <div class="card">
            <div class="section-head">
                <div><h2>Recent activity</h2><div class="kicker">Latest pages and captures</div></div>
            </div>
            <div class="stack">
                ${recentPreview.map(item => pageRow(item, item.updated_human || item.created_human || '')).join('') || '<div class="empty">Nothing captured yet.</div>'}
            </div>
        </div>
    </div>`;
}

function renderOverviewSearch() {
    const pages = filteredPages();
    $('panel-overview').innerHTML = `
    <div class="card">
        <div class="section-head"><div><h2>Search results</h2><div class="kicker">${pages.length} match${pages.length === 1 ? '' : 'es'}</div></div></div>
        <div class="stack">
            ${pages.slice(0, 30).map(p => `
                <div class="row" data-ref="${esc(p.ref)}">
                    <div class="row-main">
                        <div class="row-title">${esc(p.title)}</div>
                        <div class="row-meta">
                            ${badge(p.type_label, typeColor(p.type))}
                            ${p.status ? badge(p.status_label, p.status_color) : ''}
                        </div>
                        ${p.summary ? `<div class="summary-text" style="margin-top:6px">${esc(clip(p.summary, 140))}</div>` : ''}
                    </div>
                    <div class="row-right">${esc(p.updated_human || '')}</div>
                </div>
            `).join('') || '<div class="empty">No matches found.</div>'}
        </div>
    </div>`;
}

function renderFocus() {
    const tasks = filteredTasks().filter(t => t.status !== 'done').sort((a, b) => {
        const da = a.due ? new Date(a.due) : null, db = b.due ? new Date(b.due) : null;
        if (!da && db) return 1;
        if (da && !db) return -1;
        if (!da && !db) return a.title.localeCompare(b.title);
        return da - db;
    });

    const lanes = [
        { key: 'overdue', label: 'Overdue' },
        { key: 'today', label: 'Due today' },
        { key: 'week', label: 'This week' },
        { key: 'later', label: 'Later' },
        { key: 'unscheduled', label: 'No date' },
    ].map(l => ({ ...l, items: tasks.filter(t => t.due_bucket === l.key) }))
     .filter(l => l.items.length);

    $('panel-focus').innerHTML = `
    <div class="grid grid-4" style="margin-bottom:20px">
        <div class="metric-card danger"><div class="label"><span class="dot"></span>Overdue</div><div class="value">${T.overdue_tasks || 0}</div></div>
        <div class="metric-card warn"><div class="label"><span class="dot"></span>Due today</div><div class="value">${T.due_today || 0}</div></div>
        <div class="metric-card"><div class="label"><span class="dot"></span>Due this week</div><div class="value">${T.due_week || 0}</div></div>
        <div class="metric-card good"><div class="label"><span class="dot"></span>Done</div><div class="value">${T.done_tasks || 0}</div></div>
    </div>
    <div class="kanban">
        ${lanes.map(l => `
            <div class="lane">
                <h3><span>${esc(l.label)}</span><span class="count">${l.items.length}</span></h3>
                ${l.items.slice(0, 12).map(t => pageRow(t, '')).join('')}
            </div>
        `).join('') || '<div class="empty">No open tasks match your search.</div>'}
    </div>`;
}

function renderKnowledge() {
    const pages = filteredPages();
    const types = (D.analytics?.type_counts || []).filter(x => x.count > 0);
    const connected = pages.filter(x => x.link_count || x.backlink_count)
        .sort((a, b) => (b.link_count + b.backlink_count) - (a.link_count + a.backlink_count)).slice(0, 8);

    const attention = [];
    pages.forEach(p => {
        if (!p.link_count && !p.backlink_count) attention.push({ ...p, reason: 'Not linked' });
        else if (p.is_stale_summary) attention.push({ ...p, reason: 'Summary may be stale' });
    });

    $('panel-knowledge').innerHTML = `
    <div class="grid grid-4" style="margin-bottom:20px">
        ${types.map(t => `
            <div class="type-card" data-type-filter="${esc(t.label)}">
                <span class="icon">${t.icon || ''}</span>
                <div>
                    <div class="lbl">${esc(t.label)}</div>
                </div>
                <span class="n">${t.count}</span>
            </div>
        `).join('')}
    </div>
    <div class="split">
        <div class="card">
            <div class="section-head"><div><h2>Well connected</h2><div class="kicker">Pages with the most links</div></div></div>
            <div class="stack">
                ${connected.map(p => pageRow(p, `${p.link_count + p.backlink_count} links`)).join('') || '<div class="empty">No connected pages yet.</div>'}
            </div>
        </div>
        <div class="card">
            <div class="section-head"><div><h2>Needs attention</h2><div class="kicker">Isolated or possibly outdated pages</div></div></div>
            <div class="stack">
                ${attention.slice(0, 8).map(p => pageRow(p, p.reason)).join('') || '<div class="empty">Nothing needs attention.</div>'}
            </div>
        </div>
    </div>`;

    document.querySelectorAll('[data-type-filter]').forEach(el => el.addEventListener('click', () => {
        STATE.q = el.dataset.typeFilter; $('search').value = STATE.q; refresh();
    }));
}

function renderData() {
    const rows = filteredPages().slice(0, 200);
    $('panel-data').innerHTML = `
    <div class="tablewrap">
        <table>
            <thead><tr><th>Title</th><th>Type</th><th>Status</th><th>Due / Date</th><th>Links</th></tr></thead>
            <tbody>
                ${rows.map(item => `
                    <tr data-ref="${esc(item.ref)}">
                        <td><strong>${esc(item.title)}</strong><div class="sub">${esc(item.ref)}</div></td>
                        <td>${badge(item.type_label, typeColor(item.type))}</td>
                        <td>${item.status ? badge(item.status_label, item.status_color) : '&mdash;'}</td>
                        <td>${esc(item.due || item.date || '\u2014')}<div class="sub">${esc(item.due_human || item.date_human || '')}</div></td>
                        <td>${(item.link_count || 0) + (item.backlink_count || 0)}</td>
                    </tr>
                `).join('') || `<tr><td colspan="5" class="empty">No pages match your search.</td></tr>`}
            </tbody>
        </table>
    </div>`;
}

function makeGraphLayout(nodes, edges, width = 800, height = 560, radiusScale = 0.36) {
    if (!nodes.length) return { nodes: [], edges: [] };
    const cx = width / 2, cy = height / 2;
    const radius = Math.min(width, height) * radiusScale;
    const deg = new Map(nodes.map(n => [n.ref, 0]));
    edges.forEach(e => { deg.set(e.source, (deg.get(e.source) || 0) + 1); deg.set(e.target, (deg.get(e.target) || 0) + 1); });
    const sorted = [...nodes].sort((a, b) => (deg.get(b.ref) || 0) - (deg.get(a.ref) || 0));
    const topDeg = deg.get(sorted[0]?.ref) || 1;
    const gNodes = sorted.map((n, i) => {
        const t = i / Math.max(1, sorted.length);
        const angle = t * Math.PI * 2;
        const d = deg.get(n.ref) || 0;
        const r = i === 0 ? 0 : radius * (0.4 + 0.6 * (1 - Math.min(1, d / topDeg)));
        return { ...n, x: i === 0 ? cx : cx + Math.cos(angle) * r, y: i === 0 ? cy : cy + Math.sin(angle) * r, r: clamp(4 + d * 1.3, 4, 16), degree: d };
    });
    return { nodes: gNodes, edges };
}

function renderGraph() {
    const container = $('panel-graph');
    const layout = makeGraphLayout(G.nodes || [], G.edges || []);
    const byRef = new Map(layout.nodes.map(n => [n.ref, n]));
    const visible = layout.nodes.filter(matchesSearch);
    const allowed = new Set(visible.map(n => n.ref));
    const visibleEdges = (G.edges || []).filter(e => allowed.has(e.source) && allowed.has(e.target));

    const w = 800, h = 560;
    const links = visibleEdges.map(e => {
        const a = byRef.get(e.source), b = byRef.get(e.target);
        if (!a || !b) return '';
        return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="var(--border)" stroke-width="1"></line>`;
    }).join('');

    const nodeEls = visible.map(n => {
        const selected = n.ref === STATE.selected;
        return `
        <g class="graph-node" data-ref="${esc(n.ref)}" transform="translate(${n.x},${n.y})">
            <circle class="node-circle" r="${n.r}" fill="${typeColor(n.type)}" stroke="var(--surface)" stroke-width="${selected ? 2.5 : 1.5}" opacity="${selected ? 1 : 0.85}"></circle>
            <text class="node-label" text-anchor="middle" y="${n.r + 13}">${esc(n.label.length > 16 ? n.label.slice(0, 16) + '\u2026' : n.label)}</text>
        </g>`;
    }).join('');

    const sel = STATE.selected ? BY_REF.get(STATE.selected) : null;
    let drawer = `<div class="drawer"><h3>${esc(sel ? sel.title : 'Connections')}</h3><div class="sub">${esc(sel ? sel.ref : 'Select a page to see how it connects.')}</div>`;
    if (sel) {
        drawer += `
            ${sel.summary ? `<div class="summary-text">${esc(sel.summary)}</div>` : ''}
            <div class="kv"><div class="k">Type</div><div class="v">${esc(sel.type_label)}</div></div>
            <div class="kv"><div class="k">Status</div><div class="v">${sel.status ? esc(sel.status_label) : '\u2014'}</div></div>
            <div class="kv"><div class="k">Links</div><div class="v">${sel.link_count || 0}</div></div>
            <div class="kv"><div class="k">Backlinks</div><div class="v">${sel.backlink_count || 0}</div></div>
            <div class="chips">
                ${(sel.related || []).map(r => `<span class="chip" data-ref="${esc(r)}">${esc(BY_REF.get(r)?.title || r)}</span>`).join('')}
                ${(sel.backlinks || []).map(r => `<span class="chip" data-ref="${esc(r)}">${esc(BY_REF.get(r)?.title || r)}</span>`).join('')}
            </div>`;
    } else if (!visible.length) {
        drawer += `<div class="empty">No pages match your search.</div>`;
    }
    drawer += `</div>`;

    container.innerHTML = `
    <div class="graphwrap">
        <div class="graphstage">
            <svg viewBox="0 0 ${w} ${h}" style="width:100%;height:100%"><g>${links}${nodeEls}</g></svg>
            <div class="graphcontrols"><button id="graph-reset">Reset</button></div>
        </div>
        <div class="graphpanel">${drawer}</div>
    </div>`;

    container.querySelectorAll('.graph-node').forEach(el => el.addEventListener('click', (e) => {
        e.stopPropagation(); STATE.selected = el.dataset.ref; renderGraph(); bindClicks();
    }));
    $('graph-reset')?.addEventListener('click', () => { STATE.selected = null; STATE.q = ''; $('search').value = ''; refresh(); });
}

function bindClicks() {
    document.querySelectorAll('[data-ref]').forEach(el => el.addEventListener('click', (e) => {
        e.stopPropagation();
        const ref = el.dataset.ref;
        STATE.selected = ref;
        if (STATE.tab === 'graph') { renderGraph(); bindClicks(); }
        else { setActiveTab('graph'); }
    }));
    document.querySelectorAll('[data-jump-tab]').forEach(el => el.addEventListener('click', () => setActiveTab(el.dataset.jumpTab)));
}

function refresh() {
    if (T.workspace_name) $('meta-workspace').textContent = T.workspace_name;
    if (STATE.tab === 'overview') renderOverview();
    if (STATE.tab === 'focus') renderFocus();
    if (STATE.tab === 'knowledge') renderKnowledge();
    if (STATE.tab === 'data') renderData();
    if (STATE.tab === 'graph') renderGraph();
    bindClicks();
}

document.querySelectorAll('.tab').forEach(btn => btn.addEventListener('click', () => setActiveTab(btn.dataset.tab)));
$('search').addEventListener('input', e => { STATE.q = e.target.value; refresh(); });
document.addEventListener('keydown', (e) => {
    if (e.key === '/' && document.activeElement !== $('search')) { e.preventDefault(); $('search').focus(); }
    if (e.key === 'Escape') { STATE.q = ''; $('search').value = ''; $('search').blur(); STATE.selected = null; refresh(); }
});

applyTheme(STATE.theme);
if (!STATE.theme) updateThemeIcon();
refresh();
setActiveTab('overview');
</script>
</body>
</html>
"""

def build_dashboard_html(cfg: Config) -> str:
    data = build_dashboard_data(cfg)
    return HTML_TEMPLATE.replace("@@DATA@@", _safe_json(data))


def write_dashboard(cfg: Config) -> Path:
    html = build_dashboard_html(cfg)
    out_dir = cfg.workspace / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dashboard.html"
    path.write_text(html, encoding="utf-8")
    return path