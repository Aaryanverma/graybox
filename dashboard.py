"""Gray Box Dashboard.

Pure read-side feature: reads existing wiki pages and inbox items and renders
one self-contained HTML file. No new storage format, no server, no JS build
step — just a static file you open in a browser. Never writes back to wiki/
or inbox/.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta, timezone
from html import escape
from pathlib import Path

from graybox.config import Config
from graybox.models import now_iso, now_readable
from graybox.storage import list_inbox_items, list_pages

STATUS_ORDER = ["open", "in-progress", "blocked", "done", ""]
PAGE_TYPE_ORDER = [
    "project", "task", "meeting", "decision", "person",
    "technology", "company", "topic", "action", "journal",
]

STATUS_META = {
    "open": {"label": "Open", "color": "#2563EB", "icon": "○"},
    "in-progress": {"label": "In progress", "color": "#D97706", "icon": "◔"},
    "blocked": {"label": "Blocked", "color": "#DC2626", "icon": "⛔"},
    "done": {"label": "Done", "color": "#059669", "icon": "✓"},
    "": {"label": "No status", "color": "#6B7280", "icon": "·"},
}

TYPE_META = {
    "project": {"label": "Projects", "icon": "🧭", "accent": "#2563EB"},
    "task": {"label": "Tasks", "icon": "✓", "accent": "#7C3AED"},
    "meeting": {"label": "Meetings", "icon": "☕", "accent": "#0EA5E9"},
    "decision": {"label": "Decisions", "icon": "⚑", "accent": "#10B981"},
    "person": {"label": "People", "icon": "👤", "accent": "#F59E0B"},
    "technology": {"label": "Technologies", "icon": "⚙", "accent": "#8B5CF6"},
    "company": {"label": "Companies", "icon": "🏢", "accent": "#14B8A6"},
    "topic": {"label": "Topics", "icon": "◌", "accent": "#EC4899"},
    "action": {"label": "Actions", "icon": "→", "accent": "#F97316"},
    "journal": {"label": "Journal", "icon": "✎", "accent": "#64748B"},
}

def _safe_date(value: str) -> date | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _safe_dt(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    # Normalize a trailing 'Z' to an explicit UTC offset so strptime's %z
    # can parse it on every Python version - datetime.fromisoformat() only
    # gained native 'Z' support in 3.11, and we don't want dashboard
    # rendering to silently degrade on older interpreters.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"

    dt = None
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            break
        except ValueError:
            continue

    if dt is None:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _rel_date(value: str, today: date) -> str:
    d = _safe_date(value)
    if d is None:
        return "Unspecified"
    if d < today:
        return f"{(today - d).days} day(s) ago"
    if d == today:
        return "Today"
    delta = (d - today).days
    return "Tomorrow" if delta == 1 else f"In {delta} day(s)"


def _type_label(page_type: str) -> str:
    return TYPE_META.get(page_type, {"label": page_type.title() or "Other"})["label"]


def _type_icon(page_type: str) -> str:
    return TYPE_META.get(page_type, {"icon": "•"})["icon"]


def _status_label(status: str) -> str:
    return STATUS_META.get(status or "", STATUS_META[""])["label"]


def _truncate(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _snippet(text: str, limit: int = 240) -> str:
    return _truncate(" ".join((text or "").split()), limit)


def _due_bucket(due: str, today: date) -> str:
    d = _safe_date(due)
    if d is None:
        return 'unscheduled'
    if d < today:
        return 'overdue'
    if d == today:
        return 'today'
    if d <= today + timedelta(days=7):
        return 'week'
    return 'later'


def _safe_json(data: object) -> str:
    import json
    return json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\u003c').replace('>', '\u003e').replace('&', '\u0026')


def build_dashboard_data(cfg: Config) -> dict:
    today = date.today()
    generated_at = now_readable()
    pages = list_pages(cfg)
    inbox = list_inbox_items(cfg)

    records: list[dict] = []
    for p in pages:
        raw = ''
        if getattr(p, 'path', ''):
            try:
                raw = Path(p.path).read_text(encoding='utf-8')
            except OSError:
                raw = ''
        updated = _safe_dt(p.updated) or _safe_dt(p.created)
        rec = {
            'ref': p.ref, 'id': p.id, 'type': p.type, 'type_label': _type_label(p.type), 'type_icon': _type_icon(p.type),
            'title': p.title, 'summary': p.summary, 'status': p.status, 'status_label': _status_label(p.status),
            'owner': p.owner, 'due': p.due, 'date': p.date, 'due_human': _rel_date(p.due, today) if p.due else 'Unspecified',
            'date_human': _rel_date(p.date, today) if p.date else 'Unspecified',
            'updated': p.updated, 'updated_human': updated.strftime('%Y-%m-%d %H:%M') if updated else p.updated,
            'created': p.created, 'aliases': p.aliases, 'related': sorted(set(p.related or [])), 'backlinks': sorted(set(p.backlinks or [])),
            'sources': sorted(set(p.sources or [])), 'tags': p.tags, 'attendees': p.attendees, 'path': p.path,
            'link_count': len(set(p.related or [])), 'backlink_count': len(set(p.backlinks or [])), 'source_count': len(set(p.sources or [])),
            'note_count': len(p.notes or []), 'raw_excerpt': _truncate(raw or p.summary or '\n'.join(p.notes or []), 1400),
            'search_text': ' '.join([p.title, p.summary, ' '.join(p.aliases), ' '.join(p.tags), ' '.join(p.related), ' '.join(p.backlinks), ' '.join(p.sources), p.owner, p.due, p.date, p.type, p.status]).lower(),
            'frontmatter': p.frontmatter(),
        }
        rec['due_bucket'] = _due_bucket(p.due, today)
        records.append(rec)

    records.sort(key=lambda r: (_safe_dt(r['updated']) or datetime.min, r['title'].lower()), reverse=True)
    tasks = [r for r in records if r['type'] == 'task']
    counts = Counter(r['type'] for r in records)
    status_counts = Counter(r['status'] or '' for r in tasks)

    def _daily(items: list[dict], field: str, days: int, forward: bool = False) -> list[dict]:
        if forward:
            start = today
            rng = [start + timedelta(days=i) for i in range(days)]
        else:
            start = today - timedelta(days=days - 1)
            rng = [start + timedelta(days=i) for i in range(days)]
        c = Counter()
        for item in items:
            d = _safe_date(item.get(field, ''))
            if d and rng[0] <= d <= rng[-1]:
                c[d.isoformat()] += 1
        return [{'label': d.strftime('%m/%d'), 'date': d.isoformat(), 'value': c.get(d.isoformat(), 0)} for d in rng]

    page_types = []
    for t in PAGE_TYPE_ORDER:
        if counts.get(t, 0):
            page_types.append({'key': t, 'label': TYPE_META.get(t, {'label': t.title()})['label'], 'icon': TYPE_META.get(t, {'icon': '•'})['icon'], 'count': counts[t]})
    for t in sorted(k for k in counts if k not in PAGE_TYPE_ORDER):
        page_types.append({'key': t, 'label': TYPE_META.get(t, {'label': t.title()})['label'], 'icon': TYPE_META.get(t, {'icon': '•'})['icon'], 'count': counts[t]})

    status_order = [('open', 'Open'), ('in-progress', 'In progress'), ('blocked', 'Blocked'), ('done', 'Done'), ('', 'No status')]
    status_list = [{'key': k, 'label': lbl, 'count': status_counts.get(k, 0)} for k, lbl in status_order if status_counts.get(k, 0)]
    pages_by_day = _daily([{'updated': r['updated']} for r in records], 'updated', 30)
    tasks_due = _daily([{'due': r['due']} for r in tasks], 'due', 14, forward=True)

    graph_nodes = []
    graph_edges = []
    edge_keys = set()
    by_ref = {r['ref']: r for r in records}
    for r in records:
        refs = sorted(set(r['related'] + r['backlinks']))
        graph_nodes.append({'ref': r['ref'], 'label': r['title'], 'type': r['type'], 'type_label': r['type_label'], 'summary': r['summary'], 'degree': 0, 'related': r['related'], 'backlinks': r['backlinks'], 'link_count': r['link_count'], 'backlink_count': r['backlink_count'], 'source_count': r['source_count']})
        for ref in refs:
            if ref not in by_ref or ref == r['ref']:
                continue
            key = tuple(sorted((r['ref'], ref)))
            if key in edge_keys:
                continue
            edge_keys.add(key)
            graph_edges.append({'source': key[0], 'target': key[1]})
    deg = Counter()
    for e in graph_edges:
        deg[e['source']] += 1
        deg[e['target']] += 1
    for n in graph_nodes:
        n['degree'] = deg[n['ref']]

    def _focus_rank(t):
        d = _safe_date(t['due'])
        return (d is None, d or date.max, t['title'].lower())
    focus = sorted([t for t in tasks if t['status'] != 'done' and _safe_date(t['due']) and _safe_date(t['due']) <= today + timedelta(days=7)], key=_focus_rank)
    if not focus:
        focus = sorted([t for t in tasks if t['status'] != 'done'], key=_focus_rank)[:6]

    recent = []
    for p in records[:10]:
        recent.append({'kind': 'page', 'title': p['title'], 'type_label': p['type_label'], 'summary': _snippet(p['summary'], 180), 'updated_human': p['updated_human'], 'ref': p['ref']})
    for item in inbox[:6]:
        created = _safe_dt(item.created)
        recent.append({'kind': 'inbox', 'id': item.id, 'created_human': created.strftime('%Y-%m-%d %H:%M') if created else item.created, 'content_excerpt': _truncate(item.content, 420)})
    recent.sort(key=lambda e: e.get('updated_human', e.get('created_human', '')), reverse=True)
    recent = recent[:12]

    connected_pages = sum(1 for r in records if r['link_count'] or r['backlink_count'])
    return {
        'meta': {'generated_at': generated_at, 'workspace': str(cfg.workspace), 'today': today.isoformat(), 'type_meta': TYPE_META, 'status_meta': STATUS_META, 'page_types': page_types},
        'summary': {
            'generated_at': generated_at, 'workspace': str(cfg.workspace), 'today': today.isoformat(), 'total_pages': len(records), 'task_total': len(tasks), 'inbox_total': len(inbox),
            'open_tasks': status_counts.get('open', 0), 'active_tasks': sum(1 for t in tasks if t['status'] != 'done'), 'done_tasks': status_counts.get('done', 0),
            'overdue_tasks': sum(1 for t in tasks if t['status'] != 'done' and _safe_date(t['due']) and _safe_date(t['due']) < today),
            'due_today': sum(1 for t in tasks if t['status'] != 'done' and _safe_date(t['due']) == today),
            'due_week': sum(1 for t in tasks if t['status'] != 'done' and _safe_date(t['due']) and today < _safe_date(t['due']) <= today + timedelta(days=7)),
            'project_total': counts.get('project', 0), 'meeting_total': counts.get('meeting', 0), 'decision_total': counts.get('decision', 0),
            'people_total': counts.get('person', 0), 'topic_total': counts.get('topic', 0), 'technology_total': counts.get('technology', 0) + counts.get('company', 0),
            'connected_pages': connected_pages, 'pages_updated_30d': sum(x['value'] for x in pages_by_day), 'inbox_30d': len(inbox),
            'focus_items': focus, 'recent_activity': recent,
        },
        'pages': records, 'inbox': [{'kind': 'inbox', 'id': i.id, 'created': i.created, 'created_human': (_safe_dt(i.created).strftime('%Y-%m-%d %H:%M') if _safe_dt(i.created) else i.created), 'content_excerpt': _truncate(i.content, 420)} for i in inbox],
        'analytics': {'total_pages': len(records), 'task_total': len(tasks), 'type_counts': page_types, 'type_max': max([x['count'] for x in page_types], default=1), 'status_counts': status_list, 'status_max': max([x['count'] for x in status_list], default=1), 'pages_by_day': pages_by_day, 'tasks_due_by_day': tasks_due},
        'graph': {'nodes': graph_nodes, 'edges': graph_edges},
    }

HTML_TEMPLATE = """<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Gray Box Dashboard</title>
<style>
:root, :root[data-theme="light"] {
    --bg: #F5F5F7;
    --panel: rgba(255, 255, 255, 0.65);
    --panel-solid: #FFFFFF;
    --border: rgba(0, 0, 0, 0.06);
    --text: #1D1D1F;
    --muted: #86868B;
    --soft: #A1A1A6;
    --shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
    --radius: 20px;
    --accent-blue: #007AFF;
    --accent-purple: #AF52DE;
}
:root[data-theme="dark"] {
    --bg: #000000;
    --panel: rgba(28, 28, 30, 0.65);
    --panel-solid: #1C1C1E;
    --border: rgba(255, 255, 255, 0.08);
    --text: #F5F5F7;
    --muted: #86868B;
    --soft: #636366;
    --shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
}
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --bg: #000000;
        --panel: rgba(28, 28, 30, 0.65);
        --panel-solid: #1C1C1E;
        --border: rgba(255, 255, 255, 0.08);
        --text: #F5F5F7;
        --muted: #86868B;
        --soft: #636366;
        --shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
}

/* Cinematic "Light Switch" Transition */
.theme-transition,
.theme-transition *,
.theme-transition *:before,
.theme-transition *:after {
    transition: background-color 0.8s cubic-bezier(0.22, 1, 0.36, 1),
                border-color 0.8s cubic-bezier(0.22, 1, 0.36, 1),
                color 0.8s cubic-bezier(0.22, 1, 0.36, 1),
                box-shadow 0.8s cubic-bezier(0.22, 1, 0.36, 1),
                fill 0.8s cubic-bezier(0.22, 1, 0.36, 1),
                stroke 0.8s cubic-bezier(0.22, 1, 0.36, 1) !important;
    transition-delay: 0s !important;
}

* { box-sizing: border-box; }
body {
    margin: 0;
    color: var(--text);
    font: 14px/1.47 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    letter-spacing: -0.015em;
    background: var(--bg);
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(0, 122, 255, 0.035), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(175, 82, 222, 0.035), transparent 25%);
    background-attachment: fixed;
}
.shell { width: min(1400px, calc(100% - 40px)); margin: 0 auto; padding: 24px 0 60px; }

/* Hero / Header Section */
.hero {
    position: sticky; top: 16px; z-index: 20;
    background: var(--panel);
    backdrop-filter: blur(24px) saturate(150%);
    -webkit-backdrop-filter: blur(24px) saturate(150%);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    border-radius: var(--radius);
    padding: 24px 28px;
    margin-bottom: 28px;
}
.hero-grid { display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: center; }
.brand { display: flex; gap: 16px; align-items: center; }
.mark {
    width: 52px; height: 52px; border-radius: 14px; display: grid; place-items: center;
    color: #fff; background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    font-size: 24px; font-weight: 600;
    box-shadow: 0 8px 16px rgba(0, 122, 255, 0.15);
}
h1 { margin: 0; font-size: 24px; font-weight: 600; letter-spacing: -0.02em; }
.subtitle { margin-top: 4px; color: var(--muted); font-size: 13px; max-width: 60ch; line-height: 1.4; }
.hero-meta {
    display: flex; gap: 10px; font-size: 12px; color: var(--muted); text-align: right; flex-wrap: wrap;
    justify-content: flex-end;
}
.meta-item {
    display: flex; flex-direction: column; gap: 4px; text-align: left;
    background: var(--panel-solid);
    border: 1px solid var(--border);
    padding: 10px 14px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}
.meta-item strong {
    color: var(--text); font-weight: 600; text-transform: uppercase;
    font-size: 10px; letter-spacing: 0.05em; opacity: 0.7;
}
.meta-item span { font-weight: 500; color: var(--text); }

.toolbar { margin-top: 20px; display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.search {
    flex: 1 1 240px; display: flex; align-items: center; gap: 10px;
    padding: 10px 16px; border: 1px solid var(--border); border-radius: 12px;
    background: var(--panel-solid);
    transition: box-shadow 0.2s;
}
.search svg { width: 16px; height: 16px; flex-shrink: 0; stroke: var(--muted); }
.search:focus-within {
    box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
    border-color: var(--accent-blue);
}
.search input { border: 0; outline: 0; background: transparent; color: var(--text); width: 100%; font-size: 14px; font-family: inherit;}
.search input::placeholder { color: var(--soft); }
.tabbar { display: flex; gap: 4px; background: rgba(142, 142, 147, 0.1); padding: 4px; border-radius: 14px; }
.btn {
    border: none; background: transparent; color: var(--text); border-radius: 10px; padding: 8px 16px;
    font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.2s; font-family: inherit;
}
.btn.active {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(0, 122, 255, 0.25);
}
.btn:hover:not(.active) { background: rgba(142, 142, 147, 0.08); }

/* Icon Buttons (Copy, Theme) */
.btn-icon {
    background: transparent; border: none; color: var(--muted); cursor: pointer;
    padding: 6px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center;
    transition: all 0.2s;
}
.btn-icon:hover { background: rgba(142, 142, 147, 0.15); color: var(--text); }
.btn-icon.copied { color: #34C759; }

/* Layout Grids */
main { display: grid; gap: 24px; }
.panel { display: none; }
.panel.active { display: grid; gap: 24px; animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1); }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }

.grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 20px; }
.grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
.grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.split { display: grid; grid-template-columns: 1.6fr 1fr; gap: 24px; }

.section, .metric, .item, .lane {
    background: var(--panel-solid);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
}
.section { padding: 24px; }
.section-head { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px; }
.title { margin: 0; font-size: 18px; font-weight: 600; }
.kicker { color: var(--muted); font-size: 13px; margin-top: 4px; }

.metric { padding: 20px; position: relative; overflow: hidden; display: flex; flex-direction: column; justify-content: center; }
.metric .l { color: var(--muted); font-size: 13px; font-weight: 500; }
.metric .v { font-size: 32px; font-weight: 600; letter-spacing: -0.03em; margin-top: 6px; color: var(--accent, var(--text)); }
.metric .f { color: var(--soft); font-size: 12px; margin-top: 4px; }

.stack { display: grid; gap: 14px; }
.item { padding: 18px; transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s; cursor: pointer; border-radius: 16px; }
.item:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(0,0,0,0.08); }
.item .t { font-weight: 600; font-size: 15px; letter-spacing: -0.01em; }
.item .m { color: var(--muted); font-size: 13px; margin-top: 4px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap;}
.item .s { color: var(--text); font-size: 14px; margin-top: 10px; line-height: 1.5; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.ref { color: var(--soft); font-size: 11px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.item .copy-btn { opacity: 0; transition: opacity 0.2s, transform 0.2s; }
.item:hover .copy-btn { opacity: 1; transform: scale(1.05); }
@media (max-width: 820px) { .item .copy-btn { opacity: 1; } }

/* Badges & Pills */
.badge {
    display: inline-flex; align-items: center; padding: 4px 10px;
    border-radius: 8px; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
}
.muted { background: rgba(142, 142, 147, 0.1); color: var(--muted); }
.overdue { background: rgba(255, 59, 48, 0.1); color: #FF3B30; }
.today { background: rgba(255, 149, 0, 0.1); color: #FF9500; }
.soon { background: rgba(0, 122, 255, 0.1); color: #007AFF; }
.done { background: rgba(52, 199, 89, 0.1); color: #34C759; }

.chips { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.chip, .pill {
    padding: 6px 12px; border-radius: 12px; border: 1px solid var(--border);
    background: var(--bg); color: var(--muted); font-size: 12px; cursor: pointer;
    transition: all 0.2s; font-family: inherit; font-weight: 500;
}
.chip:hover, .pill:hover { background: var(--border); color: var(--text); }
.pill.active { background: var(--text); color: var(--bg); border-color: transparent; }

.kanban { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 16px; align-items: start; }
.lane { padding: 16px; background: rgba(142, 142, 147, 0.04); box-shadow: none; border-color: transparent;}
.lane h3 { margin: 0 0 16px; font-size: 14px; font-weight: 600; display: flex; justify-content: space-between; align-items: center; }

.filters { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 20px; }
.filters input, .filters select {
    padding: 10px 16px; border-radius: 12px; border: 1px solid var(--border);
    background: var(--panel-solid); color: var(--text); font-size: 13px;
    font-family: inherit; outline: none; transition: box-shadow 0.2s;
}
.filters input:focus, .filters select:focus {
    box-shadow: 0 0 0 4px rgba(0, 122, 255, 0.1);
    border-color: var(--accent-blue);
}

.tablewrap { overflow: auto; border-radius: var(--radius); border: 1px solid var(--border); background: var(--panel-solid); }
table { width: 100%; min-width: 800px; border-collapse: collapse; }
th, td { padding: 14px 20px; text-align: left; border-bottom: 1px solid var(--border); font-size: 14px; }
th { position: sticky; top: 0; background: rgba(142, 142, 147, 0.06); font-weight: 500; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; backdrop-filter: blur(12px);}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(142, 142, 147, 0.03); }
.raw { white-space: pre-wrap; word-break: break-all; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; background: rgba(0,0,0,0.03); padding: 12px; border-radius: 12px; }

.graphwrap { display: grid; grid-template-columns: 1fr 320px; gap: 20px; height: 650px; }
.graphstage {
    background: var(--panel-solid); border: 1px solid var(--border); border-radius: var(--radius);
    overflow: hidden; position: relative; box-shadow: inset 0 2px 15px rgba(0,0,0,0.02);
    cursor: grab;
}
.graphstage:active { cursor: grabbing; }

.graphpanel {
    background: var(--panel-solid); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 18px;
    box-shadow: var(--shadow);
}
svg { width: 100%; height: 100%; display: block; user-select: none; }
#nodes text { 
    user-select: none; 
    pointer-events: none; 
    font-weight: 500;
    text-shadow: 0px 1px 4px var(--panel-solid), 0px -1px 4px var(--panel-solid), 1px 0px 4px var(--panel-solid), -1px 0px 4px var(--panel-solid);
}
#nodes g circle {
    transition: stroke-width 0.2s, filter 0.2s;
}
#nodes g:hover circle {
    stroke-width: 3;
    filter: brightness(1.1);
}

.graph-controls {
    position: absolute;
    bottom: 20px;
    left: 20px;
    display: flex;
    gap: 8px;
    background: var(--panel);
    backdrop-filter: blur(12px);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

.graph-btn {
    width: 32px; height: 32px;
    border-radius: 8px; border: none;
    background: var(--panel-solid);
    color: var(--text); font-weight: bold; font-size: 16px;
    cursor: pointer; display: grid; place-items: center;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
.graph-btn:hover { background: rgba(142,142,147,0.1); }

.footer { margin-top: 48px; text-align: center; color: var(--soft); font-size: 12px; }

@media(max-width:1024px) { .grid4, .kanban { grid-template-columns: repeat(2, 1fr); } .split, .graphwrap { grid-template-columns: 1fr; } .graphwrap { height: auto; } .graphstage { height: 500px; } }
@media(max-width:600px) { .hero-grid, .grid2, .grid3, .grid4, .kanban { grid-template-columns: 1fr; } .hero-meta { flex-direction: column; text-align: left; gap: 8px;} }
</style>
</head>
<body>
<div class='shell'>
<header class='hero'>
  <div class='hero-grid'>
    <div>
      <div class='brand'><div class='mark'>K</div><div><h1>Gray Box Dashboard</h1><div class='subtitle'>A read-only overview of your workspace.</div></div></div>
      <div class='toolbar'>
        <div class='search'>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
            <input id='q' type='search' placeholder='Search everything… (Press / to focus)'>
        </div>
        <div class='tabbar' id='tabs'>
          <button class='btn active' data-tab='overview'>Overview</button>
          <button class='btn' data-tab='tasks'>Tasks</button>
          <button class='btn' data-tab='knowledge'>Knowledge</button>
          <button class='btn' data-tab='graph'>Graph</button>
          <button class='btn' data-tab='advanced'>Data</button>
        </div>
        <button id="theme-toggle" class="btn-icon" style="margin-left: auto; width: 32px; height: 32px;" title="Toggle Theme"></button>
      </div>
    </div>
    <div class='hero-meta'>
        <div class='meta-item'><strong>Generated</strong><span id='gen'></span></div>
        <div class='meta-item'><strong>Workspace</strong><span id='ws'></span></div>
        <div class='meta-item'><strong>Counts</strong><span id='counts'></span></div>
    </div>
  </div>
</header>
<main>
  <section id='p-overview' class='panel active'></section>
  <section id='p-tasks' class='panel'></section>
  <section id='p-knowledge' class='panel'></section>
  <section id='p-graph' class='panel'></section>
  <section id='p-advanced' class='panel'></section>
</main>
<div id='foot' class='footer'></div>
</div>
<script id='data' type='application/json'>@@DATA@@</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent), P=D.pages, T=D.summary, TA=P.filter(x=>x.type==='task'), BY=new Map(P.map(x=>[x.ref,x]));
const S={tab:'overview', q:'', tv:'kanban', ts:'all', td:'all', to:'', kt:'all', as:'updated', gn:null};
const $=id=>document.getElementById(id);
const esc=s=>String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
const q=()=>S.q.trim().toLowerCase();
const match=(x)=>!q()||x.search_text.includes(q())||x.title.toLowerCase().includes(q())||x.summary.toLowerCase().includes(q());
const db=x=>x.due?((()=>{const d=new Date(x.due+'T00:00:00'),n=new Date(D.meta.today+'T00:00:00');if(d<n)return'overdue';if(+d===+n)return'today';const w=new Date(n);w.setDate(w.getDate()+7);if(d<=w)return'week';return'later'})()):'unscheduled';

const st=s=>({'open':'open','in-progress':'soon','blocked':'overdue','done':'done','':'muted'})[s||'']||'muted';
const cType=t=>D.meta.type_meta[t]?.accent||'var(--text)';

const copySvg = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
const checkSvg = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

document.addEventListener('click', async e => {
    const btn = e.target.closest('.copy-btn');
    if(!btn) return;
    e.stopPropagation();
    const ref = btn.dataset.ref;
    const item = BY.get(ref);
    if(!item) return;
    
    let md = '';
    if (item.type === 'task') {
        const checked = item.status === 'done' ? 'x' : ' ';
        md = `- [${checked}] **${item.title}**${item.owner ? ` (@${item.owner})` : ''}${item.due ? ` (Due: ${item.due})` : ''}\n  ${item.summary ? item.summary.replace(/\\n/g, '\\n  ') : ''}`;
    } else {
        md = `**[${item.type_label}] ${item.title}**\n${item.status_label && item.status ? `*Status:* ${item.status_label}\n` : ''}${item.summary ? `> ${item.summary}\n` : ''}*Ref:* \`${item.ref}\``;
    }
    
    try {
        await navigator.clipboard.writeText(md.trim());
        const orig = btn.innerHTML;
        btn.innerHTML = checkSvg;
        btn.classList.add('copied');
        setTimeout(() => { btn.innerHTML = orig; btn.classList.remove('copied'); }, 2000);
    } catch(err) { console.error('Copy failed', err); }
});

/* Keyboard shortcuts */
document.addEventListener('keydown', e => {
    if (e.key === '/' && document.activeElement.tagName !== 'INPUT') {
        e.preventDefault();
        $('q').focus();
    }
});

function card(x, links=true){
    return `<div class='item' data-card='1' data-ref='${esc(x.ref)}'>
        <div style='display:flex;justify-content:space-between;gap:8px'>
            <div class='t'>${esc(x.title)}</div>
            ${x.status ? `<span class='badge ${st(x.status)}'>${esc(x.status_label)}</span>` : ''}
        </div>
        <div class='m'><span>${esc(x.type_label)}</span> ${x.owner?`· <span>${esc(x.owner)}</span>`:''} ${x.due?`· <span>Due: ${esc(x.due_human)}</span>`:''}</div>
        <div class='s'>${esc(x.summary||'')}</div>
        ${links&&x.related?.length ? `<div class='chips'>${x.related.slice(0,3).map(r=>`<span class='chip' data-s='${esc(r)}'>${esc(BY.get(r)?.title||r)}</span>`).join('')}</div>`:''}
        <div style='display:flex; justify-content:flex-end; align-items:center; margin-top:14px; gap:8px;'>
            <button class='btn-icon copy-btn' data-ref='${esc(x.ref)}' title='Copy as Markdown'>${copySvg}</button>
            <div class='ref'>${esc(x.ref)}</div>
        </div>
    </div>`;
}

function overview(){
    $('gen').textContent = T.generated_at;
    $('ws').textContent = T.workspace.split('/').pop();
    $('counts').textContent = `${T.total_pages} pages`;

    $('p-overview').innerHTML = `
        <div class='section'><div class='section-head'><div><h2 class='title'>Workspace Overview</h2></div></div>
        <div class='grid4'>
            ${[['Tasks', T.task_total, 'var(--accent-purple)'], ['Projects', T.project_total, 'var(--accent-blue)'], ['Meetings', T.meeting_total, '#5AC8FA'], ['People', T.people_total, '#FF9500']]
                .map(([l,v,a])=>`<div class='metric' style='--accent:${a}'><div class='l'>${l}</div><div class='v'>${v}</div></div>`).join('')}
        </div></div>
        <div class='split'>
            <div class='section'><div class='section-head'><div><h3 class='title'>Focus</h3><div class='kicker'>Actionable tasks</div></div></div>
            <div class='stack'>${T.focus_items.slice(0,5).map(x=>card(x,false)).join('')||'<div class="item">Clear.</div>'}</div></div>
            <div class='section'><div class='section-head'><div><h3 class='title'>Recent Updates</h3></div></div>
            <div class='stack'>${T.recent_activity.slice(0,6).map(x=>
                `<div class='item'>
                    <div class='t'>${esc(x.title||('Inbox '+x.id))}</div>
                    <div class='m'>${esc(x.updated_human||x.created_human)}</div>
                    <div class='s'>${esc(x.summary||x.content_excerpt)}</div>
                </div>`
            ).join('')}</div></div>
        </div>
    `;
}

function tasks(){
    const f=TA.filter(x=>match(x) && (S.ts==='all'||x.status===S.ts) && (S.td==='all'||db(x)===S.td));
    let body='';
    if(S.tv==='kanban'){
        body=`<div class='kanban'>${['open','in-progress','blocked','done',''].map(s=>
            `<div class='lane'><h3>${({'open':'Open','in-progress':'In Progress','blocked':'Blocked','done':'Done','':'No Status'})[s]} <span style='color:var(--muted);font-weight:normal'>${f.filter(x=>(x.status||'')===s).length}</span></h3>
            <div class='stack'>${f.filter(x=>(x.status||'')===s).map(x=>card(x)).join('')}</div></div>`
        ).join('')}</div>`;
    } else {
        body=`<div class='stack'>${f.map(x=>card(x)).join('')}</div>`;
    }

    $('p-tasks').innerHTML=`
        <div class='filters'>
            <select id='ts'><option value='all'>All Status</option><option value='open'>Open</option><option value='in-progress'>In Progress</option><option value='blocked'>Blocked</option><option value='done'>Done</option></select>
            <select id='td'><option value='all'>All Dates</option><option value='overdue'>Overdue</option><option value='today'>Today</option><option value='week'>This Week</option><option value='later'>Later</option></select>
            <div class='tabbar' style='margin-left:auto'><button class='btn ${S.tv==='kanban'?'active':''}' data-tv='kanban'>Kanban</button><button class='btn ${S.tv==='list'?'active':''}' data-tv='list'>List</button></div>
        </div>
        ${body}
    `;
    $('ts').value=S.ts; $('td').value=S.td;
    $('ts').onchange=e=>{S.ts=e.target.value;tasks()}; $('td').onchange=e=>{S.td=e.target.value;tasks()};
    document.querySelectorAll('[data-tv]').forEach(b=>b.onclick=()=>{S.tv=b.dataset.tv;tasks()});
    bindCards();
}

function knowledge(){
    const types=D.meta.page_types;
    const f=P.filter(x=>match(x)&&(S.kt==='all'||x.type===S.kt));
    const groups={}; f.forEach(x=>(groups[x.type]=groups[x.type]||[]).push(x));

    $('p-knowledge').innerHTML=`
        <div class='chips' style='margin-bottom:24px'>
            <button class='pill ${S.kt==='all'?'active':''}' data-kt='all'>All</button>
            ${types.map(t=>`<button class='pill ${S.kt===t.key?'active':''}' data-kt='${esc(t.key)}'>${esc(t.label)}</button>`).join('')}
        </div>
        <div class='grid3'>
            ${types.filter(t=>groups[t.key]?.length).map(t=>
                `<div class='section'><h3 class='title' style='margin-bottom:16px;color:${cType(t.key)}'>${esc(t.label)}</h3>
                <div class='stack'>${groups[t.key].map(x=>card(x)).join('')}</div></div>`
            ).join('')}
        </div>
    `;
    document.querySelectorAll('[data-kt]').forEach(b=>b.onclick=()=>{S.kt=b.dataset.kt;knowledge()});
    bindCards();
}

function graph(){
    const qq=q();
    // Filter nodes: include if they have degrees, or if they match search.
    const nodes = D.graph.nodes.filter(n=>n.degree>0 || (qq && [n.label,n.ref,n.summary].join(' ').toLowerCase().includes(qq)));
    // Filter edges to only include those where both ends exist in our filtered node list.
    const edges = D.graph.edges.filter(e=>nodes.some(n=>n.ref===e.source)&&nodes.some(n=>n.ref===e.target));

    $('p-graph').innerHTML=`
        <div class='graphwrap'>
            <div class='graphstage' id='stage'>
                <svg id='graph-svg'>
                    <g id='graph-group'>
                        <g id='links'></g><g id='nodes'></g>
                    </g>
                </svg>
                <div class="graph-controls">
                    <button class="graph-btn" id="zoom-in" title="Zoom In">+</button>
                    <button class="graph-btn" id="zoom-out" title="Zoom Out">−</button>
                    <button class="graph-btn" id="zoom-reset" title="Reset View">⟲</button>
                </div>
            </div>
            <div class='graphpanel'>
                <div><div class='title' id='gtitle'>Select a node</div><div class='kicker' id='gmeta'>Click to inspect</div></div>
                <div class='s' id='gsum'></div>
                <div class='chips' id='gnei'></div>
            </div>
        </div>
    `;

    const svg = $('graph-svg');
    const graphGroup = $('graph-group');
    const gl = $('links');
    const gn = $('nodes');
    
    // Get actual dimensions of the container
    const rect = $('stage').getBoundingClientRect();
    const w = rect.width || 800;
    const h = rect.height || 600;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    // --- SVG Zoom & Pan Logic (Native JS) ---
    let zoomState = { scale: 1, tx: 0, ty: 0 };
    let isDragging = false;
    let dragStart = { x: 0, y: 0 };

    const updateTransform = () => {
        graphGroup.setAttribute('transform', `translate(${zoomState.tx}, ${zoomState.ty}) scale(${zoomState.scale})`);
    };

    const zoom = (factor, originX = w/2, originY = h/2) => {
        const newScale = Math.min(Math.max(0.2, zoomState.scale * factor), 5);
        if (newScale === zoomState.scale) return;
        
        // Calculate new translation to zoom towards the origin
        const dx = (originX - zoomState.tx) / zoomState.scale;
        const dy = (originY - zoomState.ty) / zoomState.scale;
        
        zoomState.scale = newScale;
        zoomState.tx = originX - dx * zoomState.scale;
        zoomState.ty = originY - dy * zoomState.scale;
        
        updateTransform();
    };

    $('zoom-in').onclick = () => zoom(1.3);
    $('zoom-out').onclick = () => zoom(0.7);
    $('zoom-reset').onclick = () => { zoomState = { scale: 1, tx: 0, ty: 0 }; updateTransform(); };

    // Mouse Wheel Zooming
    svg.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = svg.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        const factor = e.deltaY > 0 ? 0.9 : 1.1; // Smooth scrolling adjustment
        zoom(factor, mouseX, mouseY);
    }, { passive: false });

    // Panning
    svg.addEventListener('mousedown', (e) => {
        isDragging = true;
        dragStart = { x: e.clientX - zoomState.tx, y: e.clientY - zoomState.ty };
    });
    window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        zoomState.tx = e.clientX - dragStart.x;
        zoomState.ty = e.clientY - dragStart.y;
        updateTransform();
    });
    window.addEventListener('mouseup', () => isDragging = false);
    svg.addEventListener('mouseleave', () => isDragging = false);


    // --- Graph Data Initialization ---
    const N = new Map();
    // Initialize nodes around the center
    const seed = nodes.map((n,i)=>{
        const obj = {...n, x: w/2 + (Math.random()-0.5)*200, y: h/2 + (Math.random()-0.5)*200, vx:0, vy:0};
        N.set(n.ref, obj);
        return obj;
    });

    const L = edges.map(e=>({s: N.get(e.source), t: N.get(e.target)})).filter(e=>e.s&&e.t);
    const ne = new Map();
    L.forEach(e=>{
        (ne.get(e.s.ref)||ne.set(e.s.ref,new Set()).get(e.s.ref)).add(e.t.ref);
        (ne.get(e.t.ref)||ne.set(e.t.ref,new Set()).get(e.t.ref)).add(e.s.ref);
    });

    const rad = n => 6 + Math.min(12, Math.sqrt(n.degree||0)*2.5);

    // Create SVG elements
    const els = L.map(e=>{
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('stroke','var(--border)');
        line.setAttribute('stroke-width','1.5');
        gl.appendChild(line);
        return {el:line, s:e.s, t:e.t};
    });

    const nds = seed.map(n=>{
        const g = document.createElementNS('http://www.w3.org/2000/svg','g');
        const circle = document.createElementNS('http://www.w3.org/2000/svg','circle');
        const text = document.createElementNS('http://www.w3.org/2000/svg','text');
        
        circle.setAttribute('r', rad(n));
        circle.setAttribute('fill', cType(n.type));
        circle.setAttribute('stroke', 'var(--panel-solid)');
        circle.setAttribute('stroke-width', '2');
        
        text.textContent = n.label;
        text.setAttribute('font-size', '12');
        text.setAttribute('fill', 'var(--text)');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dy', rad(n) + 16);
        text.style.display = 'none'; // Hide text by default for less clutter
        
        g.append(circle, text);
        gn.appendChild(g);
        
        // Add hover interaction
        g.onmouseenter = (e) => { 
            if(isDragging) return;
            e.stopPropagation();
            hoverNode(n.ref); 
        };
        g.onmouseleave = () => hoverNode(null);
        g.onclick = (e) => {
            e.stopPropagation();
            sel(n.ref);
        }
        
        return {el:g, circle, text, n};
    });

    let hoveredNode = null;
    const hoverNode = (ref) => {
        hoveredNode = ref;
        updateVisuals();
    }

    const sel = (ref) => {
        S.gn = ref;
        const n = N.get(ref);
        if(!n) return;
        $('gtitle').textContent = n.label;
        $('gmeta').textContent = `${n.type_label} · ${n.degree} connections`;
        $('gsum').textContent = n.summary || '';
        $('gnei').innerHTML = [...(ne.get(ref)||[])].map(r=>`<span class='chip' data-s='${esc(r)}'>${esc(BY.get(r)?.title||r)}</span>`).join('');
        bindCards();
        updateVisuals();
    };

    const updateVisuals = () => {
        const qq=q();
        nds.forEach(o => {
            const isMatch = !qq || [o.n.label,o.n.ref,o.n.summary].join(' ').toLowerCase().includes(qq);
            const isSel = o.n.ref === S.gn;
            const isHov = o.n.ref === hoveredNode;
            const isNei = (S.gn && ne.get(S.gn)?.has(o.n.ref)) || (hoveredNode && ne.get(hoveredNode)?.has(o.n.ref));
            
            let opacity = 0.25;
            if(isMatch && !S.gn && !hoveredNode) opacity = 1;
            else if(isSel || isHov) opacity = 1;
            else if(isNei) opacity = 0.85;

            o.el.style.opacity = opacity;
            
            // Text logic: only show for selected, hovered, or matched elements to avoid overlap
            const showText = isSel || isHov || isNei || (qq.length > 1 && isMatch);
            o.text.style.display = showText ? 'block' : 'none';
            
            if (showText && (isSel || isHov)) {
                o.el.parentNode.appendChild(o.el); // Bring to front
            }
        });

        els.forEach(e => {
            const isSel = e.s.ref === S.gn || e.t.ref === S.gn;
            const isHov = e.s.ref === hoveredNode || e.t.ref === hoveredNode;
            const active = isSel || isHov;
            e.el.style.opacity = (!S.gn && !hoveredNode) ? 1 : (active ? 1 : 0.1);
            if(active) e.el.setAttribute('stroke', 'var(--muted)');
            else e.el.setAttribute('stroke', 'var(--border)');
        });
    };

    // Force simulation parameters (adjusted for more spacing)
    const charge = 6000;
    const spring = 0.04;
    const targetDist = 120; 
    const padding = 30;

    let animFrame;
    const tick = () => {
        // Apply repulsive charge
        for(let i=0; i<seed.length; i++){
            for(let j=i+1; j<seed.length; j++){
                const a=seed[i], b=seed[j];
                const dx=b.x-a.x, dy=b.y-a.y;
                const d2=dx*dx+dy*dy+1;
                const d=Math.sqrt(d2);
                const f=(charge/d2)*0.1;
                a.vx -= (dx/d)*f; a.vy -= (dy/d)*f;
                b.vx += (dx/d)*f; b.vy += (dy/d)*f;
            }
        }
        // Apply spring forces along links
        L.forEach(l=>{
            const dx=l.t.x-l.s.x, dy=l.t.y-l.s.y;
            const d=Math.sqrt(dx*dx+dy*dy)||1;
            const diff=d-targetDist;
            const f=(diff*spring);
            l.s.vx += (dx/d)*f; l.s.vy += (dy/d)*f;
            l.t.vx -= (dx/d)*f; l.t.vy -= (dy/d)*f;
        });

        // Update positions
        seed.forEach(n=>{
            // Gravity towards center
            n.vx += (w/2 - n.x)*0.008;
            n.vy += (h/2 - n.y)*0.008;
            
            n.vx *= 0.82; // friction
            n.vy *= 0.82;
            
            n.x += n.vx;
            n.y += n.vy;

            // Strict boundary constraint based on un-zoomed dimensions 
            // Allows panning to see pushed-out elements
            const r = rad(n);
            if (n.x < r + padding) { n.x = r + padding; n.vx *= -0.5; }
            if (n.x > w - r - padding) { n.x = w - r - padding; n.vx *= -0.5; }
            if (n.y < r + padding) { n.y = r + padding; n.vy *= -0.5; }
            if (n.y > h - r - padding) { n.y = h - r - padding; n.vy *= -0.5; }
        });

        // Update SVG attributes reading straight from the object coordinates
        els.forEach(l=>{
            l.el.setAttribute('x1', l.s.x); l.el.setAttribute('y1', l.s.y);
            l.el.setAttribute('x2', l.t.x); l.el.setAttribute('y2', l.t.y);
        });
        nds.forEach(o=>{
            o.el.setAttribute('transform', `translate(${o.n.x},${o.n.y})`);
        });

        if(S.tab === 'graph') {
            animFrame = requestAnimationFrame(tick);
        }
    };
    
    updateVisuals();
    if(seed.length > 0) {
         if(!S.gn || !N.has(S.gn)) sel(seed[0].ref);
         else sel(S.gn);
    }
    cancelAnimationFrame(animFrame);
    tick();
}

function advanced(){
    let items=P.filter(match);
    if(S.as==='title') items.sort((a,b)=>a.title.localeCompare(b.title));
    else if(S.as==='type') items.sort((a,b)=>(a.type_label).localeCompare(b.type_label)||a.title.localeCompare(b.title));
    else items.sort((a,b)=>(b.updated||'').localeCompare(a.updated||'')||a.title.localeCompare(b.title));

    $('p-advanced').innerHTML=`
        <div class='filters'>
            <select id='as'><option value='updated'>Recent First</option><option value='title'>Title A-Z</option><option value='type'>Type</option></select>
        </div>
        <div class='tablewrap'><table>
            <thead><tr><th>Title</th><th>Type</th><th>Status</th><th>Updated</th></tr></thead>
            <tbody>${items.map(x=>`
                <tr data-card='1' data-ref='${esc(x.ref)}'>
                    <td><div style='font-weight:600'>${esc(x.title)}</div><div class='muted' style='margin-top:4px;'>${esc(x.summary)}</div></td>
                    <td>${esc(x.type_label)}</td>
                    <td>${x.status?`<span class='badge ${st(x.status)}'>${esc(x.status_label)}</span>`:''}</td>
                    <td>${esc(x.updated_human)}</td>
                </tr>
            `).join('')}</tbody>
        </table></div>
    `;
    $('as').value=S.as;
    $('as').onchange=e=>{S.as=e.target.value;advanced()};
    bindCards();
}

function initTheme() {
    const root = document.documentElement;
    const toggle = document.getElementById('theme-toggle');
    const saved = localStorage.getItem('theme');
    if (saved) root.setAttribute('data-theme', saved);
    
    const updateIcon = () => {
        const isDark = root.getAttribute('data-theme') === 'dark' || (!root.hasAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
        toggle.innerHTML = isDark 
            ? `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>` 
            : `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
    };
    updateIcon();
    
    toggle.onclick = () => {
        root.classList.add('theme-transition');
        const isDark = root.getAttribute('data-theme') === 'dark' || (!root.hasAttribute('data-theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
        const newTheme = isDark ? 'light' : 'dark';
        root.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        updateIcon();
        setTimeout(() => root.classList.remove('theme-transition'), 800);
    };
}

function bindCards(){
    document.querySelectorAll('[data-card]').forEach(el=>el.onclick=e=>{
        const r = el.dataset.ref;
        if(S.tab==='graph') { sel(r); return; }
        // For other views, clicking a card searches its ref to isolate it
        S.q=r; $('q').value=r; refresh();
    });
    document.querySelectorAll('[data-s]').forEach(b=>b.onclick=e=>{
        e.stopPropagation(); S.q=b.dataset.s; $('q').value=S.q; refresh();
    });
}

function setTab(t){
    S.tab=t;
    document.querySelectorAll('.btn[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===t));
    document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===`p-${t}`));
    if(t==='graph') setTimeout(graph, 50); // slight delay to ensure container has dimensions
}

function refresh(){
    overview(); tasks(); knowledge(); advanced();
    if(S.tab==='graph') graph();
}

$('tabs').onclick=e=>{const b=e.target.closest('[data-tab]'); if(b)setTab(b.dataset.tab)};
$('q').oninput=e=>{S.q=e.target.value; refresh();};

refresh(); setTab('overview'); initTheme();
</script>
</body>
</html>
"""

def build_dashboard_html(cfg: Config) -> str:
    data = build_dashboard_data(cfg)
    return HTML_TEMPLATE.replace('@@DATA@@', _safe_json(data))


def write_dashboard(cfg: Config) -> Path:
    html = build_dashboard_html(cfg)
    out_dir = cfg.workspace / 'exports'
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'dashboard.html'
    path.write_text(html, encoding='utf-8')
    return path