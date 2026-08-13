from __future__ import annotations

import json

from graybox.migrate_obsidian import (
    _parse_note,
    _rewrite_links,
    migrate_vault,
)
from graybox.models import Page, now_iso
from graybox.storage import list_inbox_items, read_page, write_page


class StubLLM:
    def __init__(self, classify=None):
        self.classify = classify or (lambda _title: {"type": "topic", "summary": "", "aliases": []})

    def llm_call(self, *, system_prompt, prompt):
        title = next(line[7:] for line in prompt.splitlines() if line.startswith("Title: "))
        return {"response": json.dumps(self.classify(title))}


def _write_note(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_duplicate_titles_get_distinct_pages_and_path_keys(tmp_path, temp_cfg):
    vault = tmp_path / "vault"
    _write_note(vault, "one/note.md", "---\ntitle: Same\n---\nFirst")
    _write_note(vault, "two/note.md", "---\ntitle: Same\n---\nSecond")

    report = migrate_vault(temp_cfg, StubLLM(), vault)

    assert len(report.created) == 2
    assert {item.ref for item in report.created} == {"topic/same", "topic/same-two"}
    assert read_page(temp_cfg, "topic", "same").notes
    assert read_page(temp_cfg, "topic", "same-two").notes


def test_links_use_actual_fuzzy_dedup_target(tmp_path, temp_cfg):
    existing = Page(
        id="john-smith",
        type="person",
        title="John Smith",
        created=now_iso(),
        updated=now_iso(),
    )
    write_page(temp_cfg, existing)
    vault = tmp_path / "vault"
    _write_note(vault, "john.md", "---\ntitle: John Smit\n---\nSee [[John Smit]].")

    report = migrate_vault(
        temp_cfg,
        StubLLM(lambda _title: {"type": "person", "summary": "", "aliases": []}),
        vault,
    )

    assert len(report.merged) == 1
    loaded = read_page(temp_cfg, "person", "john-smith")
    assert "[[person/john-smith]]" in loaded.notes[0]
    assert "[[person/john-smit]]" not in loaded.notes[0]


def test_company_and_technology_slugs_do_not_overwrite(tmp_path, temp_cfg):
    vault = tmp_path / "vault"
    _write_note(vault, "technology.md", "---\ntitle: Acme\n---\nTechnology")
    _write_note(vault, "company.md", "---\ntitle: Acme!\n---\nCompany")

    def classify(title):
        page_type = "company" if title == "Acme!" else "technology"
        return {"type": page_type, "summary": "", "aliases": []}

    report = migrate_vault(temp_cfg, StubLLM(classify), vault)

    assert {item.ref.split("/", 1)[0] for item in report.created} == {
        "technology",
        "company",
    }
    for item in report.created:
        page_type, slug = item.ref.split("/", 1)
        assert read_page(temp_cfg, page_type, slug) is not None


def test_multiline_markdown_stays_one_note_after_round_trip(tmp_path, temp_cfg):
    vault = tmp_path / "vault"
    _write_note(
        vault,
        "note.md",
        "---\ntitle: Markdown\n---\nIntro\n- a list item\n## A heading\nThe rest",
    )

    migrate_vault(temp_cfg, StubLLM(), vault)
    loaded = read_page(temp_cfg, "topic", "markdown")

    assert len(loaded.notes) == 1
    assert "- a list item" in loaded.notes[0]
    assert "## A heading" in loaded.notes[0]


def test_frontmatter_and_inbox_provenance_are_preserved(tmp_path, temp_cfg):
    vault = tmp_path / "vault"
    _write_note(
        vault,
        "note.md",
        "---\n"
        "title: Metadata\n"
        "aliases: [Meta]\n"
        "tags: [imported, notes]\n"
        "date: 2026-08-13\n"
        "custom: keep-me\n"
        "---\nBody",
    )

    migrate_vault(temp_cfg, StubLLM(), vault)
    page = read_page(temp_cfg, "topic", "metadata")
    inbox = list_inbox_items(temp_cfg)[0]

    assert page.aliases == ["Meta"]
    assert page.tags == ["imported", "notes"]
    assert page.date == "2026-08-13"
    assert page.extra["obsidian_frontmatter"]["custom"] == "keep-me"
    assert inbox.extra["migrated_from"].endswith("/note.md")
    assert "custom: keep-me" in inbox.content


def test_malformed_frontmatter_does_not_abort_parsing(tmp_path):
    path = _write_note(tmp_path, "broken.md", "---\n- not-a-map\n---\nBody")

    note = _parse_note(path)

    assert note is not None
    assert note.title == "broken"
    assert note.body.strip() == "Body"


def test_wikilinks_preserve_ambiguous_and_unresolved_targets():
    body = "[[Missing]] ![[image.png]] [[Folder/Note#Intro|read this]]"

    rewritten = _rewrite_links(
        body,
        {"folder/note": "topic/note", "missing": None},
    )

    assert "[[Missing]]" in rewritten
    assert "![[image.png]]" in rewritten
    assert "[[topic/note#Intro|read this]]" in rewritten
