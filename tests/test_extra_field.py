from __future__ import annotations

from graybox.models import Page, now_iso
from graybox.storage import write_page, read_page


class TestPageExtraField:
    def test_extra_field_exists_and_defaults_empty(self):
        page = Page(id="x", type="topic", title="X", created=now_iso(), updated=now_iso())
        assert page.extra == {}

    def test_extra_data_survives_round_trip(self, temp_cfg):
        page = Page(
            id="x", type="topic", title="X", created=now_iso(), updated=now_iso(),
            extra={"source_system": "slack", "external_id": "T123"},
        )
        write_page(temp_cfg, page)
        loaded = read_page(temp_cfg, "topic", "x")
        assert loaded.extra == {"source_system": "slack", "external_id": "T123"}