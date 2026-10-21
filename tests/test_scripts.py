"""Tests for the release helper scripts in ``scripts/``."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if not SCRIPTS.is_dir():  # e.g. testing from the sdist, which ships tests but not scripts
    pytest.skip("scripts/ is not available", allow_module_level=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


citation = _load("sync_citation")
notes = _load("release_notes")


class TestCitationCheck:
    def test_latest_release_passes(self):
        assert citation.check("0.2.1", "2021-03-24", ("0.2.1", "2021-03-24")) is None

    def test_date_may_precede_the_tagged_commit(self):
        # The release is prepared (and dated) before the merge commit is tagged.
        assert citation.check("0.3.0", "2026-10-20", ("0.3.0", "2026-10-22")) is None

    @pytest.mark.parametrize("date", [None, "2026-13-01", "October 2026", "20261020", "2026-W43-2"])
    def test_invalid_date_fails(self, date):
        assert "date-released" in citation.check("0.3.0", date, ("0.2.1", "2021-03-24"))

    def test_release_in_preparation_passes(self):
        # A release-prep commit cites the new version before it is tagged.
        assert citation.check("0.3.0", "2026-10-21", ("0.2.1", "2021-03-24")) is None

    def test_older_than_latest_release_fails(self):
        assert "older than" in citation.check("0.2.0", "2021-03-01", ("0.2.1", "2021-03-24"))

    def test_versions_compare_numerically(self):
        assert citation.check("0.10.0", "2025-02-01", ("0.9.0", "2025-01-01")) is None

    def test_no_release_yet_passes(self):
        assert citation.check("0.1.0", "2020-01-01", None) is None

    def test_non_final_cited_version_fails(self):
        assert "not a final" in citation.check("0.3.0rc1", "2026-10-20", ("0.2.1", "2021-03-24"))


class TestCitationReleaseCheck:
    @pytest.mark.parametrize("tag", ["v0.3.0", "v0.3.0rc1", "v0.3.0a2", "v0.3.0b1"])
    def test_tag_of_cited_version_passes(self, tag):
        assert citation.check_release("0.3.0", tag) is None

    def test_tag_of_other_version_fails(self):
        assert "releases 0.3.1" in citation.check_release("0.3.0", "v0.3.1")

    @pytest.mark.parametrize("tag", ["v0.3", "v0.3.0.post1", "v0.3.0-rc1", "latest"])
    def test_non_release_tag_fails(self, tag):
        assert "not a release tag" in citation.check_release("0.3.0", tag)

    def test_read_fields(self):
        text = "cff-version: 1.2.0\nversion: 0.3.0\ndate-released: '2026-10-21'\n"
        assert citation.read_fields(text) == ("0.3.0", "2026-10-21")


CHANGELOG = """# Changelog

## [0.4.0] - unreleased

- Upcoming.

## [0.3.0] - 2026-10-21

### Added

- Everything.

## [0.2.0] - 2021-03

- Old.
"""


class TestReleaseNotes:
    def test_section_body_stops_at_next_heading(self):
        heading, body = notes.section(CHANGELOG, "0.3.0")
        assert heading == "## [0.3.0] - 2026-10-21"
        assert body == "### Added\n\n- Everything."

    def test_last_section_runs_to_end(self):
        assert notes.section(CHANGELOG, "0.2.0")[1] == "- Old."

    def test_ready_section_has_no_problems(self):
        assert notes.problems(CHANGELOG, "0.3.0") == []

    def test_unreleased_section_is_not_ready(self):
        found = notes.problems(CHANGELOG, "0.4.0")
        assert any("unreleased" in p for p in found)
        assert any("release date" in p for p in found)

    def test_missing_section_is_not_ready(self):
        assert notes.problems(CHANGELOG, "0.5.0") == ["CHANGELOG.md has no '## [0.5.0]' section"]

    @pytest.mark.parametrize("tag", ["v0.3.0", "0.3.0", "v0.3.0rc1", "v0.3.0a2", "v0.3.0b1"])
    def test_tags_map_to_their_final_version(self, tag):
        assert notes.final_version(tag) == "0.3.0"

    def test_version_is_not_a_regex(self):
        # "0.3.0" must not match a "0x3y0" heading through regex wildcards.
        with pytest.raises(LookupError):
            notes.section("## [0x3y0] - 2026-01-01\n- x\n", "0.3.0")
