"""Extract the release notes for a version from CHANGELOG.md.

The release workflow uses this to fill in the GitHub Release (which Zenodo archives),
and to refuse to publish a version whose changelog section is missing or still marked
``unreleased``.

A pre-release tag refers to the section of its final version (``v0.3.0rc1`` -> 0.3.0),
so a release-candidate dry run checks the notes the final release will publish.

Usage::

    python scripts/release_notes.py v0.3.0           # print the notes for 0.3.0
    python scripts/release_notes.py v0.3.0 --check   # exit 1 unless they are ready
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"

_PRE_SUFFIX = re.compile(r"(?:a|b|rc)\d+$")


def final_version(tag: str) -> str:
    """Return the final version a tag refers to (``v0.3.0rc1`` -> ``0.3.0``)."""
    return _PRE_SUFFIX.sub("", tag.removeprefix("v"))


def section(text: str, version: str) -> tuple[str, str]:
    """Return ``(heading, body)`` of the ``## [version]`` section of a changelog."""
    heading = re.search(rf"(?m)^## \[{re.escape(version)}\][^\n]*$", text)
    if heading is None:
        raise LookupError(f"CHANGELOG.md has no '## [{version}]' section")
    rest = text[heading.end() :]
    following = re.search(r"(?m)^## ", rest)
    body = rest[: following.start()] if following else rest
    return heading.group(0), body.strip()


def problems(text: str, version: str) -> list[str]:
    """Return the reasons the notes for ``version`` are not ready to publish."""
    try:
        heading, body = section(text, version)
    except LookupError as err:
        return [str(err)]
    found = []
    if not re.search(r"\d{4}-\d{2}-\d{2}", heading):
        found.append(f"the heading '{heading}' needs the release date (YYYY-MM-DD)")
    if "unreleased" in heading.lower():
        found.append(f"the heading '{heading}' is still marked unreleased")
    if not body:
        found.append(f"the '{heading}' section is empty")
    return found


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tag", help="release tag or version, e.g. v0.3.0")
    parser.add_argument("--check", action="store_true", help="only check the notes are ready")
    args = parser.parse_args(argv)
    version = final_version(args.tag)
    text = CHANGELOG.read_text(encoding="utf-8")
    if args.check:
        found = problems(text, version)
        for problem in found:
            print(f"CHANGELOG.md: {problem}")
        return 1 if found else 0
    try:
        print(section(text, version)[1])
    except LookupError as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
