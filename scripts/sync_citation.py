"""Keep the ``version`` and ``date-released`` fields of CITATION.cff in step with git tags.

The citation file describes the latest *released* version, so it is synchronised with
the newest ``v*`` tag reachable from the current commit (the same tags that
``hatch-vcs`` derives the package version from).

Usage::

    python scripts/sync_citation.py            # rewrite CITATION.cff from the latest tag
    python scripts/sync_citation.py --check    # exit 1 if CITATION.cff is out of date
    python scripts/sync_citation.py 0.3.0      # use an explicit version (date = today)
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CITATION = ROOT / "CITATION.cff"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def latest_tag() -> tuple[str, str]:
    """Return ``(version, date)`` of the newest ``v*`` tag reachable from HEAD."""
    tag = _git("describe", "--tags", "--abbrev=0", "--match", "v*")
    date = _git("log", "-1", "--format=%cs", tag)
    return tag.removeprefix("v"), date


def render(text: str, version: str, date: str) -> str:
    """Return ``text`` with the version and date-released fields set."""
    text, n_version = re.subn(r"(?m)^version:.*$", f"version: {version}", text)
    if n_version != 1:
        raise SystemExit("CITATION.cff must contain exactly one top-level 'version:' line")
    line = f"date-released: '{date}'"
    text, n_date = re.subn(r"(?m)^date-released:.*$", line, text)
    if n_date == 0:
        text = re.sub(r"(?m)^(version:.*$)", rf"\1\n{line}", text)
    return text


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("version", nargs="?", help="version to write (default: latest tag)")
    parser.add_argument("--check", action="store_true", help="only report whether in sync")
    args = parser.parse_args(argv)

    if args.version:
        version, date = args.version.removeprefix("v"), dt.date.today().isoformat()
    else:
        version, date = latest_tag()
    old = CITATION.read_text(encoding="utf-8")
    new = render(old, version, date)
    if args.check:
        if new != old:
            print(f"CITATION.cff is out of date: expected version {version} ({date})")
            return 1
        print(f"CITATION.cff is in sync with tag v{version}")
        return 0
    if new != old:
        CITATION.write_text(new, encoding="utf-8")
        print(f"CITATION.cff updated to version {version} ({date})")
    else:
        print(f"CITATION.cff already at version {version} ({date})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
