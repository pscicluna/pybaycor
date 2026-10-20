"""Keep the ``version`` and ``date-released`` fields of CITATION.cff in step with releases.

The citation file describes the latest *final* release (``vX.Y.Z`` tags; pre-release
tags such as ``v0.3.0rc1`` are ignored), or the release being prepared: a release is
cut by first committing CITATION.cff with the new version (``sync_citation.py 0.3.0``)
and then tagging that commit, so the tagged sources, the sdist and the Zenodo archive
all carry the right metadata.

Usage::

    python scripts/sync_citation.py 0.3.0      # prepare a release (date = today)
    python scripts/sync_citation.py            # rewrite from the latest final release tag
                                               # (dated with the tagged commit)
    python scripts/sync_citation.py --check    # CI: exit 1 unless CITATION.cff names the
                                               # latest release or a newer one
    python scripts/sync_citation.py --check-release v0.3.0rc1
                                               # release workflow: exit 1 unless
                                               # CITATION.cff names 0.3.0
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

_FINAL = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")
# PEP 440 pre-releases as used in tags: v0.3.0a1, v0.3.0b2, v0.3.0rc1.
_PRE = re.compile(r"^v?(\d+\.\d+\.\d+)(?:a|b|rc)\d+$")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def parse_version(text: str) -> tuple[int, int, int]:
    """Return ``(major, minor, patch)`` of a final ``X.Y.Z`` (or ``vX.Y.Z``) version."""
    match = _FINAL.match(text.strip())
    if match is None:
        raise ValueError(f"{text!r} is not a final X.Y.Z version")
    major, minor, patch = (int(g) for g in match.groups())
    return major, minor, patch


def release_version(tag: str) -> str:
    """Return the final version a release tag belongs to (``v0.3.0rc1`` -> ``0.3.0``)."""
    if (match := _FINAL.match(tag)) is not None:
        return ".".join(match.groups())
    if (match := _PRE.match(tag)) is not None:
        return match.group(1)
    raise ValueError(f"{tag!r} is not a release tag (vX.Y.Z, or vX.Y.Z{{a,b,rc}}N)")


def latest_release() -> tuple[str, str] | None:
    """Return ``(version, date)`` of the newest final ``vX.Y.Z`` tag reachable from HEAD."""
    tags = _git("tag", "--list", "v*", "--merged", "HEAD").split()
    finals = [t for t in tags if _FINAL.match(t)]
    if not finals:
        return None
    tag = max(finals, key=parse_version)
    return tag.removeprefix("v"), _git("log", "-1", "--format=%cs", tag)


def read_fields(text: str) -> tuple[str, str | None]:
    """Return the ``version`` and ``date-released`` values of a CITATION.cff text."""
    version = re.search(r"(?m)^version:\s*['\"]?([^'\"\s]+)", text)
    if version is None:
        raise SystemExit("CITATION.cff has no top-level 'version:' line")
    date = re.search(r"(?m)^date-released:\s*['\"]?([^'\"\s]+)", text)
    return version.group(1), date.group(1) if date else None


def check(cited: str, cited_date: str | None, latest: tuple[str, str] | None) -> str | None:
    """Return an error message if the cited version is behind the latest release.

    The citation may name the latest final release or a newer version that is being
    prepared for release. ``date-released`` must be a valid date but is not compared
    with the tag: it is set when the release is prepared, which can be a day or two
    before the release commit is tagged.
    """
    try:
        cited_key = parse_version(cited)
    except ValueError as err:
        return f"CITATION.cff version: {err}"
    try:
        # fromisoformat also accepts e.g. 20261020 and 2026-W43-2, which CFF does not.
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", cited_date or ""):
            raise ValueError
        dt.date.fromisoformat(cited_date or "")
    except ValueError:
        return f"CITATION.cff date-released must be a YYYY-MM-DD date; got {cited_date!r}"
    if latest is not None and cited_key < parse_version(latest[0]):
        return f"CITATION.cff names {cited}, older than the latest release {latest[0]}"
    return None


def check_release(cited: str, tag: str) -> str | None:
    """Return an error message unless the cited version is the one ``tag`` releases."""
    try:
        expected = release_version(tag)
    except ValueError as err:
        return str(err)
    if cited != expected:
        return (
            f"CITATION.cff names {cited} but the tag {tag} releases {expected}; commit "
            f"`python scripts/sync_citation.py {expected}` before tagging"
        )
    return None


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
    parser.add_argument("version", nargs="?", help="version to write (default: latest release)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if behind the latest release")
    mode.add_argument("--check-release", metavar="TAG", help="fail unless TAG's version is cited")
    args = parser.parse_args(argv)
    text = CITATION.read_text(encoding="utf-8")

    if args.check or args.check_release:
        if args.version:
            parser.error("a version cannot be combined with --check/--check-release")
        cited, cited_date = read_fields(text)
        if args.check_release:
            error = check_release(cited, args.check_release)
        else:
            error = check(cited, cited_date, latest_release())
        if error:
            print(error)
            return 1
        print(f"CITATION.cff is consistent (version {cited})")
        return 0

    if args.version:
        try:
            parse_version(args.version)  # only final versions are cited
        except ValueError as err:
            raise SystemExit(str(err)) from err
        version, date = args.version.removeprefix("v"), dt.date.today().isoformat()
    else:
        latest = latest_release()
        if latest is None:
            raise SystemExit("No vX.Y.Z release tag is reachable from HEAD")
        version, date = latest
    new = render(text, version, date)
    if new != text:
        CITATION.write_text(new, encoding="utf-8")
        print(f"CITATION.cff updated to version {version} ({date})")
    else:
        print(f"CITATION.cff already at version {version} ({date})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
