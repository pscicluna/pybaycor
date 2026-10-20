# Releasing pybaycor

Releases are cut by pushing a tag. The `Release` workflow
(`.github/workflows/release.yml`) then builds the sdist and wheel, tests the wheel in a
plain `pip` environment, and publishes:

| Tag | Example | Published to | GitHub Release / Zenodo DOI |
| --- | --- | --- | --- |
| final | `v0.3.0` | PyPI | yes |
| pre-release | `v0.3.0rc1`, `v0.3.0b1`, `v0.3.0a1` | TestPyPI only (a dry run) | no |

A pre-release runs the same checks (citation and changelog of the final version), the
build, the wheel tests and a trusted-publishing upload. It does not exercise the PyPI
publisher and `pypi` environment, the GitHub Release or Zenodo; those first run on the
final tag.

Uploads use PyPI trusted publishing, so no API token is stored in the repository.
The package version comes from the tag (`hatch-vcs`); nothing in the sources needs to
be bumped except the citation metadata and the changelog.

## One-time setup

These need maintainer access to the PyPI, TestPyPI, GitHub and Zenodo accounts.

1. **PyPI**: on <https://pypi.org/manage/project/pybaycor/settings/publishing/> add a
   trusted publisher: owner `pscicluna`, repository `pybaycor`, workflow
   `release.yml`, environment `pypi`.
2. **TestPyPI**: pybaycor already exists there (0.1.0, uploaded in 2021), so log in
   with the account that owns it and add a trusted publisher on
   <https://test.pypi.org/manage/project/pybaycor/settings/publishing/> with the same
   details and environment `testpypi`.
3. **GitHub**: create the environments `pypi` and `testpypi` (Settings →
   Environments). Optionally add yourself as a required reviewer on `pypi`, so a final
   release waits for a click before it uploads, and restrict both to `v*` tags. A tag
   ruleset (Settings → Rules → Rulesets) that lets only maintainers create `v*` tags
   additionally controls *who* can start a release.
4. **Zenodo**: log in to <https://zenodo.org> with GitHub and switch on
   `pscicluna/pybaycor` under *GitHub*. This must happen **before** the first release
   is published, or that release is not archived. Zenodo reads its metadata from
   `CITATION.cff`.

## Cutting a release

1. **Prepare** on a branch, and merge it to `main` through a PR as usual:
   - `pixi run sync-citation 0.3.0` sets `version` and `date-released` in
     `CITATION.cff` (CI accepts a citation that names a version newer than the latest
     release, so this can be merged before tagging);
   - in `CHANGELOG.md`, replace `unreleased` in the `## [0.3.0]` heading with the
     release date (`## [0.3.0] - 2026-10-21`); this section becomes the GitHub Release
     notes.
2. **Dry run** (recommended for the first release, and after changes to the
   workflow): tag the merge commit as a release candidate and push the tag.

   ```bash
   git switch main && git pull
   git tag -a v0.3.0rc1 -m "pybaycor 0.3.0rc1" && git push origin v0.3.0rc1
   ```

   Check the run in the Actions tab and the upload on
   <https://test.pypi.org/project/pybaycor/>. To try it, install the dependencies from
   PyPI and only pybaycor from TestPyPI (anyone can upload look-alike dependencies to
   TestPyPI, so don't let pip resolve them there):

   ```bash
   pip install "pymc>=6.1" "arviz>=1.1" numpy scipy pandas xarray matplotlib  # from PyPI
   pip install --no-deps -i https://test.pypi.org/simple/ pybaycor==0.3.0rc1
   ```

   If something needs fixing, merge the fix and tag `v0.3.0rc2`. Don't create a
   GitHub Release for a pre-release by hand: Zenodo would archive it and mint a DOI.
3. **Release**: tag the same commit as the final version and push the tag.

   ```bash
   git tag -a v0.3.0 -m "pybaycor 0.3.0" && git push origin v0.3.0
   ```

   Always use annotated tags (`-a`), as here. The build pins the version to the tag
   being released, so an rc and the final tag can share a commit.

   The workflow publishes to PyPI (after your approval, if the `pypi` environment
   requires it) and creates the GitHub Release with the changelog section as its notes
   and the distributions attached; Zenodo then mints the DOI.
4. **Afterwards**:
   - after the first Zenodo release, add the *concept* DOI (the one that always
     resolves to the latest version) to `CITATION.cff` (`doi:`) and a DOI badge to the
     README;
   - start the next changelog section, `## [0.3.1] - unreleased` (or `0.4.0`).

## If something goes wrong

- **The workflow fails before publishing** (metadata check, build or tests): fix it on
  `main`, then move the tag (`git tag -d v0.3.0 && git push origin :refs/tags/v0.3.0`)
  and tag the fixed commit. Nothing has been uploaded, so the version can be reused.
- **The upload succeeded**: PyPI never accepts the same version twice, even after a
  deletion. Fix the problem and release the next patch version instead.
- **Only part of the upload succeeded** (e.g. the wheel but not the sdist): re-running
  the job fails with "File already exists" for the uploaded file (a re-run uses the
  workflow as tagged, so it cannot be told to skip it). Either upload the missing file
  by hand from the run's `dist` artifact (`twine upload dist/<file>` with a PyPI API
  token scoped to pybaycor), or release the next patch version.
- **The GitHub Release step failed after the PyPI upload**: re-run that job from the
  Actions tab. To create the release by hand instead, from a checkout of the tag with
  the files from the run's `dist` artifact in `dist/`:

  ```bash
  python scripts/release_notes.py v0.3.0 > notes.md
  gh release create v0.3.0 dist/* --verify-tag --title "pybaycor 0.3.0" --notes-file notes.md
  ```
- **Zenodo did not archive the release**: check that the repository is switched on at
  Zenodo, then in the repository's Settings → Webhooks → Zenodo → Recent Deliveries,
  find the `release` event and use **Redeliver**.
