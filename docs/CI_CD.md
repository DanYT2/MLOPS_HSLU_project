# CI/CD Architecture

This document describes the continuous-integration and continuous-delivery
setup for the Customer Churn Prediction stack. It is the source of truth
for **what runs when, why each job exists, what it gates, and how to
extend or debug it**. The workflow YAML files in `.github/workflows/` are
heavily commented and are the second source — this document is the
big-picture view that those comments can't easily express.

---

## 1. Goals

The pipeline is designed around four goals, in priority order:

1. **Keep `main` always shippable.** No commit can land on `main` without
   passing lint, unit tests with ≥50% coverage, image builds, secret
   scanning, and the GitHub-managed CodeQL default scan.
2. **Make every artifact traceable.** Every published container image is
   tagged with the commit SHA that produced it; every release attaches an
   SBOM derived from the same lockfile that fed the image build.
3. **Fail fast where it's cheap.** Lint runs before tests; PR-time scans
   run in parallel so code-level and image-level regressions are reported
   simultaneously.
4. **Be explicit about what is enforcing vs. informational.** Hard gates
   (lint, tests, gitleaks) fail PRs. Soft signals (CodeQL alerts, format
   check) surface in the Security tab without blocking — calibrated so
   teams aren't pressured into rubber-stamping noisy findings.

---

## 2. Pipeline at a glance

```
                           push to feature branch
                           ───────────────────────
                                    │
                                    ▼
                  ┌──────────────────────────────────────┐
                  │ CI (ci.yml)                          │
                  │   lint  →  tests + coverage gate     │
                  └──────────────────────────────────────┘

                       PR opened against main
                       ──────────────────────
                                    │
                ┌───────────────────┼───────────────────────┐
                ▼                   ▼                       ▼
        ┌───────────────┐   ┌────────────────┐     ┌────────────────────┐
        │ CI (ci.yml)   │   │ Docker build   │     │ Secret Scan        │
        │ lint + tests  │   │ (docker.yml)   │     │ (secret-scan.yml)  │
        │ + coverage    │   │ build only on  │     │ Gitleaks           │
        └───────────────┘   │ PR (no push)   │     └────────────────────┘
                            └────────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │ CodeQL (GitHub default setup)   │
                    │ Auto-managed by GitHub          │
                    └─────────────────────────────────┘

                            merge to main
                            ─────────────
                                    │
                                    ▼
                ┌──────────────────────────────────────────────┐
                │ Docker push (docker.yml on push:main)        │
                │   tags: main, sha-<short>, latest            │
                │   includes SBOM + SLSA provenance            │
                └──────────────────────────────────────────────┘

                            tag v1.2.3
                            ──────────
                                    │
                ┌───────────────────┴────────────────────────────┐
                ▼                                                ▼
        ┌───────────────┐                              ┌──────────────────┐
        │ docker.yml    │                              │ release.yml      │
        │  builds with  │                              │  generate notes  │
        │  semver tags  │                              │  + SBOM artifact │
        └───────────────┘                              │  + promote image │
                                                       │    tags via      │
                                                       │   imagetools     │
                                                       └──────────────────┘

                            cron / manual
                            ─────────────
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
                  secret-scan.yml          train.yml
                  (weekly Mon 05Z)         (manual dispatch only)
```

---

## 3. Workflow matrix

The full list of workflows in `.github/workflows/` and their roles.

| Workflow | File | Triggers | Gate? | Purpose |
|---|---|---|---|---|
| CI | `ci.yml` | every push, PR → main | **enforcing** | Ruff lint, pytest with `--cov-fail-under=50` |
| Build & Push | `docker.yml` | push:main, push:tag, PR → main, manual | **enforcing for build**; push is conditional | Build api/mlflow/monitor images with SBOM + SLSA provenance |
| Release | `release.yml` | push of `v*.*.*` tag | n/a (post-merge) | Generate release notes, attach CycloneDX SBOM, promote image tags |
| Secret Scan | `secret-scan.yml` | push:main, PR → main, weekly cron | **enforcing** | Gitleaks history scan |
| Train Model | `train.yml` | manual dispatch only | n/a | Full training run, uploads model artifacts |

**Outside of `.github/workflows/`:** CodeQL runs under GitHub's
*default setup* (Settings → Code security → Code scanning). This is
GitHub-managed — there is no YAML to maintain, no query pack to
configure. Findings appear in the Security → Code scanning tab. To
move CodeQL into a workflow file with custom queries instead, switch
to advanced setup in the same UI page.

---

## 4. Trigger reference

A different view: what fires on which event.

| Event | Workflows |
|---|---|
| Push to feature branch | `ci.yml` |
| Open or update a PR → `main` | `ci.yml`, `docker.yml` (build-only), `secret-scan.yml` + GitHub-managed CodeQL |
| Merge to `main` | `ci.yml`, `docker.yml` (push), `secret-scan.yml` + GitHub-managed CodeQL |
| Push of `v*.*.*` tag | `docker.yml` (push with semver tags), `release.yml` |
| Weekly cron (Mon 05Z) | `secret-scan.yml` |
| Manual dispatch | `train.yml`, `docker.yml`, `secret-scan.yml` |

---

## 5. Branching and release model

The model is **trunk-based with semver tags**:

1. Develop on feature branches. CI runs lint + tests on every push.
2. Open a PR to `main`. The full PR suite runs. Required checks prevent
   merge until enforcing jobs pass.
3. Merge to `main`. `docker.yml` republishes images under `main`,
   `latest`, and `sha-<short>` tags. These are the rolling-deploy
   candidates if you operate continuous delivery.
4. To cut a release, push an annotated tag `vX.Y.Z` from `main`:

   ```bash
   git tag -a v1.4.2 -m "1.4.2"
   git push origin v1.4.2
   ```

   Both `docker.yml` and `release.yml` fire on the tag push. `docker.yml`
   builds + pushes the same images under semver tags. `release.yml`
   then re-tags the `sha-<short>` digest under additional aliases
   (`v_major`, `v_minor`, `v_full`, `latest`) using
   `docker buildx imagetools create` — this is a manifest-only operation,
   so the digest and its SBOM/provenance attestations carry over.

The tag push is also the only path that produces a GitHub Release.
Release notes are generated automatically from PR titles since the
previous tag.

### Why `imagetools create` instead of rebuilding?

Rebuilding at release time risks pulling new base-image SHAs (Debian
slim, Python 3.13-slim) or updated PyPI metadata that could change the
image bit-for-bit without changing the source. Re-tagging an existing
digest guarantees the artifact in production is exactly the artifact
that passed all gates on `main`.

---

## 6. Image references

All images are published to GitHub Container Registry under the repo
owner's namespace, lowercased:

```
ghcr.io/<owner>/churn-api
ghcr.io/<owner>/churn-mlflow
ghcr.io/<owner>/churn-monitor
```

Tag conventions:

| Tag pattern | Produced by | Mutable? |
|---|---|---|
| `sha-<7char>` | docker.yml (always) | no — keyed on git SHA |
| `pr-<num>` | docker.yml (PR only, build-only, never pushed) | n/a — never published |
| `main` | docker.yml on push:main | yes — moves with main |
| `latest` | docker.yml on push:main, release.yml on tag | yes — moves with main / latest tag |
| `<branch-name>` | docker.yml on push to a non-main branch (build-only) | yes |
| `<major>` (e.g. `1`) | release.yml | yes — moves with each minor |
| `<major>.<minor>` (e.g. `1.4`) | release.yml | yes — moves with each patch |
| `<full>` (e.g. `1.4.2`) | release.yml | no — immutable |

For reproducible deploys, **always reference images by `sha-<7char>` or
by full `<major>.<minor>.<patch>` tag**, never `latest` or `main`.

### Attestations

Each pushed image carries two attestations under its referrers list:

- **CycloneDX SBOM** — the dependency manifest, produced by
  `docker/build-push-action` with `sbom: true`.
- **SLSA v1 provenance** — builder identity + git SHA, produced with
  `provenance: mode=max`.

Verify with `cosign verify-attestation` against the GHCR issuer.

---

## 7. Secrets

| Secret | Used by | Required? | Purpose |
|---|---|---|---|
| `GITHUB_TOKEN` | all workflows | auto-injected | GHCR auth, code-scanning uploads, release publish |
| `TEST_DATASET_URL` | `train.yml` | only for training | URL to download `test.csv` if not provided as `dataset_url` input |

No long-lived registry credentials are stored — GHCR auth flows through
the workflow's `GITHUB_TOKEN`, scoped only to the running repo.

---

## 8. Caching strategy

Two cache layers, each keyed deliberately:

- **uv resolver cache** — `astral-sh/setup-uv@v6` with
  `cache-dependency-glob: "uv.lock"`. Invalidated by any change to the
  lockfile. A hot cache makes `uv sync` complete in 5–10 seconds.

- **Docker buildx GHA cache** — `cache-from`/`cache-to` of
  `type=gha,scope=<image-name>`. Per-image scope so a churn-api build
  doesn't evict churn-mlflow layers. `mode=max` exports every
  intermediate stage's layers, not just the final image.

---

## 9. Gating logic — what actually blocks a merge

Configure these as required status checks on the `main` branch in
**Settings → Branches → Branch protection rules**:

- `CI / Lint (ruff)`
- `CI / Tests (pytest) (3.13)`
- `Build and Push Images / Build churn-api`
- `Build and Push Images / Build churn-mlflow`
- `Build and Push Images / Build churn-monitor`
- `Secret Scan / Gitleaks`

The following are **deliberately not** required, because their failure
modes don't justify blocking PRs:

- GitHub-managed CodeQL findings — surface in the Security tab; query
  packs evolve weekly, so new findings on unchanged code shouldn't
  block unrelated PRs.
- Ruff format check (within ci.yml) — `continue-on-error: true`.

---

## 10. Training workflow — when and how

`train.yml` is **manual only**. There is no path that triggers it
automatically. To run a training:

1. Go to **Actions → Train Model → Run workflow**.
2. Optionally set `dataset_url` if you want to use a non-default
   training CSV. The companion `TEST_DATASET_URL` must be set as a
   repo secret if you go this route.
3. The job spins up MLflow locally on the runner, runs `train.py`,
   and uploads:
   - `training-artifacts` (submission.csv, optuna_studies.db, mlflow.db)
   - `mlruns` (the full local artifact tree, including the registered
     ensemble pyfunc model)

Replay locally:

```bash
gh run download <run-id> -n mlruns -D ./mlruns
gh run download <run-id> -n training-artifacts -D .
docker compose up mlflow   # serves the downloaded mlruns/ via the image
```

> **Note:** `train.py` currently hardcodes `MLFLOW_TRACKING_URI=http://localhost:5001`
> in its top-of-file constants. The workflow exports the same URL for
> consistency but the env var is not yet consumed by the script. Update
> the script to read `MLFLOW_TRACKING_URI` and the env var becomes
> effective.

---

## 11. Local validation before pushing

The same checks that CI runs are reproducible locally:

```bash
# Lint
uv run ruff check .
uv run ruff format --check .

# Tests with coverage (matches ci.yml exactly)
PYTHONPATH=project uv run pytest project/tests \
  --cov=project --cov-report=term --cov-fail-under=50

# Image builds (matches docker.yml's build step)
docker compose -f project/docker-compose.yml build
```

---

## 12. Troubleshooting

**`ci.yml` fails on import of `schemas.schemas`.**
The PYTHONPATH wiring is in three places — pyproject.toml's pytest
config, conftest.py, and the CI env. All three must agree on `project/`
being on the path. See the "Import layout (gotcha)" section of
`CLAUDE.md`.

**`docker.yml` fails to push to GHCR with 403.**
The repo's package settings must allow GitHub Actions to publish. Check
**Settings → Actions → General → Workflow permissions** is set to
"Read and write".

**`release.yml`'s `promote-images` step can't find the source tag.**
The `sha-<short>` source tag is computed from `github.sha`. It is
produced by docker.yml on the same tag-push event. If docker.yml's
build for that SHA failed or was skipped, the promotion fails. Re-run
docker.yml for that ref and re-run release.yml.

**CodeQL workflow fails with "CodeQL analyses from advanced configurations cannot be processed when the default setup is enabled".**
Default setup and advanced (workflow-based) setup are mutually
exclusive. Either delete the workflow-based codeql.yml, or switch
default setup off in **Settings → Code security → Code scanning**.

**CodeQL marks a finding I know is safe.**
Dismiss it from the Security tab with a reason. Don't add suppression
comments to source — those persist beyond the fix and rot.

---

## 13. Extending the pipeline

**Add a new container image:**
1. Add the Dockerfile + Compose service.
2. Append the image to the matrix in `docker.yml` and `release.yml`
   (promote-images job).

**Add a new gate (e.g. type checks with mypy):**
1. Add `mypy` to `[dependency-groups.dev]` in `pyproject.toml`.
2. Add a new job to `ci.yml` mirroring the structure of `lint`.
3. Once the project is mypy-clean, add the new job to the required
   status checks (section 9).

**Bump the coverage threshold:**
Edit `--cov-fail-under=50` in `ci.yml`. The threshold should track
realistic project coverage — bumping it ahead of test growth is a
recipe for blocked PRs. Most of the uncovered surface is `monitor.py`'s
`run()` replay loop; raise the floor as the unit-testable surface
grows.

**Bump Python:**
Update `.python-version`, `requires-python` in `pyproject.toml`, and
the matrix in `ci.yml`. The setup-uv action installs the requested
version on the fly; no other workflow needs editing.

**Re-add the smoke/security/CodeQL workflows:**
The earlier iteration of this pipeline included `smoke-test.yml`,
`security.yml`, and `codeql.yml`. They were removed because:
- CodeQL conflicted with GitHub's default setup (see §12).
- security.yml's pip-audit / Trivy gates produced too much CVE noise
  for the project's scope at the time.
- smoke-test.yml's compose boot was redundant with image-build success.

Git history has working versions of all three if you want to revive
them. CodeQL specifically requires disabling default setup first.
