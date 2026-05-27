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
   passing lint, unit tests with ≥60% coverage, container image builds,
   a stack-level smoke test, and security scans.
2. **Make every artifact traceable.** Every published container image is
   tagged with the commit SHA that produced it; every release attaches an
   SBOM derived from the same lockfile that fed the image build.
3. **Fail fast where it's cheap.** Lint runs before tests; tests run
   before the heavyweight Trivy/CodeQL scans on pushes. Smoke tests boot
   in parallel so an image-level regression and a code-level regression
   are reported simultaneously.
4. **Be explicit about what is enforcing vs. informational.** Hard gates
   (lint, tests, pip-audit, gitleaks) fail PRs. Soft signals (Trivy,
   CodeQL alerts, format check) surface in the Security tab or run summary
   without blocking — calibrated so teams aren't pressured into rubber-
   stamping noisy findings.

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
                ┌───────────────────┼───────────────────────────┐
                ▼                   ▼                           ▼
        ┌───────────────┐   ┌───────────────┐         ┌───────────────────┐
        │ CI (ci.yml)   │   │ Smoke Test    │         │ Security Scans    │
        │ lint + tests  │   │ (smoke-       │         │ (security.yml)    │
        │ + coverage    │   │  test.yml)    │         │   pip-audit + Trivy
        └───────────────┘   │ compose boot  │         └───────────────────┘
                            │ + import smoke│
                ┌───────────┴───────────────┴───────────────────┐
                ▼                                               ▼
        ┌───────────────┐                               ┌───────────────┐
        │ CodeQL        │                               │ Secret Scan   │
        │ (codeql.yml)  │                               │ (secret-      │
        │ Python SAST   │                               │  scan.yml)    │
        └───────────────┘                               │ Gitleaks      │
                                                        └───────────────┘
                                    │
                                    ▼
                ┌──────────────────────────────────────────────┐
                │ Docker build & push (docker.yml)             │
                │   builds api/mlflow/monitor images           │
                │   push: false on PRs (build-only)            │
                └──────────────────────────────────────────────┘

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

                            cron (off-hours)
                            ────────────────
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
   security.yml                codeql.yml                  secret-scan.yml
   (weekly Mon 06Z)           (weekly Sun 04Z)            (weekly Mon 05Z)
```

---

## 3. Workflow matrix

The full list of workflows and their roles. Every workflow declares the
minimum permissions required and explains those declarations in its
own header comment.

| Workflow | File | Triggers | Gate? | Purpose |
|---|---|---|---|---|
| CI | `.github/workflows/ci.yml` | every push, PR → main | **enforcing** | Ruff lint, pytest with `--cov-fail-under=60` |
| Smoke Test | `.github/workflows/smoke-test.yml` | push:main, PR → main | **enforcing** | Compose build + boot infra + import-smoke api/monitor |
| Build & Push | `.github/workflows/docker.yml` | push:main, push:tag, PR → main, manual | **enforcing for build**; push is conditional | Build api/mlflow/monitor images with SBOM + SLSA provenance |
| Release | `.github/workflows/release.yml` | push of `v*.*.*` tag | n/a (post-merge) | Generate release notes, attach CycloneDX SBOM, promote image tags |
| Security Scans | `.github/workflows/security.yml` | PR → main, weekly cron, manual | pip-audit **enforcing**, Trivy **informational** | Python deps audit + image CVE scan, both surfaced to code-scanning |
| CodeQL | `.github/workflows/codeql.yml` | push:main, PR → main, weekly cron | informational | Python SAST, results in Security tab |
| Secret Scan | `.github/workflows/secret-scan.yml` | push:main, PR → main, weekly cron | **enforcing** | Gitleaks history scan |
| Train Model | `.github/workflows/train.yml` | manual dispatch only | n/a | Full training run, uploads model artifacts |

---

## 4. Trigger reference

A different view: what fires on which event.

| Event | Workflows |
|---|---|
| Push to feature branch | `ci.yml` |
| Open or update a PR → `main` | `ci.yml`, `smoke-test.yml`, `docker.yml` (build-only), `security.yml`, `codeql.yml`, `secret-scan.yml` |
| Merge to `main` | `ci.yml`, `smoke-test.yml`, `docker.yml` (push), `codeql.yml`, `secret-scan.yml` |
| Push of `v*.*.*` tag | `docker.yml` (push with semver tags), `release.yml` |
| Weekly cron (Sun 04Z) | `codeql.yml` |
| Weekly cron (Mon 05Z) | `secret-scan.yml` |
| Weekly cron (Mon 06Z) | `security.yml` |
| Manual dispatch | `train.yml`, `docker.yml`, `security.yml`, `secret-scan.yml`, `codeql.yml` |

---

## 5. Branching and release model

The model is **trunk-based with semver tags**:

1. Develop on feature branches. CI runs lint + tests on every push.
2. Open a PR to `main`. The full suite (above) runs. Required checks
   prevent merge until enforcing jobs pass.
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

Workflows use these repository secrets. None are required for the
default CI path on a fresh fork — only the manual training workflow
needs additional configuration to source its dataset.

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
  lockfile, shared across all workflows running on the same runner image.
  A hot cache makes `uv sync` complete in 5–10 seconds.

- **Docker buildx GHA cache** — `cache-from`/`cache-to` of
  `type=gha,scope=<image-name>`. Per-image scope so a churn-api build
  doesn't evict churn-mlflow layers. `mode=max` exports every
  intermediate stage's layers, not just the final image.

The smoke test workflow shares the same `type=gha` cache backend as
docker.yml without specifying a scope explicitly — Compose builds key
the cache on service name automatically.

---

## 9. Gating logic — what actually blocks a merge

Configure these as required status checks on the `main` branch in
**Settings → Branches → Branch protection rules**:

- `CI / Lint (ruff)`
- `CI / Tests (pytest) (3.13)`
- `Smoke Test / Compose stack smoke`
- `Build and Push Images / Build churn-api`
- `Build and Push Images / Build churn-mlflow`
- `Build and Push Images / Build churn-monitor`
- `Security Scans / pip-audit (uv lockfile)`
- `Secret Scan / Gitleaks`

The following are **deliberately not** required, because their failure
modes don't justify blocking PRs:

- `Security Scans / Trivy image scan (...)` — `exit-code: "0"` upstream,
  findings surface in the Security tab.
- `CodeQL / Analyze (python)` — query packs evolve weekly; new findings
  on unchanged code shouldn't block unrelated PRs.
- Ruff format check (within ci.yml) — `continue-on-error: true`.

Flip any of these to enforcing by removing `continue-on-error` or
setting `exit-code: "1"` in the relevant action input.

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

# Tests with coverage
PYTHONPATH=project uv run pytest project/tests \
  --cov=project --cov-report=term --cov-fail-under=60

# Image builds (matches docker.yml's build step)
docker compose -f project/docker-compose.yml build

# Compose smoke (matches smoke-test.yml)
cd project
docker compose up -d postgres mlflow
docker compose exec -T postgres psql -U monitor -d monitoring -c '\dt monitoring_metrics'
curl -fsS http://localhost:5001/health
docker compose run --rm --no-deps --entrypoint python api -c "import web_service"
docker compose down -v

# pip-audit (matches security.yml)
uv export --frozen --no-hashes --no-dev -o /tmp/req.txt
uvx pip-audit --requirement /tmp/req.txt --strict
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

**`smoke-test.yml`'s postgres healthcheck times out.**
Postgres init.sql runs on first startup only — if a previous run left a
volume with an incompatible schema, `init.sql` will not re-run. Check
the dump-logs step output for migration errors. The teardown step uses
`down -v` to drop volumes, so this should not happen run-to-run, but
can if a workflow is cancelled mid-init.

**`security.yml`'s Trivy step shows findings I can't fix.**
Findings come in two flavors. **OS package CVEs** are fixed by bumping
the base image (e.g. `python:3.13-slim` → newer point release) — the
fix is in the Dockerfile, not in this workflow. **Python package CVEs**
are fixed by `uv lock --upgrade-package <name>` and committing the
updated `uv.lock`.

**CodeQL marks a finding I know is safe.**
Dismiss it from the Security tab with a reason. Don't add suppression
comments to source — those persist beyond the fix and rot. If a query
produces too many false positives, lower the query suite from
`security-extended` back to `security-and-quality` in `codeql.yml`.

---

## 13. Extending the pipeline

**Add a new container image:**
1. Add the Dockerfile + Compose service.
2. Append the image to the matrix in `docker.yml`, `security.yml` (Trivy
   job), and `release.yml` (promote-images job). The smoke test pulls
   from the Compose file, so no edit needed there if the service has a
   `build:` stanza.

**Add a new gate (e.g. type checks with mypy):**
1. Add `mypy` to `[dependency-groups.dev]` in `pyproject.toml`.
2. Add a new job to `ci.yml` mirroring the structure of `lint`.
3. Once the project is mypy-clean, add the new job to the required
   status checks (section 9).

**Bump the coverage threshold:**
Edit `--cov-fail-under=60` in `ci.yml`. The threshold should track
realistic project coverage — bumping it ahead of test growth is a
recipe for blocked PRs.

**Bump Python:**
Update `.python-version`, `requires-python` in `pyproject.toml`, and
the matrix in `ci.yml`. The setup-uv action installs the requested
version on the fly; no other workflow needs editing.
