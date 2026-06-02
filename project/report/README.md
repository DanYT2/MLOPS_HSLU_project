# MLOps Report — Build Instructions

This directory contains an IEEE-style academic report on the Customer
Churn MLOps pipeline. The deliverables are:

```
report/
├── report.tex            Main LaTeX source (IEEEtran, conference, A4)
├── references.bib        BibTeX bibliography
├── figures/              Reused MLflow artefacts, notebook figures,
│                         generated architecture diagrams, and (after
│                         you add them) screenshots from the running
│                         Docker stack.
├── figures/PLACEHOLDERS.md
│                         Catalogue of every screenshot placeholder
│                         used in the report, with capture instructions.
├── figures/_gen_diagrams.py
│                         The Python script that produced the
│                         architecture, training-pipeline and CI/CD
│                         diagrams. Re-run with `python _gen_diagrams.py`
│                         if you edit the source.
└── README.md             This file.
```

## Compiling to PDF

LaTeX is **not** installed in the project's `uv` environment, so the
compile step has to run against an external LaTeX toolchain. Pick one:

### Option A — local TeX Live / MacTeX

```bash
brew install --cask basictex          # macOS, ~100 MB
# or: brew install --cask mactex      # full distribution, ~4 GB
eval "$(/usr/libexec/path_helper)"    # picks up new PATH
sudo tlmgr update --self
sudo tlmgr install ieeetran cite latexmk

cd project/report
latexmk -pdf -bibtex report.tex
```

### Option B — Docker (no host install required)

```bash
cd project/report
docker run --rm -v "$PWD":/work -w /work texlive/texlive:latest \
  latexmk -pdf -bibtex report.tex
```

### Option C — Overleaf

1. Upload `report.tex`, `references.bib` and the entire `figures/`
   folder to a new Overleaf project.
2. Set the compiler to `pdfLaTeX` and the bibliography tool to
   `BibTeX`.
3. Click Recompile.

## Adding screenshots

Bring up the local stack:

```bash
cd project
docker compose up --build
```

Then follow `figures/PLACEHOLDERS.md` — each row tells you which URL to
visit, what to capture, and which filename to save the PNG under. The
LaTeX `\screenshot` macro detects the file automatically: if the PNG
exists in `figures/` it is included; otherwise a labelled placeholder
box is rendered in its place. No edits to `report.tex` are needed.

## Constraints honoured

- Hard page cap: **≤ 15 pages** in IEEE conference two-column A4.
- No code blocks and no configuration snippets (YAML, TOML, SQL DDL,
  env vars) — descriptions are in prose, tables and figures only.
- Bibliography: 11 academic references + 7 tool-documentation URLs.
- Every figure either renders from an existing artefact or is a
  clearly labelled screenshot placeholder.

## Regenerating the architecture diagrams

```bash
cd project/report/figures
uv run python _gen_diagrams.py
```

This refreshes `fig_architecture.png`, `fig_training_pipeline.png` and
`fig_cicd.png`.
