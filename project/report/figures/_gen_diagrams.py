"""Generate architecture / pipeline / CI diagrams for the report.

Run: python _gen_diagrams.py
Outputs: fig_architecture.png, fig_training_pipeline.png, fig_cicd.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PALETTE = {
    "data": "#E8F0FE",
    "compute": "#D2E3FC",
    "store": "#FCE8E6",
    "serve": "#E6F4EA",
    "ops": "#FEF7E0",
    "edge": "#1A73E8",
    "text": "#202124",
}


def box(ax, x, y, w, h, label, color, fontsize=9):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=PALETTE["edge"],
        facecolor=color,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["text"],
        wrap=True,
    )


def arrow(ax, x1, y1, x2, y2, label=None, style="-|>", color="#5F6368", rad=0.0):
    a = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=1.1,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    if label:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2 + 0.08,
            label,
            ha="center",
            va="center",
            fontsize=7.5,
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
        )


# =====================================================================
# Figure 1 — System Architecture (Docker Compose topology)
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title(
    "Customer Churn MLOps — System Architecture", fontsize=13, fontweight="bold", pad=10
)

# Data layer
box(ax, 0.3, 5.6, 2.0, 0.9, "Raw Data\ntrain.csv / test.csv", PALETTE["data"])

# Training (offline)
box(
    ax,
    0.3,
    3.8,
    2.0,
    1.2,
    "Training Container\n(train.py)\nLightGBM + XGBoost\n5-fold CV + Optuna",
    PALETTE["compute"],
    fontsize=8.5,
)

# MLflow
box(
    ax,
    3.0,
    3.8,
    2.3,
    1.2,
    "MLflow Server\n(mlflow:5001)\nTracking + Registry",
    PALETTE["store"],
)

# Postgres
box(
    ax,
    3.0,
    1.6,
    2.3,
    1.0,
    "PostgreSQL\n(postgres:5432)\nmonitoring_metrics",
    PALETTE["store"],
)

# API
box(
    ax,
    6.0,
    3.8,
    2.3,
    1.2,
    "FastAPI Service\n(api:8000)\n/predict, /predict/batch\nChurnEnsembleModel",
    PALETTE["serve"],
    fontsize=8.5,
)

# Monitor
box(ax, 6.0, 1.6, 2.3, 1.0, "Monitor Worker\n(replay + Evidently)", PALETTE["compute"])

# Grafana
box(
    ax,
    9.0,
    3.8,
    2.5,
    1.2,
    "Grafana\n(grafana:3000)\nDrift + Quality Dashboard",
    PALETTE["ops"],
)

# Adminer
box(ax, 9.0, 1.6, 2.5, 1.0, "Adminer\n(adminer:8080)\nDB UI", PALETTE["ops"])

# Client
box(ax, 9.0, 5.6, 2.5, 0.9, "End User / Client\n(HTTP request)", PALETTE["data"])

# Arrows
arrow(ax, 1.3, 5.6, 1.3, 5.0)  # data -> train
arrow(ax, 2.3, 4.4, 3.0, 4.4, label="log runs")  # train -> mlflow
arrow(ax, 5.3, 4.4, 6.0, 4.4, label="@champion")  # mlflow -> api
arrow(ax, 9.0, 6.0, 8.3, 4.6, label="POST /predict")  # client -> api
arrow(ax, 8.3, 4.4, 9.0, 4.4, label="response")  # api -> client (via grafana col)
arrow(ax, 6.0, 2.1, 5.3, 2.1, label="INSERT metrics")  # monitor -> postgres
arrow(
    ax, 5.3, 2.4, 6.0, 4.0, rad=0.1, label="batch replay"
)  # postgres? actually api->monitor
arrow(ax, 7.1, 2.6, 7.1, 3.8, label="/predict/batch", rad=-0.1)
arrow(
    ax, 8.3, 2.1, 9.0, 2.1, label="SQL"
)  # postgres -> adminer? actually we go pg->grafana
arrow(ax, 5.3, 2.2, 9.0, 4.0, rad=-0.2, label="SELECT")  # postgres -> grafana

# Legend
legend_y = 0.3
for i, (lbl, col) in enumerate(
    [
        ("Data", PALETTE["data"]),
        ("Compute", PALETTE["compute"]),
        ("Storage", PALETTE["store"]),
        ("Serving", PALETTE["serve"]),
        ("Ops UI", PALETTE["ops"]),
    ]
):
    box(ax, 0.5 + i * 2.0, legend_y, 0.4, 0.4, "", col)
    ax.text(1.0 + i * 2.0, legend_y + 0.2, lbl, fontsize=8, va="center")

plt.tight_layout()
plt.savefig("fig_architecture.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_architecture.png")

# =====================================================================
# Figure 2 — Training pipeline flow
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title(
    "Training Pipeline: CSV → Feature Engineering → HPO → CV → Registry",
    fontsize=13,
    fontweight="bold",
    pad=10,
)

stages = [
    (0.2, "Raw CSV\n(train + test)", PALETTE["data"]),
    (1.9, "Feature\nEngineering\n(derived + OHE)", PALETTE["compute"]),
    (3.6, "Optuna HPO\n(TPE, 20 trials\nper model)", PALETTE["compute"]),
    (5.3, "5-fold\nStratified CV\n(LGBM + XGB)", PALETTE["compute"]),
    (7.0, "Pyfunc Wrap\n(10 fold models\n→ ensemble)", PALETTE["compute"]),
    (8.7, "MLflow\nRegistry\n@champion", PALETTE["store"]),
    (10.4, "FastAPI\nServing", PALETTE["serve"]),
]
for x, label, col in stages:
    box(ax, x, 2.0, 1.5, 1.4, label, col, fontsize=8.5)

for i in range(len(stages) - 1):
    x1 = stages[i][0] + 1.5
    x2 = stages[i + 1][0]
    arrow(ax, x1, 2.7, x2, 2.7)

# Side artifacts
box(ax, 3.6, 0.4, 1.5, 1.0, "optuna_studies.db\n(SQLite)", PALETTE["store"], fontsize=8)
box(
    ax,
    5.3,
    0.4,
    1.5,
    1.0,
    "Per-fold metrics,\nconfusion matrix,\nfeat importance",
    PALETTE["store"],
    fontsize=7.5,
)
arrow(ax, 4.35, 2.0, 4.35, 1.4, style="-|>", color="#5F6368")
arrow(ax, 6.05, 2.0, 6.05, 1.4, style="-|>", color="#5F6368")

plt.tight_layout()
plt.savefig("fig_training_pipeline.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_training_pipeline.png")

# =====================================================================
# Figure 3 — CI/CD pipeline (GitHub Actions workflows)
# =====================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis("off")
ax.set_title("CI/CD: GitHub Actions Workflows", fontsize=13, fontweight="bold", pad=10)

# Trigger column
box(ax, 0.3, 4.5, 2.2, 0.9, "Push / PR to main", PALETTE["data"])
box(ax, 0.3, 3.3, 2.2, 0.9, "Tag v*.*.*", PALETTE["data"])
box(
    ax,
    0.3,
    2.1,
    2.2,
    0.9,
    "Weekly cron\n(Mon 05:00 UTC)",
    PALETTE["data"],
    fontsize=8.5,
)
box(ax, 0.3, 0.9, 2.2, 0.9, "Manual dispatch", PALETTE["data"])

# Workflow column
box(
    ax,
    4.0,
    4.8,
    2.5,
    0.7,
    "ci.yml\nRuff + pytest + cov ≥ 50%",
    PALETTE["compute"],
    fontsize=8.5,
)
box(
    ax,
    4.0,
    3.9,
    2.5,
    0.7,
    "docker.yml\nBuild + push 3 GHCR images",
    PALETTE["compute"],
    fontsize=8.5,
)
box(
    ax,
    4.0,
    3.0,
    2.5,
    0.7,
    "release.yml\nSemver tag & release",
    PALETTE["compute"],
    fontsize=8.5,
)
box(
    ax,
    4.0,
    2.1,
    2.5,
    0.7,
    "secret-scan.yml\nGitleaks SARIF report",
    PALETTE["compute"],
    fontsize=8.5,
)
box(
    ax,
    4.0,
    1.2,
    2.5,
    0.7,
    "train.yml\nEnd-to-end training run",
    PALETTE["compute"],
    fontsize=8.5,
)
box(ax, 4.0, 0.3, 2.5, 0.7, "CodeQL (default setup)", PALETTE["compute"], fontsize=8.5)

# Outputs column
box(
    ax,
    8.5,
    4.5,
    3.2,
    0.9,
    "PR check status\n(blocks merge if red)",
    PALETTE["serve"],
    fontsize=8.5,
)
box(
    ax,
    8.5,
    3.3,
    3.2,
    0.9,
    "GHCR images +\nSBOM + SLSA provenance",
    PALETTE["store"],
    fontsize=8.5,
)
box(
    ax,
    8.5,
    2.1,
    3.2,
    0.9,
    "Code-scanning alerts\n(SARIF in Security tab)",
    PALETTE["ops"],
    fontsize=8.5,
)
box(
    ax,
    8.5,
    0.9,
    3.2,
    0.9,
    "submission.csv +\nmlruns artifact",
    PALETTE["store"],
    fontsize=8.5,
)

# Trigger -> workflow arrows
for ty, wy in [
    (4.95, 5.15),
    (4.95, 4.25),
    (3.75, 3.35),
    (3.75, 2.45),
    (2.55, 3.35),
    (2.55, 0.65),
    (1.35, 1.55),
]:
    arrow(ax, 2.5, ty, 4.0, wy)
# Workflow -> output
arrow(ax, 6.5, 5.15, 8.5, 4.95)
arrow(ax, 6.5, 4.25, 8.5, 3.75)
arrow(ax, 6.5, 2.45, 8.5, 2.55)
arrow(ax, 6.5, 1.55, 8.5, 1.35)

plt.tight_layout()
plt.savefig("fig_cicd.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Wrote fig_cicd.png")
