# Screenshot Placeholders

Each entry below corresponds to a `\fbox{...}` placeholder in `report.tex`.
Bring up the stack with `docker compose up --build` from `project/`, then capture each screenshot at the listed URL/area and drop the PNG into this folder using the listed filename.

The LaTeX source already references these filenames; once you save the PNGs the placeholder boxes will be replaced automatically on the next compile (the `\fbox` macro falls back to `\includegraphics` when the file exists — see `report.tex` preamble).

| # | Filename to save | What to capture | URL / view |
|---|---|---|---|
| S1 | `screenshot_mlflow_registry.png` | MLflow Registry view of `CustomerChurnEnsemble` with the `@champion` alias visible. | http://localhost:5001 → Models → CustomerChurnEnsemble |
| S2 | `screenshot_mlflow_run.png` | One MLflow run page showing the parent run with its 10 nested child runs (LightGBM + XGBoost folds), metrics tab open. | http://localhost:5001 → Experiments → most recent run |
| S3 | `screenshot_fastapi_swagger.png` | Swagger UI listing `GET /`, `POST /predict`, `POST /predict/batch`, `GET /model/info`. | http://localhost:8000/docs |
| S4 | `screenshot_fastapi_response.png` | Successful `POST /predict` request/response example with `churn_probability` and `prediction` fields. | Use the Swagger "Try it out" panel with the example payload. |
| S5 | `screenshot_grafana_dashboard.png` | Full Customer Churn Monitoring dashboard with all eight panels populated by the monitor replay (run the stack for a few minutes first). | http://localhost:3000/d/customer-churn-monitoring (anonymous viewer enabled) |
| S6 | `screenshot_grafana_drift.png` | Close-up of the "Drifted Columns" + "Share of Drifted Columns" time-series panels, ideally with `DRIFT_MODE=gradual` enabled so the curve rises. | Grafana dashboard, scroll to drift panels. |
| S7 | `screenshot_adminer.png` | Adminer table view of `monitoring_metrics` showing the most recent rows written by the monitor. | http://localhost:8080 → Server `postgres`, user/pass `monitor/monitor`, db `monitoring`, table `monitoring_metrics`. |
| S8 | `screenshot_github_actions.png` | GitHub Actions tab showing a successful PR build (green ci.yml + docker.yml + secret-scan.yml). | https://github.com/DanYT2/<repo>/actions on any merged PR. |
| S9 | `screenshot_codeql_alerts.png` *(optional)* | Security tab → Code scanning alerts (empty list is fine — proves CodeQL is wired up). | https://github.com/DanYT2/<repo>/security/code-scanning |

## Tips

- **Browser zoom**: set to 90–100 % before capturing so the figures stay legible at half-column width.
- **Window size**: 1440 × 900 produces crisp screenshots that scale well to the IEEE column width (~252 pt).
- **Crop tightly**: keep ~10 px of whitespace around the relevant UI; don't include the OS chrome.
- **Format**: PNG (lossless) is preferred; the `.tex` includegraphics calls assume `.png`.
- **If a screenshot is unavailable**: leave the placeholder box in place — the report still compiles and the placeholder text tells a reader what should appear there.
