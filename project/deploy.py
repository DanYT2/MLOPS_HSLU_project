"""Prefect deployment definitions with schedules.

Run this script to start serving the training and batch prediction flows.
The serve() call blocks and polls for scheduled runs, executing them
in-process. It also registers the deployments with the Prefect server
so they appear in the UI at http://localhost:4200.

Usage:
  python deploy.py                          # local development
  # Or in Docker via docker-compose (see Dockerfile.worker)

Environment variables:
  PREFECT_API_URL   – Prefect server URL  (Docker: http://prefect-server:4200/api)
  MLFLOW_TRACKING_URI – MLflow server URL (Docker: http://mlflow:5001)
  DATA_DIR          – path to dataset/    (Docker: /app/dataset)
  OUTPUT_DIR        – path for predictions (Docker: /app/predictions)
"""

from __future__ import annotations

import os
import time
from typing import Optional

from prefect import serve

from flows import batch_prediction_flow, training_flow


def _wait_for_prefect_api(api_url: Optional[str], *, timeout_seconds: int = 120) -> None:
    """Block until the Prefect server is reachable.

    `serve()` registers deployments by calling the Prefect API; in Docker
    compose, `depends_on` does not guarantee the server is ready yet.
    """
    if not api_url:
        return

    import httpx

    start = time.monotonic()
    last_exc: Exception | None = None

    while time.monotonic() - start < timeout_seconds:
        try:
            # A successful TCP connection is enough; the endpoint may return
            # 404/405 depending on routing, but we only care that it's reachable.
            with httpx.Client(timeout=5) as client:
                client.get(api_url)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2)

    raise RuntimeError(
        f"Prefect server not reachable at {api_url!r} after {timeout_seconds}s"
    ) from last_exc


if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIR", "project/dataset")
    output_dir = os.environ.get("OUTPUT_DIR", "project/predictions")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
    prefect_api_url = os.environ.get("PREFECT_API_URL")

    _wait_for_prefect_api(prefect_api_url)

    training_dep = training_flow.to_deployment(
        name="scheduled-training",
        cron="0 2 * * 0",  # Every Sunday at 2 AM
        parameters={
            "data_dir": data_dir,
            "tracking_uri": tracking_uri,
            "n_trials": 10,
        },
    )

    batch_dep = batch_prediction_flow.to_deployment(
        name="scheduled-batch-prediction",
        cron="0 6 * * 1",  # Every Monday at 6 AM
        parameters={
            "input_path": f"{data_dir}/test.csv",
            "output_path": f"{output_dir}/batch_output.csv",
            "tracking_uri": tracking_uri,
        },
    )

    serve(training_dep, batch_dep)
