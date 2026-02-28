import os
import subprocess
import time
from pathlib import Path

import pytest
import requests

from mlproject.src.pipeline.dag_run import (
    run_eval,
    run_generate_configs,
    run_training,
    run_tune,
)


@pytest.fixture(scope="module")
def full_lifecycle_configs(tmp_path_factory):
    """Generate all configs needed for the full lifecycle test."""
    out_dir = tmp_path_factory.mktemp("configs")
    exp_path = "mlproject/configs/experiments/etth3.yaml"
    train_pipe = "mlproject/configs/pipelines/standard_train.yaml"

    # 1. Run training to produce artifacts
    run_training(experiment_path=exp_path, pipeline_path=train_pipe)

    from mlproject.src.generator.orchestrator import ConfigGenerator

    generator = ConfigGenerator(train_pipe, experiment_config_path=exp_path)
    paths = generator.generate_all(
        output_dir=str(out_dir), alias="latest", include_tune=True
    )

    return {
        "exp_path": exp_path,
        "train_pipe": train_pipe,
        "eval_pipe": paths.get("eval"),
        "serve_pipe": paths.get("serve"),
        "tune_pipe": paths.get("tune"),
    }


def test_full_lifecycle_dag(full_lifecycle_configs):
    """Test the entire DAG lifecycle."""
    c = full_lifecycle_configs

    # 3. Verify Evaluation Run
    run_eval(experiment_path=c["exp_path"], pipeline_path=c["eval_pipe"])

    # 4. Verify Tuning Run (1 trial for speed)
    run_tune(experiment_path=c["exp_path"], pipeline_path=c["tune_pipe"], n_trials=1)

    # 5. Verify Serving API Health (Start server process)
    port = 8085
    server_cmd = [
        "bash",
        "./mlproject/serve_api.sh",
        "-e",
        c["exp_path"],
        "-a",
        "latest",
        c["serve_pipe"],
        "--port",
        str(port),
    ]

    # Kill any existing process on the port just in case
    subprocess.run(["fuser", "-k", f"{port}/tcp"], stderr=subprocess.DEVNULL)

    server_env = os.environ.copy()
    server_env["PYTHONPATH"] = "."

    log_file_path = "/tmp/server_test.log"
    log_file = open(log_file_path, "w")

    process = subprocess.Popen(
        server_cmd, env=server_env, stdout=log_file, stderr=subprocess.STDOUT
    )

    try:
        # Wait for initialize
        healthy = False
        for _ in range(30):
            try:
                resp = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if resp.status_code == 200 and resp.json().get("status") == "healthy":
                    healthy = True
                    break
            except requests.ConnectionError:
                pass
            time.sleep(2)

        assert (
            healthy
        ), f"Serving Health Check FAILED after 30 attempts. Check {log_file_path} for logs."

        # 6. Verify Standard Prediction
        predict_payload = {
            "data": {
                "date": [
                    "2020-01-01 00:00:00",
                    "2020-01-01 01:00:00",
                    "2020-01-01 02:00:00",
                    "2020-01-01 03:00:00",
                    "2020-01-01 04:00:00",
                    "2020-01-01 05:00:00",
                    "2020-01-01 06:00:00",
                    "2020-01-01 07:00:00",
                    "2020-01-01 08:00:00",
                    "2020-01-01 09:00:00",
                    "2020-01-01 10:00:00",
                    "2020-01-01 11:00:00",
                    "2020-01-01 12:00:00",
                    "2020-01-01 13:00:00",
                    "2020-01-01 14:00:00",
                    "2020-01-01 15:00:00",
                    "2020-01-01 16:00:00",
                    "2020-01-01 17:00:00",
                    "2020-01-01 18:00:00",
                    "2020-01-01 19:00:00",
                    "2020-01-01 20:00:00",
                    "2020-01-01 21:00:00",
                    "2020-01-01 22:00:00",
                    "2020-01-01 23:00:00",
                ],
                "HUFL": [
                    5.827,
                    5.8,
                    5.969,
                    6.372,
                    7.153,
                    7.976,
                    8.715,
                    9.340,
                    9.763,
                    9.986,
                    10.040,
                    9.916,
                    9.609,
                    9.156,
                    8.591,
                    7.970,
                    7.338,
                    6.745,
                    6.233,
                    5.838,
                    5.582,
                    5.465,
                    5.465,
                    5.557,
                ],
                "MUFL": [
                    1.599,
                    1.492,
                    1.492,
                    1.492,
                    1.492,
                    1.509,
                    1.582,
                    1.711,
                    1.896,
                    2.113,
                    2.337,
                    2.552,
                    2.742,
                    2.902,
                    3.024,
                    3.104,
                    3.137,
                    3.125,
                    3.067,
                    2.969,
                    2.838,
                    2.683,
                    2.515,
                    2.346,
                ],
                "mobility_inflow": [
                    1.234,
                    1.456,
                    1.678,
                    1.890,
                    2.123,
                    2.456,
                    2.789,
                    3.012,
                    3.234,
                    3.456,
                    3.678,
                    3.890,
                    4.012,
                    4.123,
                    4.234,
                    4.345,
                    4.456,
                    4.567,
                    4.678,
                    4.789,
                    4.890,
                    4.901,
                    4.912,
                    4.923,
                ],
            }
        }

        pred_resp = requests.post(
            f"http://127.0.0.1:{port}/predict", json=predict_payload
        )
        assert pred_resp.status_code == 200
        preds = pred_resp.json().get("predictions", {})
        assert len(preds) > 0, "No predictions found"
        for k, v in preds.items():
            assert v is not None
            assert len(v) > 0

    finally:
        log_file.close()
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                print(f"Server logs:\n{f.read()}")
        process.terminate()
        process.wait(timeout=5)
        subprocess.run(["fuser", "-k", f"{port}/tcp"], stderr=subprocess.DEVNULL)
