from mlproject.src.pipeline.compat.v1.run import main_run


def test_run_pipeline_smoke():
    # run with default config (creates synthetic data) to smoke test
    cfg_path = "mlproject/configs/experiments/etth1.yaml"
    main_run("train", cfg_path=cfg_path)
    main_run("eval", cfg_path=cfg_path)
    assert True
