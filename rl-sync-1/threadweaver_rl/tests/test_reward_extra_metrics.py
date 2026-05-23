import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

MODULE_ROOT = Path(__file__).resolve().parents[1]


def _load_metric_utils():
    stub_names = ["verl", "verl.utils", "verl.utils.import_utils", "torch"]
    previous_modules = {name: sys.modules.get(name) for name in stub_names}

    try:
        verl_stub = types.ModuleType("verl")
        verl_stub.DataProto = object
        sys.modules["verl"] = verl_stub

        sys.modules["verl.utils"] = types.ModuleType("verl.utils")

        import_utils_stub = types.ModuleType("verl.utils.import_utils")
        import_utils_stub.deprecated = lambda _message: lambda fn: fn
        sys.modules["verl.utils.import_utils"] = import_utils_stub

        if "torch" not in sys.modules:
            sys.modules["torch"] = types.ModuleType("torch")

        spec = importlib.util.spec_from_file_location(
            "test_metric_utils",
            MODULE_ROOT / "verl/trainer/ppo/metric_utils.py",
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


metric_utils = _load_metric_utils()
summarize_reward_extra = metric_utils.summarize_reward_extra


def test_summarize_reward_extra_logs_numeric_and_bool_scalars():
    metrics = summarize_reward_extra(
        {
            "correct": [True, False, np.bool_(True)],
            "acceleration_ratio": [0.1, np.float32(0.3), np.nan, None],
            "ground_truth": ["1", "2", "3"],
        },
        prefix="reward_extra",
    )

    assert metrics["reward_extra/correct/mean"] == pytest.approx(2 / 3)
    assert metrics["reward_extra/correct/max"] == 1.0
    assert metrics["reward_extra/correct/min"] == 0.0
    assert metrics["reward_extra/acceleration_ratio/mean"] == pytest.approx(0.2)
    assert metrics["reward_extra/acceleration_ratio/max"] == pytest.approx(0.3)
    assert metrics["reward_extra/acceleration_ratio/min"] == pytest.approx(0.1)
    assert not any(key.startswith("reward_extra/ground_truth/") for key in metrics)


def test_summarize_reward_extra_does_not_fan_out_categorical_values():
    metrics = summarize_reward_extra(
        {
            "pred": ["x = 1", "x = 2", "x = 3"],
            "error": ["timeout: request 1", "timeout: request 2"],
            "details": [{"raw": "free-form"}],
            "mixed": [1.0, "sample-specific-label", None],
        },
        prefix="reward_extra",
    )

    assert not any(key.startswith("reward_extra/pred/") for key in metrics)
    assert not any(key.startswith("reward_extra/error/") for key in metrics)
    assert not any(key.startswith("reward_extra/details/") for key in metrics)
    assert metrics["reward_extra/mixed/numeric_mean"] == 1.0
    assert metrics["reward_extra/mixed/numeric_max"] == 1.0
    assert metrics["reward_extra/mixed/numeric_min"] == 1.0


def test_summarize_reward_extra_keeps_parallel_aliases():
    train_metrics = summarize_reward_extra(
        {
            "parallel_ratio": [0.1, 0.3],
            "subtask_ratio": [0.2, 0.4],
            "trial_parallel_ratio": [0.5, 0.7],
        },
        prefix="reward_extra",
    )
    val_metrics = summarize_reward_extra(
        {
            "parallel_ratio": [0.2, 0.4],
            "subtask_parallel_ratio": [0.3, 0.5],
            "trial_ratio": [0.6, 0.8],
        },
        prefix="val-extra",
    )

    assert train_metrics["metrics/parallel_ratio"] == pytest.approx(0.2)
    assert train_metrics["metrics/subtask_parallel_ratio"] == pytest.approx(0.3)
    assert train_metrics["metrics/trial_parallel_ratio"] == pytest.approx(0.6)
    assert val_metrics["val/parallel_ratio"] == pytest.approx(0.3)
    assert val_metrics["val/subtask_parallel_ratio"] == pytest.approx(0.4)
    assert val_metrics["val/trial_parallel_ratio"] == pytest.approx(0.7)
