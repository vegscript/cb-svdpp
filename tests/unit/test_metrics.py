import numpy as np
import pytest

from recsys_lab.metrics import rmse


def test_rmse_matches_known_value() -> None:
    truth = np.asarray([3.0, 4.0, 5.0], dtype=np.float32)
    pred = np.asarray([2.0, 4.0, 4.0], dtype=np.float32)
    assert rmse(truth, pred) == pytest.approx((2.0 / 3.0) ** 0.5)


def test_rmse_rejects_empty_arrays() -> None:
    with pytest.raises(ValueError):
        rmse(np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32))
