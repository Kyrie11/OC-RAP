import numpy as np

from ocrap.algorithms.ocmero import oc_mero


def test_ocmero_detects_oracle_artifact_under_shared_observation():
    M = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    p = np.array([0.5, 0.5])
    C = np.ones((2, 2), dtype=float)
    res = oc_mero(M, p, C, alpha=0.5, beta=0.5, option_valid=np.array([True, True]))
    assert res.r_orc > 0.9
    assert res.r_dep < -0.9
    assert res.gap > 1.9


def test_invalid_option_masked_and_no_oracle_assert():
    M = np.array([[1.0, 10.0], [-1.0, -10.0]], dtype=float)
    res = oc_mero(M, np.array([0.5, 0.5]), np.eye(2), option_valid=np.array([True, False]))
    assert np.all(res.q[:, 1] < -1e8)
    assert isinstance(res.gap, float)
