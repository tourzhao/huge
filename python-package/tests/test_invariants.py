"""Property-based invariant tests: random inputs, universal properties.

Complements fixed-fixture tests — hunts unknown bugs rather than guarding
known fixes. Seeded loops keep runtime bounded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyhuge import huge

METHODS = ("mb", "glasso", "ct", "tiger")


def _random_shape(seed: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    return int(rng.integers(15, 121)), int(rng.integers(4, 51)), int(rng.integers(1, 9))


@pytest.mark.parametrize("seed", range(6))
def test_outputs_symmetric_sparsity_monotone(seed: int) -> None:
    n, d, nlam = _random_shape(seed)
    x = np.random.default_rng(seed + 100).normal(size=(n, d))
    for m in METHODS:
        with warnings.catch_warnings():
            if m == "tiger":
                warnings.simplefilter("ignore", RuntimeWarning)
            fit = huge(x, method=m, nlambda=nlam, verbose=False)
        if m == "tiger":
            assert 1 <= len(fit.path) <= nlam
        else:
            assert len(fit.path) == nlam
        for p in fit.path:
            pm = p.toarray()
            assert np.abs(pm - pm.T).max() == 0.0
            assert np.abs(np.diag(pm)).max() == 0.0
        assert ((fit.sparsity >= 0) & (fit.sparsity <= 1)).all()
        if nlam > 1:
            assert (np.diff(fit.sparsity) >= -1e-12).all()


@pytest.mark.parametrize("seed", (7, 8, 9))
def test_ct_path_nested(seed: int) -> None:
    n, d, _ = _random_shape(seed)
    x = np.random.default_rng(seed + 100).normal(size=(n, d))
    fit = huge(x, method="ct", nlambda=4, verbose=False)
    for i in range(3):
        a = fit.path[i].toarray() != 0
        b = fit.path[i + 1].toarray() != 0
        assert b[a].all()


@pytest.mark.parametrize("seed", (10, 11, 12))
def test_glasso_icov_pd_and_path_consistent(seed: int) -> None:
    n, d, _ = _random_shape(seed)
    x = np.random.default_rng(seed + 100).normal(size=(n, d))
    fit = huge(x, method="glasso", nlambda=4, verbose=False)
    assert fit.icov is not None
    for i, ic in enumerate(fit.icov):
        assert np.linalg.eigvalsh(ic).min() > 0
        offdiag = ~np.eye(ic.shape[0], dtype=bool)
        assert np.array_equal(fit.path[i].toarray() != 0, (ic != 0) & offdiag)


def test_extreme_column_scales_invariant() -> None:
    rng = np.random.default_rng(31)
    x = rng.normal(size=(60, 12))
    x_scaled = x * (10.0 ** np.linspace(-6, 6, 12))
    f1 = huge(x, method="mb", nlambda=4, verbose=False)
    f2 = huge(x_scaled, method="mb", nlambda=4, verbose=False)
    for a, b in zip(f1.path, f2.path):
        assert np.array_equal(a.toarray(), b.toarray())


def test_near_singular_correlation_stays_finite() -> None:
    rng = np.random.default_rng(32)
    base = rng.normal(size=80)
    x = np.column_stack([base + rng.normal(scale=0.01, size=80) for _ in range(10)])
    for m in ("mb", "glasso", "tiger"):
        if m == "tiger":
            with pytest.warns(RuntimeWarning, match="certified prefix"):
                fit = huge(x, method=m, nlambda=4, verbose=False)
        else:
            fit = huge(x, method=m, nlambda=4, verbose=False)
        assert np.isfinite(fit.sparsity).all()
        if fit.icov is not None:
            for ic in fit.icov:
                assert np.isfinite(ic).all()


def test_nlambda_one_everywhere() -> None:
    x = np.random.default_rng(33).normal(size=(50, 8))
    for m in METHODS:
        fit = huge(x, method=m, nlambda=1, verbose=False)
        assert len(fit.path) == 1
