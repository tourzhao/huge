"""Parity checks between pyhuge native outputs and R huge references."""

from __future__ import annotations

import numpy as np
import pytest

from pyhuge import huge, huge_inference, huge_select

from pyhuge.parity import (
    has_r_huge,
    run_r_ct_default_reference,
    run_r_ct_reference,
    run_r_glasso_reference,
    run_r_inference_reference,
    run_r_tiger_reference,
)


HAS_R_HUGE = has_r_huge()


def _edge_counts(path) -> np.ndarray:
    return np.asarray([np.count_nonzero(np.triu(m.toarray() != 0, 1)) for m in path], dtype=float)


@pytest.mark.skipif(not HAS_R_HUGE, reason="requires local R with package huge")
def test_parity_ct_path_and_stars_selection():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(120, 20))
    lam = np.linspace(0.6, 0.05, 8)

    r_ref = run_r_ct_reference(x, lam, rep_num=6, stars_thresh=0.1, seed=123)

    fit = huge(x, method="ct", lambda_=lam, verbose=False)
    sel = huge_select(fit, criterion="stars", rep_num=6, stars_thresh=0.1, verbose=False)

    # For CT thresholding, path parity should be very close under identical lambda path.
    assert np.allclose(fit.lambda_path, r_ref["lambda"], rtol=0.0, atol=1e-12)
    assert np.max(np.abs(fit.sparsity - r_ref["sparsity"])) <= 1e-10
    assert np.array_equal(_edge_counts(fit.path), r_ref["edges"])

    # Selection rules are not bitwise-identical across implementations; allow small index drift.
    assert abs((sel.opt_index or 1) - int(r_ref["opt_index"])) <= 2


@pytest.mark.skipif(not HAS_R_HUGE, reason="requires local R with package huge")
def test_parity_ct_default_rank_path():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(150, 24))

    r_ref = run_r_ct_default_reference(x, nlambda=10, lambda_min_ratio=0.05)
    fit = huge(x, method="ct", nlambda=10, lambda_min_ratio=0.05, verbose=False)

    path_py = np.stack([m.toarray() for m in fit.path], axis=0)

    assert np.allclose(fit.lambda_path, r_ref["lambda"], rtol=0.0, atol=1e-12)
    assert np.allclose(fit.sparsity, r_ref["sparsity"], rtol=0.0, atol=1e-12)
    assert np.array_equal(path_py != 0, r_ref["path"] != 0)


@pytest.mark.skipif(not HAS_R_HUGE, reason="requires local R with package huge")
def test_parity_glasso_path_and_ebic_selection():
    rng = np.random.default_rng(321)
    x = rng.normal(size=(120, 20))
    lam = np.geomspace(0.5, 0.05, 8)

    r_ref = run_r_glasso_reference(x, lam)

    fit = huge(x, method="glasso", lambda_=lam, verbose=False)
    sel = huge_select(fit, criterion="ebic", verbose=False)

    assert np.allclose(fit.lambda_path, r_ref["lambda"], rtol=0.0, atol=1e-12)

    max_edges = x.shape[1] * (x.shape[1] - 1) / 2.0
    edge_gap = np.mean(np.abs(_edge_counts(fit.path) - r_ref["edges"])) / max_edges
    sparsity_gap = float(np.mean(np.abs(fit.sparsity - r_ref["sparsity"])))

    # Implementations are aligned but not strictly bitwise-identical; enforce coarse parity.
    assert edge_gap <= 0.25
    assert sparsity_gap <= 0.25
    assert abs((sel.opt_index or 1) - int(r_ref["opt_index"])) <= 4


@pytest.mark.skipif(not HAS_R_HUGE, reason="requires local R with package huge")
def test_parity_tiger_correlation_solver():
    rng = np.random.default_rng(91)
    x = rng.normal(size=(100, 16))
    lam = np.asarray([0.4, 0.25, 0.15, 0.1])

    r_ref = run_r_tiger_reference(x, lam)
    fit = huge(x, method="tiger", lambda_=lam, verbose=False)
    path = np.stack([matrix.toarray() for matrix in fit.path], axis=0)
    icov = np.stack(fit.icov, axis=0)

    assert np.array_equal(fit.lambda_path, r_ref["lambda"])
    assert np.allclose(fit.sparsity, r_ref["sparsity"], rtol=0.0, atol=1e-12)
    assert np.array_equal(path, r_ref["path"])
    assert np.allclose(icov, r_ref["icov"], rtol=0.0, atol=1e-10)


@pytest.mark.skipif(not HAS_R_HUGE, reason="requires local R with package huge")
@pytest.mark.parametrize(
    ("type_", "method"),
    [
        ("Gaussian", "score"),
        ("Nonparanormal", "score"),
        ("Nonparanormal", "wald"),
    ],
)
def test_parity_inference(type_, method):
    rng = np.random.default_rng(17)
    x = np.round(rng.normal(size=(18, 4)), 1)
    gram = np.cov(x, rowvar=False) + 0.5 * np.eye(x.shape[1])
    t_mat = np.linalg.inv(gram)
    adj = (np.abs(t_mat) > 0.2).astype(float)
    np.fill_diagonal(adj, 0.0)

    r_ref = run_r_inference_reference(
        x,
        t_mat,
        adj,
        alpha=0.05,
        type_=type_,
        method=method,
    )
    py = huge_inference(
        x,
        t_mat,
        adj,
        alpha=0.05,
        type_=type_,
        method=method,
    )

    assert np.allclose(py.p, r_ref["p"], rtol=1e-10, atol=1e-12, equal_nan=True)
    assert np.allclose(py.data, r_ref["data"], rtol=0.0, atol=1e-12)
    assert py.error == pytest.approx(r_ref["error"], abs=1e-15)

    if type_ == "Nonparanormal":
        other = "wald" if method == "score" else "score"
        other_py = huge_inference(x, t_mat, adj, type_=type_, method=other)
        assert not np.allclose(py.p, other_py.p, rtol=1e-8, atol=1e-10, equal_nan=True)
