from __future__ import annotations

import numpy as np
import pytest

from pyhuge import (
    huge,
    huge_generator,
    huge_glasso,
    huge_inference,
    huge_mb,
    huge_npn,
    huge_roc,
    huge_select,
    huge_tiger,
)


def test_native_ct_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 12))

    fit_ct = huge(x, method="ct", nlambda=5, backend="native")
    assert fit_ct.method == "ct"
    assert len(fit_ct.path) == 5


def test_native_mb_glasso_select_runs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 12))

    fit_mb = huge_mb(x, nlambda=4, backend="native")
    sel_ric = huge_select(fit_mb, criterion="ric", backend="native")
    assert sel_ric.opt_lambda > 0
    assert 0.0 <= sel_ric.opt_sparsity <= 1.0

    fit_gl = huge_glasso(x, nlambda=4, cov_output=True, backend="native")
    sel_ebic = huge_select(fit_gl, criterion="ebic", backend="native")
    assert fit_gl.icov is not None and len(fit_gl.icov) == 4
    assert fit_gl.cov is not None and len(fit_gl.cov) == 4
    assert sel_ebic.opt_icov is not None


def test_native_tiger_runs():
    x = np.random.default_rng(4).normal(size=(60, 12))

    fit = huge_tiger(x, lambda_=[0.5], backend="native", verbose=False)

    assert fit.method == "tiger"
    assert len(fit.path) == 1
    assert fit.icov is not None and len(fit.icov) == 1
    np.testing.assert_array_equal(fit.lambda_path, np.asarray([0.5]))


def test_native_generator_roc_inference_runs():
    sim = huge_generator(n=70, d=10, graph="hub", g=2, random_state=1)
    fit = huge(sim.data, method="ct", nlambda=4, backend="native")

    roc = huge_roc(fit.path, sim.theta)
    assert 0.0 <= roc.auc <= 1.0

    t = np.eye(sim.data.shape[1], dtype=float)
    out = huge_inference(sim.data, t, sim.theta, alpha=0.05, type_="Gaussian")
    assert out.p.shape == (10, 10)
    assert 0.0 <= out.error <= 1.0


def test_native_inference_matches_r_fixture():
    x = np.asarray(
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
            [3, 2, 1],
            [1, 2, 3],
            [2, 2, 1],
        ],
        dtype=float,
    )
    t_mat = np.asarray(
        [[1.5, 0.2, -0.1], [0.2, 1.2, 0.15], [-0.1, 0.15, 1.4]],
        dtype=float,
    )
    adj = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    expected = {
        ("Gaussian", "score"): (
            np.asarray(
                [
                    [0.7585391313693459, 0.12818790275067804, 0.07685037046833898],
                    [0.12818790275067804, 0.3922925958633643, 0.4309175708833086],
                    [0.07685037046833898, 0.4309175708833086, 0.5091503125479249],
                ]
            ),
            2.0 / 9.0,
        ),
        ("Nonparanormal", "score"): (
            np.asarray(
                [
                    [0.7731215776133444, 0.020526136230479386, 1.8427772836782097e-5],
                    [0.013640212724985279, 0.5574997855046242, 0.32142866625856015],
                    [1.4224743399404716e-5, 0.345341799696653, 0.8921437867572235],
                ]
            ),
            2.0 / 9.0,
        ),
        ("Nonparanormal", "wald"): (
            np.asarray(
                [
                    [2.637889906509372e-13, 2.217154559533974e-4, 0.04457812277415374],
                    [2.217154559533974e-4, 0.0, 0.0],
                    [0.04457812277415374, 0.0, 7.038813976123493e-14],
                ]
            ),
            4.0 / 9.0,
        ),
    }

    for key, (expected_p, expected_error) in expected.items():
        out = huge_inference(x, t_mat, adj, alpha=0.1, type_=key[0], method=key[1])
        assert np.allclose(out.p, expected_p, rtol=2e-11, atol=5e-13)
        assert out.error == pytest.approx(expected_error, abs=1e-15)
        assert np.array_equal(out.data, x)


def test_native_npn_modes():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 8)) ** 3

    z1 = huge_npn(x, npn_func="shrinkage")
    z2 = huge_npn(x, npn_func="truncation")
    z3 = huge_npn(x, npn_func="skeptic")

    assert z1.shape == x.shape
    assert z2.shape == x.shape
    assert z3.shape == (8, 8)
