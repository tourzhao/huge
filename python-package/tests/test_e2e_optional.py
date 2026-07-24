"""End-to-end native tests for pyhuge."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


HAS_MPL = importlib.util.find_spec("matplotlib") is not None
HAS_NX = importlib.util.find_spec("networkx") is not None


def test_e2e_mb_select_ric():
    from pyhuge import huge_mb, huge_select

    rng = np.random.default_rng(11)
    x = rng.normal(size=(60, 12))
    fit = huge_mb(x, nlambda=4, verbose=False)
    sel = huge_select(fit, criterion="ric", rep_num=5, verbose=False)

    assert fit.method == "mb"
    assert len(fit.path) == 4
    assert sel.opt_lambda > 0


def test_e2e_ct_stars():
    from pyhuge import huge_ct, huge_select

    rng = np.random.default_rng(17)
    x = rng.normal(size=(50, 10))
    fit = huge_ct(x, nlambda=4, verbose=False)
    sel = huge_select(
        fit,
        criterion="stars",
        stars_thresh=0.1,
        stars_subsample_ratio=0.7,
        rep_num=3,
        verbose=False,
    )

    assert fit.method == "ct"
    assert sel.criterion == "stars"
    assert 0.0 <= sel.opt_sparsity <= 1.0


def test_e2e_tiger_runs():
    from pyhuge import huge_tiger

    rng = np.random.default_rng(23)
    x = rng.normal(size=(50, 10))
    fit = huge_tiger(x, nlambda=4, verbose=False)

    assert fit.method == "tiger"
    assert len(fit.path) == 4
    assert fit.icov is not None


def test_e2e_npn_glasso_ebic():
    from pyhuge import huge_glasso, huge_npn, huge_select

    rng = np.random.default_rng(31)
    x = rng.normal(size=(70, 12))
    x_npn = huge_npn(x, npn_func="shrinkage", verbose=False)
    fit = huge_glasso(x_npn, nlambda=4, verbose=False)
    sel = huge_select(fit, criterion="ebic", ebic_gamma=0.5, verbose=False)

    assert fit.method == "glasso"
    assert fit.loglik is not None
    assert sel.opt_icov is not None


def test_e2e_generator_roc_inference():
    from pyhuge import huge, huge_generator, huge_inference, huge_roc

    sim = huge_generator(n=80, d=10, graph="hub", g=2, verbose=False)
    fit = huge(sim.data, method="ct", nlambda=4, verbose=False)
    roc = huge_roc(fit.path, sim.theta, verbose=False, plot=False)
    inf = huge_inference(
        data=sim.data,
        t=np.eye(sim.data.shape[1]),
        adj=sim.theta,
        alpha=0.05,
        type_="Gaussian",
        method="score",
    )

    assert 0.0 <= roc.auc <= 1.0
    assert roc.fp.shape == roc.tp.shape
    assert 0.0 <= inf.error <= 1.0


@pytest.mark.skipif(not (HAS_MPL and HAS_NX), reason="requires matplotlib+networkx")
def test_e2e_summary_and_network_plot():
    from pyhuge import huge_glasso, huge_plot_network, huge_select, huge_select_summary, huge_summary

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(41)
    x = rng.normal(size=(60, 10))
    fit = huge_glasso(x, nlambda=4, verbose=False)
    sel = huge_select(fit, criterion="ebic", verbose=False)

    s1 = huge_summary(fit)
    s2 = huge_select_summary(sel)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    huge_plot_network(fit, index=-1, ax=ax, layout="spring")
    plt.close(fig)

    assert s1.n_features == 10
    assert s2.refit_n_features == 10
