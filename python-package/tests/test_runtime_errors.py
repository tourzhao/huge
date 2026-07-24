"""Runtime and validation error tests for pyhuge."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse

from pyhuge import (
    PyHugeError,
    huge,
    huge_generator,
    huge_inference,
    huge_npn,
    huge_plot,
    huge_roc,
    huge_select,
)


def test_huge_invalid_method_raises():
    x = np.random.default_rng(0).normal(size=(20, 6))
    with pytest.raises(PyHugeError, match="`method` must be one of"):
        huge(x, method="bad", verbose=False)


@pytest.mark.parametrize("method", ("ct", "mb", "glasso", "tiger"))
@pytest.mark.parametrize(
    "grid",
    (
        pytest.param(
            {"nlambda": 3, "lambda_min_ratio": 0.2}, id="automatic"
        ),
        pytest.param({"lambda_": [0.5, 0.2]}, id="explicit"),
    ),
)
def test_huge_rejects_raw_data_with_one_observation(method, grid):
    x = np.asarray([[1.0, 2.0, 3.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PyHugeError, match="at least two observations"):
            huge(x, method=method, verbose=False, **grid)


@pytest.mark.parametrize("method", ("ct", "mb", "glasso", "tiger"))
@pytest.mark.parametrize(
    "grid",
    (
        pytest.param(
            {"nlambda": 3, "lambda_min_ratio": 0.2}, id="automatic"
        ),
        pytest.param({"lambda_": [0.5, 0.2]}, id="explicit"),
    ),
)
def test_huge_rejects_raw_data_with_a_constant_column(method, grid):
    varying = np.linspace(-1.0, 1.0, 8)
    x = np.column_stack((varying, np.ones_like(varying), varying**2))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PyHugeError, match="constant column"):
            huge(x, method=method, verbose=False, **grid)


def test_huge_npn_invalid_func_raises():
    x = np.random.default_rng(0).normal(size=(20, 6))
    with pytest.raises(PyHugeError, match="`npn_func` must be one of"):
        huge_npn(x, npn_func="bad", verbose=False)


def test_huge_plot_non_square_raises():
    g = np.ones((4, 3), dtype=float)
    with pytest.raises(PyHugeError, match="`g` must be square"):
        huge_plot(g)


def test_secondary_matrices_reject_nonfinite_values():
    theta = np.zeros((3, 3), dtype=float)
    theta[0, 1] = theta[1, 0] = 1.0
    bad_theta = theta.copy()
    bad_theta[0, 2] = np.nan
    with pytest.raises(PyHugeError, match="theta.*non-finite"):
        huge_roc([np.zeros((3, 3))], bad_theta, plot=False)

    bad_path = sparse.csc_matrix(
        np.array([[0.0, np.inf, 0.0], [np.inf, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    with pytest.raises(PyHugeError, match=r"path\[1\].*non-finite"):
        huge_roc([bad_path], theta, plot=False)

    x = np.random.default_rng(10).normal(size=(30, 3))
    bad_t = np.eye(3)
    bad_t[0, 0] = np.nan
    with pytest.raises(PyHugeError, match="t.*non-finite"):
        huge_inference(x, bad_t, np.zeros((3, 3)))

    bad_adj = sparse.eye(3, format="csc")
    bad_adj.data[0] = np.inf
    with pytest.raises(PyHugeError, match="adj.*non-finite"):
        huge_inference(x, np.eye(3), bad_adj)

    bad_graph = np.zeros((3, 3))
    bad_graph[0, 1] = np.inf
    with pytest.raises(PyHugeError, match="g.*non-finite"):
        huge_plot(bad_graph)


def test_secondary_matrix_conversion_errors_are_normalized():
    with pytest.raises(PyHugeError, match="g.*numeric.*matrix"):
        huge_plot([["not-a-number"]])


def test_gaussian_inference_ignores_nonparanormal_method():
    x = np.random.default_rng(11).normal(size=(40, 3))
    t = np.eye(3)
    adj = np.zeros((3, 3))

    default = huge_inference(x, t, adj, type_="Gaussian")
    ignored = huge_inference(x, t, adj, type_="Gaussian", method="unused")

    assert np.array_equal(default.p, ignored.p, equal_nan=True)
    assert default.error == ignored.error

    with pytest.raises(PyHugeError, match="method.*one of"):
        huge_inference(
            x,
            t,
            adj,
            type_="Nonparanormal",
            method="unused",
        )


def test_gaussian_inference_supports_one_variable():
    x = np.arange(4.0).reshape(-1, 1)
    result = huge_inference(x, np.eye(1), np.zeros((1, 1)))

    assert result.p.shape == (1, 1)
    assert result.p[0, 0] == pytest.approx(
        0.15729920705028505, abs=1e-15
    )
    assert result.error == 0.0


@pytest.mark.parametrize("type_", ("Gaussian", "Nonparanormal"))
def test_inference_rejects_one_observation_without_runtime_warnings(type_):
    x = np.asarray([[1.0, 2.0, 3.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PyHugeError, match="at least two observations"):
            huge_inference(x, np.eye(3), np.zeros((3, 3)), type_=type_)


def test_nonparanormal_inference_rejects_one_variable():
    x = np.arange(4.0).reshape(-1, 1)

    with pytest.raises(PyHugeError, match="at least two variables"):
        huge_inference(
            x,
            np.eye(1),
            np.zeros((1, 1)),
            type_="Nonparanormal",
        )


@pytest.mark.parametrize("type_", ("Gaussian", "Nonparanormal"))
def test_inference_rejects_constant_columns_without_runtime_warnings(type_):
    x = np.column_stack((np.arange(4.0), np.ones(4)))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PyHugeError, match="constant column"):
            huge_inference(x, np.eye(2), np.zeros((2, 2)), type_=type_)


@pytest.mark.parametrize("type_", ("Gaussian", "Nonparanormal"))
@pytest.mark.parametrize("bad_diagonal", (0.0, -1.0))
def test_inference_requires_a_positive_t_diagonal(type_, bad_diagonal):
    x = np.random.default_rng(12).normal(size=(8, 2))
    t = np.diag([bad_diagonal, 1.0])

    with pytest.raises(PyHugeError, match="positive diagonal"):
        huge_inference(x, t, np.zeros((2, 2)), type_=type_)


@pytest.mark.parametrize(
    ("type_", "method"),
    (
        ("Gaussian", "score"),
        ("Nonparanormal", "score"),
        ("Nonparanormal", "wald"),
    ),
)
def test_inference_rejects_nonfinite_edge_p_values(type_, method):
    x = np.random.default_rng(13).normal(size=(8, 2))
    t = np.eye(2) * 1e-200

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PyHugeError, match="finite|numerically"):
            huge_inference(
                x,
                t,
                np.zeros((2, 2)),
                type_=type_,
                method=method,
            )


@pytest.mark.parametrize("method", ("score", "wald"))
def test_nonparanormal_inference_keeps_finite_edge_limits(method):
    x = np.asarray([[0.0, 0.0], [1.0, 1.0]])

    result = huge_inference(
        x,
        np.eye(2),
        np.zeros((2, 2)),
        type_="Nonparanormal",
        method=method,
    )

    assert np.isfinite(result.p[0, 1])
    assert np.isfinite(result.p[1, 0])


def test_gaussian_inference_normalizes_extreme_finite_data():
    x = np.asarray([[1e308, 1e308], [-1e308, -1e308]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = huge_inference(x, np.eye(2), np.zeros((2, 2)))

    assert np.all(np.isfinite(result.p))
    assert np.array_equal(result.p, result.p.T)
    assert np.isfinite(result.error)


@pytest.mark.parametrize("method", ("score", "wald"))
def test_nonparanormal_inference_handles_extreme_finite_ranks(method):
    x = np.asarray([[1e308, 1e308], [-1e308, -1e308]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = huge_inference(
            x,
            np.eye(2),
            np.zeros((2, 2)),
            type_="Nonparanormal",
            method=method,
        )

    assert np.isfinite(result.p[0, 1])
    assert np.isfinite(result.p[1, 0])


def test_non_eps_plot_ignores_file_naming_arguments(tmp_path):
    graph = np.zeros((3, 3))
    assert (
        huge_plot(
            graph,
            epsflag=False,
            graph_name="",
            cur_num=0,
            location=object(),
        )
        is None
    )

    with pytest.raises(PyHugeError, match="cur_num.*positive integer"):
        huge_plot(graph, epsflag=True, cur_num=0)
    with pytest.raises(PyHugeError, match="graph_name.*non-empty"):
        huge_plot(graph, epsflag=True, graph_name="")
    with pytest.raises(PyHugeError, match="location.*existing directory"):
        huge_plot(graph, epsflag=True, location=str(tmp_path / "missing"))


def test_huge_select_ebic_requires_glasso():
    x = np.random.default_rng(1).normal(size=(30, 8))
    fit = huge(x, method="ct", nlambda=4, verbose=False)
    with pytest.raises(PyHugeError, match="requires a glasso fit"):
        huge_select(fit, criterion="ebic", verbose=False)


@pytest.mark.parametrize("value", (True, 1.5, "2", np.nan, np.inf, [2]))
def test_positive_integer_inputs_are_strict_numeric_scalars(value):
    x = np.random.default_rng(2).normal(size=(30, 6))

    with pytest.raises(PyHugeError, match="positive integer"):
        huge(x, method="ct", nlambda=value, verbose=False)


@pytest.mark.parametrize(
    "value", (True, "0.5", [0.5], np.nan, np.inf, 0.0, -0.1, 1.1)
)
def test_ratio_inputs_are_strict_numeric_scalars(value):
    x = np.random.default_rng(3).normal(size=(30, 6))

    with pytest.raises(PyHugeError, match="lambda_min_ratio"):
        huge(x, method="ct", lambda_min_ratio=value, verbose=False)


@pytest.mark.parametrize("method", ("mb", "ct", "glasso", "tiger"))
def test_explicit_lambda_ignores_inactive_grid_arguments(method):
    x = np.random.default_rng(4).normal(size=(50, 6))

    fit = huge(
        x,
        method=method,
        lambda_=[0.5],
        nlambda=0,
        lambda_min_ratio=np.nan,
        verbose=False,
    )

    assert np.array_equal(fit.lambda_path, np.array([0.5]))


def test_ct_allows_zero_threshold_but_other_methods_do_not():
    x = np.random.default_rng(5).normal(size=(40, 6))
    fit = huge(x, method="ct", lambda_=[0.2, 0.0], verbose=False)
    assert np.array_equal(fit.lambda_path, np.array([0.2, 0.0]))

    for method in ("mb", "glasso", "tiger"):
        with pytest.raises(PyHugeError, match="positive"):
            huge(x, method=method, lambda_=[0.0], verbose=False)


@pytest.mark.parametrize("value", (True, "0.2", ["0.2"], 0.2 + 0.0j))
def test_lambda_requires_a_real_numeric_sequence(value):
    x = np.random.default_rng(8).normal(size=(30, 6))

    with pytest.raises(PyHugeError, match="numeric sequence"):
        huge(x, method="ct", lambda_=value, verbose=False)


@pytest.mark.parametrize("value", (True, "0.5", None, [0.5], np.nan, np.inf))
def test_ebic_gamma_requires_a_finite_numeric_scalar(value):
    x = np.random.default_rng(6).normal(size=(40, 6))
    fit = huge(x, method="glasso", nlambda=3, verbose=False)

    with pytest.raises(PyHugeError, match="ebic_gamma.*finite numeric scalar"):
        huge_select(fit, criterion="ebic", ebic_gamma=value, verbose=False)


@pytest.mark.parametrize(
    ("name", "kwargs"),
    (
        ("v", {"v": np.nan}),
        ("u", {"u": np.inf}),
        ("prob", {"prob": np.nan}),
        ("v", {"v": "0.3"}),
        ("prob", {"prob": True}),
    ),
)
def test_generator_numeric_parameters_are_finite_scalars(name, kwargs):
    with pytest.raises(PyHugeError, match=rf"{name}.*finite numeric scalar"):
        huge_generator(n=20, d=6, graph="random", verbose=False, **kwargs)


def test_generator_default_probability_is_capped_for_small_dimensions():
    for graph in ("random", "cluster"):
        for d in (1, 2, 4):
            result = huge_generator(
                n=10, d=d, graph=graph, random_state=7, verbose=False
            )
            assert result.data.shape == (10, d)
            assert result.sigmahat.shape == (d, d)
