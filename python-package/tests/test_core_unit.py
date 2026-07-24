"""Core unit tests for pyhuge native implementation."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import sparse

from pyhuge import core


@pytest.mark.parametrize(
    ("wrapper_name", "method", "wrapper_kwargs", "forwarded"),
    (
        (
            "huge_mb",
            "mb",
            {"scr": True, "scr_num": 3, "sym": "and"},
            {
                "scr": True,
                "scr_num": 3,
                "cov_output": False,
                "sym": "and",
            },
        ),
        (
            "huge_glasso",
            "glasso",
            {"scr": True, "cov_output": True},
            {"scr": True, "cov_output": True, "sym": "or"},
        ),
        ("huge_ct", "ct", {}, {"sym": "or"}),
        ("huge_tiger", "tiger", {"sym": "and"}, {"sym": "and"}),
    ),
)
def test_method_wrappers_forward_exact_kwargs(
    monkeypatch, wrapper_name, method, wrapper_kwargs, forwarded
):
    x = np.arange(24.0).reshape(6, 4)
    lambda_path = np.asarray([0.4, 0.2])
    result_marker = object()
    calls = []

    def fake_huge(**kwargs):
        calls.append(kwargs)
        return result_marker

    monkeypatch.setattr(core, "huge", fake_huge)

    result = getattr(core, wrapper_name)(
        x,
        lambda_=lambda_path,
        nlambda=7,
        lambda_min_ratio=0.37,
        verbose=False,
        backend="native",
        input_type="data",
        **wrapper_kwargs,
    )

    assert result is result_marker
    assert len(calls) == 1
    actual = calls[0]
    assert actual.pop("x") is x
    assert actual.pop("lambda_") is lambda_path
    assert actual == {
        "nlambda": 7,
        "lambda_min_ratio": 0.37,
        "method": method,
        "verbose": False,
        "backend": "native",
        "input_type": "data",
        **forwarded,
    }


@pytest.mark.parametrize(
    ("wrapper", "parameter_names"),
    (
        (
            core.huge_mb,
            (
                "x", "lambda_", "nlambda", "lambda_min_ratio", "scr",
                "scr_num", "sym", "verbose", "backend", "input_type",
            ),
        ),
        (
            core.huge_glasso,
            (
                "x", "lambda_", "nlambda", "lambda_min_ratio", "scr",
                "cov_output", "verbose", "backend", "input_type",
            ),
        ),
        (
            core.huge_ct,
            (
                "x", "lambda_", "nlambda", "lambda_min_ratio", "verbose",
                "backend", "input_type",
            ),
        ),
        (
            core.huge_tiger,
            (
                "x", "lambda_", "nlambda", "lambda_min_ratio", "sym",
                "verbose", "backend", "input_type",
            ),
        ),
    ),
)
def test_method_wrapper_signatures_are_explicit(wrapper, parameter_names):
    signature = inspect.signature(wrapper)

    assert tuple(signature.parameters) == parameter_names
    assert (
        signature.parameters["input_type"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert all(
        parameter.kind
        not in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize(
    "call",
    (
        lambda: core.huge(np.ones((5, 4)), backend="legacy"),
        lambda: core.huge_mb(np.ones((5, 4)), backend="legacy"),
        lambda: core.huge_glasso(np.ones((5, 4)), backend="legacy"),
        lambda: core.huge_ct(np.ones((5, 4)), backend="legacy"),
        lambda: core.huge_tiger(np.ones((5, 4)), backend="legacy"),
        lambda: core.huge_select(object(), backend="legacy"),
    ),
)
def test_public_estimators_reject_invalid_backend_without_stale_version(call):
    with pytest.raises(core.PyHugeError) as error:
        call()

    assert "native" in str(error.value)
    assert "0.3" not in str(error.value)


def test_mb_screening_size_matches_r_contract():
    rng = np.random.default_rng(51)

    with pytest.raises(core.PyHugeError, match="scr_num.*< d"):
        core.huge_mb(
            rng.normal(size=(12, 4)),
            scr=True,
            scr_num=4,
            nlambda=2,
            verbose=False,
        )

    wide = core.huge_mb(
        rng.normal(size=(5, 8)),
        scr=True,
        nlambda=2,
        verbose=False,
    )
    assert wide.raw["scr"] is True
    assert wide.raw["scr_num"] == 4

    tall = core.huge_mb(
        rng.normal(size=(8, 4)),
        scr=True,
        nlambda=2,
        verbose=False,
    )
    assert tall.raw["scr"] is False
    assert tall.raw["scr_num"] is None


def test_ct_native_path_is_converted_one_lambda_at_a_time(monkeypatch):
    correlation = np.asarray(
        [
            [1.0, 0.7, -0.4],
            [0.7, 1.0, 0.2],
            [-0.4, 0.2, 1.0],
        ]
    )
    lambda_path = np.asarray([0.5, 0.0, 0.5, 0.25])

    class FakeNative:
        def __init__(self):
            self.calls = []

        def threshold_path(self, corr, lambdas):
            self.calls.append(np.asarray(lambdas).copy())
            adjacency = (np.abs(corr) > float(lambdas[0])).astype(np.uint8)
            np.fill_diagonal(adjacency, 0)
            return [adjacency]

    native = FakeNative()
    monkeypatch.setattr(core, "_CPP", native)

    result = core._run_ct(correlation, lambda_path)

    assert [call.tolist() for call in native.calls] == [
        [0.5], [0.0], [0.5], [0.25],
    ]
    assert len(result) == len(lambda_path)
    for matrix, threshold in zip(result, lambda_path):
        expected = (np.abs(correlation) > threshold).astype(float)
        np.fill_diagonal(expected, 0.0)
        assert sparse.isspmatrix_csc(matrix)
        assert matrix.dtype == np.float64
        assert np.array_equal(matrix.toarray(), expected)


def test_ct_native_partial_failure_rebuilds_full_python_path(monkeypatch):
    correlation = np.asarray(
        [
            [1.0, 0.8, 0.1],
            [0.8, 1.0, -0.4],
            [0.1, -0.4, 1.0],
        ]
    )
    lambda_path = np.asarray([0.7, 0.3, 0.0])

    class FailingNative:
        def __init__(self):
            self.call_count = 0

        def threshold_path(self, corr, lambdas):
            self.call_count += 1
            if self.call_count == 2:
                raise RuntimeError("synthetic native failure")
            return [np.ones_like(corr, dtype=np.uint8)]

    native = FailingNative()
    monkeypatch.setattr(core, "_CPP", native)

    result = core._run_ct(correlation, lambda_path)

    assert native.call_count == 2
    assert len(result) == len(lambda_path)
    for matrix, threshold in zip(result, lambda_path):
        expected = (np.abs(correlation) > threshold).astype(float)
        np.fill_diagonal(expected, 0.0)
        assert np.array_equal(matrix.toarray(), expected)


def test_summary_helpers():
    fit = core.HugeResult(
        method="mb",
        lambda_path=np.array([0.3, 0.2, 0.1]),
        sparsity=np.array([0.05, 0.08, 0.12]),
        path=[sparse.csc_matrix(np.eye(4)) for _ in range(3)],
        cov_input=False,
        data=np.ones((20, 4)),
        raw=None,
    )
    sel = core.HugeSelectResult(
        criterion="ric",
        opt_lambda=0.2,
        opt_sparsity=0.08,
        refit=sparse.csc_matrix(np.eye(4)),
        raw=None,
    )

    s1 = core.huge_summary(fit)
    s2 = core.huge_select_summary(sel)

    assert s1.path_length == 3
    assert s1.n_samples == 20
    assert s2.criterion == "ric"
    assert s2.refit_n_features == 4


def test_stockdata_loader_shape():
    stock = core.huge_stockdata()
    assert stock.data.shape == (1258, 452)
    assert stock.info.shape == (452, 3)


def test_ric_select_opt_index_is_one_based():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(80, 12))
    fit = core.huge(x, method="mb", nlambda=5, verbose=False)
    sel = core.huge_select(fit, criterion="ric", verbose=False)

    assert sel.opt_index is not None
    assert 1 <= sel.opt_index <= len(fit.path)


def test_roc_shapes():
    theta = sparse.csc_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float))
    path = [
        sparse.csc_matrix(np.zeros((3, 3), dtype=float)),
        sparse.csc_matrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)),
    ]
    roc = core.huge_roc(path, theta, plot=False)
    assert roc.f1.shape == (2,)
    assert roc.fp.shape == (2,)
    assert roc.tp.shape == (2,)
    assert np.all((roc.fp >= 0.0) & (roc.fp <= 1.0))
    assert np.all((roc.tp >= 0.0) & (roc.tp <= 1.0))
    assert 0.0 <= roc.auc <= 1.0


def test_roc_auc_supports_numpy_before_trapezoid(monkeypatch):
    if hasattr(np, "trapezoid"):
        legacy_trapz = np.trapezoid
    else:
        legacy_trapz = np.trapz
    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.setattr(np, "trapz", legacy_trapz, raising=False)

    theta = sparse.csc_matrix(
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    )
    path = [
        sparse.csc_matrix(np.zeros((3, 3), dtype=float)),
        sparse.csc_matrix(np.ones((3, 3), dtype=float) - np.eye(3)),
    ]

    roc = core.huge_roc(path, theta, plot=False)
    assert roc.auc == pytest.approx(0.5)


def test_roc_auc_is_invariant_to_equal_fpr_order_and_duplicates():
    def graph(edges):
        matrix = np.zeros((4, 4), dtype=float)
        for left, right in edges:
            matrix[left, right] = 1.0
            matrix[right, left] = 1.0
        return sparse.csc_matrix(matrix)

    truth = graph([(0, 1), (0, 2)])
    low = graph([(2, 3)])
    high = graph([(0, 1), (0, 2), (2, 3)])
    complete = sparse.csc_matrix(np.ones((4, 4)) - np.eye(4))

    paths = (
        [low, high, complete],
        [high, low, complete],
        [low, high, high, complete],
    )
    auc = [
        core.huge_roc(path, truth, plot=False).auc for path in paths
    ]

    assert auc == pytest.approx([0.75, 0.75, 0.75])


@pytest.mark.parametrize(
    "theta",
    (np.zeros((3, 3), dtype=float), np.ones((3, 3), dtype=float) - np.eye(3)),
)
def test_roc_rejects_one_class_truth(theta):
    path = [sparse.csc_matrix(np.zeros((3, 3), dtype=float))]

    with pytest.raises(core.PyHugeError, match="ROC/AUC"):
        core.huge_roc(path, sparse.csc_matrix(theta), plot=False)


def test_sparse_path_metrics_do_not_densify(monkeypatch):
    one_edge = sparse.csc_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=float,
        )
    )
    three_edges = sparse.csc_matrix(
        np.array(
            [
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=float,
        )
    )

    def reject_dense_conversion(*args, **kwargs):
        raise AssertionError("sparse path metrics must not densify")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_dense_conversion)
    path = [one_edge, three_edges]

    assert core._edge_count(path) == pytest.approx([1.0, 3.0])
    assert core._path_sparsity(path) == pytest.approx([1.0 / 6.0, 0.5])
