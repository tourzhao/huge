"""Guard: the native extension must export its full expected surface.

Deleting an `m.def` in native_core_bindings.cpp would otherwise only be
caught by whichever test happens to call that function.
"""

from __future__ import annotations

import numpy as np
import pytest

_native_core = pytest.importorskip("pyhuge._native_core")

EXPECTED = {
    "threshold_path",
    "sparsity_path",
    "spmb_graph",
    "spmb_scr",
    "spmb_graphsqrt",
    "hugeglasso",
    "ric",
    "sfgen",
    "omp_max_threads",
}


def test_native_core_exports_expected_symbols() -> None:
    actual = {n for n in dir(_native_core) if not n.startswith("_")}
    missing = EXPECTED - actual
    unexpected = actual - EXPECTED
    assert not missing, f"native symbols missing: {sorted(missing)}"
    assert not unexpected, (
        f"new native symbols {sorted(unexpected)} — add them to EXPECTED "
        "so this guard keeps tracking the full surface"
    )


def test_sparsity_path_matches_valid_square_adjacencies() -> None:
    matrices = [
        np.asfortranarray(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            dtype=float,
        ),
        np.ones((3, 3), dtype=np.uint8) - np.eye(3, dtype=np.uint8),
        np.eye(3, dtype=np.uint8),
    ]

    actual = np.asarray(_native_core.sparsity_path(matrices), dtype=float)

    np.testing.assert_allclose(actual, np.asarray([2.0 / 3.0, 1.0, 0.0]))


def test_sparsity_path_preserves_empty_and_singleton_boundaries() -> None:
    empty = np.asarray(_native_core.sparsity_path([]), dtype=float)
    singleton = np.asarray(
        _native_core.sparsity_path([np.ones((1, 1), dtype=np.uint8)]),
        dtype=float,
    )

    assert empty.shape == (0,)
    np.testing.assert_array_equal(singleton, np.asarray([0.0]))


@pytest.mark.parametrize("shape", ((2, 3), (3, 2), (0, 0)))
def test_sparsity_path_rejects_rectangular_or_empty_matrices(shape) -> None:
    matrix = np.ones(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="non-empty square"):
        _native_core.sparsity_path([matrix])


@pytest.mark.parametrize("shape", ((3,), (2, 2, 1)))
def test_sparsity_path_rejects_rank_mismatch(shape) -> None:
    matrix = np.ones(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="two-dimensional"):
        _native_core.sparsity_path([matrix])


def test_sparsity_path_rejects_non_array_elements() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        _native_core.sparsity_path([object()])


def test_sparsity_path_rejects_nonconvertible_array_elements() -> None:
    matrix = np.asarray([["not-a-number"]])

    with pytest.raises(ValueError, match="convertible to uint8"):
        _native_core.sparsity_path([matrix])


def test_ric_rotation_zero_and_n_are_equivalent() -> None:
    x = np.asarray(
        [[-2.0, 3.0], [-1.0, -1.0], [1.0, 2.0], [2.0, -2.0]]
    )

    at_zero = _native_core.ric(x, np.asarray([0.0]))
    at_n = _native_core.ric(x, np.asarray([float(x.shape[0])]))

    assert np.isfinite(at_zero)
    assert at_zero == pytest.approx(at_n, rel=0.0, abs=0.0)


@pytest.mark.parametrize("shape", ((4,), (2, 2, 1)))
def test_ric_rejects_x_rank_mismatch(shape) -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        _native_core.ric(np.ones(shape), np.asarray([0.0]))


@pytest.mark.parametrize("shape", ((0, 2), (2, 0)))
def test_ric_rejects_empty_x(shape) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _native_core.ric(np.ones(shape), np.asarray([0.0]))


def test_ric_rejects_nonfinite_x() -> None:
    x = np.asarray([[0.0, 1.0], [2.0, np.nan]])
    with pytest.raises(ValueError, match="finite"):
        _native_core.ric(x, np.asarray([0.0]))


@pytest.mark.parametrize(
    "r",
    (
        np.asarray([]),
        np.asarray([[0.0]]),
        np.asarray([-1.0]),
        np.asarray([5.0]),
        np.asarray([1.5]),
        np.asarray([np.nan]),
        np.asarray([np.inf]),
        np.asarray([2.0**40]),
    ),
)
def test_ric_rejects_invalid_rotation_inputs(r) -> None:
    x = np.arange(8.0).reshape(4, 2)
    with pytest.raises(ValueError, match="rotation|one-dimensional|non-empty"):
        _native_core.ric(x, r)


def _reconstruct_sparse_support(out, nlambda: int, d: int) -> np.ndarray:
    indptr = np.asarray(out["support_indptr"], dtype=np.int64)
    indices = np.asarray(out["support_indices"], dtype=np.int32)
    assert indptr.shape == (nlambda, d + 1)

    support = np.zeros((nlambda, d, d), dtype=bool)
    offset = 0
    for path_index in range(nlambda):
        local = indptr[path_index]
        count = int(local[-1])
        for column in range(d):
            start = offset + int(local[column])
            end = offset + int(local[column + 1])
            support[path_index, indices[start:end], column] = True
        offset += count
    assert offset == indices.size
    return support


def test_mb_sparse_support_matches_compatible_dense_output() -> None:
    corr = np.asarray(
        [[1.0, 0.7, 0.2], [0.7, 1.0, 0.6], [0.2, 0.6, 1.0]]
    )
    lambdas = np.asarray([0.5, 0.2])

    dense = _native_core.spmb_graph(corr, lambdas)
    compressed = _native_core.spmb_graph(
        corr, lambdas, dense_output=False
    )
    support = _reconstruct_sparse_support(compressed, 2, 3)
    expected = np.abs(np.asarray(dense["beta"])).transpose(0, 2, 1) > 0

    assert compressed["beta"] is None
    np.testing.assert_array_equal(support, expected)
    np.testing.assert_array_equal(compressed["df"], dense["df"])
    for mode in ("or", "and"):
        actual_graph = (
            np.logical_or(support, support.transpose(0, 2, 1))
            if mode == "or"
            else np.logical_and(support, support.transpose(0, 2, 1))
        )
        dense_graph = (
            np.logical_or(expected, expected.transpose(0, 2, 1))
            if mode == "or"
            else np.logical_and(expected, expected.transpose(0, 2, 1))
        )
        np.testing.assert_array_equal(actual_graph, dense_graph)


def test_tiger_sparse_support_matches_compatible_dense_output() -> None:
    x = np.random.default_rng(57).normal(size=(80, 5))
    lambdas = np.asarray([0.5])

    dense = _native_core.spmb_graphsqrt(x, lambdas)
    compressed = _native_core.spmb_graphsqrt(
        x, lambdas, dense_output=False
    )
    support = _reconstruct_sparse_support(compressed, 1, 5)
    expected = np.abs(np.asarray(dense["beta"])).transpose(0, 2, 1) > 0

    assert compressed["beta"] is None
    np.testing.assert_array_equal(support, expected)
    np.testing.assert_array_equal(compressed["df"], dense["df"])


def test_sparse_support_avoids_zero_path_beta_cube() -> None:
    dimension = 500
    out = _native_core.spmb_graph(
        np.eye(dimension),
        np.ones(10),
        dense_output=False,
    )

    assert out["beta"] is None
    assert np.asarray(out["support_indices"]).size == 0
    assert np.asarray(out["support_indptr"]).shape == (10, dimension + 1)
    np.testing.assert_array_equal(
        np.asarray(out["df"]), np.zeros((dimension, 10))
    )
