"""Regression tests pinning bugs fixed in 2.0.0.

Each test names the defect it guards; a failure means the fix regressed.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyhuge import (
    PyHugeError,
    huge,
    huge_ct,
    huge_generator,
    huge_glasso,
    huge_mb,
    huge_npn,
    huge_select,
    huge_tiger,
)


def test_tiger_icov_exactly_symmetric():
    # 2.0.0: shared-core in-place symmetrization read already-averaged values
    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 30))
    fit = huge(x, method="tiger", nlambda=5, verbose=False)
    for ic in fit.icov:
        assert np.abs(ic - ic.T).max() == 0.0


def test_glasso_loglik_and_ebic_match_returned_precision_matrices():
    covariance = np.diag([2.0, 3.0])
    diagonal_fit = huge(
        covariance, method="glasso", lambda_=[0.5], verbose=False
    )
    precision = diagonal_fit.icov[0]
    direct = np.linalg.slogdet(precision)[1] - np.trace(covariance @ precision)
    assert diagonal_fit.loglik[0] == pytest.approx(direct, abs=1e-12)

    rng = np.random.default_rng(3)
    x = rng.normal(size=(35, 10))
    fit = huge(x, method="glasso", nlambda=10, verbose=False)
    sample_correlation = np.corrcoef(x, rowvar=False)
    direct_path = np.asarray(
        [
            np.linalg.slogdet(item)[1] - np.trace(sample_correlation @ item)
            for item in fit.icov
        ]
    )
    assert fit.loglik == pytest.approx(direct_path, abs=1e-10)

    edge_count = np.asarray([item.nnz / 2.0 for item in fit.path])
    expected_ebic = (
        -x.shape[0] * direct_path
        + np.log(x.shape[0]) * edge_count
        + 4.0 * 0.5 * np.log(x.shape[1]) * edge_count
    )
    selected = huge_select(fit, criterion="ebic", verbose=False)
    assert selected.ebic_score == pytest.approx(expected_ebic, abs=1e-9)
    assert selected.opt_index == int(np.argmin(expected_ebic) + 1)


def test_glasso_precision_path_and_edge_metadata_are_undirected():
    from pyhuge import core

    covariance = np.asarray(
        [
            [1, 0.517180, -0.517349, 0.326007, 0.263074, -0.434894, -0.594540],
            [0.517180, 1, -0.007732, 0.807446, -0.171208, -0.164722, 0.110568],
            [-0.517349, -0.007732, 1, 0.108058, -0.413103, 0.652356, 0.356858],
            [0.326007, 0.807446, 0.108058, 1, 0.131565, -0.283979, -0.107923],
            [0.263074, -0.171208, -0.413103, 0.131565, 1, -0.473447, -0.695847],
            [-0.434894, -0.164722, 0.652356, -0.283979, -0.473447, 1, 0.390406],
            [-0.594540, 0.110568, 0.356858, -0.107923, -0.695847, 0.390406, 1],
        ],
        dtype=float,
    )
    fit = huge(
        covariance,
        method="glasso",
        nlambda=8,
        lambda_min_ratio=0.005,
        cov_output=True,
        verbose=False,
    )
    native = core._CPP.hugeglasso(
        covariance, fit.lambda_path, False, False
    )
    native_path_raw = np.asarray(native["path"])
    assert native_path_raw.shape == (len(fit.lambda_path), *covariance.shape)
    assert native_path_raw.dtype == np.uint8
    native_path = native_path_raw.astype(bool, copy=False)
    native_icov = np.asarray(native["icov"], dtype=float)
    native_df = np.asarray(native["df"], dtype=float)
    native_sparsity = np.asarray(native["sparsity"], dtype=float)
    d = covariance.shape[0]

    for index, (path, precision) in enumerate(zip(native_path, native_icov)):
        assert np.array_equal(precision, precision.T)
        assert np.array_equal(path, path.T)
        support = precision != 0.0
        np.fill_diagonal(support, False)
        assert np.array_equal(path, support)
        edge_count = int(np.count_nonzero(np.triu(path, k=1)))
        assert native_df[index] == edge_count
        assert native_sparsity[index] == 2.0 * edge_count / (d * (d - 1))
        sign, logdet = np.linalg.slogdet(precision)
        assert sign > 0
        inverse_residual = np.linalg.norm(
            fit.cov[index] @ precision - np.eye(d), ord=np.inf
        )
        assert inverse_residual <= 1e-2
        direct = logdet - np.trace(covariance @ precision)
        assert native["loglik"][index] == pytest.approx(direct, abs=1e-10)
        assert np.array_equal(fit.path[index].toarray() != 0, path)
        assert np.array_equal(fit.icov[index], precision)


def test_glasso_precision_symmetrization_avoids_finite_overflow():
    covariance = 1e-307 * np.asarray(
        [[1.0, 0.99], [0.99, 1.0]]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = huge(
            covariance,
            method="glasso",
            lambda_=[1e-309],
            cov_output=True,
            verbose=False,
        )

    precision = fit.icov[0]
    assert np.all(np.isfinite(precision))
    assert np.array_equal(precision, precision.T)
    assert np.all(np.isfinite(fit.cov[0]))
    assert np.all(np.isfinite(fit.loglik))
    direct = np.linalg.slogdet(precision)[1] - np.trace(
        covariance @ precision
    )
    assert fit.loglik[0] == pytest.approx(direct, abs=1e-10)


def test_glasso_loglik_has_no_absolute_pivot_scale():
    covariance = 1e20 * np.asarray(
        [[1.0, 0.5], [0.5, 1.0]]
    )
    fit = huge(
        covariance,
        method="glasso",
        lambda_=[1e19],
        cov_output=True,
        verbose=False,
    )

    precision = fit.icov[0]
    assert np.all(np.isfinite(precision))
    assert np.all(np.isfinite(fit.cov[0]))
    direct = np.linalg.slogdet(precision)[1] - np.trace(
        covariance @ precision
    )
    assert np.isfinite(direct)
    assert fit.loglik[0] == pytest.approx(direct, abs=1e-10)


@pytest.mark.parametrize("cov_output", (False, True))
def test_glasso_rejects_non_spd_precision_paths(cov_output):
    rho = 0.9999
    covariance = np.full((3, 3), rho)
    np.fill_diagonal(covariance, 1.0)

    with pytest.raises(PyHugeError, match="not positive definite"):
        huge(
            covariance,
            method="glasso",
            lambda_=[0.1, 0.0002],
            cov_output=cov_output,
            verbose=False,
        )


def test_glasso_refines_precision_after_symmetrization():
    rho = 0.9999
    covariance = np.full((3, 3), rho)
    np.fill_diagonal(covariance, 1.0)

    fit = huge(
        covariance,
        method="glasso",
        lambda_=[0.01],
        cov_output=True,
        verbose=False,
    )
    fit_without_cov = huge(
        covariance,
        method="glasso",
        lambda_=[0.01],
        cov_output=False,
        verbose=False,
    )
    precision = fit.icov[0]
    assert np.array_equal(precision, precision.T)
    assert fit_without_cov.icov[0] == pytest.approx(precision, abs=1e-12)
    assert np.linalg.slogdet(precision)[0] > 0
    residual = np.linalg.norm(
        fit.cov[0] @ precision - np.eye(3), ord=np.inf
    )
    assert residual <= 1e-2


def test_glasso_returns_coherent_pair_for_ill_conditioned_toeplitz_input():
    d = 20
    rho = 0.99
    index = np.arange(d)
    covariance = rho ** np.abs(index[:, None] - index[None, :])

    fit = huge(
        covariance,
        method="glasso",
        lambda_=[0.001],
        cov_output=True,
        verbose=False,
    )

    precision = fit.icov[0]
    estimated_covariance = fit.cov[0]
    assert np.array_equal(precision, precision.T)
    assert np.array_equal(estimated_covariance, estimated_covariance.T)
    assert np.linalg.slogdet(precision)[0] > 0
    assert np.linalg.slogdet(estimated_covariance)[0] > 0
    residual = np.linalg.norm(
        estimated_covariance @ precision - np.eye(d), ord=np.inf
    )
    assert residual <= 1e-2


def test_glasso_refines_a_finite_iteration_limit_candidate():
    covariance = np.asarray(
        [
            [
                1.0000000000000002, 0.18009786022575919,
                -0.10400558762095684, 0.76774091516199638,
                0.18742182119843603,
            ],
            [
                0.18009786022575919, 1, 0.83834121674080364,
                0.48252423066148048, -0.066223216563695037,
            ],
            [
                -0.10400558762095684, 0.83834121674080364, 1,
                0.097388083759665525, -0.56897959252915542,
            ],
            [
                0.76774091516199638, 0.48252423066148048,
                0.097388083759665525, 1.0000000000000002,
                0.44589188604391788,
            ],
            [
                0.18742182119843603, -0.066223216563695037,
                -0.56897959252915542, 0.44589188604391788,
                0.99999999999999989,
            ],
        ],
        dtype=float,
    )

    with pytest.warns(RuntimeWarning, match="iteration limit"):
        fit = huge(
            covariance,
            method="glasso",
            lambda_=[0.00083834121674080362],
            cov_output=True,
            input_type="covariance",
            verbose=False,
        )

    assert np.linalg.slogdet(fit.icov[0])[0] > 0
    residual = np.linalg.norm(
        fit.cov[0] @ fit.icov[0] - np.eye(5), ord=np.inf
    )
    assert residual <= 1e-2


@pytest.mark.parametrize("cov_output", (False, True))
@pytest.mark.parametrize("scale", (1e-320, 1e308))
def test_glasso_rejects_nonfinite_native_results(scale, cov_output):
    covariance = np.eye(2) * scale

    with pytest.raises(PyHugeError, match="non-finite"):
        huge(
            covariance,
            method="glasso",
            lambda_=[scale],
            cov_output=cov_output,
            verbose=False,
        )


def test_mb_screen_builder_excludes_response_under_ties():
    from pyhuge import core

    corr = np.ones((4, 4), dtype=float)
    idx = core._build_screen_idx(corr, 1)

    assert idx.shape == (1, 4)
    for response in range(corr.shape[0]):
        assert response not in idx[:, response]


def test_native_mb_screen_rejects_unsafe_indices():
    from pyhuge import core

    if core._CPP is None:
        pytest.skip("requires native extension")

    corr = np.eye(4)
    lambdas = np.asarray([0.5])
    valid = np.asarray(
        [[1, 0, 0, 0], [2, 2, 1, 1]], dtype=np.int32
    )
    unsafe = {
        "wrong rank": np.asarray([1, 0, 0, 0], dtype=np.int32),
        "wrong columns": valid[:, :3],
        "empty": np.empty((0, 4), dtype=np.int32),
        "out of range": np.where(
            np.arange(valid.size).reshape(valid.shape) == 0, 4, valid
        ).astype(np.int32),
        "internal sentinel": np.where(
            np.arange(valid.size).reshape(valid.shape) == 0, -1, valid
        ).astype(np.int32),
        "negative": np.where(
            np.arange(valid.size).reshape(valid.shape) == 0, -2, valid
        ).astype(np.int32),
        "duplicate": valid.copy(),
        "self index": valid.copy(),
    }
    unsafe["duplicate"][:, 0] = [1, 1]
    unsafe["self index"][0, 0] = 0

    for case, idx in unsafe.items():
        with pytest.raises(ValueError, match="idx_scr"):
            core._CPP.spmb_scr(corr, lambdas, idx)

    out = core._CPP.spmb_scr(corr, lambdas, valid)
    assert np.asarray(out["beta"]).shape == (1, 4, 4)


def test_ct_default_path_is_undirected_tie_safe_and_refittable():
    def covariance(edge_weights):
        matrix = np.eye(4)
        upper = np.triu_indices(4, 1)
        matrix[upper] = np.asarray(edge_weights, dtype=float)
        matrix[(upper[1], upper[0])] = matrix[upper]
        return matrix

    cases = (
        (
            covariance([0.30, 0.25, 0.20, 0.15, 0.10, 0.05]),
            3,
            0.5,
            np.asarray([0.25, 0.20, 0.15]),
            np.asarray([1, 2, 3]),
        ),
        (
            covariance([0.30, 0.20, 0.20, 0.10, 0.08, 0.05]),
            3,
            0.5,
            np.asarray([0.20, 0.20, 0.10]),
            np.asarray([1, 1, 3]),
        ),
        (
            covariance([0.10] * 6),
            2,
            1.0,
            np.asarray([0.10, 0.0]),
            np.asarray([0, 6]),
        ),
        (
            np.eye(4),
            3,
            0.5,
            np.asarray([0.0, 0.0, 0.0]),
            np.asarray([0, 0, 0]),
        ),
    )

    for matrix, nlambda, ratio, expected_lambda, expected_edges in cases:
        fit = huge(
            matrix,
            method="ct",
            nlambda=nlambda,
            lambda_min_ratio=ratio,
            verbose=False,
        )
        refit = huge(
            matrix,
            method="ct",
            lambda_=fit.lambda_path,
            verbose=False,
        )
        paths = [item.toarray() for item in fit.path]
        edge_count = np.asarray(
            [np.count_nonzero(np.triu(item, 1)) for item in paths]
        )

        assert np.array_equal(fit.lambda_path, expected_lambda)
        assert np.all(fit.lambda_path >= 0)
        assert np.array_equal(edge_count, expected_edges)
        assert fit.sparsity == pytest.approx(expected_edges / 6.0, abs=0.0)
        assert fit.sparsity == pytest.approx(refit.sparsity, abs=0.0)
        for index, path in enumerate(paths):
            assert np.array_equal(path, path.T)
            assert np.count_nonzero(np.diag(path)) == 0
            assert np.array_equal(path, refit.path[index].toarray())
            if index:
                assert np.all(paths[index - 1] <= path)


def test_ct_small_dimensions_and_literal_strict_threshold():
    singleton = huge(
        np.ones((1, 1)),
        method="ct",
        nlambda=3,
        lambda_min_ratio=0.5,
        verbose=False,
    )
    assert np.array_equal(singleton.lambda_path, np.zeros(3))
    assert np.array_equal(singleton.sparsity, np.zeros(3))
    assert all(item.nnz == 0 for item in singleton.path)

    pair_matrix = np.asarray([[1.0, 0.3], [0.3, 1.0]])
    pair = huge(
        pair_matrix,
        method="ct",
        nlambda=3,
        lambda_min_ratio=0.5,
        verbose=False,
    )
    pair_refit = huge(
        pair_matrix,
        method="ct",
        lambda_=pair.lambda_path,
        verbose=False,
    )
    assert np.array_equal(pair.lambda_path, np.zeros(3))
    assert np.array_equal(pair.sparsity, np.ones(3))
    for path, refit_path in zip(pair.path, pair_refit.path):
        assert np.array_equal(path.toarray(), refit_path.toarray())

    boundary = np.eye(3)
    boundary[0, 1] = boundary[1, 0] = 0.25 + 1e-15
    boundary[0, 2] = boundary[2, 0] = 0.25
    strict = huge(
        boundary, method="ct", lambda_=[0.25], verbose=False
    ).path[0].toarray()
    assert np.count_nonzero(np.triu(strict, 1)) == 1
    assert strict[0, 1] == 1
    assert strict[0, 2] == 0


def test_ct_default_full_graph_endpoint_retains_subnormal_correlation():
    rho = np.nextafter(0.0, 1.0)
    covariance = np.asarray([[1.0, rho], [rho, 1.0]])

    fit = huge(
        covariance,
        method="ct",
        nlambda=1,
        lambda_min_ratio=1.0,
        verbose=False,
    )
    refit = huge(
        covariance, method="ct", lambda_=fit.lambda_path, verbose=False
    )

    assert np.array_equal(fit.lambda_path, np.asarray([0.0]))
    assert np.array_equal(
        fit.path[0].toarray(), np.asarray([[0.0, 1.0], [1.0, 0.0]])
    )
    assert np.array_equal(fit.path[0].toarray(), refit.path[0].toarray())


def test_ct_default_intermediate_threshold_retains_subnormal_ordering():
    smallest = np.nextafter(0.0, 1.0)
    covariance = np.eye(3)
    upper = np.triu_indices(3, 1)
    covariance[upper] = np.asarray([3.0, 2.0, 1.0]) * smallest
    covariance[(upper[1], upper[0])] = covariance[upper]

    fit = huge(
        covariance,
        method="ct",
        nlambda=2,
        lambda_min_ratio=2.0 / 3.0,
        verbose=False,
    )
    refit = huge(
        covariance, method="ct", lambda_=fit.lambda_path, verbose=False
    )
    edge_count = np.asarray(
        [
            np.count_nonzero(np.triu(path.toarray(), k=1))
            for path in fit.path
        ]
    )

    assert np.array_equal(
        fit.lambda_path, np.asarray([2.0, 1.0]) * smallest
    )
    assert np.array_equal(edge_count, np.asarray([1, 2]))
    for actual, replay in zip(fit.path, refit.path):
        assert np.array_equal(actual.toarray(), replay.toarray())


def test_ct_covariance_conversion_is_scale_invariant():
    covariance = np.asarray([[1.0, 0.5], [0.5, 1.0]])
    reference = huge(
        covariance, method="ct", lambda_=[0.25], verbose=False
    )
    tiny = huge(
        covariance * 1e-100,
        method="ct",
        lambda_=[0.25],
        verbose=False,
    )

    assert reference.cov_input is True
    assert tiny.cov_input is True
    assert reference.path[0].nnz == 2
    np.testing.assert_array_equal(
        reference.path[0].toarray(), tiny.path[0].toarray()
    )


def test_covariance_normalization_preserves_extreme_scale_symmetry():
    from pyhuge import core

    rho = 1e-200
    variances = np.asarray([np.finfo(float).max, 1e-300])
    covariance_value = (
        rho * np.sqrt(variances[0])
    ) * np.sqrt(variances[1])
    covariance = np.asarray(
        [[variances[0], covariance_value],
         [covariance_value, variances[1]]]
    )

    for current in (covariance, covariance[::-1, ::-1]):
        correlation = core._cov_to_corr(current)
        assert np.array_equal(correlation, correlation.T)
        assert correlation[0, 1] == pytest.approx(rho, rel=1e-14)

        fit = huge(
            current, method="ct", lambda_=[rho / 2.0], verbose=False
        )
        assert np.array_equal(
            fit.path[0].toarray(),
            np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        )


def test_covariance_symmetrization_avoids_finite_overflow():
    covariance = 1e308 * np.asarray([[1.0, 0.5], [0.5, 1.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = huge(
            covariance, method="ct", lambda_=[0.25], verbose=False
        )

    assert fit.path[0].nnz == 2


def test_covariance_projection_preserves_exact_subnormal_diagonal():
    from pyhuge import core

    smallest = np.nextafter(0.0, 1.0)
    covariance = np.eye(2) * smallest
    assert smallest > 0.0
    assert np.array_equal(core._cov_to_corr(covariance), np.eye(2))

    for method in ("ct", "glasso", "tiger"):
        fit = huge(
            covariance,
            method=method,
            lambda_=[0.25],
            verbose=False,
        )
        assert fit.cov_input is True
        assert fit.sparsity[0] == 0.0
        assert fit.path[0].nnz == 0
        if fit.icov is not None:
            assert np.all(np.isfinite(fit.icov[0]))

    native = core._CPP.spmb_graphsqrt(
        covariance, np.asarray([0.25]), 1, 0.1, True
    )
    assert np.all(np.isfinite(native["icov"]))
    assert np.array_equal(native["lambda"], [0.25])


def test_covariance_detection_matches_r_symmetry_tolerance():
    from pyhuge import core

    roundoff = np.asarray([[1.0, 0.5], [0.5 + 1e-15, 1.0]])
    material_asymmetry = np.asarray(
        [[1.0, 0.5], [0.500001, 1.0]]
    )

    assert core._is_covariance_input(roundoff)
    assert not core._is_covariance_input(material_asymmetry)


def test_covariance_detection_is_invariant_to_finite_uniform_scale():
    from pyhuge import core

    raw = np.asarray([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0],
                      [7.0, 8.0, 10.0]])
    reference = huge(
        raw, method="ct", lambda_=[0.95], verbose=False
    )
    tiger_reference = huge(
        raw, method="tiger", nlambda=1, verbose=False
    )

    for factor in (1.0, 1e-15, 1e-200, np.nextafter(0.0, 1.0)):
        current = raw * factor
        assert not core._is_covariance_input(current)

        for method in ("mb", "glasso"):
            routed = huge(
                current, method=method, lambda_=[0.95], verbose=False
            )
            assert routed.cov_input is False

        fit = huge(
            current, method="ct", lambda_=[0.95], verbose=False
        )
        assert fit.cov_input is False
        assert np.array_equal(
            fit.path[0].toarray(), reference.path[0].toarray()
        )

        tiger = huge(
            current, method="tiger", nlambda=1, verbose=False
        )
        assert tiger.cov_input is False
        assert np.allclose(
            tiger.lambda_path, tiger_reference.lambda_path,
            rtol=0.0, atol=1e-14,
        )

        with pytest.raises(ValueError, match="symmetric"):
            core._CPP.spmb_graphsqrt(
                current, np.asarray([1.0]), 1, 0.1, True
            )

    near = np.asarray([[1.0, 0.5], [0.5 + 1e-15, 1.0]]) * 1e-200
    assert core._is_covariance_input(near)
    fit = huge(near, method="tiger", lambda_=[1.0], verbose=False)
    assert fit.cov_input is True
    core._CPP.spmb_graphsqrt(
        near, np.asarray([1.0]), 1, 0.1, True
    )

    extreme = np.asarray(
        [[np.finfo(float).max, 1.0], [-1.0, 1e-300]]
    )
    assert not core._is_covariance_input(extreme)


def test_covariance_symmetry_tolerance_uses_correlation_scale():
    from pyhuge import core

    epsilon = np.finfo(float).eps
    tolerance = 100.0 * epsilon
    for variance in (
        np.finfo(float).tiny,
        1.0,
        np.finfo(float).max,
    ):
        within = np.asarray(
            [[variance, 0.0], [tolerance * variance, variance]]
        )
        outside = np.asarray(
            [[variance, 0.0], [101.0 * epsilon * variance, variance]]
        )

        assert core._is_covariance_input(within)
        core._CPP.spmb_graphsqrt(
            within, np.asarray([1.0]), 1, 0.1, True
        )
        assert not core._is_covariance_input(outside)
        with pytest.raises(ValueError, match="symmetric"):
            core._CPP.spmb_graphsqrt(
                outside, np.asarray([1.0]), 1, 0.1, True
            )

    maximum = np.finfo(float).max
    negligible = np.asarray([[maximum, 1.0], [-1.0, maximum]])
    assert core._is_covariance_input(negligible)
    fit = huge(
        negligible, method="tiger", lambda_=[1.0], verbose=False
    )
    assert fit.cov_input is True
    core._CPP.spmb_graphsqrt(
        negligible, np.asarray([1.0]), 1, 0.1, True
    )

    extreme = np.asarray([[maximum, 1.0], [-1.0, 1e-300]])
    assert not core._is_covariance_input(extreme)
    with pytest.raises(ValueError, match="symmetric"):
        core._CPP.spmb_graphsqrt(
            extreme, np.asarray([1.0]), 1, 0.1, True
        )

    invalid_diagonal = np.asarray([[0.0, 1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError, match="positive.*diagonal"):
        core._CPP.spmb_graphsqrt(
            invalid_diagonal, np.asarray([1.0]), 1, 0.1, True
        )


def test_nearly_symmetric_square_raw_data_is_not_misrouted():
    x = np.asarray([[1.0, 0.5], [0.500001, 1.0]])

    for method in ("ct", "mb"):
        fit = huge(
            x, method=method, lambda_=[0.75], verbose=False
        )
        assert fit.cov_input is False
        assert fit.path[0].nnz == 2


def test_input_type_disambiguates_square_symmetric_observations():
    x = np.asarray(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
    )
    wrappers = {
        "ct": huge_ct,
        "mb": huge_mb,
        "glasso": huge_glasso,
        "tiger": huge_tiger,
    }

    for method, wrapper in wrappers.items():
        routed = huge(
            x,
            method=method,
            lambda_=[1.0],
            input_type="data",
            verbose=False,
        )
        direct = wrapper(
            x, lambda_=[1.0], input_type="data", verbose=False
        )
        assert routed.cov_input is False
        assert direct.cov_input is False

    for method in ("ct", "glasso", "tiger"):
        automatic = huge(
            x, method=method, lambda_=[1.0], verbose=False
        )
        covariance = huge(
            x,
            method=method,
            lambda_=[1.0],
            input_type="covariance",
            verbose=False,
        )
        assert automatic.cov_input is True
        assert covariance.cov_input is True


def test_tiger_input_type_keeps_correlation_and_lambda_native(monkeypatch):
    from pyhuge import core

    x = np.asarray(
        [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
    )
    raw_native = core._CPP.spmb_graphsqrt(x, None, 1, 0.1, False)
    covariance_native = core._CPP.spmb_graphsqrt(
        x, None, 1, 0.1, True
    )

    def fail_frontend_conversion(_):
        raise AssertionError("TIGER must not build correlation in Python")

    monkeypatch.setattr(core, "_cov_to_corr", fail_frontend_conversion)
    raw_fit = huge_tiger(
        x, nlambda=1, input_type="data", verbose=False
    )
    covariance_fit = huge_tiger(
        x, nlambda=1, input_type="covariance", verbose=False
    )
    automatic = huge_tiger(x, nlambda=1, verbose=False)

    np.testing.assert_array_equal(
        raw_fit.lambda_path, np.asarray(raw_native["lambda"])
    )
    np.testing.assert_array_equal(
        covariance_fit.lambda_path,
        np.asarray(covariance_native["lambda"]),
    )
    np.testing.assert_array_equal(
        automatic.lambda_path, covariance_fit.lambda_path
    )
    assert not np.array_equal(
        raw_fit.lambda_path, covariance_fit.lambda_path
    )


def test_input_type_rejects_invalid_and_incompatible_declarations():
    x = np.arange(6.0).reshape(3, 2)

    for value in ("raw", None, 1, ("data", "auto")):
        with pytest.raises(PyHugeError, match="input_type"):
            huge_ct(
                x, lambda_=[1.0], input_type=value, verbose=False
            )
    with pytest.raises(PyHugeError, match="square"):
        huge_ct(
            x,
            lambda_=[1.0],
            input_type="covariance",
            verbose=False,
        )
    with pytest.raises(PyHugeError, match="symmetric"):
        huge_tiger(
            np.asarray([[1.0, 1.0], [0.0, 1.0]]),
            lambda_=[1.0],
            input_type="covariance",
            verbose=False,
        )


def test_select_refits_preserve_explicit_square_data_routing(monkeypatch):
    from pyhuge import core

    x = np.asarray([[3.0, 2.0, 2.0], [2.0, 4.0, 4.0], [2.0, 4.0, 7.0]])
    fit = huge(
        x,
        method="tiger",
        lambda_=[1.0],
        input_type="data",
        verbose=False,
    )
    selected = huge_select(
        fit, criterion="ric", rep_num=x.shape[0], verbose=False
    )
    data_refit = huge_tiger(
        x,
        lambda_=[selected.opt_lambda],
        input_type="data",
        verbose=False,
    )
    automatic_refit = huge_tiger(
        x, lambda_=[selected.opt_lambda], verbose=False
    )

    np.testing.assert_array_equal(
        selected.refit.toarray(), data_refit.path[0].toarray()
    )
    assert not np.array_equal(
        selected.refit.toarray(), automatic_refit.path[0].toarray()
    )

    stars_data = np.random.default_rng(4701).normal(size=(12, 3))
    stars_fit = huge(
        stars_data,
        method="ct",
        lambda_=[0.8, 0.4],
        input_type="data",
        verbose=False,
    )
    original_huge = core.huge
    recursive_input_types = []

    def recording_huge(*args, **kwargs):
        recursive_input_types.append(kwargs.get("input_type"))
        return original_huge(*args, **kwargs)

    monkeypatch.setattr(core, "huge", recording_huge)
    huge_select(
        stars_fit,
        criterion="stars",
        rep_num=2,
        n_jobs=1,
        verbose=False,
    )
    assert recursive_input_types == ["data", "data"]


def test_nearly_symmetric_raw_tiger_lambda_stays_native():
    x = np.asarray([[1.0, 0.5], [0.500001, 1.0]])

    fit = huge(x, method="tiger", nlambda=1, verbose=False)

    assert fit.cov_input is False
    assert fit.lambda_path == pytest.approx(np.asarray([1.0]), abs=1e-12)


@pytest.mark.parametrize("method", ("ct", "glasso"))
@pytest.mark.parametrize("diagonal", (0.0, -1.0))
def test_covariance_conversion_rejects_nonpositive_diagonal(method, diagonal):
    covariance = np.diag([diagonal, 1.0])

    with pytest.raises(PyHugeError, match="positive.*diagonal"):
        huge(covariance, method=method, lambda_=[0.25], verbose=False)


@pytest.mark.parametrize("method", ("ct", "glasso", "tiger"))
def test_covariance_input_rejects_cauchy_schwarz_violation(method):
    covariance = np.asarray([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(PyHugeError, match="valid covariance"):
        huge(covariance, method=method, lambda_=[0.25], verbose=False)


@pytest.mark.parametrize("method", ("ct", "tiger"))
def test_covariance_input_rejects_non_positive_semidefinite_matrix(method):
    covariance = np.asarray(
        [[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]]
    )

    with pytest.raises(PyHugeError, match="positive semidefinite"):
        huge(covariance, method=method, lambda_=[1.0], verbose=False)


def test_glasso_regularizes_indefinite_covariance_input():
    covariance = np.asarray(
        [[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]]
    )

    fits = (
        huge(
            covariance,
            method="glasso",
            lambda_=[0.5],
            cov_output=True,
            verbose=False,
        ),
        huge_glasso(
            covariance,
            lambda_=[0.5],
            cov_output=True,
            verbose=False,
        ),
    )
    for fit in fits:
        precision = fit.icov[0]
        estimate = fit.cov[0]
        assert np.linalg.eigvalsh(precision).min() > 0.0
        assert np.linalg.eigvalsh(estimate).min() > 0.0
        residual = np.linalg.norm(
            estimate @ precision - np.eye(3), ord=np.inf
        )
        assert residual <= 1e-2


def test_tiger_native_rejects_non_positive_semidefinite_covariance():
    from pyhuge import core

    covariance = np.asarray(
        [[1.0, 0.9, 0.9], [0.9, 1.0, 0.0], [0.9, 0.0, 1.0]]
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        core._CPP.spmb_graphsqrt(
            covariance, np.asarray([1.0]), 1, 0.1, True
        )


def test_tiger_native_covariance_input_validates_symmetry():
    from pyhuge import core

    asymmetric = np.asarray([[1.0, 1.0], [-1.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        core._CPP.spmb_graphsqrt(
            asymmetric, np.asarray([0.25]), 1, 0.1, True
        )

    near = np.asarray([[1.0, 0.5], [0.5 + 1e-15, 1.0]])
    out = core._CPP.spmb_graphsqrt(
        near, np.asarray([0.25]), 1, 0.1, True
    )
    assert np.array_equal(out["lambda"], [0.25])


@pytest.mark.parametrize("method", ("ct", "glasso", "tiger"))
@pytest.mark.parametrize(
    "covariance",
    (
        pytest.param(np.ones((2, 2)), id="singular"),
        pytest.param(
            np.asarray(
                [
                    [1.0, -0.5 - 1e-14, -0.5 - 1e-14],
                    [-0.5 - 1e-14, 1.0, -0.5 - 1e-14],
                    [-0.5 - 1e-14, -0.5 - 1e-14, 1.0],
                ]
            ),
            id="spectral-roundoff",
        ),
    ),
)
def test_covariance_input_accepts_psd_and_spectral_roundoff(
    method, covariance
):
    fit = huge(
        covariance, method=method, lambda_=[1.0], verbose=False
    )

    assert fit.cov_input is True
    assert fit.path[0].nnz == 0


def test_covariance_conversion_clips_only_roundoff_excess():
    covariance = np.asarray([[1.0, 1.0 + 5e-9], [1.0 + 5e-9, 1.0]])
    fit = huge(covariance, method="ct", lambda_=[0.9], verbose=False)

    assert fit.path[0].nnz == 2


@pytest.mark.parametrize("method", ("mb", "glasso"))
def test_weak_nonzero_correlation_sets_default_lambda(method):
    rho = 5e-4
    first = np.asarray([-1.0, -1.0, 1.0, 1.0])
    orthogonal = np.asarray([-1.0, 1.0, -1.0, 1.0])
    x = np.column_stack(
        (first, rho * first + np.sqrt(1.0 - rho**2) * orthogonal)
    )
    expected = np.geomspace(rho, 0.2 * rho, num=3)

    fit = huge(
        x,
        method=method,
        nlambda=3,
        lambda_min_ratio=0.2,
        verbose=False,
    )

    assert fit.lambda_path == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(
    ("method", "x"),
    (
        pytest.param(
            "mb", np.asarray([[-1.0], [0.0], [1.0]]), id="single-variable"
        ),
        pytest.param("glasso", np.eye(2), id="identity-covariance"),
    ),
)
def test_zero_offdiagonal_retains_default_lambda_fallback(method, x):
    expected = np.geomspace(1e-3, 2e-4, num=3)

    fit = huge(
        x,
        method=method,
        nlambda=3,
        lambda_min_ratio=0.2,
        verbose=False,
    )

    assert fit.lambda_path == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("method", ("mb", "glasso"))
def test_ratio_one_keeps_a_repeatable_nonincreasing_path(method):
    rng = np.random.default_rng(321)
    x = rng.normal(size=(60, 6))
    correlation = np.corrcoef(x, rowvar=False)
    offdiagonal = ~np.eye(correlation.shape[0], dtype=bool)
    lambda_max = np.max(np.abs(correlation[offdiagonal]))
    fit = huge(
        x,
        method=method,
        nlambda=3,
        lambda_min_ratio=1.0,
        verbose=False,
    )

    assert fit.lambda_path[0] == pytest.approx(lambda_max, abs=1e-15)
    assert np.array_equal(
        fit.lambda_path, np.full(3, fit.lambda_path[0])
    )
    assert all((item != fit.path[0]).nnz == 0 for item in fit.path[1:])
    if method == "mb":
        assert np.array_equal(fit.df, np.repeat(fit.df[:, :1], 3, axis=1))
    else:
        assert np.array_equal(fit.df, np.repeat(fit.df[:1], 3))
        assert np.array_equal(fit.loglik, np.repeat(fit.loglik[:1], 3))
        assert all(
            np.array_equal(item, fit.icov[0]) for item in fit.icov[1:]
        )

    selected = huge_select(
        fit,
        criterion="stars",
        rep_num=2,
        n_jobs=1,
        verbose=False,
    )
    assert selected.variability.shape == (3,)


@pytest.mark.parametrize("method", ("mb", "glasso"))
def test_regularization_lambda_allows_ties_but_rejects_increases(method):
    x = np.random.default_rng(322).normal(size=(50, 5))

    tied = huge(
        x, method=method, lambda_=[0.5, 0.5, 0.2], verbose=False
    )
    assert np.array_equal(tied.lambda_path, np.asarray([0.5, 0.5, 0.2]))

    with pytest.raises(PyHugeError, match="non-increasing"):
        huge(x, method=method, lambda_=[0.2, 0.5], verbose=False)


def test_ric_zero_lower_endpoint_uses_safe_refit_or_certified_fallback():
    x = np.column_stack(
        ([-1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, 1.0])
    )

    for method in ("ct", "mb", "glasso", "tiger"):
        fit = huge(
            x, method=method, lambda_=[1.0], verbose=False
        )
        if method in {"glasso", "tiger"}:
            with pytest.warns(
                RuntimeWarning,
                match="RIC selected lambda = 0.*original fitted path",
            ):
                selected = huge_select(
                    fit, criterion="ric", rep_num=x.shape[0],
                    verbose=False,
                )
            assert selected.raw["ric_fallback"] is True
            assert np.array_equal(
                selected.refit.toarray(), fit.path[0].toarray()
            )
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                selected = huge_select(
                    fit, criterion="ric", rep_num=x.shape[0],
                    verbose=False,
                )
            assert not any(
                "original fitted path" in str(item.message)
                for item in caught
            )
            assert selected.refit.nnz == 2
            assert selected.raw["ric_fallback"] is False

        assert selected.opt_lambda == 0.0
        assert np.isfinite(selected.opt_sparsity)


def test_ric_zero_does_not_erase_representable_weak_correlation():
    first = np.asarray([-1.0, -1.0, 1.0, 1.0])
    orthogonal = np.asarray([-1.0, 1.0, -1.0, 1.0])
    # Below the former fixed 8-epsilon cutoff, but safely above the
    # pair-specific dot-product roundoff bound.
    x = np.column_stack((first, orthogonal + 1.5e-15 * first))
    fit = huge(
        x, method="ct", lambda_=[0.5], verbose=False
    )
    selected = huge_select(
        fit, criterion="ric", rep_num=x.shape[0], verbose=False
    )

    assert selected.opt_lambda == 0.0
    assert selected.refit.nnz == 2


def test_explicit_lambda_domain_and_error_priority():
    x = np.random.default_rng(324).normal(size=(20, 3))

    for method in ("ct", "mb", "glasso", "tiger"):
        with pytest.raises(PyHugeError, match="at least one"):
            huge(x, method=method, lambda_=[], verbose=False)
        with pytest.raises(PyHugeError, match="non-finite"):
            huge(x, method=method, lambda_=[np.inf], verbose=False)

    with pytest.raises(PyHugeError, match="non-negative"):
        huge(x, method="ct", lambda_=[-0.1], verbose=False)
    for method in ("mb", "glasso", "tiger"):
        with pytest.raises(PyHugeError, match="positive"):
            huge(x, method=method, lambda_=[0.0], verbose=False)

    for method in ("mb", "glasso", "tiger"):
        with pytest.raises(PyHugeError, match="non-finite"):
            huge(
                x, method=method, lambda_=[0.1, 0.2, np.nan],
                verbose=False,
            )
        with pytest.raises(PyHugeError, match="positive"):
            huge(
                x, method=method, lambda_=[0.1, 0.2, -1.0],
                verbose=False,
            )


def test_public_lambda_is_scalar_or_one_dimensional():
    raw = np.random.default_rng(325).normal(size=(30, 3))
    covariance = np.asarray(
        [[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]]
    )
    valid = (
        0.9,
        np.asarray(0.9),
        np.asarray([0.9, 0.8, 0.7, 0.6]),
    )
    invalid = (
        np.asarray([[0.9, 0.8, 0.7, 0.6]]),
        np.asarray([[0.9], [0.8], [0.7], [0.6]]),
        np.asarray([[0.9, 0.8], [0.7, 0.6]]),
        np.asarray([[[0.9, 0.8]], [[0.7, 0.6]]]),
        [[0.9, 0.8], [0.7, 0.6]],
    )

    for method in ("ct", "mb", "glasso", "tiger"):
        x = raw if method == "mb" else covariance
        for value in valid:
            fit = huge(x, method=method, lambda_=value, verbose=False)
            assert np.array_equal(
                fit.lambda_path, np.asarray(value, dtype=float).reshape(-1)
            )
        for value in invalid:
            with pytest.raises(PyHugeError, match="one-dimensional"):
                huge(x, method=method, lambda_=value, verbose=False)

    with pytest.raises(PyHugeError, match="numeric sequence"):
        huge(
            covariance, method="ct",
            lambda_=[[0.9, 0.8], [0.7]], verbose=False,
        )


def test_python_native_lambda_entries_require_one_dimension():
    from pyhuge import core

    correlation = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    raw = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    index = np.asarray([[1, 0]], dtype=np.int32)
    entries = {
        "ct": lambda grid: core._CPP.threshold_path(correlation, grid),
        "mb": lambda grid: core._CPP.spmb_graph(correlation, grid),
        "mb_scr": lambda grid: core._CPP.spmb_scr(
            correlation, grid, index
        ),
        "glasso": lambda grid: core._CPP.hugeglasso(
            correlation, grid, False, False
        ),
        "tiger": lambda grid: core._CPP.spmb_graphsqrt(
            raw, grid, 2, 0.1, False
        ),
    }

    for entry in entries.values():
        entry(np.asarray([0.9, 0.8]))
        for value in (np.asarray(0.9), np.asarray([[0.9, 0.8]])):
            with pytest.raises(
                (ValueError, RuntimeError),
                match="dimension|one-dimensional|expected 1",
            ):
                entry(value)


def test_native_regularization_entries_fail_closed_on_unsafe_lambda():
    from pyhuge import core

    correlation = np.asarray([[1.0, 0.3], [0.3, 1.0]])
    index = np.asarray([[1, 0]], dtype=np.int32)
    entries = {
        "mb": lambda grid: core._CPP.spmb_graph(correlation, grid),
        "mb_scr": lambda grid: core._CPP.spmb_scr(
            correlation, grid, index
        ),
        "glasso": lambda grid: core._CPP.hugeglasso(
            correlation, grid, False, False
        ),
    }

    for name, entry in entries.items():
        entry(np.asarray([0.2, 0.2, 0.1]))
        with pytest.raises(ValueError, match="non-increasing"):
            entry(np.asarray([0.1, 0.2]))
        for value in (0.0, -0.1, np.nan, np.inf):
            with pytest.raises(ValueError, match="positive and finite"):
                entry(np.asarray([value]))
        with pytest.raises(ValueError, match="positive and finite"):
            entry(np.asarray([0.1, 0.2, np.nan]))
        with pytest.raises(ValueError, match="positive and finite"):
            entry(np.asarray([0.1, 0.2, -0.1]))
        with pytest.raises(ValueError, match="nlambda.*positive"):
            entry(np.asarray([]))


def test_native_threshold_path_validates_shape_and_lambda_domain():
    from pyhuge import core

    correlation = np.asarray(
        [[1.0, 0.4, 0.1], [0.4, 1.0, 0.2], [0.1, 0.2, 1.0]]
    )
    supplied = np.asarray([0.1, 0.5, 0.2, 0.0])
    path = core._CPP.threshold_path(correlation, supplied)

    assert len(path) == supplied.size
    for matrix, threshold in zip(path, supplied):
        expected = (np.abs(correlation) > threshold).astype(np.uint8)
        np.fill_diagonal(expected, 0)
        assert np.array_equal(matrix, expected)

    # The rectangular probe deliberately provides extra storage. If the
    # square guard regresses, the old implementation still cannot read past
    # this buffer while the test reports the missing exception.
    for shape in ((2, 3), (0, 0)):
        with pytest.raises(ValueError, match="non-empty square"):
            core._CPP.threshold_path(np.ones(shape), np.asarray([0.1]))
    with pytest.raises(ValueError, match="one-dimensional"):
        core._CPP.threshold_path(correlation, np.asarray([[0.1, 0.2]]))
    with pytest.raises(ValueError, match="at least one"):
        core._CPP.threshold_path(correlation, np.asarray([]))
    for value in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="finite and non-negative"):
            core._CPP.threshold_path(correlation, np.asarray([value]))


def test_ct_preserves_supplied_lambda_order_and_stars_checks_it():
    x = np.random.default_rng(323).normal(size=(40, 4))
    supplied = np.asarray([0.1, 0.5, 0.2, 0.0])
    fit = huge(x, method="ct", lambda_=supplied, verbose=False)

    assert np.array_equal(fit.lambda_path, supplied)
    for index, threshold in enumerate(supplied):
        single = huge(
            x, method="ct", lambda_=[threshold], verbose=False
        )
        assert (fit.path[index] != single.path[0]).nnz == 0
        assert fit.sparsity[index] == single.sparsity[0]

    ascending = huge(
        x, method="ct", lambda_=[0.1, 0.3, 0.5], verbose=False
    )
    unordered = huge(
        x, method="ct", lambda_=[0.5, 0.1, 0.3], verbose=False
    )
    for candidate in (ascending, unordered):
        with pytest.raises(PyHugeError, match="StARS.*non-increasing"):
            huge_select(
                candidate, criterion="stars", rep_num=2,
                n_jobs=1, verbose=False,
            )

    tied = huge(
        x, method="ct", lambda_=[0.5, 0.5, 0.2], verbose=False
    )
    selected = huge_select(
        tied, criterion="stars", rep_num=2, n_jobs=1, verbose=False
    )
    assert selected.variability.shape == (3,)
    assert huge_select(
        ascending, criterion="ric", rep_num=2, verbose=False
    ).criterion == "ric"


def test_weak_nonzero_covariance_sets_glasso_default_lambda():
    covariance = 1e-20 * np.asarray([[1.0, 0.5], [0.5, 1.0]])
    expected = np.geomspace(5e-21, 1e-21, num=3)

    fit = huge(
        covariance,
        method="glasso",
        nlambda=3,
        lambda_min_ratio=0.2,
        input_type="covariance",
        verbose=False,
    )

    assert fit.lambda_path == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_glasso_auto_covariance_matches_r_historical_default_lambda_scale():
    covariance = np.asarray([[0.001, 0.002], [0.002, 0.02]])
    legacy_max = float(np.max(np.abs(covariance - np.eye(2))))

    automatic = huge(
        covariance,
        method="glasso",
        nlambda=2,
        lambda_min_ratio=0.1,
        verbose=False,
    )
    explicit = huge(
        covariance,
        method="glasso",
        nlambda=2,
        lambda_min_ratio=0.1,
        input_type="covariance",
        verbose=False,
    )

    assert automatic.lambda_path == pytest.approx(
        [legacy_max, 0.1 * legacy_max], abs=1e-15
    )
    assert explicit.lambda_path == pytest.approx([0.002, 0.0002], abs=1e-15)


def test_default_regularization_path_retains_small_and_subnormal_tails():
    from pyhuge import core

    base = np.asarray([[1.0, 1e-200], [1e-200, 1.0]])
    path = core._build_lambda_path(
        base_matrix=base,
        method="mb",
        lambda_=None,
        nlambda=3,
        lambda_min_ratio=1e-200,
    )

    assert path.shape == (3,)
    assert np.isfinite(path).all()
    assert (path > 0.0).all()
    assert np.all(np.diff(path) <= 0.0)
    assert path[0] == 1e-200
    assert np.log(path[1]) == pytest.approx(np.log(1e-300), abs=1e-12)
    assert path[2] == np.nextafter(0.0, 1.0)

    representable = core._build_lambda_path(
        base_matrix=np.asarray([[1.0, 0.5], [0.5, 1.0]]),
        method="glasso",
        lambda_=None,
        nlambda=2,
        lambda_min_ratio=1e-8,
    )
    assert representable[-1] == pytest.approx(5e-9, rel=1e-14)

    near_one = core._build_lambda_path(
        base_matrix=np.asarray(
            [[1.0, np.finfo(float).tiny],
             [np.finfo(float).tiny, 1.0]]
        ),
        method="mb",
        lambda_=None,
        nlambda=10,
        lambda_min_ratio=np.nextafter(1.0, 0.0),
    )
    assert near_one[0] == np.finfo(float).tiny
    assert np.all(np.diff(near_one) <= 0.0)


@pytest.mark.parametrize("method", ("ct", "mb", "glasso", "tiger"))
@pytest.mark.parametrize(
    ("grid", "path_length"),
    (
        pytest.param(
            {"nlambda": 3, "lambda_min_ratio": 0.2}, 3, id="automatic"
        ),
        pytest.param({"lambda_": [0.5, 0.2]}, 2, id="explicit"),
    ),
)
def test_raw_single_variable_estimators_return_empty_paths(
    method, grid, path_length
):
    x = np.asarray([[-1.0], [0.0], [1.0]])
    fit = huge(x, method=method, verbose=False, **grid)

    assert fit.cov_input is False
    assert fit.lambda_path.shape == (path_length,)
    assert np.isfinite(fit.lambda_path).all()
    if method == "ct" and "nlambda" in grid:
        assert np.array_equal(fit.lambda_path, np.zeros(path_length))
    else:
        assert (fit.lambda_path > 0).all()
    assert fit.sparsity.shape == (path_length,)
    assert np.array_equal(fit.sparsity, np.zeros(path_length))
    assert len(fit.path) == path_length
    assert all(graph.shape == (1, 1) and graph.nnz == 0 for graph in fit.path)

    expected_df_shape = {
        "ct": None,
        "mb": (1, path_length),
        "glasso": (path_length,),
        "tiger": (1, path_length),
    }[method]
    if expected_df_shape is None:
        assert fit.df is None
    else:
        assert fit.df.shape == expected_df_shape
        assert np.isfinite(fit.df).all()
        assert np.count_nonzero(fit.df) == 0

    if method in {"glasso", "tiger"}:
        precision = np.stack(fit.icov)
        assert precision.shape == (path_length, 1, 1)
        assert np.isfinite(precision).all()
        assert (precision[:, 0, 0] > 0).all()
    else:
        assert fit.icov is None

    if method == "glasso":
        assert fit.loglik.shape == (path_length,)
        assert np.isfinite(fit.loglik).all()
    else:
        assert fit.loglik is None

    if method != "mb":
        sample_cov = np.atleast_2d(np.cov(x, rowvar=False))
        assert np.array_equal(sample_cov, np.ones((1, 1)))
        reference = huge(sample_cov, method=method, verbose=False, **grid)
        assert reference.cov_input is True
        assert np.array_equal(fit.lambda_path, reference.lambda_path)
        assert np.array_equal(fit.sparsity, reference.sparsity)
        assert all(
            (left != right).nnz == 0
            for left, right in zip(fit.path, reference.path)
        )
        if fit.df is not None:
            assert np.array_equal(fit.df, reference.df)
        if fit.icov is not None:
            assert np.array_equal(np.stack(fit.icov), np.stack(reference.icov))
        if fit.loglik is not None:
            assert np.array_equal(fit.loglik, reference.loglik)


def test_mb_rejects_single_variable_covariance_input():
    with pytest.raises(PyHugeError, match="requires raw data matrix"):
        huge(np.ones((1, 1)), method="mb", verbose=False)


def test_tiger_native_lambda_and_square_root_lasso_kkt():
    from pyhuge import core

    if core._CPP is None:
        pytest.skip("requires native extension")

    rng = np.random.default_rng(9)
    x = rng.normal(size=(80, 15))
    lam = 0.2
    out = core._CPP.spmb_graphsqrt(x, np.asarray([lam]))
    beta = np.asarray(out["beta"], dtype=float)[0].T
    corr = np.corrcoef(x, rowvar=False)

    errors = []
    for response in range(corr.shape[0]):
        coefficient = beta[:, response]
        q = (
            corr[response, response]
            - 2.0 * corr[:, response] @ coefficient
            + coefficient @ corr @ coefficient
        )
        tau = np.sqrt(max(float(q), np.finfo(float).eps))
        score = (corr[:, response] - corr @ coefficient) / tau
        score[response] = 0.0
        active = np.abs(coefficient) > 1e-8
        active[response] = False
        active_error = (
            np.max(np.abs(score[active] - lam * np.sign(coefficient[active])))
            if np.any(active)
            else 0.0
        )
        inactive = ~active
        inactive[response] = False
        inactive_error = (
            np.max(np.maximum(np.abs(score[inactive]) - lam, 0.0))
            if np.any(inactive)
            else 0.0
        )
        errors.append(max(float(active_error), float(inactive_error)))

    assert np.count_nonzero(np.abs(beta) > 1e-8) > 0
    assert max(errors) <= 1e-6

    default_fit = huge(x, method="tiger", nlambda=6,
                       lambda_min_ratio=0.2, verbose=False)
    offdiag = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
    lambda_max = float(np.max(offdiag))
    if lambda_max == 0.0:
        lambda_max = 1e-3
    assert default_fit.lambda_path[0] == pytest.approx(lambda_max, abs=1e-12)
    assert default_fit.lambda_path[-1] == pytest.approx(0.2 * lambda_max, abs=1e-12)
    covariance_default = huge(
        np.cov(x, rowvar=False),
        method="tiger",
        nlambda=6,
        lambda_min_ratio=0.2,
        verbose=False,
    )
    assert np.allclose(
        covariance_default.lambda_path,
        default_fit.lambda_path,
        rtol=0.0,
        atol=1e-12,
    )

    custom = np.asarray([0.35, 0.2, 0.1])
    custom_fit = huge(x, method="tiger", lambda_=custom, verbose=False)
    assert np.array_equal(custom_fit.lambda_path, custom)

    covariance_fit = huge(
        np.cov(x, rowvar=False), method="tiger", lambda_=custom, verbose=False
    )
    assert covariance_fit.cov_input is True
    assert np.array_equal(covariance_fit.lambda_path, custom_fit.lambda_path)
    assert np.array_equal(covariance_fit.df, custom_fit.df)
    for raw_path, cov_path in zip(custom_fit.path, covariance_fit.path):
        assert np.array_equal(raw_path.toarray(), cov_path.toarray())
    for raw_icov, cov_icov in zip(custom_fit.icov, covariance_fit.icov):
        assert np.allclose(raw_icov, cov_icov, rtol=0.0, atol=1e-10)


def test_tiger_covariance_symmetrization_avoids_finite_overflow():
    from pyhuge import core

    correlation = np.asarray([[1.0, 0.6], [0.6, 1.0]])
    covariance = np.finfo(float).max * correlation
    lambda_path = np.asarray([0.25])

    reference = huge(
        correlation, method="tiger", lambda_=lambda_path, verbose=False
    )
    fit = huge(
        covariance, method="tiger", lambda_=lambda_path, verbose=False
    )
    assert np.allclose(
        np.stack(fit.icov), np.stack(reference.icov), rtol=0.0, atol=1e-12
    )
    assert all(
        (left != right).nnz == 0
        for left, right in zip(fit.path, reference.path)
    )

    native_reference = core._CPP.spmb_graphsqrt(
        correlation, lambda_path, 1, 0.1, True
    )
    native_fit = core._CPP.spmb_graphsqrt(
        covariance, lambda_path, 1, 0.1, True
    )
    for field in ("beta", "df", "icov", "lambda"):
        assert np.allclose(
            native_fit[field], native_reference[field],
            rtol=0.0, atol=1e-12,
        )

    auto_reference = huge(
        correlation,
        method="tiger",
        nlambda=3,
        lambda_min_ratio=0.5,
        verbose=False,
    )
    auto_fit = huge(
        covariance,
        method="tiger",
        nlambda=3,
        lambda_min_ratio=0.5,
        verbose=False,
    )
    assert np.allclose(
        auto_fit.lambda_path, auto_reference.lambda_path,
        rtol=0.0, atol=1e-14,
    )
    assert auto_fit.lambda_path[0] == pytest.approx(0.6, abs=1e-14)

    native_auto_reference = core._CPP.spmb_graphsqrt(
        correlation, None, 3, 0.5, True
    )
    native_auto_fit = core._CPP.spmb_graphsqrt(
        covariance, None, 3, 0.5, True
    )
    assert np.allclose(
        native_auto_fit["lambda"], native_auto_reference["lambda"],
        rtol=0.0, atol=1e-14,
    )

    nearly_symmetric = covariance.copy()
    nearly_symmetric[1, 0] = np.nextafter(nearly_symmetric[0, 1], 0.0)
    assert nearly_symmetric[1, 0] != nearly_symmetric[0, 1]
    projected = nearly_symmetric.copy()
    off_diagonal = (
        0.5 * nearly_symmetric[0, 1]
        + 0.5 * nearly_symmetric[1, 0]
    )
    projected[0, 1] = off_diagonal
    projected[1, 0] = off_diagonal

    near_fit = huge(
        nearly_symmetric,
        method="tiger",
        nlambda=3,
        lambda_min_ratio=0.5,
        verbose=False,
    )
    projected_fit = huge(
        projected,
        method="tiger",
        nlambda=3,
        lambda_min_ratio=0.5,
        verbose=False,
    )
    assert np.allclose(
        near_fit.lambda_path, projected_fit.lambda_path,
        rtol=0.0, atol=1e-14,
    )
    assert np.allclose(
        np.stack(near_fit.icov), np.stack(projected_fit.icov),
        rtol=0.0, atol=1e-12,
    )

    native_near = core._CPP.spmb_graphsqrt(
        nearly_symmetric, None, 3, 0.5, True
    )
    native_projected = core._CPP.spmb_graphsqrt(
        projected, None, 3, 0.5, True
    )
    for field in ("beta", "df", "icov", "lambda"):
        assert np.allclose(
            native_near[field], native_projected[field],
            rtol=0.0, atol=1e-12,
        )


def test_tiger_covariance_normalization_preserves_weak_extreme_scale_correlation():
    from pyhuge import core

    rho = 1e-200
    high = np.finfo(float).max
    low = 1e-300
    off_diagonal = rho * np.sqrt(high) * np.sqrt(low)
    covariance = np.asarray(
        [[high, off_diagonal], [off_diagonal, low]]
    )
    correlation = np.asarray([[1.0, rho], [rho, 1.0]])

    reference = huge(
        correlation,
        method="tiger",
        nlambda=3,
        lambda_min_ratio=0.5,
        verbose=False,
    )
    for current in (covariance, covariance[::-1, ::-1].copy()):
        fit = huge(
            current,
            method="tiger",
            nlambda=3,
            lambda_min_ratio=0.5,
            verbose=False,
        )
        native_fit = core._CPP.spmb_graphsqrt(
            current, None, 3, 0.5, True
        )
        assert np.allclose(
            fit.lambda_path / rho,
            reference.lambda_path / rho,
            rtol=1e-12,
            atol=0.0,
        )
        assert fit.lambda_path[0] / rho == pytest.approx(1.0, rel=1e-12)
        assert np.allclose(
            np.asarray(native_fit["lambda"]) / rho,
            reference.lambda_path / rho,
            rtol=1e-12,
            atol=0.0,
        )


def test_tiger_native_lambda_preserves_weak_nonzero_correlations():
    rho = 5e-4
    a = np.asarray([-1.0, -1.0, 1.0, 1.0])
    b = np.asarray([-1.0, 1.0, -1.0, 1.0])
    x = np.column_stack((a, rho * a + np.sqrt(1.0 - rho**2) * b))
    ratio = 0.2
    expected = np.geomspace(rho, rho * ratio, 3)

    raw_fit = huge(
        x, method="tiger", nlambda=3,
        lambda_min_ratio=ratio, verbose=False,
    )
    covariance_fit = huge(
        np.asarray([[1.0, rho], [rho, 1.0]]),
        method="tiger", nlambda=3,
        lambda_min_ratio=ratio, verbose=False,
    )

    assert np.corrcoef(x, rowvar=False)[0, 1] == pytest.approx(
        rho, abs=1e-15
    )
    assert np.allclose(raw_fit.lambda_path, expected, rtol=1e-12, atol=0.0)
    assert np.allclose(
        covariance_fit.lambda_path, expected, rtol=1e-12, atol=0.0
    )
    assert np.allclose(
        raw_fit.lambda_path, covariance_fit.lambda_path,
        rtol=1e-12, atol=0.0,
    )


def test_tiger_native_lambda_path_stays_positive_at_subnormal_limit():
    from pyhuge import core

    minimum = np.nextafter(0.0, 1.0)
    covariance = np.asarray([[1.0, minimum], [minimum, 1.0]])

    fit = huge(
        covariance, method="tiger", nlambda=3,
        lambda_min_ratio=0.1, verbose=False,
    )
    replay = huge(
        covariance, method="tiger", lambda_=fit.lambda_path,
        verbose=False,
    )
    native = core._CPP.spmb_graphsqrt(
        covariance, None, 3, 0.1, True
    )
    assert np.array_equal(fit.lambda_path, np.full(3, minimum))
    assert np.array_equal(replay.lambda_path, fit.lambda_path)
    assert np.array_equal(native["lambda"], fit.lambda_path)
    assert native["path_truncated"] is False
    assert np.isfinite(np.stack(fit.icov)).all()

    identity = huge(
        np.eye(2), method="tiger", nlambda=3,
        lambda_min_ratio=minimum, verbose=False,
    )
    native_identity = core._CPP.spmb_graphsqrt(
        np.eye(2), None, 3, minimum, True
    )
    assert identity.lambda_path[0] == 1e-3
    assert identity.lambda_path[-1] == minimum
    assert np.array_equal(native_identity["lambda"], identity.lambda_path)
    assert native_identity["path_truncated"] is False
    assert np.all(np.isfinite(identity.lambda_path))
    assert np.all(identity.lambda_path > 0.0)
    assert np.all(np.diff(identity.lambda_path) <= 0.0)
    assert np.log(identity.lambda_path[1]) == pytest.approx(
        np.log(1e-3) + np.log(minimum) / 2.0,
        abs=1e-12,
    )

    singleton = huge(
        np.eye(1), method="tiger", nlambda=3,
        lambda_min_ratio=minimum, verbose=False,
    )
    native_singleton = core._CPP.spmb_graphsqrt(
        np.eye(1), None, 3, minimum, True
    )
    assert np.array_equal(singleton.lambda_path, identity.lambda_path)
    assert np.array_equal(
        native_singleton["lambda"], singleton.lambda_path
    )
    assert native_singleton["path_truncated"] is False


def test_tiger_explicit_lambda_allows_ties_and_rejects_increases():
    from pyhuge import core

    covariance = np.asarray([[1.0, 0.3], [0.3, 1.0]])
    automatic = huge(
        covariance, method="tiger", nlambda=3,
        lambda_min_ratio=1.0, verbose=False,
    )
    replay = huge(
        covariance, method="tiger", lambda_=automatic.lambda_path,
        verbose=False,
    )
    native_replay = core._CPP.spmb_graphsqrt(
        covariance, automatic.lambda_path, 3, 0.1, True
    )

    assert np.all(np.diff(automatic.lambda_path) == 0.0)
    assert np.array_equal(replay.lambda_path, automatic.lambda_path)
    assert np.array_equal(native_replay["lambda"], automatic.lambda_path)
    assert all(
        (left != right).nnz == 0
        for left, right in zip(replay.path, automatic.path)
    )
    assert np.array_equal(
        np.stack(replay.icov), np.stack(automatic.icov)
    )

    with pytest.raises(PyHugeError, match="non-increasing"):
        huge(
            covariance, method="tiger", lambda_=[0.1, 0.2],
            verbose=False,
        )
    with pytest.raises(ValueError, match="non-increasing"):
        core._CPP.spmb_graphsqrt(
            covariance, np.asarray([0.1, 0.2]), 2, 0.1, True
        )
    with pytest.raises(ValueError, match="positive and finite"):
        core._CPP.spmb_graphsqrt(
            covariance, np.asarray([0.1, 0.2, np.nan]), 3, 0.1, True
        )


def test_tiger_raw_correlation_is_stable_across_finite_column_scales():
    from pyhuge import core

    x = np.column_stack(([-1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]))
    ratio = 0.5
    reference = huge(
        x, method="tiger", nlambda=3,
        lambda_min_ratio=ratio, verbose=False,
    )
    native_reference = core._CPP.spmb_graphsqrt(
        x, None, 3, ratio, False
    )
    explicit_reference = core._CPP.spmb_graphsqrt(
        x, np.asarray([0.25]), 1, ratio, False
    )
    assert reference.lambda_path[0] == pytest.approx(0.5, abs=1e-15)

    minimum = np.nextafter(0.0, 1.0)
    inputs = (
        x * 1e308,
        x * 1e-200,
        x * minimum,
        x * np.asarray([1e308, 1e-200]),
    )
    for current in inputs:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit = huge(
                current, method="tiger", nlambda=3,
                lambda_min_ratio=ratio, verbose=False,
            )
        native_fit = core._CPP.spmb_graphsqrt(
            current, None, 3, ratio, False
        )
        explicit_fit = core._CPP.spmb_graphsqrt(
            current, np.asarray([0.25]), 1, ratio, False
        )
        assert np.allclose(
            fit.lambda_path, reference.lambda_path,
            rtol=0.0, atol=1e-14,
        )
        assert all(
            (left != right).nnz == 0
            for left, right in zip(fit.path, reference.path)
        )
        assert np.allclose(
            np.stack(fit.icov), np.stack(reference.icov),
            rtol=0.0, atol=1e-12,
        )
        for field in ("beta", "df", "icov", "lambda"):
            assert np.allclose(
                native_fit[field], native_reference[field],
                rtol=0.0, atol=1e-12,
            )
            assert np.allclose(
                explicit_fit[field], explicit_reference[field],
                rtol=0.0, atol=1e-12,
            )

    maximum = np.finfo(float).max
    previous_1 = np.nextafter(maximum, 0.0)
    previous_2 = np.nextafter(previous_1, 0.0)
    previous_3 = np.nextafter(previous_2, 0.0)
    adjacent = np.column_stack(
        (
            [maximum, previous_1, previous_2, previous_3],
            [maximum, previous_2, previous_1, previous_3],
        )
    )
    adjacent_fit = huge(
        adjacent, method="tiger", nlambda=1, verbose=False
    )
    adjacent_native = core._CPP.spmb_graphsqrt(
        adjacent, None, 1, 0.1, False
    )
    assert adjacent_fit.lambda_path[0] == pytest.approx(
        0.8, rel=1e-14, abs=0.0
    )
    assert adjacent_native["lambda"][0] == pytest.approx(
        0.8, rel=1e-14, abs=0.0
    )


@pytest.mark.parametrize("method", ("ct", "mb", "glasso"))
def test_python_raw_estimators_preserve_extreme_finite_scales(method):
    x = np.column_stack(([-1.0, 0.0, 1.0], [-1.0, 1.0, 0.0]))
    minimum = np.nextafter(0.0, 1.0)
    inputs = (
        x * 1e308,
        x * 1e-200,
        x * minimum,
        x * np.asarray([1e308, 1e-200]),
    )
    explicit_reference = huge(
        x, method=method, lambda_=[0.4], verbose=False
    )
    automatic_reference = huge(
        x, method=method, nlambda=3,
        lambda_min_ratio=0.5, verbose=False,
    )

    for current in inputs:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            explicit = huge(
                current, method=method, lambda_=[0.4], verbose=False
            )
            automatic = huge(
                current, method=method, nlambda=3,
                lambda_min_ratio=0.5, verbose=False,
            )
        assert explicit.cov_input is False
        assert np.array_equal(
            explicit.lambda_path, explicit_reference.lambda_path
        )
        assert np.array_equal(explicit.sparsity, explicit_reference.sparsity)
        assert np.allclose(
            automatic.lambda_path, automatic_reference.lambda_path,
            rtol=0.0, atol=1e-14,
        )
        assert np.array_equal(
            automatic.sparsity, automatic_reference.sparsity
        )
        assert all(
            (left != right).nnz == 0
            for left, right in zip(explicit.path, explicit_reference.path)
        )
        assert all(
            (left != right).nnz == 0
            for left, right in zip(automatic.path, automatic_reference.path)
        )
        if explicit.df is not None:
            assert np.array_equal(explicit.df, explicit_reference.df)
        if explicit.icov is not None:
            assert np.allclose(
                np.stack(explicit.icov), np.stack(explicit_reference.icov),
                rtol=0.0, atol=1e-12,
            )


def test_python_standardization_preserves_adjacent_maximum_scale_ulps():
    from pyhuge import core

    maximum = np.finfo(float).max
    levels = [maximum]
    for _ in range(6):
        levels.append(np.nextafter(levels[-1], 0.0))
    indices = np.column_stack(
        (np.arange(7.0), np.asarray([0.0, 1.0, 3.0, 6.0, 2.0, 5.0, 4.0]))
    )
    x = np.column_stack(
        (
            np.asarray(levels),
            np.asarray(levels)[indices[:, 1].astype(int)],
        )
    )

    standardized = core._standardize(x)
    correlation = np.corrcoef(standardized, rowvar=False)
    expected = np.corrcoef(indices, rowvar=False)
    assert np.allclose(correlation, expected, rtol=0.0, atol=1e-14)

    three_point = np.asarray(levels[:3]).reshape(-1, 1)
    three_standardized = core._standardize(three_point)[:, 0]
    assert np.allclose(
        three_standardized, [1.0, 0.0, -1.0], rtol=0.0, atol=1e-15
    )
    assert abs(float(np.mean(three_standardized))) <= 1e-16

    fit = huge(x, method="ct", lambda_=[0.64], verbose=False)
    assert fit.path[0].toarray()[0, 1] == 1.0


def test_tiger_stops_before_interpolation_degeneracy():
    signal = np.arange(1.0, 31.0)
    duplicated_data = np.column_stack((signal, signal))

    with pytest.warns(RuntimeWarning, match="certified prefix"):
        fit = huge(
            duplicated_data,
            method="tiger",
            nlambda=2,
            lambda_min_ratio=0.99,
            verbose=False,
        )
    assert fit.lambda_path == pytest.approx(np.asarray([1.0]), abs=1e-12)
    returned_nlambda = fit.lambda_path.size
    assert len(fit.path) == returned_nlambda
    assert len(fit.icov) == returned_nlambda
    assert fit.sparsity.shape == (returned_nlambda,)
    assert fit.df.shape[1] == returned_nlambda
    assert np.all(np.isfinite(fit.icov[0]))
    assert fit.icov[0] == pytest.approx(np.eye(2), abs=1e-12)

    with pytest.raises(PyHugeError, match="could not certify a supplied lambda"):
        huge(
            duplicated_data,
            method="tiger",
            lambda_=[0.99],
            verbose=False,
        )


def test_tiger_selection_keeps_ric_and_rejects_unsafe_stars_grids():
    x = np.random.default_rng(321).normal(size=(120, 20))
    fit = huge(
        x,
        method="tiger",
        nlambda=4,
        lambda_min_ratio=0.4,
        verbose=False,
    )

    with pytest.raises(
        PyHugeError, match="TIGER.*StARS|common certified prefix"
    ):
        huge_select(
            fit, criterion="stars", rep_num=3, n_jobs=1, verbose=False
        )

    ric = huge_select(fit, criterion="ric", rep_num=10, verbose=False)
    assert np.isfinite(ric.opt_lambda) and ric.opt_lambda > 0.0
    assert ric.refit.shape == (20, 20)


@pytest.mark.parametrize(
    ("method", "expected"),
    (("mb", "ric"), ("ct", "stars"), ("glasso", "ebic"), ("tiger", "ric")),
)
def test_selector_default_matches_r_method_contract(method, expected):
    x = np.random.default_rng(19).normal(size=(60, 8))
    fit = huge(x, method=method, nlambda=3, verbose=False)

    selected = huge_select(
        fit, criterion=None, rep_num=3, n_jobs=1, verbose=False
    )
    explicit = huge_select(
        fit, criterion=expected, rep_num=3, n_jobs=1, verbose=False
    )

    assert selected.criterion == expected
    assert selected.opt_index == explicit.opt_index
    assert selected.opt_lambda == explicit.opt_lambda
    assert np.array_equal(selected.refit.toarray(), explicit.refit.toarray())
    if method == "ct":
        assert selected.variability is not None
    if method == "glasso":
        assert selected.ebic_score is not None


def test_selector_ignores_inactive_criterion_parameters():
    x = np.random.default_rng(20).normal(size=(50, 7))
    fit = huge(x, method="glasso", nlambda=3, verbose=False)

    selected = huge_select(
        fit,
        criterion=None,
        stars_thresh=np.nan,
        stars_subsample_ratio=np.nan,
        rep_num=0,
        n_jobs=0,
        verbose=False,
    )

    assert selected.criterion == "ebic"


@pytest.mark.parametrize(
    ("method", "criterion"),
    (
        ("ct", "ric"),
        ("mb", "ric"),
        ("glasso", "ric"),
        ("tiger", "ric"),
        ("mb", None),
        ("tiger", None),
    ),
)
def test_ric_handles_single_variable_empty_graph_boundary(method, criterion):
    x = np.asarray([[-1.0], [0.0], [1.0]])
    fit = huge(x, method=method, nlambda=3, verbose=False)

    selected = huge_select(
        fit, criterion=criterion, rep_num=x.shape[0], verbose=False
    )

    assert selected.criterion == "ric"
    assert selected.opt_lambda == 0.0
    assert selected.opt_sparsity == 0.0
    assert selected.refit.shape == (1, 1)
    assert selected.refit.nnz == 0
    assert selected.opt_index == int(np.argmin(np.abs(fit.lambda_path))) + 1
    assert selected.opt_icov is None
    assert selected.opt_cov is None
    assert selected.raw["ric_fallback"] is False
    assert selected.raw["ric_refit_lambda"] is None


@pytest.mark.parametrize("method", ("ct", "mb", "glasso", "tiger"))
def test_ric_handles_exact_zero_correlation_boundary(method):
    x = np.column_stack(
        (
            np.asarray([-1.0, -1.0, 1.0, 1.0]),
            np.asarray([-1.0, 1.0, -1.0, 1.0]),
        )
    )
    fit = huge(x, method=method, nlambda=3, verbose=False)

    selected = huge_select(
        fit, criterion="ric", rep_num=x.shape[0], verbose=False
    )

    assert selected.opt_lambda == 0.0
    assert selected.opt_sparsity == 0.0
    assert selected.refit.shape == (2, 2)
    assert selected.refit.nnz == 0
    assert selected.opt_index == int(np.argmin(np.abs(fit.lambda_path))) + 1
    assert selected.opt_icov is None
    assert selected.opt_cov is None
    assert selected.raw["ric_fallback"] is False
    assert selected.raw["ric_refit_lambda"] is None


@pytest.mark.parametrize("method", ("ct", "mb", "glasso", "tiger"))
def test_ric_selection_scale_invariant_at_tiny_scales(method):
    # 2.0.0: an absolute sd floor broke tiny-scale standardization
    rng = np.random.default_rng(9)
    x = rng.normal(size=(80, 12))
    x[:, 1] = 0.75 * x[:, 0] + 0.25 * x[:, 1]
    x[:, 3] = -0.65 * x[:, 2] + 0.35 * x[:, 3]

    lambda_path = [0.8, 0.5, 0.3]
    fit1 = huge(x, method=method, lambda_=lambda_path, verbose=False)
    fit2 = huge(
        x * 1e-15, method=method, lambda_=lambda_path, verbose=False
    )
    s1 = huge_select(
        fit1, criterion="ric", rep_num=x.shape[0], verbose=False
    )
    s2 = huge_select(
        fit2, criterion="ric", rep_num=x.shape[0], verbose=False
    )

    assert s1.opt_lambda == pytest.approx(s2.opt_lambda, rel=1e-12)
    assert s1.opt_sparsity == s2.opt_sparsity
    np.testing.assert_array_equal(s1.refit.toarray(), s2.refit.toarray())


def test_npn_matches_r_formula():
    # 2.0.0: missing per-column sd normalization, wrong truncation ranks
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 12)) ** 3
    n = x.shape[0]

    z = huge_npn(x, npn_func="shrinkage", verbose=False)
    assert z.std(axis=0, ddof=1) == pytest.approx(np.ones(12), abs=1e-12)

    zt = huge_npn(x, npn_func="truncation", verbose=False)
    assert zt.std(axis=0, ddof=1) == pytest.approx(np.ones(12), abs=1e-12)

    # spot-check the shrinkage formula itself on one column
    from scipy import stats

    ranks = stats.rankdata(x[:, 0])
    ref = stats.norm.ppf(ranks / (n + 1.0))
    ref = ref / ref.std(ddof=1)
    assert z[:, 0] == pytest.approx(ref, abs=1e-12)


def test_npn_skeptic_handles_one_and_two_variables():
    one_variable = np.asarray([[3.0], [1.0], [2.0]])
    singleton = huge_npn(
        one_variable, npn_func="skeptic", verbose=False
    )
    np.testing.assert_array_equal(singleton, np.ones((1, 1)))

    general_pair = np.asarray(
        [[1.0, 1.0], [2.0, 3.0], [3.0, 2.0], [4.0, 4.0]]
    )
    general = huge_npn(
        general_pair, npn_func="skeptic", verbose=False
    )
    expected_off_diagonal = 2.0 * np.sin((np.pi / 6.0) * 0.8)
    assert general == pytest.approx(
        np.asarray(
            [
                [1.0, expected_off_diagonal],
                [expected_off_diagonal, 1.0],
            ]
        ),
        abs=1e-15,
    )

    perfect_negative = np.asarray(
        [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]]
    )
    pair = huge_npn(
        perfect_negative, npn_func="skeptic", verbose=False
    )
    assert pair == pytest.approx(
        np.asarray([[1.0, -1.0], [-1.0, 1.0]]), abs=1e-15
    )


@pytest.mark.parametrize(
    ("x", "message"),
    (
        pytest.param(
            np.asarray([[1.0, 2.0, 3.0]]),
            "at least two observations",
            id="one-observation",
        ),
        pytest.param(
            np.asarray([[1.0]]),
            "at least two observations",
            id="one-observation-one-variable",
        ),
        pytest.param(
            np.ones((5, 1)),
            "constant column",
            id="constant-single-variable",
        ),
        pytest.param(
            np.column_stack((np.ones(5), np.arange(5.0), np.arange(5.0) ** 2)),
            "constant column",
            id="constant-first-column",
        ),
        pytest.param(
            np.column_stack((np.arange(5.0), np.ones(5), np.arange(5.0) ** 2)),
            "constant column",
            id="constant-middle-column",
        ),
        pytest.param(
            np.column_stack((np.arange(5.0), np.arange(5.0) ** 2, np.ones(5))),
            "constant column",
            id="constant-last-column",
        ),
    ),
)
def test_npn_skeptic_rejects_undefined_correlations(x, message):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PyHugeError, match=message):
            huge_npn(x, npn_func="skeptic", verbose=False)


def test_npn_constant_guard_is_skeptic_only():
    x = np.column_stack((np.arange(5.0), np.ones(5)))

    for npn_func in ("shrinkage", "truncation"):
        out = huge_npn(x, npn_func=npn_func, verbose=False)
        assert out.shape == x.shape
        assert np.all(np.isfinite(out))


def test_generator_sigma_is_correlation_matrix():
    # 2.0.0: sigma was the raw precision inverse (non-unit diagonal)
    r = huge_generator(n=50, d=25, graph="hub", verbose=False)
    assert r.sigma.diagonal() == pytest.approx(np.ones(25), abs=0)
    assert np.abs(r.omega - r.omega.T).max() == 0.0
    assert np.abs(r.sigma @ r.omega - np.eye(25)).max() < 1e-10

    small = huge_generator(
        n=8,
        d=4,
        graph="band",
        random_state=123,
        verbose=False,
    )
    expected = np.corrcoef(small.data, rowvar=False)
    assert small.sigmahat == pytest.approx(expected, abs=1e-12)
    assert np.array_equal(small.sigmahat, small.sigmahat.T)
    assert np.array_equal(np.diag(small.sigmahat), np.ones(4))

    singleton = huge_generator(
        n=8, d=1, graph="band", random_state=123, verbose=False
    )
    assert np.array_equal(singleton.sigmahat, np.ones((1, 1)))

    with pytest.raises(PyHugeError, match="n.*at least 2"):
        huge_generator(n=1, d=4, graph="band", verbose=False)


def test_stars_n_jobs_reproduces_serial_and_caps_workers(monkeypatch):
    # Pre-drawn index sets make thread-pool fits identical to serial, while
    # excessive n_jobs requests must not create more workers than subsamples.
    import concurrent.futures

    real_executor = concurrent.futures.ThreadPoolExecutor
    worker_counts = []

    def recording_executor(*args, **kwargs):
        worker_counts.append(kwargs["max_workers"])
        return real_executor(*args, **kwargs)

    monkeypatch.setattr(
        concurrent.futures, "ThreadPoolExecutor", recording_executor
    )

    rng = np.random.default_rng(3)
    x = rng.normal(size=(120, 30))
    fit = huge(x, method="mb", nlambda=5, verbose=False)
    s1 = huge_select(fit, criterion="stars", rep_num=4, n_jobs=1, verbose=False)
    with pytest.warns(RuntimeWarning, match="OpenMP or BLAS"):
        s2 = huge_select(
            fit, criterion="stars", rep_num=4, n_jobs=2, verbose=False
        )
    with pytest.warns(RuntimeWarning, match="OpenMP or BLAS"):
        s3 = huge_select(
            fit, criterion="stars", rep_num=4, n_jobs=20, verbose=False
        )
    assert worker_counts == [2, 4]
    for parallel in (s2, s3):
        assert s1.opt_index == parallel.opt_index
        assert np.array_equal(s1.variability, parallel.variability)
        assert (s1.refit != parallel.refit).nnz == 0


@pytest.mark.parametrize(
    ("rep_num", "expected_dtype"),
    (
        (1, np.uint8),
        (255, np.uint8),
        (256, np.uint16),
        (65535, np.uint16),
        (65536, np.uint32),
        (2**32 - 1, np.uint32),
    ),
)
def test_stars_count_dtype_is_smallest_safe_unsigned_type(
    rep_num, expected_dtype
):
    from pyhuge import core

    dtype = core._stars_count_dtype(rep_num)
    assert dtype == np.dtype(expected_dtype)
    count = np.zeros(1, dtype=dtype)
    count[0] = rep_num
    assert int(count[0]) == rep_num


def test_stars_count_dtype_rejects_unrepresentable_replications():
    from pyhuge import core

    with pytest.raises(PyHugeError, match="rep_num.*count range"):
        core._stars_count_dtype(2**32)


@pytest.mark.parametrize("rep_num", (0, -1))
def test_stars_count_dtype_requires_positive_replications(rep_num):
    from pyhuge import core

    with pytest.raises(PyHugeError, match="rep_num.*positive integer"):
        core._stars_count_dtype(rep_num)


def test_stars_rejects_replications_above_count_range_before_subsampling():
    x = np.random.default_rng(17).normal(size=(12, 4))
    fit = huge(x, method="ct", lambda_=[0.25], verbose=False)

    with pytest.raises(PyHugeError, match="rep_num.*count range"):
        huge_select(
            fit,
            criterion="stars",
            rep_num=2**32,
            verbose=False,
        )


def test_stars_requires_at_least_two_variables():
    from scipy import sparse
    from pyhuge import core

    x = np.arange(10.0).reshape(-1, 1)
    fit = core.HugeResult(
        method="ct",
        lambda_path=np.asarray([0.5]),
        sparsity=np.asarray([0.0]),
        path=[sparse.csc_matrix((1, 1), dtype=float)],
        cov_input=False,
        data=x,
        raw={"backend": "native"},
    )

    with pytest.raises(PyHugeError, match="StARS.*at least two variables"):
        core.huge_select(fit, criterion="stars", rep_num=2, verbose=False)


@pytest.mark.parametrize(
    ("n", "ratio"),
    (
        pytest.param(2, None, id="default-ratio-n2"),
        pytest.param(3, 0.5, id="explicit-ratio-n3"),
        pytest.param(
            4,
            np.nextafter(0.5, 0.0),
            id="just-below-two-observations",
        ),
    ),
)
def test_stars_rejects_subsamples_smaller_than_two_before_fitting(
    monkeypatch, n, ratio
):
    from scipy import sparse
    from pyhuge import core

    fit = core.HugeResult(
        method="ct",
        lambda_path=np.asarray([0.5]),
        sparsity=np.asarray([0.0]),
        path=[sparse.csc_matrix((3, 3), dtype=float)],
        cov_input=False,
        data=np.arange(n * 3.0).reshape(n, 3),
        raw={"backend": "native"},
    )

    def reject_subfit(*args, **kwargs):
        raise AssertionError("invalid StARS subsample reached a model fit")

    monkeypatch.setattr(core, "huge", reject_subfit)
    with pytest.raises(
        PyHugeError,
        match="stars_subsample_ratio.*at least two observations",
    ):
        core.huge_select(
            fit,
            criterion="stars",
            stars_subsample_ratio=ratio,
            rep_num=2,
            verbose=False,
        )


@pytest.mark.parametrize(
    ("n", "ratio"),
    (
        pytest.param(3, None, id="default-ratio"),
        pytest.param(4, 0.5, id="exact-boundary"),
    ),
)
def test_stars_accepts_subsamples_of_exactly_two_observations(
    monkeypatch, n, ratio
):
    from types import SimpleNamespace
    from scipy import sparse
    from pyhuge import core

    empty = sparse.csc_matrix((3, 3), dtype=float)
    fit = core.HugeResult(
        method="ct",
        lambda_path=np.asarray([0.5]),
        sparsity=np.asarray([0.0]),
        path=[empty],
        cov_input=False,
        data=np.arange(n * 3.0).reshape(n, 3),
        raw={"backend": "native"},
    )
    subfit_sizes = []

    def fixed_subfit(x, **kwargs):
        subfit_sizes.append(x.shape[0])
        return SimpleNamespace(path=[empty])

    monkeypatch.setattr(core, "huge", fixed_subfit)
    selected = core.huge_select(
        fit,
        criterion="stars",
        stars_subsample_ratio=ratio,
        rep_num=2,
        verbose=False,
    )

    assert subfit_sizes == [2, 2]
    np.testing.assert_array_equal(selected.variability, np.zeros(1))


@pytest.mark.parametrize("dimension", (1, 2, 5, 17))
def test_stars_upper_triangle_indices_are_contiguous(dimension):
    from pyhuge import core

    rows, cols = np.triu_indices(dimension, k=1)
    packed = core._stars_upper_triangle_indices(rows, cols, dimension)
    expected = np.arange(dimension * (dimension - 1) // 2, dtype=np.int64)
    assert np.array_equal(packed, expected)


def test_stars_upper_triangle_indices_avoid_int32_overflow():
    from pyhuge import core

    dimension = 100_000
    rows = np.asarray([0, 1, dimension - 2], dtype=np.int32)
    cols = np.asarray([1, dimension - 1, dimension - 1], dtype=np.int32)
    packed = core._stars_upper_triangle_indices(rows, cols, dimension)
    expected = np.asarray(
        [0, 2 * dimension - 4, dimension * (dimension - 1) // 2 - 1],
        dtype=np.int64,
    )
    assert np.array_equal(packed, expected)


def test_stars_glasso_keeps_one_sided_frequency_semantics(monkeypatch):
    from types import SimpleNamespace
    from scipy import sparse
    from pyhuge import core

    empty = sparse.csc_matrix((3, 3), dtype=float)
    upper = sparse.csc_matrix(([1.0], ([0], [1])), shape=(3, 3))
    lower = sparse.csc_matrix(([1.0], ([1], [0])), shape=(3, 3))
    estimate = core.HugeResult(
        method="glasso",
        lambda_path=np.asarray([0.5]),
        sparsity=np.asarray([0.0]),
        path=[empty],
        cov_input=False,
        data=np.zeros((10, 3), dtype=float),
        raw={"backend": "native", "scr": False},
    )
    subpaths = iter((upper, lower, lower, empty, empty))
    calls = []

    def fixed_subfit(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(path=[next(subpaths)])

    monkeypatch.setattr(core, "huge", fixed_subfit)

    selected = core.huge_select(
        estimate,
        criterion="stars",
        rep_num=5,
        n_jobs=1,
        verbose=False,
    )
    probability = np.zeros((3, 3), dtype=float)
    probability[0, 1] = 1.0
    probability[1, 0] = 2.0
    probability /= 5.0
    p_mat = 0.5 * (probability + probability.T)
    expected = np.asarray(
        [4.0 * np.sum(p_mat * (1.0 - p_mat)) / (3 * 2)]
    )
    assert len(calls) == 5
    assert np.array_equal(selected.variability, expected)


@pytest.mark.parametrize("method", ("ct", "mb"))
def test_stars_streaming_matches_dense_reference(method):
    x = np.random.default_rng(12).normal(size=(50, 8))
    fit = huge(x, method=method, nlambda=3, verbose=False)
    rep_num = 3
    selected = huge_select(
        fit,
        criterion="stars",
        rep_num=rep_num,
        n_jobs=1,
        verbose=False,
    )

    # Reproduce the pre-streaming dense accumulation on the same pre-drawn
    # subsamples. This guards the variability formula and selected index while
    # the production path remains sparse.
    m = max(2, int(x.shape[0] * 0.8))
    rng = np.random.default_rng(0)
    freq = np.zeros((len(fit.path), x.shape[1], x.shape[1]), dtype=float)
    for _ in range(rep_num):
        idx = rng.choice(x.shape[0], size=m, replace=False)
        subfit = huge(
            x[idx],
            lambda_=fit.lambda_path,
            method=method,
            sym=fit.sym,
            verbose=False,
        )
        for li, path in enumerate(subfit.path):
            freq[li] += (path.toarray() != 0).astype(float)

    freq /= float(rep_num)
    expected = np.zeros(len(fit.path), dtype=float)
    for li in range(len(fit.path)):
        p_mat = 0.5 * (freq[li] + freq[li].T)
        np.fill_diagonal(p_mat, 0.0)
        expected[li] = 4.0 * np.sum(p_mat * (1.0 - p_mat)) / (
            x.shape[1] * (x.shape[1] - 1)
        )

    assert np.any(expected > 0)
    assert np.array_equal(selected.variability, expected)
    crossings = np.flatnonzero(expected >= 0.1)
    expected_index = (
        len(expected) - 1
        if crossings.size == 0
        else max(crossings[0] - 1, 0)
    )
    assert selected.opt_index == expected_index + 1


@pytest.mark.parametrize("n_jobs", (1, 2))
def test_stars_accumulation_does_not_densify_sparse_paths(monkeypatch, n_jobs):
    from scipy import sparse

    x = np.random.default_rng(11).normal(size=(60, 10))
    fit = huge(x, method="mb", nlambda=3, verbose=False)

    def reject_dense_conversion(*args, **kwargs):
        raise AssertionError("StARS accumulation must not densify sparse paths")

    monkeypatch.setattr(sparse.csc_matrix, "toarray", reject_dense_conversion)
    monkeypatch.setattr(sparse.csc_matrix, "todense", reject_dense_conversion)

    def select():
        return huge_select(
            fit,
            criterion="stars",
            rep_num=3,
            n_jobs=n_jobs,
            verbose=False,
        )

    if n_jobs > 1:
        with pytest.warns(RuntimeWarning, match="OpenMP or BLAS"):
            selected = select()
    else:
        selected = select()

    assert selected.variability.shape == (3,)
    assert np.all(np.isfinite(selected.variability))


def test_mb_path_column_indices_sorted():
    # core collect_sorted: csc structures must be canonically ordered
    rng = np.random.default_rng(5)
    x = rng.normal(size=(100, 30))
    for m in ("mb", "tiger"):
        fit = huge(x, method=m, nlambda=5, verbose=False)
        for p in fit.path:
            assert p.has_sorted_indices


def test_solvers_are_silent_normally_and_glasso_rejects_uncertified_limit_case():
    # 2.0.0: convergence status channel (hit_max_iter) from the shared core
    import warnings

    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 30))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for m in ("mb", "glasso", "tiger"):
            huge(x, method=m, nlambda=5, verbose=False)

    # This pathological iteration-limit result is not a usable inverse pair.
    rng = np.random.default_rng(50)
    base = rng.normal(size=(40, 5))
    idx = rng.integers(0, 5, size=60)
    xp = base[:, idx] + rng.normal(scale=0.05, size=(40, 60))
    xp = (xp - xp.mean(0)) / xp.std(0, ddof=1)
    with pytest.raises(
        PyHugeError, match="inconsistent precision and covariance"
    ):
        huge(xp, method="glasso", lambda_=[0.001], verbose=False)
