"""Native pyhuge core implementation (no rpy2 dependency)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from importlib import resources

import numpy as np
from scipy import sparse, stats
from scipy.spatial.distance import squareform


LambdaInput = Union[float, np.number, Sequence[float], np.ndarray]


class PyHugeError(RuntimeError):
    """Raised when pyhuge encounters invalid inputs or backend failures."""


@dataclass
class HugeResult:
    """Result from native ``huge()``."""

    method: str
    lambda_path: np.ndarray
    sparsity: np.ndarray
    path: list[sparse.csc_matrix]
    cov_input: bool
    data: np.ndarray
    sym: str = "or"
    df: Optional[np.ndarray] = None
    loglik: Optional[np.ndarray] = None
    icov: Optional[list[np.ndarray]] = None
    cov: Optional[list[np.ndarray]] = None
    idx_mat: Optional[np.ndarray] = None
    raw: Any = None


@dataclass
class HugeSelectResult:
    """Result from native ``huge_select()``."""

    criterion: str
    opt_lambda: float
    opt_sparsity: float
    refit: sparse.csc_matrix
    opt_index: Optional[int] = None
    variability: Optional[np.ndarray] = None
    ebic_score: Optional[np.ndarray] = None
    opt_icov: Optional[np.ndarray] = None
    opt_cov: Optional[np.ndarray] = None
    raw: Any = None


@dataclass
class HugeGeneratorResult:
    """Result from native ``huge_generator()``."""

    data: np.ndarray
    sigma: np.ndarray
    omega: np.ndarray
    sigmahat: np.ndarray
    theta: sparse.csc_matrix
    sparsity: float
    graph_type: str
    raw: Any = None


@dataclass
class HugeInferenceResult:
    """Result from native ``huge_inference()``."""

    data: np.ndarray
    p: np.ndarray
    error: float
    raw: Any = None


@dataclass
class HugeRocResult:
    """Result from native ``huge_roc()``."""

    f1: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    auc: float
    raw: Any = None


@dataclass
class HugeStockDataResult:
    """Built-in stock dataset result."""

    data: np.ndarray
    info: np.ndarray
    raw: Any = None


@dataclass
class HugeSummary:
    """Compact summary for HugeResult."""

    method: str
    n_samples: int
    n_features: int
    path_length: int
    sparsity_min: float
    sparsity_max: float
    cov_input: bool
    has_icov: bool
    has_cov: bool


@dataclass
class HugeSelectSummary:
    """Compact summary for HugeSelectResult."""

    criterion: str
    opt_lambda: float
    opt_sparsity: float
    refit_n_features: int
    has_opt_icov: bool
    has_opt_cov: bool


_ALLOWED_METHODS = {"mb", "glasso", "ct", "tiger"}
_ALLOWED_SYM = {"and", "or"}
_ALLOWED_CRITERIA = {"ric", "stars", "ebic"}
_ALLOWED_GRAPH_TYPES = {"random", "hub", "cluster", "band", "scale-free"}
_ALLOWED_NPN_FUNCS = {"shrinkage", "truncation", "skeptic"}
_ALLOWED_INFERENCE_TYPES = {"Gaussian", "Nonparanormal"}
_ALLOWED_INFERENCE_METHODS = {"score", "wald"}


try:  # optional acceleration
    from . import _native_core as _CPP
except Exception:  # pragma: no cover - extension optional
    _CPP = None


def _ensure_backend_native(backend: str) -> None:
    if backend != "native":
        raise PyHugeError("pyhuge supports only `backend=\"native\"`.")


def _ensure_2d_array(name: str, value: Any, finite: bool = True) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise PyHugeError(f"`{name}` must be a 2D array.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise PyHugeError(f"`{name}` must be non-empty.")
    if finite and not np.isfinite(arr).all():
        raise PyHugeError(f"`{name}` contains non-finite values.")
    return arr


def _to_dense_matrix(value: Any, name: str) -> np.ndarray:
    try:
        raw = value.toarray() if sparse.issparse(value) else value
        arr = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PyHugeError(f"`{name}` must be a numeric 2D matrix.") from exc
    if arr.ndim != 2:
        raise PyHugeError(f"`{name}` must be a 2D array-like matrix.")
    if not np.isfinite(arr).all():
        raise PyHugeError(f"`{name}` contains non-finite values.")
    return arr


def _ensure_finite_numeric_scalar(name: str, value: Any) -> float:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PyHugeError(f"`{name}` must be a finite numeric scalar.") from exc
    if arr.ndim != 0 or arr.dtype.kind not in {"i", "u", "f"}:
        raise PyHugeError(f"`{name}` must be a finite numeric scalar.")
    fvalue = float(arr)
    if not np.isfinite(fvalue):
        raise PyHugeError(f"`{name}` must be a finite numeric scalar.")
    return fvalue


def _ensure_positive_int(name: str, value: Any) -> int:
    try:
        fvalue = _ensure_finite_numeric_scalar(name, value)
    except PyHugeError as exc:
        raise PyHugeError(f"`{name}` must be a finite positive integer.") from exc
    if fvalue < 1 or fvalue != math.floor(fvalue):
        raise PyHugeError(f"`{name}` must be a finite positive integer.")
    return int(fvalue)


def _stars_count_dtype(rep_num: int) -> np.dtype:
    if rep_num < 1:
        raise PyHugeError("`rep_num` must be a positive integer.")
    for dtype in (np.uint8, np.uint16, np.uint32):
        if rep_num <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise PyHugeError("`rep_num` exceeds the supported StARS count range.")


def _stars_upper_triangle_indices(
    rows: np.ndarray, cols: np.ndarray, dimension: int
) -> np.ndarray:
    """Row-major condensed indices for coordinates strictly above diagonal."""
    rows64 = np.asarray(rows, dtype=np.int64)
    cols64 = np.asarray(cols, dtype=np.int64)
    return (
        rows64 * (2 * int(dimension) - rows64 - 1) // 2
        + cols64
        - rows64
        - 1
    )


def _ensure_ratio(name: str, value: float, low_open: float = 0.0, high_closed: float = 1.0) -> float:
    fval = _ensure_finite_numeric_scalar(name, value)
    if not (fval > low_open and fval <= high_closed):
        raise PyHugeError(f"`{name}` must satisfy {low_open} < {name} <= {high_closed}.")
    return fval


def _ensure_lambda_sequence(
    lambda_: LambdaInput, *, allow_ties: bool = False,
    allow_zero: bool = False, enforce_order: bool = True
) -> np.ndarray:
    try:
        raw = np.asarray(lambda_)
    except (TypeError, ValueError) as exc:
        raise PyHugeError("`lambda_` must be a numeric sequence.") from exc
    if raw.dtype.kind not in {"i", "u", "f"}:
        raise PyHugeError("`lambda_` must be a numeric sequence.")
    if raw.ndim > 1:
        raise PyHugeError(
            "`lambda_` must be a numeric scalar or one-dimensional sequence."
        )
    lam = np.asarray(raw, dtype=float).reshape(-1)
    if lam.size == 0:
        raise PyHugeError("`lambda_` must contain at least one value.")
    if not np.isfinite(lam).all():
        raise PyHugeError("`lambda_` contains non-finite values.")
    if allow_zero and np.any(lam < 0):
        raise PyHugeError("`lambda_` must contain non-negative values for method `ct`.")
    if not allow_zero and np.any(lam <= 0):
        raise PyHugeError("`lambda_` must contain positive values.")
    if enforce_order and lam.size > 1:
        if allow_ties and np.any(np.diff(lam) > 0):
            raise PyHugeError(
                "`lambda_` must be non-increasing (ties are allowed)."
            )
        elif not allow_ties and np.any(np.diff(lam) >= 0):
            raise PyHugeError("`lambda_` must be strictly decreasing.")
    return lam


def _ensure_ct_lambda_sequence(lambda_: LambdaInput) -> np.ndarray:
    return _ensure_lambda_sequence(
        lambda_, allow_ties=True, allow_zero=True, enforce_order=False
    )


def _is_covariance_input(x: np.ndarray) -> bool:
    if x.shape[0] != x.shape[1]:
        return False
    values = np.asarray(x, dtype=float)
    if not np.isfinite(values).all():
        return False

    # Compare off-diagonal entries in implied-correlation units.  This keeps
    # covariance routing invariant to a finite uniform rescaling without
    # subtracting large values before they have been normalized.
    tolerance = 100.0 * np.finfo(float).eps
    dimension = int(x.shape[0])
    if dimension <= 1:
        return True

    diagonal = np.abs(np.diag(values))
    diagonal_root = np.sqrt(diagonal)
    for column in range(1, dimension):
        left = values[:column, column]
        right = values[column, :column]
        different = left != right
        if not np.any(different):
            continue

        rows = np.flatnonzero(different)
        left = left[different]
        right = right[different]
        with np.errstate(under="ignore"):
            covariance_scale = diagonal_root[rows] * diagonal_root[column]
        equal_diagonal = diagonal[rows] == diagonal[column]
        covariance_scale[equal_diagonal] = diagonal[column]
        reference = np.maximum(
            np.maximum(np.abs(left), np.abs(right)), covariance_scale
        )
        with np.errstate(under="ignore"):
            normalized_left = left / reference
            normalized_right = right / reference
            difference = np.abs(normalized_left - normalized_right)
        scale = np.maximum(np.abs(normalized_left), np.abs(normalized_right))
        threshold = np.full_like(scale, tolerance)
        relative = scale > tolerance
        threshold[relative] = tolerance * scale[relative]
        if np.any(difference > threshold):
            return False
    return True


def _resolve_input_type(x: np.ndarray, input_type: str) -> bool:
    allowed = {"auto", "data", "covariance"}
    if not isinstance(input_type, str) or input_type not in allowed:
        raise PyHugeError(
            "`input_type` must be exactly one of "
            "'auto', 'data', or 'covariance'."
        )
    if input_type == "auto":
        return _is_covariance_input(x)
    if input_type == "data":
        return False
    if x.shape[0] != x.shape[1]:
        raise PyHugeError(
            "`input_type='covariance'` requires a square matrix."
        )
    if not _is_covariance_input(x):
        raise PyHugeError(
            "Covariance input must be symmetric within numeric tolerance."
        )
    return True


def _standardize(x: np.ndarray) -> np.ndarray:
    column_scale = np.max(np.abs(x), axis=0)
    if np.any(~np.isfinite(column_scale)) or np.any(column_scale <= 0.0):
        raise PyHugeError(
            "Data must have finite, positive sample standard deviations."
        )

    # Binary-power scaling is exact, bounds every column, and retains
    # representable ULP differences near the floating-point maximum.
    _, exponents = np.frexp(column_scale)
    with np.errstate(under="ignore"):
        centered = np.ldexp(x, -exponents)
    reference = centered[0:1, :].copy()
    centered -= reference
    centered -= np.mean(centered, axis=0)

    centered_scale = np.max(np.abs(centered), axis=0)
    if np.any(~np.isfinite(centered_scale)) or np.any(centered_scale <= 0.0):
        raise PyHugeError(
            "Data must have finite, positive sample standard deviations."
        )
    centered /= centered_scale
    sd = np.std(centered, axis=0, ddof=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 0.0):
        raise PyHugeError(
            "Data must have finite, positive sample standard deviations."
        )
    centered /= sd
    return centered


def _cov_to_corr(
    cov: np.ndarray, *, require_psd: bool = True
) -> np.ndarray:
    diagonal = np.diag(cov)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise PyHugeError(
            "Covariance input must have positive finite diagonal entries."
        )
    inv_sd = 1.0 / np.sqrt(diagonal)
    dimension = int(cov.shape[0])
    corr = np.eye(dimension, dtype=float)
    for column in range(1, dimension):
        for row in range(column):
            inv_large = max(inv_sd[row], inv_sd[column])
            inv_small = min(inv_sd[row], inv_sd[column])
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                value = (cov[row, column] * inv_large) * inv_small
            if not np.isfinite(value):
                raise PyHugeError(
                    "Covariance input cannot be converted to a finite "
                    "correlation matrix."
                )
            if abs(value) > 1.0 + 1e-8:
                raise PyHugeError(
                    "Covariance input is not a valid covariance matrix."
                )
            value = max(-1.0, min(1.0, float(value)))
            corr[row, column] = value
            corr[column, row] = value

    if require_psd:
        spectral_bound = max(1.0, float(np.linalg.norm(corr, ord=np.inf)))
        tolerance = (
            100.0
            * np.finfo(float).eps
            * max(1, dimension)
            * spectral_bound
        )
        shifted = corr.copy()
        shifted.flat[:: dimension + 1] += tolerance
        try:
            np.linalg.cholesky(shifted)
        except np.linalg.LinAlgError as exc:
            raise PyHugeError(
                "Covariance input must be positive semidefinite."
            ) from exc
    return corr


def _offdiag_abs_max(mat: np.ndarray) -> float:
    d = mat.shape[0]
    if d <= 1:
        return 1e-3
    m = np.max(np.abs(mat[~np.eye(d, dtype=bool)]))
    return 1e-3 if m == 0.0 else float(m)


def _default_nlambda(method: str) -> int:
    return 20 if method == "ct" else 10


def _default_lambda_min_ratio(method: str) -> float:
    return 0.05 if method == "ct" else 0.1


def _build_lambda_path(
    *,
    base_matrix: np.ndarray,
    method: str,
    lambda_: Optional[LambdaInput],
    nlambda: Optional[int],
    lambda_min_ratio: Optional[float],
    legacy_glasso_covariance: bool = False,
) -> np.ndarray:
    if lambda_ is not None:
        return _ensure_lambda_sequence(lambda_, allow_ties=True)

    nlam = _default_nlambda(method) if nlambda is None else _ensure_positive_int("nlambda", nlambda)
    ratio = (
        _default_lambda_min_ratio(method)
        if lambda_min_ratio is None
        else _ensure_ratio("lambda_min_ratio", lambda_min_ratio)
    )
    if legacy_glasso_covariance:
        legacy_matrix = base_matrix - np.eye(base_matrix.shape[0])
        legacy_max = float(np.max(np.abs(legacy_matrix)))
        lam_max = 1e-3 if legacy_max == 0.0 else legacy_max
    else:
        lam_max = _offdiag_abs_max(base_matrix)
    if ratio == 1.0:
        # Equal geomspace endpoints can wobble by one ulp and appear to
        # increase. Preserve the documented path length with exact ties.
        lam = np.full(nlam, lam_max, dtype=float)
    elif method == "ct":
        lam = np.linspace(lam_max, lam_max * ratio, nlam)
    else:
        lam_min = lam_max * ratio
        if lam_min > 0.0:
            lam = np.geomspace(lam_max, lam_min, nlam)
        else:
            # Keep representable interior points when the requested endpoint
            # underflows, then saturate only the unrepresentable tail.
            fractions = np.linspace(0.0, 1.0, nlam)
            with np.errstate(under="ignore"):
                lam = np.exp(
                    np.log(lam_max) + fractions * np.log(ratio)
                )
            lam[~np.isfinite(lam) | (lam <= 0.0)] = np.nextafter(0.0, 1.0)
        lam[0] = lam_max
        lam = np.minimum.accumulate(lam)
    return _ensure_lambda_sequence(lam, allow_ties=True)


def _spd_inverse(a: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive-definite matrix via Cholesky.

    Mirrors R's ``chol2inv(chol(a))``: faster than a general LU inverse and
    symmetric by construction.
    """
    from scipy.linalg import cho_factor, cho_solve

    return cho_solve(cho_factor(a, lower=True), np.eye(a.shape[0]))


def _adj_sparsity(adj: np.ndarray) -> float:
    d = adj.shape[0]
    if d <= 1:
        return 0.0
    edges = float(np.count_nonzero(np.triu(adj, 1)))
    return (2.0 * edges) / (d * (d - 1))


def _path_sparsity(path: list[sparse.csc_matrix]) -> np.ndarray:
    if len(path) == 0:
        return np.asarray([], dtype=float)
    d = int(path[0].shape[0])
    denom = float(d * (d - 1))
    if denom <= 0:
        return np.zeros(len(path), dtype=float)
    return 2.0 * _edge_count(path) / denom


def _symmetrize(directed: np.ndarray, sym: str) -> np.ndarray:
    if sym == "or":
        adj = np.logical_or(directed, directed.T)
    else:
        adj = np.logical_and(directed, directed.T)
    np.fill_diagonal(adj, False)
    return adj


def _run_ct_native(
    corr: np.ndarray, lambda_path: np.ndarray
) -> list[sparse.csc_matrix]:
    """Build CT matrices one lambda at a time to bound dense native memory."""
    out: list[sparse.csc_matrix] = []
    for index in range(lambda_path.size):
        dense = _CPP.threshold_path(
            corr, lambda_path[index : index + 1]
        )[0]
        out.append(sparse.csc_matrix(dense, dtype=float))
    return out


def _run_ct(corr: np.ndarray, lambda_path: np.ndarray) -> list[sparse.csc_matrix]:
    if _CPP is not None:
        try:
            return _run_ct_native(
                np.asarray(corr, dtype=float),
                np.asarray(lambda_path, dtype=float),
            )
        except Exception:
            pass

    out: list[sparse.csc_matrix] = []
    abs_corr = np.abs(corr)
    for lam in lambda_path:
        adj = abs_corr > float(lam)
        np.fill_diagonal(adj, False)
        out.append(sparse.csc_matrix(adj.astype(float)))
    return out


def _run_ct_default_rank(
    corr: np.ndarray,
    nlambda: int,
    lambda_min_ratio: float,
) -> tuple[np.ndarray, list[sparse.csc_matrix], np.ndarray]:
    # Build a conservative density schedule over undirected edges.  Each
    # returned lambda is then passed through the same strict-threshold helper
    # as an explicit path, so ties are never split and refits are identical.
    d = int(corr.shape[0])
    if d <= 1:
        lam = np.zeros(nlambda, dtype=float)
        path = [sparse.csc_matrix((d, d), dtype=float) for _ in range(nlambda)]
        sparsity = np.zeros(nlambda, dtype=float)
        return lam, path, sparsity

    s = np.abs(np.asarray(corr, dtype=float))
    np.fill_diagonal(s, 0.0)
    upper = np.triu_indices(d, 1)
    edge_weights = np.sort(s[upper])[::-1]
    edge_total = int(edge_weights.size)
    target_edges = np.ceil(
        np.linspace(
            1.0,
            float(lambda_min_ratio) * edge_total,
            num=nlambda,
        )
    ).astype(np.int64)
    target_edges = np.clip(target_edges, 0, edge_total)

    lambda_path = np.empty(nlambda, dtype=float)
    for index, target in enumerate(target_edges):
        if int(target) < edge_total:
            next_edge = max(int(target), 0)
            lambda_path[index] = float(edge_weights[next_edge])
        else:
            lambda_path[index] = 0.0

    # These ranking buffers can total roughly 20*d^2 bytes.  They are no
    # longer needed once the thresholds are fixed, so release them before
    # native CT allocates and converts its path matrices.
    del s, upper, edge_weights, target_edges
    path = _run_ct(corr, lambda_path)
    sparsity = _ct_path_sparsity(path)
    return lambda_path, path, sparsity


def _ct_path_sparsity(path: list[sparse.csc_matrix]) -> np.ndarray:
    if len(path) == 0:
        return np.asarray([], dtype=float)
    d = int(path[0].shape[0])
    denom = float(d * (d - 1))
    if denom <= 0:
        return np.zeros(len(path), dtype=float)
    return np.asarray([float(p.nnz) / denom for p in path], dtype=float)


def _require_native_core(component: str) -> None:
    if _CPP is None:
        raise PyHugeError(
            f"`{component}` requires native C++ core (`pyhuge._native_core`). "
            "Reinstall pyhuge with extension build enabled."
        )


def _warn_if_not_converged(out: dict, solver: str) -> None:
    if bool(out.get("hit_max_iter", False)):
        warnings.warn(
            f"{solver} solver reached its iteration limit; "
            "estimates may not be fully converged",
            RuntimeWarning,
            stacklevel=3,
        )


def _run_glasso(
    s_mat: np.ndarray,
    lambda_path: np.ndarray,
    scr: bool,
    cov_output: bool,
) -> tuple[list[sparse.csc_matrix], list[np.ndarray], Optional[list[np.ndarray]], np.ndarray, np.ndarray]:
    _require_native_core("glasso")
    try:
        out = _CPP.hugeglasso(
            np.asarray(s_mat, dtype=float),
            np.asarray(lambda_path, dtype=float),
            bool(scr),
            bool(cov_output),
        )
    except Exception as exc:
        raise PyHugeError(f"native glasso backend failed: {exc}") from exc

    _warn_if_not_converged(out, "glasso")

    path_cube = np.asarray(out["path"], dtype=np.uint8)
    icov_cube = np.asarray(out["icov"], dtype=float)
    loglik = np.asarray(out["loglik"], dtype=float).reshape(-1)
    df = np.asarray(out["df"], dtype=float).reshape(-1)

    path: list[sparse.csc_matrix] = []
    icov: list[np.ndarray] = []
    cov_list: list[np.ndarray] = []
    for i in range(path_cube.shape[0]):
        adj = path_cube[i] != 0
        np.fill_diagonal(adj, False)
        path.append(sparse.csc_matrix(adj.astype(float)))
        icov.append(np.asarray(icov_cube[i], dtype=float))

    cov_raw = out.get("cov", None)
    if cov_output and cov_raw is not None:
        cov_cube = np.asarray(cov_raw, dtype=float)
        for i in range(cov_cube.shape[0]):
            cov_list.append(np.asarray(cov_cube[i], dtype=float))
        cov_ret: Optional[list[np.ndarray]] = cov_list
    else:
        cov_ret = None

    return path, icov, cov_ret, df, loglik


def _build_screen_idx(corr: np.ndarray, scr_num: int) -> np.ndarray:
    d = corr.shape[0]
    if scr_num <= 0 or scr_num >= d:
        raise PyHugeError("`scr_num` must satisfy 1 <= scr_num < d.")
    scores = np.abs(corr).copy()
    np.fill_diagonal(scores, -np.inf)
    order = np.argsort(-scores, axis=0, kind="stable")
    return np.asarray(order[:scr_num, :], dtype=np.int32)


def _run_mb(
    corr: np.ndarray,
    lambda_path: np.ndarray,
    sym: str,
    scr: bool,
    scr_num: Optional[int],
) -> tuple[list[sparse.csc_matrix], np.ndarray]:
    _require_native_core("mb")
    try:
        if scr:
            if scr_num is None:
                raise PyHugeError("`scr=True` requires `scr_num` in native MB C++ core.")
            idx_mat = _build_screen_idx(corr, scr_num)
            out = _CPP.spmb_scr(
                np.asarray(corr, dtype=float),
                np.asarray(lambda_path, dtype=float),
                idx_mat,
                False,
            )
        else:
            out = _CPP.spmb_graph(
                np.asarray(corr, dtype=float),
                np.asarray(lambda_path, dtype=float),
                False,
            )
    except Exception as exc:
        raise PyHugeError(f"native mb backend failed: {exc}") from exc

    _warn_if_not_converged(out, "mb")

    df = np.asarray(out["df"], dtype=float)
    return _column_support_to_path(
        out, int(lambda_path.size), int(corr.shape[0]), sym
    ), df


def _beta_cube_to_path(beta: np.ndarray, sym: str) -> list[sparse.csc_matrix]:
    """Symmetrized adjacency path from a (nlambda, d, d) coefficient cube."""
    path: list[sparse.csc_matrix] = []
    for li in range(beta.shape[0]):
        directed = np.abs(beta[li]) > 0
        np.fill_diagonal(directed, False)
        adj = _symmetrize(directed, sym)
        path.append(sparse.csc_matrix(adj.astype(float)))
    return path


def _column_support_to_path(
    out: dict, nlambda: int, d: int, sym: str
) -> list[sparse.csc_matrix]:
    indptr = np.asarray(out["support_indptr"], dtype=np.int64)
    indices = np.asarray(out["support_indices"], dtype=np.int32)
    if indptr.shape != (nlambda, d + 1):
        raise PyHugeError("Native sparse support has an invalid indptr shape.")
    if indices.ndim != 1:
        raise PyHugeError("Native sparse support indices must be one-dimensional.")

    path: list[sparse.csc_matrix] = []
    offset = 0
    for lambda_index in range(nlambda):
        local_indptr = indptr[lambda_index]
        if (
            local_indptr[0] != 0
            or np.any(local_indptr < 0)
            or np.any(np.diff(local_indptr) < 0)
        ):
            raise PyHugeError("Native sparse support has an invalid indptr.")
        count = int(local_indptr[-1])
        end = offset + count
        if end > indices.size:
            raise PyHugeError("Native sparse support indices are truncated.")
        local_indices = indices[offset:end]
        if np.any(local_indices < 0) or np.any(local_indices >= d):
            raise PyHugeError("Native sparse support contains an invalid index.")

        directed = sparse.csc_matrix(
            (
                np.ones(count, dtype=float),
                local_indices,
                local_indptr,
            ),
            shape=(d, d),
        )
        directed.setdiag(0)
        directed.eliminate_zeros()
        if sym == "or":
            adjacency = directed.maximum(directed.T)
        else:
            adjacency = directed.multiply(directed.T)
        adjacency = sparse.csc_matrix(adjacency, dtype=float)
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()
        if adjacency.nnz:
            adjacency.data.fill(1.0)
        adjacency.sum_duplicates()
        adjacency.sort_indices()
        path.append(adjacency)
        offset = end

    if offset != indices.size:
        raise PyHugeError("Native sparse support contains trailing indices.")
    return path


def _run_tiger(
    x_data: np.ndarray,
    lambda_path: Optional[np.ndarray],
    nlambda: int,
    lambda_min_ratio: float,
    covariance_input: bool,
    sym: str,
) -> tuple[list[sparse.csc_matrix], np.ndarray, list[np.ndarray], np.ndarray]:
    _require_native_core("tiger")
    try:
        native_lambda = None if lambda_path is None else np.asarray(lambda_path, dtype=float)
        out = _CPP.spmb_graphsqrt(
            np.asarray(x_data, dtype=float),
            native_lambda,
            int(nlambda),
            float(lambda_min_ratio),
            bool(covariance_input),
            False,
        )
    except Exception as exc:
        raise PyHugeError(f"native tiger backend failed: {exc}") from exc

    actual_lambda = np.asarray(out["lambda"], dtype=float)
    if bool(out.get("path_truncated", False)):
        warnings.warn(
            "tiger returned the "
            f"{actual_lambda.size}-value certified prefix of the "
            f"{nlambda}-value native lambda path; smaller lambda values did "
            "not converge or were numerically degenerate",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        _warn_if_not_converged(out, "tiger")

    df = np.asarray(out["df"], dtype=float)
    icov_cube = np.asarray(out["icov"], dtype=float)

    path = _column_support_to_path(
        out, int(actual_lambda.size), int(x_data.shape[1]), sym
    )
    icov = [np.asarray(icov_cube[i], dtype=float) for i in range(icov_cube.shape[0])]
    return path, df, icov, actual_lambda


def _edge_count(path: Sequence[sparse.csc_matrix]) -> np.ndarray:
    return np.asarray(
        [sparse.triu(p, k=1, format="csc").count_nonzero() for p in path],
        dtype=float,
    )


def _selected_index_ric(sparsity: np.ndarray) -> int:
    if sparsity.size <= 2:
        return int(sparsity.size - 1)
    curv = np.zeros_like(sparsity)
    curv[1:-1] = np.abs(np.diff(sparsity, n=2))
    return int(np.argmax(curv))


def _ebic_from_fit(est: HugeResult, ebic_gamma: float) -> np.ndarray:
    if est.method != "glasso":
        raise PyHugeError("`criterion='ebic'` requires a glasso fit.")

    d = est.path[0].shape[0]
    n = est.data.shape[0] if not est.cov_input else d
    edge_k = _edge_count(est.path)

    loglik = est.loglik
    if loglik is None:
        if est.icov is None:
            raise PyHugeError("`criterion='ebic'` requires glasso fit with log-likelihood information.")
        cov_mat = np.asarray(np.cov(est.data, rowvar=False), dtype=float) if not est.cov_input else est.data
        vals = np.zeros(len(est.icov), dtype=float)
        for i, prec in enumerate(est.icov):
            sign, logdet = np.linalg.slogdet(np.asarray(prec, dtype=float))
            vals[i] = -np.inf if sign <= 0 else float(logdet - np.trace(cov_mat @ prec))
        loglik = vals

    l = np.asarray(loglik, dtype=float) * (n / 2.0)
    return -2.0 * l + edge_k * np.log(max(n, 2)) + 4.0 * ebic_gamma * edge_k * np.log(max(d, 2))


def huge(
    x: np.ndarray,
    lambda_: Optional[LambdaInput] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    method: str = "mb",
    scr: Optional[bool] = None,
    scr_num: Optional[int] = None,
    cov_output: bool = False,
    sym: str = "or",
    verbose: bool = True,
    backend: str = "native",
    *,
    input_type: str = "auto",
) -> HugeResult:
    """Native graph path estimation.

    ``verbose`` and ``backend`` are accepted for R-API compatibility.
    Only the ``native`` backend is currently supported; ``verbose`` output
    is not yet implemented. ``input_type`` may be ``"auto"``, ``"data"``,
    or ``"covariance"``; use ``"data"`` when observations form a square
    symmetric matrix. For glasso covariance input with no explicit lambda,
    ``"auto"`` matches R's historical diagonal-sensitive lambda scale;
    ``"covariance"`` uses only off-diagonal entries. The automatic ``ct``
    path uses undirected edge weights and a literal strict threshold; tied
    weights are never split, and reusing ``lambda_path`` reconstructs the
    same graphs.
    """
    _ensure_backend_native(backend)

    if method not in _ALLOWED_METHODS:
        raise PyHugeError(f"`method` must be one of {sorted(_ALLOWED_METHODS)}.")
    if sym not in _ALLOWED_SYM:
        raise PyHugeError(f"`sym` must be one of {sorted(_ALLOWED_SYM)}.")

    x = _ensure_2d_array("x", x, finite=True)

    if lambda_ is None:
        if nlambda is not None:
            nlambda = _ensure_positive_int("nlambda", nlambda)
        if lambda_min_ratio is not None:
            lambda_min_ratio = _ensure_ratio("lambda_min_ratio", lambda_min_ratio)
    else:
        if method == "ct":
            lambda_ = _ensure_ct_lambda_sequence(lambda_)
        elif method in {"mb", "glasso", "tiger"}:
            lambda_ = _ensure_lambda_sequence(lambda_, allow_ties=True)
        else:
            lambda_ = _ensure_lambda_sequence(lambda_)
        nlambda = None
        lambda_min_ratio = None

    if method in {"ct", "tiger"} and scr is not None:
        raise PyHugeError("`scr` is only applicable for method `mb` and `glasso`.")
    if method != "mb" and scr_num is not None:
        raise PyHugeError("`scr_num` is only applicable for method `mb`.")

    if method == "mb":
        scr = False if scr is None else bool(scr)
        if scr_num is not None:
            scr_num = _ensure_positive_int("scr_num", scr_num)
            if not scr:
                raise PyHugeError("`scr_num` requires `scr=True`.")
    elif method == "glasso":
        scr = False if scr is None else bool(scr)
    else:
        scr = None

    if cov_output and method != "glasso":
        raise PyHugeError("`cov_output=True` is only valid for method `glasso`.")
    if method in {"ct", "glasso"} and sym != "or":
        raise PyHugeError("`sym` is only applicable to method `mb` and `tiger`.")

    cov_input = _resolve_input_type(x, input_type)
    if not cov_input and x.shape[0] < 2:
        raise PyHugeError("Raw data `x` must contain at least two observations.")
    if not cov_input and np.any(np.all(x == x[0:1, :], axis=0)):
        raise PyHugeError("Raw data `x` contains a constant column.")
    if cov_input and method == "mb":
        raise PyHugeError(f"`method={method}` requires raw data matrix (n x d), not covariance matrix.")

    if method == "tiger":
        nlam = _default_nlambda("tiger") if nlambda is None else int(nlambda)
        ratio = (
            _default_lambda_min_ratio("tiger")
            if lambda_min_ratio is None
            else float(lambda_min_ratio)
        )
        requested_lambda = None if lambda_ is None else np.asarray(lambda_, dtype=float)
        path, df, icov, lambda_path = _run_tiger(
            np.asarray(x, dtype=float),
            requested_lambda,
            nlam,
            ratio,
            cov_input,
            sym=sym,
        )
        return HugeResult(
            method=method,
            lambda_path=lambda_path,
            sparsity=_path_sparsity(path),
            path=path,
            cov_input=cov_input,
            data=np.asarray(x, dtype=float),
            sym=sym,
            df=df,
            icov=icov,
            raw={"backend": "native"},
        )

    cov_mat: Optional[np.ndarray] = None
    corr: Optional[np.ndarray] = None
    if cov_input:
        transpose = x.T
        cov_mat = x.copy()
        different = x != transpose
        cov_mat[different] = (
            0.5 * x[different] + 0.5 * transpose[different]
        )
        corr = _cov_to_corr(cov_mat, require_psd=method != "glasso")

    if cov_input:
        assert cov_mat is not None and corr is not None
        s_glasso = cov_mat
    else:
        # For data input every method works on the correlation matrix
        # (matching R's .huge_preprocess); no covariance is needed.
        x_std = _standardize(x)
        if x.shape[1] == 1:
            # numpy.corrcoef squeezes a one-column input to a scalar.  Keep the
            # correlation-domain contract used by every downstream solver.
            corr = np.ones((1, 1), dtype=float)
        else:
            corr = np.asarray(np.corrcoef(x_std, rowvar=False), dtype=float)
        s_glasso = corr

    if method == "mb" and bool(scr) and scr_num is None:
        n, d = x.shape
        if n < d:
            scr_num = n - 1
        else:
            # Match huge: without explicit scr_num in n>=d, lossy screening is skipped.
            scr = False
    if method == "mb" and bool(scr) and scr_num is not None:
        d = corr.shape[0]
        if int(scr_num) >= d:
            raise PyHugeError("`scr_num` must satisfy 1 <= scr_num < d.")

    if method == "ct":
        if lambda_ is None:
            nlam = _default_nlambda("ct") if nlambda is None else _ensure_positive_int("nlambda", nlambda)
            ratio = _default_lambda_min_ratio("ct") if lambda_min_ratio is None else float(lambda_min_ratio)
            lambda_path, path, sparsity = _run_ct_default_rank(corr, nlam, ratio)
        else:
            lambda_path = _ensure_ct_lambda_sequence(lambda_)
            path = _run_ct(corr, lambda_path)
            sparsity = _ct_path_sparsity(path)
        return HugeResult(
            method=method,
            lambda_path=lambda_path,
            sparsity=sparsity,
            path=path,
            cov_input=cov_input,
            data=np.asarray(x, dtype=float),
            sym="or",
            raw={"backend": "native"},
        )

    base = corr if method == "mb" else s_glasso
    lambda_path = _build_lambda_path(
        base_matrix=base,
        method=method,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        legacy_glasso_covariance=(
            method == "glasso"
            and cov_input
            and input_type == "auto"
        ),
    )

    if method == "glasso":
        path, icov, cov_list, df, loglik = _run_glasso(
            s_glasso,
            lambda_path,
            scr=bool(scr),
            cov_output=cov_output,
        )
        return HugeResult(
            method=method,
            lambda_path=lambda_path,
            sparsity=_path_sparsity(path),
            path=path,
            cov_input=cov_input,
            data=np.asarray(x, dtype=float),
            sym="or",
            df=df,
            loglik=loglik,
            icov=icov,
            cov=cov_list,
            raw={"backend": "native", "scr": bool(scr), "cov_output": bool(cov_output)},
        )

    path, df = _run_mb(
        corr=np.asarray(corr, dtype=float),
        lambda_path=lambda_path,
        sym=sym,
        scr=bool(scr),
        scr_num=scr_num,
    )
    icov: Optional[list[np.ndarray]] = None
    raw = {"backend": "native", "scr": bool(scr), "scr_num": scr_num}

    return HugeResult(
        method=method,
        lambda_path=lambda_path,
        sparsity=_path_sparsity(path),
        path=path,
        cov_input=False,
        data=np.asarray(x, dtype=float),
        sym=sym,
        df=df,
        icov=icov,
        raw=raw,
    )


def huge_mb(
    x: np.ndarray,
    lambda_: Optional[LambdaInput] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    scr: Optional[bool] = None,
    scr_num: Optional[int] = None,
    sym: str = "or",
    verbose: bool = True,
    backend: str = "native",
    *,
    input_type: str = "auto",
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method='mb')``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="mb",
        scr=scr,
        scr_num=scr_num,
        cov_output=False,
        sym=sym,
        verbose=verbose,
        backend=backend,
        input_type=input_type,
    )


def huge_glasso(
    x: np.ndarray,
    lambda_: Optional[LambdaInput] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    scr: Optional[bool] = None,
    cov_output: bool = False,
    verbose: bool = True,
    backend: str = "native",
    *,
    input_type: str = "auto",
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method='glasso')``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="glasso",
        scr=scr,
        cov_output=cov_output,
        sym="or",
        verbose=verbose,
        backend=backend,
        input_type=input_type,
    )


def huge_ct(
    x: np.ndarray,
    lambda_: Optional[LambdaInput] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    verbose: bool = True,
    backend: str = "native",
    *,
    input_type: str = "auto",
) -> HugeResult:
    """Convenience wrapper for ``huge(..., method='ct')``."""

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="ct",
        sym="or",
        verbose=verbose,
        backend=backend,
        input_type=input_type,
    )


def huge_tiger(
    x: np.ndarray,
    lambda_: Optional[LambdaInput] = None,
    nlambda: Optional[int] = None,
    lambda_min_ratio: Optional[float] = None,
    sym: str = "or",
    verbose: bool = True,
    backend: str = "native",
    *,
    input_type: str = "auto",
) -> HugeResult:
    """Run native TIGER on observations or covariance/correlation input.

    Correlation construction, covariance validation, and automatic lambda
    selection occur together in the C++ core after ``input_type`` is resolved.
    """

    return huge(
        x=x,
        lambda_=lambda_,
        nlambda=nlambda,
        lambda_min_ratio=lambda_min_ratio,
        method="tiger",
        sym=sym,
        verbose=verbose,
        backend=backend,
        input_type=input_type,
    )


def huge_select(
    est: HugeResult,
    criterion: Optional[str] = None,
    ebic_gamma: float = 0.5,
    stars_thresh: float = 0.1,
    stars_subsample_ratio: Optional[float] = None,
    rep_num: int = 20,
    n_jobs: int = 1,
    verbose: bool = True,
    backend: str = "native",
) -> HugeSelectResult:
    """Native model selection for ``HugeResult``.

    ``n_jobs`` > 1 fits the stars subsamplings in a thread pool, capped at
    ``rep_num`` workers (results are identical to the serial path; mirrors R's
    ``num.cores``). Each fit may also start OpenMP or BLAS threads, so
    ``n_jobs=1`` is the portable choice for a bounded thread budget. Only
    applicable when ``criterion="stars"``. TIGER with StARS is rejected
    because subsample fits do not yet expose a common certified lambda-path
    prefix; use RIC for TIGER. With ``criterion=None``, defaults match R:
    RIC for MB/TIGER, StARS for CT, and EBIC for graphical lasso.
    ``verbose`` is accepted for R-API compatibility but not yet implemented.
    """
    _ensure_backend_native(backend)

    if not isinstance(est, HugeResult):
        raise PyHugeError("`est` must be HugeResult in native backend.")
    if est.cov_input:
        raise PyHugeError("Model selection is not available when using covariance matrix as input.")
    if est.method not in _ALLOWED_METHODS:
        raise PyHugeError(f"`est.method` must be one of {sorted(_ALLOWED_METHODS)}.")

    default_criteria = {
        "mb": "ric",
        "ct": "stars",
        "glasso": "ebic",
        "tiger": "ric",
    }
    crit = default_criteria[est.method] if criterion is None else str(criterion)
    if crit not in _ALLOWED_CRITERIA:
        raise PyHugeError(f"`criterion` must be one of {sorted(_ALLOWED_CRITERIA)}.")
    if crit == "ebic" and est.method != "glasso":
        raise PyHugeError("`criterion='ebic'` requires a glasso fit.")
    if crit == "stars" and est.method == "tiger":
        raise PyHugeError(
            "TIGER with StARS is unavailable until subsample fits can share "
            "a common certified prefix; use criterion='ric' for TIGER."
        )
    if (
        crit == "stars"
        and est.lambda_path.size > 1
        and np.any(np.diff(est.lambda_path) > 0)
    ):
        raise PyHugeError(
            "StARS requires `est.lambda_path` to be non-increasing; "
            "refit with `lambda_` in decreasing order."
        )

    if crit in {"ric", "stars"}:
        rep_num = _ensure_positive_int("rep_num", rep_num)
    if crit == "stars":
        n_jobs = _ensure_positive_int("n_jobs", n_jobs)
        stars_thresh = _ensure_ratio("stars_thresh", stars_thresh)
        if stars_subsample_ratio is not None:
            stars_subsample_ratio = _ensure_ratio(
                "stars_subsample_ratio", stars_subsample_ratio
            )
    if crit == "ebic":
        ebic_gamma = _ensure_finite_numeric_scalar("ebic_gamma", ebic_gamma)

    nlam = est.lambda_path.size
    if nlam == 0 or len(est.path) == 0:
        raise PyHugeError("`est` has an empty path.")
    if len(est.path) != nlam:
        raise PyHugeError("`est.path` length must match `est.lambda_path` length.")

    scr_meta = False
    scr_num_meta: Optional[int] = None
    if isinstance(est.raw, dict):
        scr_meta = bool(est.raw.get("scr", False))
        if est.raw.get("scr_num") is not None:
            try:
                scr_num_meta = int(est.raw.get("scr_num"))
            except Exception:
                scr_num_meta = None

    opt_idx = 0
    variability: Optional[np.ndarray] = None
    ebic_score: Optional[np.ndarray] = None

    if crit == "ric":
        x = _ensure_2d_array("est.data", est.data, finite=True)
        n = x.shape[0]
        if _CPP is not None:
            if n > rep_num:
                rng = np.random.default_rng(0)
                r = np.asarray(rng.choice(n, size=rep_num, replace=False), dtype=np.int32)
            else:
                r = np.arange(n, dtype=np.int32)

            # RIC must see standardized data: the lambda path is defined on the
            # correlation scale, and the rotated inner products are otherwise
            # scale-dependent (multiplying x by c scales opt_lambda by c^2).
            x_std = _standardize(x)
            opt_lambda = float(_CPP.ric(x_std, r)) / float(max(n, 1))
            if not np.isfinite(opt_lambda) or opt_lambda < 0.0:
                raise PyHugeError("Native RIC returned an invalid lambda.")
            nearest_idx = int(np.argmin(np.abs(est.lambda_path - opt_lambda)))
            d = x.shape[1]
            if d <= 1:
                max_offdiag = 0.0
            else:
                denominator = float(max(n - 1, 1))
                corr = (x_std.T @ x_std) / denominator
                abs_corr = np.abs(corr)
                abs_std = np.abs(x_std)
                absolute_cross = (abs_std.T @ abs_std) / denominator
                scaled_eps = float(n) * np.finfo(float).eps
                dot_gamma = (
                    scaled_eps / (1.0 - scaled_eps)
                    if scaled_eps < 1.0
                    else np.inf
                )
                roundoff_bound = (
                    dot_gamma * absolute_cross
                    + np.finfo(float).eps * abs_corr
                )
                resolvable_corr = np.where(
                    abs_corr > roundoff_bound, abs_corr, 0.0
                )
                offdiag = ~np.eye(d, dtype=bool)
                max_offdiag = float(np.max(resolvable_corr[offdiag]))
            # Match huge.select's empty-graph boundary.  In particular, RIC
            # intentionally returns zero for d=1 and exact zero correlation;
            # do not send that boundary value to solvers requiring lambda > 0.
            # The pair-specific dot-product bound above filters only numerical
            # cancellation, unlike an absolute tolerance that loses weak edges.
            if opt_lambda >= max_offdiag:
                return HugeSelectResult(
                    criterion=crit,
                    opt_lambda=float(opt_lambda),
                    opt_sparsity=0.0,
                    refit=sparse.csc_matrix((d, d), dtype=float),
                    opt_index=int(nearest_idx + 1),
                    variability=None,
                    ebic_score=None,
                    raw={
                        "backend": "native",
                        "criterion": crit,
                        "ric_fallback": False,
                        "ric_refit_lambda": None,
                    },
                )

            refit_lambda = opt_lambda
            zero_proxy = opt_lambda == 0.0 and est.method != "ct"
            if zero_proxy:
                refit_lambda = float(np.finfo(float).tiny)

            try:
                refit_fit = huge(
                    x=x,
                    lambda_=[refit_lambda],
                    method=est.method,
                    scr=scr_meta if est.method in {"mb", "glasso"} else None,
                    scr_num=scr_num_meta if est.method == "mb" else None,
                    cov_output=(
                        est.method == "glasso" and est.cov is not None
                    ),
                    sym=est.sym,
                    verbose=False,
                    backend="native",
                    input_type="data",
                )
            except PyHugeError:
                if not zero_proxy:
                    raise
                fallback_lambda = float(est.lambda_path[nearest_idx])
                warnings.warn(
                    "RIC selected lambda = 0, but the method could not "
                    "certify the smallest positive proxy; the original "
                    f"fitted path at lambda {fallback_lambda:.17g} was used.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                out = HugeSelectResult(
                    criterion=crit,
                    opt_lambda=float(opt_lambda),
                    opt_sparsity=float(
                        _path_sparsity([est.path[nearest_idx]])[0]
                    ),
                    refit=est.path[nearest_idx],
                    opt_index=int(nearest_idx + 1),
                    variability=None,
                    ebic_score=None,
                    raw={
                        "backend": "native",
                        "criterion": crit,
                        "ric_fallback": True,
                        "ric_refit_lambda": fallback_lambda,
                    },
                )
                if est.icov is not None:
                    out.opt_icov = np.asarray(
                        est.icov[nearest_idx], dtype=float
                    )
                if est.cov is not None:
                    out.opt_cov = np.asarray(
                        est.cov[nearest_idx], dtype=float
                    )
                return out

            out = HugeSelectResult(
                criterion=crit,
                opt_lambda=float(opt_lambda),
                opt_sparsity=float(refit_fit.sparsity[0]),
                refit=refit_fit.path[0],
                opt_index=int(nearest_idx + 1),
                variability=None,
                ebic_score=None,
                raw={
                    "backend": "native",
                    "criterion": crit,
                    "ric_fallback": False,
                    "ric_refit_lambda": float(refit_lambda),
                },
            )
            if refit_fit.icov is not None:
                out.opt_icov = np.asarray(refit_fit.icov[0], dtype=float)
            if refit_fit.cov is not None:
                out.opt_cov = np.asarray(refit_fit.cov[0], dtype=float)
            return out

        # Fallback when C++ core is unavailable.
        opt_idx = _selected_index_ric(est.sparsity)

    elif crit == "ebic":
        ebic_score = _ebic_from_fit(est, float(ebic_gamma))
        opt_idx = int(np.argmin(ebic_score))

    else:  # stars
        x = _ensure_2d_array("est.data", est.data, finite=True)
        n = x.shape[0]
        d = x.shape[1]
        if d < 2:
            raise PyHugeError("StARS requires at least two variables.")
        if min(n_jobs, rep_num) > 1:
            warnings.warn(
                "`n_jobs > 1` runs StARS fits concurrently, and each fit may "
                "also start OpenMP or BLAS threads; the total thread count "
                "can multiply. Use `n_jobs=1` for a portable single-level "
                "thread budget.",
                RuntimeWarning,
                stacklevel=2,
            )

        ratio = stars_subsample_ratio
        if ratio is None:
            ratio = 0.8 if n <= 144 else min(0.99, 10.0 * math.sqrt(n) / n)
        m = int(math.floor(n * ratio))
        if m < 2:
            raise PyHugeError(
                "`stars_subsample_ratio` must select at least two observations."
            )

        rng = np.random.default_rng(0)
        packed_frequency = est.method in {"ct", "mb"}
        if packed_frequency:
            frequency_shape = (nlam, d * (d - 1) // 2)
        else:
            # Glasso precision iterates can contain a numerically one-sided
            # edge. Keep both directions so its existing StARS semantics are
            # unchanged; CT and MB construct exactly symmetric paths.
            frequency_shape = (nlam, d, d)
        freq = np.zeros(frequency_shape, dtype=_stars_count_dtype(rep_num))

        # Pre-draw all subsample index sets (fits consume no RNG), keeping
        # results identical whether fits run serially or in parallel threads.
        # Threads suffice: the native solvers hold the GIL only in the
        # binding layer, and OpenMP does the real work.
        index_sets = [rng.choice(n, size=m, replace=False) for _ in range(rep_num)]

        def _stars_subfit(idx: np.ndarray) -> list:
            return huge(
                x[idx],
                lambda_=est.lambda_path,
                method=est.method,
                sym=est.sym,
                scr=scr_meta if est.method in {"mb", "glasso"} else None,
                scr_num=scr_num_meta if est.method == "mb" else None,
                backend="native",
                input_type="data",
            ).path

        def _accumulate_stars_path(path_list: list[sparse.csc_matrix]) -> None:
            if len(path_list) != nlam:
                raise PyHugeError(
                    "A StARS subsample returned a path with an unexpected length."
                )
            for li, p in enumerate(path_list):
                rows, cols = p.nonzero()
                if packed_frequency:
                    upper = rows < cols
                    packed = _stars_upper_triangle_indices(
                        rows[upper], cols[upper], d
                    )
                    freq[li, packed] += 1
                else:
                    freq[li, rows, cols] += 1

        if n_jobs > 1:
            from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

            worker_count = min(int(n_jobs), len(index_sets))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                index_iter = iter(index_sets)
                pending = set()
                for _ in range(worker_count):
                    pending.add(pool.submit(_stars_subfit, next(index_iter)))

                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        _accumulate_stars_path(future.result())
                        try:
                            next_index = next(index_iter)
                        except StopIteration:
                            continue
                        pending.add(pool.submit(_stars_subfit, next_index))
        else:
            for idx in index_sets:
                _accumulate_stars_path(_stars_subfit(idx))

        variability = np.zeros(nlam, dtype=float)
        for li in range(nlam):
            probability = freq[li].astype(np.float64)
            probability /= float(rep_num)
            if packed_frequency:
                p_mat = squareform(probability)
            else:
                p_mat = 0.5 * (probability + probability.T)
            np.fill_diagonal(p_mat, 0.0)
            variability[li] = float(4.0 * np.sum(p_mat * (1.0 - p_mat)) / (d * (d - 1)))

        stars_cross = np.where(variability >= stars_thresh)[0]
        if stars_cross.size == 0:
            opt_idx = int(nlam - 1)
        else:
            opt_idx = max(int(stars_cross[0]) - 1, 0)

    refit = est.path[opt_idx]
    out = HugeSelectResult(
        criterion=crit,
        opt_lambda=float(est.lambda_path[opt_idx]),
        opt_sparsity=float(est.sparsity[opt_idx]),
        refit=refit,
        opt_index=int(opt_idx + 1),
        variability=variability,
        ebic_score=ebic_score,
        raw={"backend": "native", "criterion": crit},
    )

    if est.icov is not None:
        out.opt_icov = np.asarray(est.icov[opt_idx], dtype=float)
    if est.cov is not None:
        out.opt_cov = np.asarray(est.cov[opt_idx], dtype=float)
    return out


def huge_npn(
    x: np.ndarray,
    npn_func: str = "shrinkage",
    verbose: bool = True,
) -> np.ndarray:
    """Native nonparanormal transformation.

    ``verbose`` is accepted for R-API compatibility but not yet implemented.
    """
    if npn_func not in _ALLOWED_NPN_FUNCS:
        raise PyHugeError(f"`npn_func` must be one of {sorted(_ALLOWED_NPN_FUNCS)}.")

    x = _ensure_2d_array("x", x, finite=True)
    n, d = x.shape

    if npn_func == "skeptic":
        if n < 2:
            raise PyHugeError(
                "NPN skeptic requires at least two observations."
            )
        if np.any(np.all(x == x[0:1, :], axis=0)):
            raise PyHugeError(
                "NPN skeptic cannot estimate a constant column."
            )
        if d == 1:
            return np.ones((1, 1), dtype=float)

        ranks = np.apply_along_axis(
            stats.rankdata, 0, np.asarray(x, dtype=float)
        )
        standardized_ranks = _standardize(ranks)
        rho = (standardized_ranks.T @ standardized_ranks) / float(n - 1)
        np.fill_diagonal(rho, 1.0)
        out = 2.0 * np.sin((np.pi / 6.0) * rho)
        np.fill_diagonal(out, 1.0)
        return out

    # Match R's huge.npn exactly:
    #   shrinkage:  qnorm(rank/(n+1))                       (no extra clipping)
    #   truncation: qnorm(clip(rank/n, thresh, 1-thresh))
    # followed by dividing each column by its sample sd.
    ranks = np.apply_along_axis(stats.rankdata, 0, np.asarray(x, dtype=float))
    if npn_func == "shrinkage":
        z = stats.norm.ppf(ranks / (n + 1.0))
    else:
        trunc = 1.0 / (4.0 * (n ** 0.25) * math.sqrt(np.pi * np.log(max(n, 2))))
        z = stats.norm.ppf(np.clip(ranks / n, trunc, 1.0 - trunc))

    col_sd = z.std(axis=0, ddof=1)
    col_sd[~np.isfinite(col_sd) | (col_sd == 0)] = 1.0
    return z / col_sd


def _group_partitions(d: int, g: int) -> list[np.ndarray]:
    idx = np.arange(d)
    return [arr for arr in np.array_split(idx, g) if arr.size > 0]


def _theta_random(d: int, prob: float, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    tri = np.triu(rng.random((d, d)) < prob, 1)
    a[tri] = 1.0
    a = a + a.T
    return a


def _theta_hub(d: int, g: int) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    for grp in _group_partitions(d, g):
        if grp.size <= 1:
            continue
        c = int(grp[0])
        for j in grp[1:]:
            a[c, j] = 1.0
            a[j, c] = 1.0
    return a


def _theta_cluster(d: int, g: int, prob: float, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    for grp in _group_partitions(d, g):
        m = grp.size
        if m <= 1:
            continue
        mask = np.triu(rng.random((m, m)) < prob, 1)
        for i in range(m):
            for j in range(i + 1, m):
                if mask[i, j]:
                    u = int(grp[i])
                    v = int(grp[j])
                    a[u, v] = 1.0
                    a[v, u] = 1.0
    return a


def _theta_band(d: int, g: int) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    for i in range(d):
        for j in range(max(0, i - g), min(d, i + g + 1)):
            if i != j:
                a[i, j] = 1.0
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    return a


def _theta_scale_free(d: int, rng: np.random.Generator) -> np.ndarray:
    a = np.zeros((d, d), dtype=float)
    if d <= 1:
        return a
    a[0, 1] = 1.0
    a[1, 0] = 1.0
    degrees = np.sum(a, axis=0)
    for new_node in range(2, d):
        total_deg = float(np.sum(degrees[:new_node]))
        if total_deg <= 0:
            target = int(rng.integers(0, new_node))
        else:
            prob = degrees[:new_node] / total_deg
            target = int(rng.choice(np.arange(new_node), p=prob))
        a[new_node, target] = 1.0
        a[target, new_node] = 1.0
        degrees = np.sum(a, axis=0)
    return a


def huge_generator(
    n: int = 200,
    d: int = 50,
    graph: str = "random",
    v: Optional[float] = None,
    u: Optional[float] = None,
    g: Optional[int] = None,
    prob: Optional[float] = None,
    vis: bool = False,
    verbose: bool = True,
    random_state: Optional[int] = None,
) -> HugeGeneratorResult:
    """Native data generator.

    ``sigmahat`` is the empirical correlation matrix, matching R.
    ``vis`` and ``verbose`` are accepted for R-API compatibility but not
    yet implemented.
    """
    n = _ensure_positive_int("n", n)
    if n < 2:
        raise PyHugeError("`n` must be at least 2 to estimate correlation.")
    d = _ensure_positive_int("d", d)
    if graph not in _ALLOWED_GRAPH_TYPES:
        raise PyHugeError(f"`graph` must be one of {sorted(_ALLOWED_GRAPH_TYPES)}.")

    rng = np.random.default_rng(random_state)
    v_val = 0.3 if v is None else _ensure_finite_numeric_scalar("v", v)
    u_val = 0.1 if u is None else _ensure_finite_numeric_scalar("u", u)

    if v_val <= 0 or u_val <= 0:
        raise PyHugeError("`v` and `u` must be positive.")

    if graph in {"hub", "cluster"}:
        g_val = 2 if d < 40 else max(2, d // 20)
        if g is not None:
            g_val = _ensure_positive_int("g", g)
    elif graph == "band":
        g_val = 1 if g is None else _ensure_positive_int("g", g)
    else:
        g_val = 1

    if graph == "random":
        p_val = min(1.0, 3.0 / d) if prob is None else _ensure_finite_numeric_scalar("prob", prob)
    elif graph == "cluster":
        p_val = min(1.0, 6.0 * g_val / d) if (d / max(g_val, 1)) <= 30 else 0.3
        if prob is not None:
            p_val = _ensure_finite_numeric_scalar("prob", prob)
    else:
        p_val = 0.0

    if p_val < 0.0 or p_val > 1.0:
        raise PyHugeError("`prob` must satisfy 0 <= prob <= 1.")

    if graph == "random":
        theta = _theta_random(d, p_val, rng)
    elif graph == "hub":
        theta = _theta_hub(d, g_val)
    elif graph == "cluster":
        theta = _theta_cluster(d, g_val, p_val, rng)
    elif graph == "band":
        theta = _theta_band(d, g_val)
    else:
        if _CPP is not None:
            try:
                seed = None if random_state is None else int(random_state)
                theta = np.asarray(_CPP.sfgen(2, int(d), seed), dtype=float)
            except Exception as exc:
                warnings.warn(
                    f"native sfgen failed ({exc!r}); falling back to the "
                    "Python scale-free generator (different edge sequence)",
                    RuntimeWarning,
                    stacklevel=2,
                )
                theta = _theta_scale_free(d, rng)
        else:
            theta = _theta_scale_free(d, rng)

    base = v_val * theta
    min_eig = float(np.min(np.linalg.eigvalsh(base)))
    shift = abs(min_eig) + 0.1 + u_val
    omega = base + np.eye(d) * shift
    # Match R: standardize sigma to a correlation matrix (cov2cor) and take
    # omega as its inverse, so the generated model has unit-variance margins.
    # Previously sigma was the raw inverse (diagonal 0.9-2.7), a different
    # model from the R package's. SPD inverses via Cholesky.
    sigma_raw = _spd_inverse(omega)
    dinv = 1.0 / np.sqrt(np.diag(sigma_raw))
    sigma = sigma_raw * np.outer(dinv, dinv)
    sigma = (sigma + sigma.T) / 2.0
    np.fill_diagonal(sigma, 1.0)
    omega = _spd_inverse(sigma)
    omega = (omega + omega.T) / 2.0

    data = rng.multivariate_normal(mean=np.zeros(d), cov=sigma, size=n)
    if d == 1:
        sigmahat = np.ones((1, 1), dtype=float)
    else:
        sigmahat = np.asarray(np.corrcoef(data, rowvar=False), dtype=float)
        sigmahat = (sigmahat + sigmahat.T) / 2.0
        np.fill_diagonal(sigmahat, 1.0)

    return HugeGeneratorResult(
        data=np.asarray(data, dtype=float),
        sigma=np.asarray(sigma, dtype=float),
        omega=np.asarray(omega, dtype=float),
        sigmahat=sigmahat,
        theta=sparse.csc_matrix(theta),
        sparsity=_adj_sparsity(theta != 0),
        graph_type=graph,
        raw={"backend": "native"},
    )


def huge_roc(
    path: Sequence[np.ndarray | sparse.spmatrix],
    theta: np.ndarray | sparse.spmatrix,
    verbose: bool = True,
    plot: bool = False,
) -> HugeRocResult:
    """Native ROC metrics for graph path.

    ``verbose`` is accepted for R-API compatibility but not yet implemented.
    """
    if len(path) == 0:
        raise PyHugeError("`path` must contain at least one adjacency matrix.")

    theta_dense = _to_dense_matrix(theta, "theta")
    if theta_dense.shape[0] != theta_dense.shape[1]:
        raise PyHugeError("`theta` must be square.")

    d = theta_dense.shape[0]
    truth = np.asarray(theta_dense != 0, dtype=bool)
    np.fill_diagonal(truth, False)
    truth_u = np.triu(truth, 1)

    total_pos = int(np.count_nonzero(truth_u))
    total_pairs = d * (d - 1) // 2
    total_neg = total_pairs - total_pos
    if total_pos == 0 or total_neg == 0:
        raise PyHugeError(
            "`theta` must contain at least one edge and at least one absent "
            "off-diagonal edge; ROC/AUC is undefined for a one-class truth."
        )

    tp = np.zeros(len(path), dtype=float)
    fp = np.zeros(len(path), dtype=float)
    f1 = np.zeros(len(path), dtype=float)

    for i, p in enumerate(path):
        pred = _to_dense_matrix(p, f"path[{i + 1}]")
        if pred.shape != (d, d):
            raise PyHugeError(f"`path[{i + 1}]` must have shape ({d}, {d}).")

        pred_u = np.triu(pred != 0, 1)
        tp_count = int(np.count_nonzero(pred_u & truth_u))
        fp_count = int(np.count_nonzero(pred_u & (~truth_u)))
        pred_count = int(np.count_nonzero(pred_u))

        tp[i] = tp_count / total_pos
        fp[i] = fp_count / total_neg

        precision = tp_count / max(pred_count, 1)
        recall = tp[i]
        denom = precision + recall
        f1[i] = 0.0 if denom <= 0 else (2.0 * precision * recall / denom)

    order = np.lexsort((tp, fp))
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0
        trapezoid = np.trapz
    auc = float(trapezoid(tp[order], fp[order]))
    out = HugeRocResult(f1=f1, tp=tp, fp=fp, auc=auc, raw={"backend": "native"})

    if plot:
        huge_plot_roc(out)

    return out


def huge_inference(
    data: np.ndarray,
    t: np.ndarray,
    adj: np.ndarray | sparse.spmatrix,
    alpha: float = 0.05,
    type_: str = "Gaussian",
    method: str = "score",
) -> HugeInferenceResult:
    """Edge-wise inference matching R ``huge.inference``.

    Gaussian inference uses the de-biased precision estimator.  For a
    nonparanormal model, ``method`` selects the score or Wald statistic from
    the R implementation.  The latter methods are substantially more
    expensive because they estimate a ``d^2`` by ``d^2`` covariance matrix.
    Data must contain at least two observations and no constant columns;
    nonparanormal inference additionally requires at least two variables.
    The supplied precision-like matrix must have a positive diagonal.
    """

    if type_ not in _ALLOWED_INFERENCE_TYPES:
        raise PyHugeError(f"`type_` must be one of {sorted(_ALLOWED_INFERENCE_TYPES)}.")
    if type_ == "Nonparanormal" and method not in _ALLOWED_INFERENCE_METHODS:
        raise PyHugeError(f"`method` must be one of {sorted(_ALLOWED_INFERENCE_METHODS)}.")
    alpha = _ensure_ratio("alpha", alpha)

    x = _ensure_2d_array("data", data, finite=True)
    n, d = x.shape
    if n < 2:
        raise PyHugeError("Inference requires at least two observations.")
    if type_ == "Nonparanormal" and d < 2:
        raise PyHugeError(
            "Nonparanormal inference requires at least two variables."
        )
    if np.any(np.all(x == x[0:1, :], axis=0)):
        raise PyHugeError("Inference data contains a constant column.")

    t_mat = _to_dense_matrix(t, "t")
    if t_mat.shape != (d, d):
        raise PyHugeError(f"`t` must have shape ({d}, {d}).")

    adj_mat = _to_dense_matrix(adj, "adj")
    if adj_mat.shape != (d, d):
        raise PyHugeError(f"`adj` must have shape ({d}, {d}).")
    if np.any(np.diag(t_mat) <= 0.0):
        raise PyHugeError("`t` must have a positive diagonal.")

    if type_ == "Gaussian":
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            u_mat = np.atleast_2d(
                np.corrcoef(_standardize(x), rowvar=False)
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            w_mat = 2.0 * t_mat - t_mat @ u_mat @ t_mat
            variance = (
                np.outer(np.diag(t_mat), np.diag(t_mat)) + t_mat * t_mat
            )
        if np.any(~np.isfinite(variance)) or np.any(variance <= 0.0):
            raise PyHugeError(
                "Gaussian inference variance must be finite and positive; "
                "check the scale of `t`."
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            statistic = np.sqrt(float(n)) * w_mat / np.sqrt(variance)
            p = 2.0 * (1.0 - stats.norm.cdf(np.abs(statistic)))
    else:
        # First pass: build the transformed Kendall matrix U.  Streaming over
        # observations keeps memory at O(n d + d^2), rather than materializing
        # the n pairwise-sign matrices or R's d^2 by d^2 covariance matrix.
        concordance_sum = np.zeros((d, d), dtype=float)
        for i in range(n):
            # The sign remains exact when a finite difference overflows to
            # +/-Inf, which can occur for observations near float limits.
            with np.errstate(over="ignore", invalid="ignore"):
                signed = np.sign(x[i, None, :] - x)
            concordance_sum += signed.T @ signed

        scale = np.pi / (2.0 * float(n - 1))
        tau_denom = float(n * (n - 1))
        u_mat = np.sin((np.pi / 2.0) * concordance_sum / tau_denom)
        np.fill_diagonal(u_mat, 1.0)
        f_mat = np.sqrt(np.maximum(0.0, 1.0 - u_mat * u_mat))
        asin_u = np.arcsin(u_mat)
        with np.errstate(over="ignore", invalid="ignore"):
            diag_outer = np.outer(np.diag(t_mat), np.diag(t_mat))
        if np.any(~np.isfinite(diag_outer)) or np.any(diag_outer <= 0.0):
            raise PyHugeError(
                "Products of `t` diagonal entries must remain finite and positive."
            )

        # Second pass: the R quadratic form for coordinate (j, k) reduces to
        # the squared (j, k) entry of T' H_i T / (T_jj T_kk).  This computes
        # exactly the same variance without allocating O(d^4) R or kron(T,T).
        sigma_sq = np.zeros((d, d), dtype=float)
        for i in range(n):
            with np.errstate(over="ignore", invalid="ignore"):
                signed = np.sign(x[i, None, :] - x)
            g_mat = asin_u - scale * (signed.T @ signed)
            np.fill_diagonal(g_mat, 0.0)
            h_mat = f_mat * g_mat
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                transformed = t_mat.T @ h_mat @ t_mat
                standardized = transformed / diag_outer
                sigma_sq += standardized * standardized
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sigma = np.sqrt(sigma_sq / float(n))

        if method == "score":
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                t_u = t_mat.T @ u_mat
                u_t = u_mat @ t_mat
                base = t_u @ t_mat
                numerator = base - t_mat * np.diag(t_u)[:, None]
                diag_idx = np.diag_indices(d)
                numerator[diag_idx] = (
                    np.diag(base)
                    - np.diag(t_mat) * np.diag(t_u)
                    - np.diag(t_mat) * np.diag(u_t)
                    + np.diag(t_mat) ** 2 * np.diag(u_mat)
                )
                score = numerator / diag_outer
                statistic = score * np.sqrt(float(n)) / (2.0 * sigma)
        else:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                t_u = t_mat @ u_mat
                u_t = u_mat @ t_mat
                numerator = t_mat * (t_u + u_t) - t_mat.T @ u_t
                denominator = t_u + u_t - 1.0
                t_wald = numerator / denominator
                statistic = (
                    t_wald * np.sqrt(float(n)) / (2.0 * sigma * diag_outer)
                )
        p = 2.0 * (1.0 - stats.norm.cdf(np.abs(statistic)))

    offdiag = ~np.eye(d, dtype=bool)
    if type_ == "Gaussian":
        finite_p = np.isfinite(p).all()
    else:
        # Score inference can have an undefined diagonal even when every edge
        # p-value is a valid finite limit.  Only off-diagonal values represent
        # tested graph edges.
        finite_p = np.isfinite(p[offdiag]).all()
    if not finite_p:
        raise PyHugeError(
            "Inference produced non-finite edge p-values; "
            "the inputs are numerically degenerate."
        )
    false_rejections = (p < alpha) & (adj_mat == 0) & offdiag
    error = float(np.count_nonzero(false_rejections)) / float(d * d)

    return HugeInferenceResult(
        data=np.asarray(x, dtype=float),
        p=np.asarray(p, dtype=float),
        error=error,
    )


def huge_stockdata() -> HugeStockDataResult:
    """Load packaged stock dataset (converted from R ``stockdata``)."""

    try:
        stock_path = resources.files("pyhuge").joinpath("data/stockdata.npz")
        with resources.as_file(stock_path) as resolved:
            payload = np.load(resolved, allow_pickle=True)
            data = np.asarray(payload["data"], dtype=float)
            info = np.asarray(payload["info"])
    except FileNotFoundError as exc:
        raise PyHugeError("Built-in stock dataset is missing from the package installation.") from exc
    except Exception as exc:
        raise PyHugeError("Failed to load built-in stock dataset.") from exc

    return HugeStockDataResult(data=data, info=info, raw={"source": str(stock_path)})


def huge_summary(fit: HugeResult) -> HugeSummary:
    """Return a concise summary of ``HugeResult``."""

    n_samples, n_features = fit.data.shape
    return HugeSummary(
        method=fit.method,
        n_samples=int(n_samples),
        n_features=int(n_features),
        path_length=int(len(fit.lambda_path)),
        sparsity_min=float(np.min(fit.sparsity)),
        sparsity_max=float(np.max(fit.sparsity)),
        cov_input=bool(fit.cov_input),
        has_icov=fit.icov is not None,
        has_cov=fit.cov is not None,
    )


def huge_select_summary(sel: HugeSelectResult) -> HugeSelectSummary:
    """Return a concise summary of ``HugeSelectResult``."""

    return HugeSelectSummary(
        criterion=sel.criterion,
        opt_lambda=float(sel.opt_lambda),
        opt_sparsity=float(sel.opt_sparsity),
        refit_n_features=int(sel.refit.shape[1]),
        has_opt_icov=sel.opt_icov is not None,
        has_opt_cov=sel.opt_cov is not None,
    )


def _mpl_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PyHugeError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from exc
    return plt


def _networkx_pkg() -> Any:
    try:
        import networkx as nx
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PyHugeError("networkx is required for network plotting. Install with `pip install networkx`.") from exc
    return nx


def huge_plot_sparsity(fit: HugeResult, ax: Optional[Any] = None, show_points: bool = True) -> Any:
    """Plot sparsity level versus regularization path."""

    if fit.lambda_path.size == 0:
        raise PyHugeError("`fit.lambda_path` is empty.")
    if fit.sparsity.size != fit.lambda_path.size:
        raise PyHugeError("`fit.sparsity` length must match `fit.lambda_path` length.")

    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    ax.plot(fit.lambda_path, fit.sparsity, "-", lw=1.6)
    if show_points:
        ax.plot(fit.lambda_path, fit.sparsity, "o", ms=3)
    if np.all(fit.lambda_path > 0):
        ax.set_xscale("log")
        ax.set_xlim(np.max(fit.lambda_path), np.min(fit.lambda_path))
    ax.set_xlabel("Regularization Parameter")
    ax.set_ylabel("Sparsity Level")
    ax.set_title(f"Sparsity Path ({fit.method})")
    return ax


def huge_plot_roc(roc: HugeRocResult, ax: Optional[Any] = None) -> Any:
    """Plot ROC curve from ``HugeRocResult``."""

    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    order = np.lexsort((roc.tp, roc.fp))
    ax.plot(roc.fp[order], roc.tp[order], "-o", ms=3, lw=1.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC={roc.auc:.4f})")
    return ax


def huge_plot_graph_matrix(
    fit: HugeResult,
    index: int = -1,
    ax: Optional[Any] = None,
) -> Any:
    """Visualize one adjacency matrix on the path as a heatmap."""

    if len(fit.path) == 0:
        raise PyHugeError("`fit.path` is empty.")

    if index < 0:
        index = len(fit.path) + index
    if index < 0 or index >= len(fit.path):
        raise PyHugeError(f"`index` out of range: got {index}, path length={len(fit.path)}")

    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    mat = fit.path[index].toarray()
    ax.imshow(mat, cmap="Greys", interpolation="nearest")
    if index < fit.lambda_path.size:
        title = f"Graph Matrix (idx={index}, lambda={fit.lambda_path[index]:.4g})"
    else:
        title = f"Graph Matrix (idx={index})"
    ax.set_title(title)
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    return ax


def huge_plot_network(
    fit: HugeResult,
    index: int = -1,
    ax: Optional[Any] = None,
    layout: str = "spring",
    with_labels: bool = False,
    node_size: float = 120.0,
    node_color: str = "#c44e52",
    edge_color: str = "#4d4d4d",
    min_abs_weight: float = 0.0,
) -> Any:
    """Plot one estimated graph as a node-edge network."""

    if len(fit.path) == 0:
        raise PyHugeError("`fit.path` is empty.")
    if min_abs_weight < 0:
        raise PyHugeError("`min_abs_weight` must be non-negative.")

    if index < 0:
        index = len(fit.path) + index
    if index < 0 or index >= len(fit.path):
        raise PyHugeError(f"`index` out of range: got {index}, path length={len(fit.path)}")

    dense = fit.path[index].toarray().astype(float)
    dense = 0.5 * (dense + dense.T)
    if min_abs_weight > 0:
        dense[np.abs(dense) < min_abs_weight] = 0.0

    nx = _networkx_pkg()
    plt = _mpl_pyplot()
    if ax is None:
        _, ax = plt.subplots(1, 1)

    g = nx.from_numpy_array(dense)
    layout_map = {
        "spring": nx.spring_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "circular": nx.circular_layout,
        "spectral": nx.spectral_layout,
        "shell": nx.shell_layout,
    }
    if layout not in layout_map:
        raise PyHugeError(f"`layout` must be one of {sorted(layout_map)}.")

    pos = layout_map[layout](g)
    nx.draw_networkx(
        g,
        pos=pos,
        ax=ax,
        with_labels=with_labels,
        node_size=float(node_size),
        node_color=node_color,
        edge_color=edge_color,
        width=1.2,
        alpha=1.0,
    )
    if index < fit.lambda_path.size:
        title = f"Network (idx={index}, lambda={fit.lambda_path[index]:.4g})"
    else:
        title = f"Network (idx={index})"
    ax.set_title(title)
    ax.set_axis_off()
    return ax


def huge_plot(
    g: np.ndarray | sparse.spmatrix,
    epsflag: bool = False,
    graph_name: str = "default",
    cur_num: int = 1,
    location: Optional[str] = None,
) -> Optional[str]:
    """R ``huge.plot``-style visualization in native Python."""

    adj = _to_dense_matrix(g, "g")
    if adj.shape[0] != adj.shape[1]:
        raise PyHugeError("`g` must be square.")

    out_path: Optional[Path] = None
    if epsflag:
        cur_num = _ensure_positive_int("cur_num", cur_num)
        if not graph_name:
            raise PyHugeError("`graph_name` must be a non-empty string.")
        if location is None:
            out_dir = Path.cwd()
        else:
            out_dir = Path(location)
            if not out_dir.is_dir():
                raise PyHugeError("`location` must be an existing directory.")
        out_path = out_dir / f"{graph_name}{int(cur_num)}.eps"

    fit = HugeResult(
        method="plot",
        lambda_path=np.asarray([1.0]),
        sparsity=np.asarray([_adj_sparsity(adj != 0)]),
        path=[sparse.csc_matrix((adj != 0).astype(float))],
        cov_input=False,
        data=np.asarray(adj, dtype=float),
    )

    ax = huge_plot_network(fit, index=0)
    plt = _mpl_pyplot()

    if not epsflag:
        plt.close(ax.figure)
        return None

    assert out_path is not None
    ax.figure.savefig(out_path, format="eps", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    return str(out_path)
