"""R-reference parity helpers for pyhuge native backend."""

from __future__ import annotations

import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np


def has_r_huge() -> bool:
    """Return True when local R and package huge are available."""

    try:
        out = subprocess.run(
            ["R", "-q", "-e", 'cat(requireNamespace("huge", quietly=TRUE))'],
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return False
    return "TRUE" in out.stdout


def _run_r_script(r_code: str, args: list[str], prefix: str = "pyhuge_rref_") -> Path:
    """Write R code to a temp script, run it, return the temp directory path."""
    td = tempfile.mkdtemp(prefix=prefix)
    out_dir = Path(td)
    script = out_dir / "run_ref.R"
    script.write_text(textwrap.dedent(r_code))
    subprocess.run(
        ["Rscript", str(script)] + args,
        check=True, text=True, capture_output=True,
    )
    return out_dir


def _load_sel_results(out_dir: Path) -> dict[str, np.ndarray | float | int]:
    """Load standard fit+select output files."""
    sel = np.loadtxt(out_dir / "sel.txt")
    return {
        "lambda": np.loadtxt(out_dir / "lambda.txt"),
        "sparsity": np.loadtxt(out_dir / "sparsity.txt"),
        "edges": np.loadtxt(out_dir / "edges.txt"),
        "opt_lambda": float(sel[0]),
        "opt_sparsity": float(sel[1]),
        "opt_index": int(sel[2]),
    }


_R_EDGE_COUNT = """
edge_count <- function(mat) { m <- as.matrix(mat); diag(m) <- 0; sum(m != 0) / 2 }
"""

_R_FIT_SELECT_OUTPUT = """
write.table(fit$lambda, file.path(out_dir, 'lambda.txt'), row.names=FALSE, col.names=FALSE)
write.table(fit$sparsity, file.path(out_dir, 'sparsity.txt'), row.names=FALSE, col.names=FALSE)
""" + _R_EDGE_COUNT + """
write.table(sapply(fit$path, edge_count), file.path(out_dir, 'edges.txt'), row.names=FALSE, col.names=FALSE)
write.table(c(sel$opt.lambda, sel$opt.sparsity, sel$opt.index), file.path(out_dir, 'sel.txt'), row.names=FALSE, col.names=FALSE)
"""


def run_r_ct_reference(
    x: np.ndarray,
    lambda_ct: np.ndarray,
    *,
    rep_num: int = 6,
    stars_thresh: float = 0.1,
    seed: int = 123,
) -> dict[str, np.ndarray | float | int]:
    """Run R huge reference for ct + stars on fixed input."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_ct_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        lam_path = out_dir / "lam.txt"
        np.savetxt(x_path, x, delimiter=",")
        np.savetxt(lam_path, lambda_ct)

        _run_r_script(
            """
            args <- commandArgs(trailingOnly = TRUE)
            x <- as.matrix(read.csv(args[1], header=FALSE))
            lam <- as.numeric(scan(args[2], quiet=TRUE))
            out_dir <- args[3]
            rep_num <- as.integer(args[4])
            stars_thresh <- as.numeric(args[5])
            seed <- as.integer(args[6])

            suppressMessages(library(huge))
            fit <- huge(x, method='ct', lambda=lam, verbose=FALSE)
            set.seed(seed)
            sel <- huge.select(fit, criterion='stars', rep.num=rep_num, stars.thresh=stars_thresh, verbose=FALSE)
            """ + _R_FIT_SELECT_OUTPUT,
            [str(x_path), str(lam_path), str(out_dir),
             str(rep_num), str(stars_thresh), str(seed)],
        )

        return _load_sel_results(out_dir)


def run_r_ct_default_reference(
    x: np.ndarray,
    *,
    nlambda: int = 20,
    lambda_min_ratio: float = 0.05,
) -> dict[str, np.ndarray]:
    """Run R huge reference for ct default rank-based path construction."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_ct_default_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        np.savetxt(x_path, x, delimiter=",")

        _run_r_script(
            """
            args <- commandArgs(trailingOnly = TRUE)
            x <- as.matrix(read.csv(args[1], header=FALSE))
            out_dir <- args[2]
            nlambda <- as.integer(args[3])
            lambda_min_ratio <- as.numeric(args[4])

            suppressMessages(library(huge))
            fit <- huge(x, method='ct', nlambda=nlambda, lambda.min.ratio=lambda_min_ratio, verbose=FALSE)

            write.table(fit$lambda, file.path(out_dir, 'lambda.txt'), row.names=FALSE, col.names=FALSE)
            write.table(fit$sparsity, file.path(out_dir, 'sparsity.txt'), row.names=FALSE, col.names=FALSE)

            d <- ncol(x)
            nlam <- length(fit$path)
            arr <- array(0, dim=c(nlam, d, d))
            for (i in seq_len(nlam)) arr[i,,] <- as.matrix(fit$path[[i]])
            write.table(as.vector(arr), file.path(out_dir, 'path_flat.txt'), row.names=FALSE, col.names=FALSE)
            """,
            [str(x_path), str(out_dir), str(nlambda), str(lambda_min_ratio)],
        )

        lam = np.atleast_1d(np.loadtxt(out_dir / "lambda.txt")).astype(float)
        sparsity = np.atleast_1d(np.loadtxt(out_dir / "sparsity.txt")).astype(float)
        flat = np.atleast_1d(np.loadtxt(out_dir / "path_flat.txt")).astype(float)
        d = int(x.shape[1])
        nlam = int(lam.size)
        path = flat.reshape((nlam, d, d), order="F")

        return {"lambda": lam, "sparsity": sparsity, "path": path}


def run_r_tiger_reference(
    x: np.ndarray,
    lambda_tiger: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run R huge TIGER on a fixed native lambda path."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_tiger_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        lam_path = out_dir / "lam.txt"
        np.savetxt(x_path, x, delimiter=",")
        np.savetxt(lam_path, lambda_tiger)

        _run_r_script(
            """
            args <- commandArgs(trailingOnly = TRUE)
            x <- as.matrix(read.csv(args[1], header=FALSE))
            lam <- as.numeric(scan(args[2], quiet=TRUE))
            out_dir <- args[3]

            suppressMessages(library(huge))
            fit <- huge(x, method='tiger', lambda=lam, verbose=FALSE)
            write.table(fit$lambda, file.path(out_dir, 'lambda.txt'),
                        row.names=FALSE, col.names=FALSE)
            write.table(fit$sparsity, file.path(out_dir, 'sparsity.txt'),
                        row.names=FALSE, col.names=FALSE)

            d <- ncol(x)
            nlam <- length(fit$path)
            path <- array(0, dim=c(nlam, d, d))
            icov <- array(0, dim=c(nlam, d, d))
            for (i in seq_len(nlam)) {
              path[i,,] <- as.matrix(fit$path[[i]])
              icov[i,,] <- fit$icov[[i]]
            }
            write.table(as.vector(path), file.path(out_dir, 'path_flat.txt'),
                        row.names=FALSE, col.names=FALSE)
            write.table(as.vector(icov), file.path(out_dir, 'icov_flat.txt'),
                        row.names=FALSE, col.names=FALSE)
            """,
            [str(x_path), str(lam_path), str(out_dir)],
        )

        lam = np.atleast_1d(np.loadtxt(out_dir / "lambda.txt")).astype(float)
        sparsity = np.atleast_1d(np.loadtxt(out_dir / "sparsity.txt")).astype(float)
        d = int(x.shape[1])
        nlam = int(lam.size)
        path = np.atleast_1d(np.loadtxt(out_dir / "path_flat.txt")).reshape(
            (nlam, d, d), order="F"
        )
        icov = np.atleast_1d(np.loadtxt(out_dir / "icov_flat.txt")).reshape(
            (nlam, d, d), order="F"
        )
        return {"lambda": lam, "sparsity": sparsity, "path": path, "icov": icov}


def run_r_glasso_reference(
    x: np.ndarray,
    lambda_gl: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    """Run R huge reference for glasso + ebic on fixed input."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_gl_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        lam_path = out_dir / "lam.txt"
        np.savetxt(x_path, x, delimiter=",")
        np.savetxt(lam_path, lambda_gl)

        _run_r_script(
            """
            args <- commandArgs(trailingOnly = TRUE)
            x <- as.matrix(read.csv(args[1], header=FALSE))
            lam <- as.numeric(scan(args[2], quiet=TRUE))
            out_dir <- args[3]

            suppressMessages(library(huge))
            fit <- huge(x, method='glasso', lambda=lam, verbose=FALSE)
            sel <- huge.select(fit, criterion='ebic', verbose=FALSE)
            """ + _R_FIT_SELECT_OUTPUT,
            [str(x_path), str(lam_path), str(out_dir)],
        )

        return _load_sel_results(out_dir)


def run_r_inference_reference(
    x: np.ndarray,
    t_mat: np.ndarray,
    adj: np.ndarray,
    *,
    alpha: float = 0.05,
    type_: str = "Gaussian",
    method: str = "score",
) -> dict[str, np.ndarray | float]:
    """Run R ``huge.inference`` on fixed matrices."""

    with tempfile.TemporaryDirectory(prefix="pyhuge_rref_inference_") as td:
        out_dir = Path(td)
        x_path = out_dir / "x.csv"
        t_path = out_dir / "t.csv"
        adj_path = out_dir / "adj.csv"
        np.savetxt(x_path, x, delimiter=",")
        np.savetxt(t_path, t_mat, delimiter=",")
        np.savetxt(adj_path, adj, delimiter=",")

        _run_r_script(
            """
            args <- commandArgs(trailingOnly = TRUE)
            x <- as.matrix(read.csv(args[1], header=FALSE))
            T <- as.matrix(read.csv(args[2], header=FALSE))
            adj <- as.matrix(read.csv(args[3], header=FALSE))
            out_dir <- args[4]
            alpha <- as.numeric(args[5])
            type <- args[6]
            method <- args[7]

            suppressMessages(library(huge))
            out <- huge.inference(x, T, adj, alpha=alpha, type=type, method=method)
            write.table(out$p, file.path(out_dir, 'p.csv'), sep=',',
                        row.names=FALSE, col.names=FALSE)
            write.table(out$data, file.path(out_dir, 'data.csv'), sep=',',
                        row.names=FALSE, col.names=FALSE)
            write.table(out$error, file.path(out_dir, 'error.txt'),
                        row.names=FALSE, col.names=FALSE)
            """,
            [
                str(x_path),
                str(t_path),
                str(adj_path),
                str(out_dir),
                str(alpha),
                type_,
                method,
            ],
        )

        return {
            "p": np.loadtxt(out_dir / "p.csv", delimiter=","),
            "data": np.loadtxt(out_dir / "data.csv", delimiter=","),
            "error": float(np.loadtxt(out_dir / "error.txt")),
        }
