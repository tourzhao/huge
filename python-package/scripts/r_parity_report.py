#!/usr/bin/env python3
"""Generate a native-vs-R parity report for pyhuge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhuge import huge, huge_select  # noqa: E402
from pyhuge.parity import has_r_huge, run_r_ct_reference, run_r_glasso_reference  # noqa: E402


def _edge_counts(path) -> np.ndarray:
    return np.asarray([np.count_nonzero(np.triu(m.toarray() != 0, 1)) for m in path], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build native-vs-R parity report")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--d", type=int, default=20)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    report: dict[str, object] = {
        "config": {"seed": args.seed, "n": args.n, "d": args.d},
        "environment": {
            "has_r_huge": has_r_huge(),
        },
    }

    if not report["environment"]["has_r_huge"]:
        report["error"] = "R with package huge is not available."
        text = json.dumps(report, indent=2)
        print(text)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        return 2

    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.n, args.d))

    lam_ct = np.linspace(0.6, 0.05, 8)
    r_ct = run_r_ct_reference(x, lam_ct, rep_num=6, stars_thresh=0.1, seed=args.seed)
    n_ct = huge(x, method="ct", lambda_=lam_ct, verbose=False)
    s_ct = huge_select(n_ct, criterion="stars", rep_num=6, stars_thresh=0.1, verbose=False)

    ct_edges_r = np.asarray(r_ct["edges"], dtype=float)
    ct_edges_n = _edge_counts(n_ct.path)
    max_edges = args.d * (args.d - 1) / 2.0
    report["ct"] = {
        "lambda_max_abs": float(np.max(np.abs(n_ct.lambda_path - r_ct["lambda"]))),
        "sparsity_max_abs": float(np.max(np.abs(n_ct.sparsity - r_ct["sparsity"]))),
        "edge_mean_abs": float(np.mean(np.abs(ct_edges_n - ct_edges_r))),
        "edge_mean_abs_norm": float(np.mean(np.abs(ct_edges_n - ct_edges_r)) / max_edges),
        "opt_index_native": int(s_ct.opt_index or 1),
        "opt_index_r": int(r_ct["opt_index"]),
        "opt_lambda_native": float(s_ct.opt_lambda),
        "opt_lambda_r": float(r_ct["opt_lambda"]),
    }

    lam_gl = np.geomspace(0.5, 0.05, 8)
    r_gl = run_r_glasso_reference(x, lam_gl)
    n_gl = huge(x, method="glasso", lambda_=lam_gl, verbose=False)
    s_gl = huge_select(n_gl, criterion="ebic", verbose=False)

    gl_edges_r = np.asarray(r_gl["edges"], dtype=float)
    gl_edges_n = _edge_counts(n_gl.path)
    report["glasso"] = {
        "lambda_max_abs": float(np.max(np.abs(n_gl.lambda_path - r_gl["lambda"]))),
        "sparsity_mean_abs": float(np.mean(np.abs(n_gl.sparsity - r_gl["sparsity"]))),
        "edge_mean_abs_norm": float(np.mean(np.abs(gl_edges_n - gl_edges_r)) / max_edges),
        "opt_index_native": int(s_gl.opt_index or 1),
        "opt_index_r": int(r_gl["opt_index"]),
        "opt_lambda_native": float(s_gl.opt_lambda),
        "opt_lambda_r": float(r_gl["opt_lambda"]),
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
