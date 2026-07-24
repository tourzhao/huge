// pybind11 thin wrapper — delegates to huge::* core functions
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "huge/huge_core.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace py = pybind11;

// Helper: copy row-major numpy 2D array to column-major vector
static std::vector<double> numpy_to_colmajor(py::array_t<double, py::array::c_style | py::array::forcecast> arr,
                                              int& rows, int& cols) {
    auto a = arr.unchecked<2>();
    rows = static_cast<int>(a.shape(0));
    cols = static_cast<int>(a.shape(1));
    std::vector<double> out(static_cast<size_t>(rows) * cols);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out[static_cast<size_t>(j) * rows + i] = a(i, j);
    return out;
}

// ---- Glasso binding ----

static py::dict py_hugeglasso(py::array_t<double, py::array::c_style | py::array::forcecast> s,
                               py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                               bool scr, bool cov_output) {
    int d, d2;
    auto S = numpy_to_colmajor(s, d, d2);
    if (d != d2) throw std::runtime_error("S must be square.");
    auto lam = lambdas.unchecked<1>();
    int nlambda = static_cast<int>(lam.shape(0));
    std::vector<double> lam_vec(nlambda);
    for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);

    huge::GlassoResult res;
    {
        // Inputs/outputs are plain C++ buffers: safe to release the GIL so
        // thread pools (e.g. huge_select stars n_jobs) can overlap fits.
        py::gil_scoped_release release;
        res = huge::glasso_compact(
            S.data(), d, lam_vec.data(), nlambda, scr, cov_output);
    }

    // Convert results
    auto loglik = py::array_t<double>(nlambda);
    auto sparsity = py::array_t<double>(nlambda);
    auto df = py::array_t<double>(nlambda);
    auto path = py::array_t<uint8_t>({nlambda, d, d});
    auto icov_out = py::array_t<double>({nlambda, d, d});

    auto LL = loglik.mutable_unchecked<1>();
    auto SP = sparsity.mutable_unchecked<1>();
    auto DF = df.mutable_unchecked<1>();
    auto P = path.mutable_unchecked<3>();
    auto I = icov_out.mutable_unchecked<3>();

    for (int k = 0; k < nlambda; k++) {
        LL(k) = res.loglik[k];
        SP(k) = res.sparsity[k];
        DF(k) = static_cast<double>(res.df[k]);
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) {
                P(k, i, j) =
                    (i != j && res.icov[k](i, j) != 0.0) ? 1 : 0;
                I(k, i, j) = res.icov[k](i, j);
            }
    }

    py::dict out;
    out["hit_max_iter"] = res.hit_max_iter;
    out["path"] = path;
    out["icov"] = icov_out;
    out["loglik"] = loglik;
    out["sparsity"] = sparsity;
    out["df"] = df;
    if (cov_output) {
        auto cov = py::array_t<double>({nlambda, d, d});
        auto C = cov.mutable_unchecked<3>();
        for (int k = 0; k < nlambda; k++)
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++)
                    C(k, i, j) = res.cov[k](i, j);
        out["cov"] = cov;
    } else {
        out["cov"] = py::none();
    }
    return out;
}

// Helper: convert ColResult columns to dense beta[nlambda,d,d] + df[d,nlambda]
static std::pair<py::array_t<double>, py::array_t<double>>
columns_to_dense(const std::vector<huge::ColResult>& columns, int d, int nlambda) {
    if (d <= 0 || nlambda <= 0 ||
        columns.size() != static_cast<size_t>(d))
        throw std::runtime_error(
            "native column output has invalid dimensions.");
    auto beta = py::array_t<double>({nlambda, d, d});
    auto df = py::array_t<double>({d, nlambda});
    std::fill_n(static_cast<double*>(beta.request().ptr), static_cast<size_t>(nlambda) * d * d, 0.0);
    std::fill_n(static_cast<double*>(df.request().ptr), static_cast<size_t>(d) * nlambda, 0.0);
    auto B = beta.mutable_unchecked<3>();
    auto DF = df.mutable_unchecked<2>();

    for (int m = 0; m < d; m++) {
        const auto& col = columns[m];
        if (col.indices.size() != col.vals.size())
            throw std::runtime_error(
                "native column indices and values have inconsistent lengths.");
        for (size_t j = 0; j < col.vals.size(); j++) {
            int encoded = col.indices[j];
            if (encoded < 0 ||
                static_cast<int64_t>(encoded) >=
                    static_cast<int64_t>(nlambda) * d)
                throw std::runtime_error(
                    "native column index is outside the requested path.");
            B(encoded / d, m, encoded % d) = col.vals[j];
        }
        for (int i = 0; i < nlambda; i++) {
            int nnz = 0;
            for (int j = 0; j < d; j++)
                if (std::fabs(B(i, m, j)) > 0.0) nnz++;
            DF(m, i) = static_cast<double>(nnz);
        }
    }
    return {beta, df};
}

// Convert sparse core columns to concatenated CSC supports, one matrix per
// lambda. support_indptr[k, ] is local to support_indices' kth contiguous
// slice; the slices are concatenated in lambda order. The CSC orientation is
// [predictor, response], the transpose of dense beta[k, response, predictor].
static std::tuple<
    py::array_t<int64_t>, py::array_t<int32_t>, py::array_t<double>>
columns_to_support(const std::vector<huge::ColResult>& columns, int d,
                   int nlambda) {
    if (d <= 0 || nlambda <= 0 ||
        columns.size() != static_cast<size_t>(d))
        throw std::runtime_error(
            "native column output has invalid dimensions.");
    if (static_cast<size_t>(nlambda) >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(d))
        throw std::overflow_error(
            "native sparse support dimensions overflowed.");
    const size_t count_size = static_cast<size_t>(nlambda) * d;
    std::vector<size_t> counts(count_size, 0);

    for (int m = 0; m < d; m++) {
        const auto& col = columns[m];
        if (col.indices.size() != col.vals.size())
            throw std::runtime_error(
                "native column indices and values have inconsistent lengths.");
        for (size_t j = 0; j < col.vals.size(); j++) {
            if (col.vals[j] == 0.0) continue;
            int encoded = col.indices[j];
            if (encoded < 0 ||
                static_cast<int64_t>(encoded) >=
                    static_cast<int64_t>(nlambda) * d)
                throw std::runtime_error(
                    "native column index is outside the requested path.");
            int lambda_index = encoded / d;
            counts[static_cast<size_t>(lambda_index) * d + m]++;
        }
    }

    auto support_indptr = py::array_t<int64_t>({nlambda, d + 1});
    auto df = py::array_t<double>({d, nlambda});
    auto indptr = support_indptr.mutable_unchecked<2>();
    auto DF = df.mutable_unchecked<2>();
    std::vector<size_t> path_offsets(static_cast<size_t>(nlambda) + 1, 0);
    std::vector<size_t> cursors(count_size, 0);

    for (int k = 0; k < nlambda; k++) {
        size_t local = 0;
        indptr(k, 0) = 0;
        for (int m = 0; m < d; m++) {
            size_t count = counts[static_cast<size_t>(k) * d + m];
            if (count > static_cast<size_t>(
                    std::numeric_limits<int64_t>::max()) - local)
                throw std::overflow_error(
                    "native sparse support exceeds int64 capacity.");
            cursors[static_cast<size_t>(k) * d + m] =
                path_offsets[static_cast<size_t>(k)] + local;
            local += count;
            indptr(k, m + 1) = static_cast<int64_t>(local);
            DF(m, k) = static_cast<double>(count);
        }
        if (local > std::numeric_limits<size_t>::max() -
                        path_offsets[static_cast<size_t>(k)])
            throw std::overflow_error(
                "native sparse support size overflowed.");
        path_offsets[static_cast<size_t>(k) + 1] =
            path_offsets[static_cast<size_t>(k)] + local;
    }

    size_t total = path_offsets.back();
    if (total > static_cast<size_t>(
            std::numeric_limits<py::ssize_t>::max()))
        throw std::overflow_error(
            "native sparse support exceeds Python array capacity.");
    auto support_indices =
        py::array_t<int32_t>(static_cast<py::ssize_t>(total));
    auto indices = support_indices.mutable_unchecked<1>();

    for (int m = 0; m < d; m++) {
        const auto& col = columns[m];
        for (size_t j = 0; j < col.vals.size(); j++) {
            if (col.vals[j] == 0.0) continue;
            int encoded = col.indices[j];
            int lambda_index = encoded / d;
            size_t cursor_index =
                static_cast<size_t>(lambda_index) * d + m;
            size_t destination = cursors[cursor_index]++;
            indices(static_cast<py::ssize_t>(destination)) =
                static_cast<int32_t>(encoded % d);
        }
    }
    return {support_indptr, support_indices, df};
}

static void append_column_output(py::dict& out,
                                 const std::vector<huge::ColResult>& columns,
                                 int d, int nlambda, bool dense_output) {
    if (dense_output) {
        auto [beta, df] = columns_to_dense(columns, d, nlambda);
        out["beta"] = beta;
        out["df"] = df;
        return;
    }
    auto [support_indptr, support_indices, df] =
        columns_to_support(columns, d, nlambda);
    out["beta"] = py::none();
    out["support_indptr"] = support_indptr;
    out["support_indices"] = support_indices;
    out["df"] = df;
}

// ---- MB binding ----

static py::dict py_spmb_graph(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                               py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                               bool dense_output) {
    int d, d2;
    auto S = numpy_to_colmajor(corr, d, d2);
    if (d != d2) throw std::runtime_error("corr must be square.");
    auto lam = lambdas.unchecked<1>();
    int nlambda = static_cast<int>(lam.shape(0));
    std::vector<double> lam_vec(nlambda);
    for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);

    huge::MBResult res;
    {
        py::gil_scoped_release release;  // plain C++ buffers only
        res = huge::mb(S.data(), d, lam_vec.data(), nlambda);
    }
    py::dict out;
    out["hit_max_iter"] = res.hit_max_iter;
    append_column_output(out, res.columns, d, nlambda, dense_output);
    return out;
}

// ---- MB with screening binding ----

static py::dict py_spmb_scr(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                              py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                              py::array_t<int, py::array::c_style | py::array::forcecast> idx_scr_arr,
                              bool dense_output) {
    int d, d2;
    auto S = numpy_to_colmajor(corr, d, d2);
    if (d != d2) throw std::runtime_error("corr must be square.");
    auto lam = lambdas.unchecked<1>();
    int nlambda = static_cast<int>(lam.shape(0));
    std::vector<double> lam_vec(nlambda);
    for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);

    if (idx_scr_arr.ndim() != 2)
        throw std::invalid_argument("idx_scr must be a two-dimensional array.");
    if (idx_scr_arr.shape(1) != d)
        throw std::invalid_argument("idx_scr must have exactly d columns.");
    int nscr = static_cast<int>(idx_scr_arr.shape(0));
    if (nscr <= 0 || nscr >= d)
        throw std::invalid_argument(
            "idx_scr must have between 1 and d - 1 rows.");
    auto idx = idx_scr_arr.unchecked<2>();
    std::vector<int> idx_scr_cm(static_cast<size_t>(nscr) * d);
    for (int m = 0; m < d; m++)
        for (int j = 0; j < nscr; j++)
            idx_scr_cm[static_cast<size_t>(m) * nscr + j] = idx(j, m);

    huge::MBResult res;
    {
        py::gil_scoped_release release;  // plain C++ buffers only
        res = huge::mb_scr(S.data(), d, lam_vec.data(), nlambda, idx_scr_cm.data(), nscr);
    }
    py::dict out;
    out["hit_max_iter"] = res.hit_max_iter;
    append_column_output(out, res.columns, d, nlambda, dense_output);
    return out;
}

// ---- TIGER binding ----

static py::dict py_spmb_graphsqrt(
        py::array_t<double, py::array::c_style | py::array::forcecast> data,
        py::object lambdas_obj, int nlambda, double lambda_min_ratio,
        bool covariance_input, bool dense_output) {
    int n, d;
    auto X = numpy_to_colmajor(data, n, d);
    std::vector<double> lam_vec;
    const double* lambda_ptr = nullptr;
    if (!lambdas_obj.is_none()) {
        auto lambdas = py::array_t<double,
            py::array::c_style | py::array::forcecast>::ensure(lambdas_obj);
        if (!lambdas || lambdas.ndim() != 1)
            throw std::runtime_error("lambdas must be a one-dimensional array.");
        auto lam = lambdas.unchecked<1>();
        nlambda = static_cast<int>(lam.shape(0));
        lam_vec.resize(nlambda);
        for (int i = 0; i < nlambda; i++) lam_vec[i] = lam(i);
        lambda_ptr = lam_vec.data();
    }

    huge::TigerResult res;
    {
        py::gil_scoped_release release;  // plain C++ buffers only
        res = huge::tiger_fit(X.data(), n, d, covariance_input, lambda_ptr,
                              nlambda, lambda_min_ratio);
    }
    nlambda = static_cast<int>(res.lambda.size());

    auto icov = py::array_t<double>({nlambda, d, d});
    auto I = icov.mutable_unchecked<3>();
    for (int k = 0; k < nlambda; k++)
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
                I(k, i, j) = res.icov[k](i, j);

    py::dict out;
    out["hit_max_iter"] = res.hit_max_iter;
    append_column_output(out, res.columns, d, nlambda, dense_output);
    out["icov"] = icov;
    out["lambda"] = py::cast(res.lambda);
    out["path_truncated"] = res.path_truncated;
    return out;
}

// ---- Threshold path (pure Python-side, no core needed) ----

static py::list py_threshold_path(py::array_t<double, py::array::c_style | py::array::forcecast> corr,
                                   py::array_t<double, py::array::c_style | py::array::forcecast> lambdas) {
    if (corr.ndim() != 2 || corr.shape(0) <= 0 ||
        corr.shape(0) != corr.shape(1))
        throw std::invalid_argument(
            "corr must be a non-empty square matrix.");
    if (lambdas.ndim() != 1)
        throw std::invalid_argument(
            "threshold lambda must be one-dimensional.");
    if (lambdas.shape(0) <= 0)
        throw std::invalid_argument(
            "threshold lambda must contain at least one value.");

    auto c = corr.unchecked<2>();
    auto l = lambdas.unchecked<1>();
    for (py::ssize_t k = 0; k < l.shape(0); k++) {
        if (!std::isfinite(l(k)) || l(k) < 0.0)
            throw std::invalid_argument(
                "threshold lambda values must be finite and non-negative.");
    }
    const py::ssize_t d = c.shape(0);
    py::list out;
    for (py::ssize_t k = 0; k < l.shape(0); k++) {
        double lam = l(k);
        auto mat = py::array_t<uint8_t>({d, d});
        auto m = mat.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < d; i++)
            for (py::ssize_t j = 0; j < d; j++)
                m(i, j) = (i != j && std::fabs(c(i, j)) > lam) ? 1 : 0;
        out.append(mat);
    }
    return out;
}

static py::array_t<double> py_sparsity_path(py::list matrices) {
    using MatrixArray = py::array_t<
        uint8_t, py::array::c_style | py::array::forcecast>;

    const py::ssize_t n = py::len(matrices);
    auto out = py::array_t<double>(n);
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t k = 0; k < n; k++) {
        auto raw = py::array::ensure(matrices[k]);
        if (!raw)
            throw std::invalid_argument(
                "`matrices` elements must be convertible to NumPy arrays.");
        if (raw.ndim() != 2)
            throw std::invalid_argument(
                "`matrices` elements must be two-dimensional.");
        if (raw.shape(0) <= 0 || raw.shape(0) != raw.shape(1))
            throw std::invalid_argument(
                "`matrices` elements must be non-empty square matrices.");

        auto mat = MatrixArray::ensure(raw);
        if (!mat)
            throw std::invalid_argument(
                "`matrices` elements must be convertible to uint8 NumPy arrays.");
        auto m = mat.unchecked<2>();
        const py::ssize_t d = m.shape(0);
        double edges = 0;
        for (py::ssize_t i = 0; i < d; i++)
            for (py::ssize_t j = i + 1; j < d; j++)
                if (m(i, j) != 0) edges += 1.0;
        o(k) = (d <= 1) ? 0.0 : 2.0 * edges / (static_cast<double>(d) * (d - 1));
    }
    return out;
}

// ---- RIC binding ----

static double py_ric(py::object x_obj, py::object r_obj) {
    using DoubleArray = py::array_t<
        double, py::array::c_style | py::array::forcecast>;

    auto x_raw = py::array::ensure(x_obj);
    if (!x_raw)
        throw std::invalid_argument(
            "RIC x must be convertible to a NumPy array.");
    if (x_raw.ndim() != 2)
        throw std::invalid_argument("RIC x must be two-dimensional.");
    if (x_raw.shape(0) <= 0 || x_raw.shape(1) <= 0)
        throw std::invalid_argument("RIC x must be a non-empty matrix.");
    if (x_raw.shape(0) > std::numeric_limits<int>::max() ||
            x_raw.shape(1) > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "RIC x dimensions exceed the C++ integer range.");

    auto x = DoubleArray::ensure(x_raw);
    if (!x)
        throw std::invalid_argument(
            "RIC x must be convertible to a numeric NumPy array.");

    auto r_raw = py::array::ensure(r_obj);
    if (!r_raw)
        throw std::invalid_argument(
            "RIC rotations must be convertible to a NumPy array.");
    if (r_raw.ndim() != 1)
        throw std::invalid_argument(
            "RIC rotations must be one-dimensional.");
    if (r_raw.shape(0) <= 0)
        throw std::invalid_argument(
            "RIC rotations must be non-empty.");
    if (r_raw.shape(0) > std::numeric_limits<int>::max())
        throw std::invalid_argument(
            "RIC rotation count exceeds the C++ integer range.");

    auto r = DoubleArray::ensure(r_raw);
    if (!r)
        throw std::invalid_argument(
            "RIC rotations must be convertible to a numeric NumPy array.");

    auto X = x.unchecked<2>();
    auto R = r.unchecked<1>();
    int n_rows = static_cast<int>(X.shape(0));
    int d = static_cast<int>(X.shape(1));
    int t = static_cast<int>(R.shape(0));

    // Convert to column-major
    std::vector<double> X_cm(static_cast<size_t>(n_rows) * d);
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < d; j++) {
            double value = X(i, j);
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "RIC x must contain only finite values.");
            X_cm[static_cast<size_t>(j) * n_rows + i] = value;
        }
    }

    std::vector<int> r_vec(t);
    for (int i = 0; i < t; i++) {
        double value = R(i);
        if (!std::isfinite(value) || value != std::floor(value))
            throw std::invalid_argument(
                "RIC rotations must contain finite integer values.");
        if (value < 0.0 || value > static_cast<double>(n_rows))
            throw std::invalid_argument(
                "RIC rotations must lie in [0, n].");
        r_vec[i] = static_cast<int>(value);
    }

    double result;
    {
        py::gil_scoped_release release;  // plain C++ buffers only
        result = huge::ric(X_cm.data(), n_rows, d, r_vec.data(), t);
    }
    if (!std::isfinite(result))
        throw std::runtime_error("RIC returned a non-finite result.");
    return result;
}

// ---- SFGen binding ----

static py::array_t<uint8_t> py_sfgen(int d0, int d, py::object seed_obj) {
    if (d <= 0 || d0 <= 0 || d0 > d) throw std::runtime_error("Invalid input: require 0 < d0 <= d.");

    int nrand = d - d0;
    std::vector<double> rands(nrand > 0 ? nrand : 0);

    std::mt19937_64 rng;
    if (seed_obj.is_none()) {
        std::random_device rd;
        rng.seed(static_cast<uint64_t>(rd()));
    } else {
        rng.seed(static_cast<uint64_t>(seed_obj.cast<unsigned long long>()));
    }
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    for (int i = 0; i < nrand; i++) rands[i] = unif(rng);

    std::vector<int> G(static_cast<size_t>(d) * d, 0);
    huge::sfgen(d0, d, G.data(), rands.data());

    // Convert to uint8 numpy
    auto out = py::array_t<uint8_t>({d, d});
    auto o = out.mutable_unchecked<2>();
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            o(i, j) = static_cast<uint8_t>(G[static_cast<size_t>(j) * d + i]);
    return out;
}

// ---- Module definition ----

PYBIND11_MODULE(_native_core, m) {
    m.doc() = "pyhuge native C++ kernels (shared core with R package)";
    m.def("threshold_path", &py_threshold_path, py::arg("corr"), py::arg("lambdas"),
          "Build adjacency path by correlation thresholding.");
    m.def("sparsity_path", &py_sparsity_path, py::arg("matrices"),
          "Compute sparsity sequence from adjacency matrices.");
    m.def("spmb_graph", &py_spmb_graph, py::arg("corr"),
          py::arg("lambdas"), py::arg("dense_output") = true,
          "MB graph path core (lossless screening).");
    m.def("spmb_scr", &py_spmb_scr, py::arg("corr"),
          py::arg("lambdas"), py::arg("idx_scr"),
          py::arg("dense_output") = true,
          "MB graph path core (lossy screening).");
    m.def("spmb_graphsqrt", &py_spmb_graphsqrt,
          py::arg("data"), py::arg("lambdas") = py::none(),
          py::arg("nlambda") = 10, py::arg("lambda_min_ratio") = 0.1,
          py::arg("covariance_input") = false,
          py::arg("dense_output") = true,
          "Correlation-domain TIGER graph path core.");
    m.def("hugeglasso", &py_hugeglasso, py::arg("s"), py::arg("lambdas"), py::arg("scr") = false,
          py::arg("cov_output") = false, "Graphical lasso path core.");
    m.def("ric", &py_ric, py::arg("x"), py::arg("r"),
          "Rotation information criterion core.");
    m.def("sfgen", &py_sfgen, py::arg("d0"), py::arg("d"), py::arg("seed") = py::none(),
          "Scale-free graph generator core.");
    m.def("omp_max_threads", []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    }, "OpenMP thread count the core will use (1 = serial build).");
}
