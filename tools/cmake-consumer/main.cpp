#include <huge/huge_core.h>

#include <cmath>
#include <iostream>

int main() {
    const double observations[] = {
        -1.0, 0.0, 1.0, 2.0,
        2.0, -1.0, 0.0, 1.0
    };
    const int rotations[] = {1};

    const double score = huge::ric(observations, 4, 2, rotations, 1);
    if (!std::isfinite(score) || score < 0.0) {
        return 1;
    }

    // Keep the original standalone C++ glasso contract while verifying that
    // language adapters can opt out of the redundant dense path matrices.
    const double covariance[] = {
        1.0, 0.5,
        0.5, 1.0
    };
    const double lambda[] = {0.1};
    const auto full = huge::glasso(covariance, 2, lambda, 1, false, true);
    const auto compact =
        huge::glasso_compact(covariance, 2, lambda, 1, false, true);
    if (full.path.size() != 1 || !compact.path.empty() ||
        full.icov.size() != 1 || compact.icov.size() != 1 ||
        full.cov.size() != 1 || compact.cov.size() != 1 ||
        full.loglik != compact.loglik || full.sparsity != compact.sparsity ||
        full.df != compact.df || full.hit_max_iter != compact.hit_max_iter) {
        return 2;
    }
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            if (full.icov[0](i, j) != compact.icov[0](i, j) ||
                full.cov[0](i, j) != compact.cov[0](i, j)) {
                return 3;
            }
            const double expected =
                (i != j && compact.icov[0](i, j) != 0.0) ? 1.0 : 0.0;
            if (full.path[0](i, j) != expected) {
                return 4;
            }
        }
    }

    std::cout << "huge CMake consumer smoke passed\n";
    return 0;
}
