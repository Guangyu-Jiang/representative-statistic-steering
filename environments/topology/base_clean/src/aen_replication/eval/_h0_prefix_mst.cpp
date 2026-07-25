#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" int h0_prefix_means(const float* points, int n_points, int n_dims, float* out) {
    if (points == nullptr || out == nullptr || n_dims <= 0) {
        return 1;
    }
    if (n_points <= 1) {
        for (int d = 0; d < n_dims; ++d) {
            out[d] = 0.0f;
        }
        return 0;
    }

    const int n = n_points;
    const int total = n * n;
    std::vector<float> sqdist(total, 0.0f);
    std::vector<float> best(n, 0.0f);
    std::vector<uint8_t> used(n, 0);

    for (int dim = 0; dim < n_dims; ++dim) {
        for (int i = 0; i < n; ++i) {
            const float xi = points[i * n_dims + dim];
            for (int j = i + 1; j < n; ++j) {
                const float diff = xi - points[j * n_dims + dim];
                const float updated = sqdist[i * n + j] + diff * diff;
                sqdist[i * n + j] = updated;
                sqdist[j * n + i] = updated;
            }
        }

        for (int i = 0; i < n; ++i) {
            best[i] = std::numeric_limits<float>::infinity();
            used[i] = 0;
        }
        best[0] = 0.0f;
        double total_weight = 0.0;

        for (int step = 0; step < n; ++step) {
            int v = -1;
            float v_best = std::numeric_limits<float>::infinity();
            for (int i = 0; i < n; ++i) {
                if (!used[i] && best[i] < v_best) {
                    v = i;
                    v_best = best[i];
                }
            }
            if (v < 0) {
                break;
            }
            used[v] = 1;
            if (step > 0 && std::isfinite(v_best) && v_best > 0.0f) {
                total_weight += std::sqrt(static_cast<double>(v_best));
            }
            for (int u = 0; u < n; ++u) {
                const float candidate = sqdist[v * n + u];
                if (!used[u] && candidate < best[u]) {
                    best[u] = candidate;
                }
            }
        }

        out[dim] = static_cast<float>(total_weight / static_cast<double>(n - 1));
    }

    return 0;
}
