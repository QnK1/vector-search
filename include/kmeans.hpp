#pragma once

#include "dataset.hpp"
#include "metrics.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace vs {

struct KMeansResult {
    std::vector<float> centroids;
    std::vector<std::uint32_t> assignments;
};

template <DistanceMetric Metric = L2Metric>
KMeansResult fit_kmeans(const Dataset& ds, std::uint32_t k, std::uint32_t max_iters = 25, Metric metric = {}) {
    KMeansResult res;
    if (ds.count == 0 || ds.dim == 0 || k == 0) {
        return res;
    }

    k = std::min(k, ds.count);
    res.centroids.resize(static_cast<std::size_t>(k) * ds.dim);
    res.assignments.assign(ds.count, 0);

    // Pick initial centroids randomly from dataset
    std::vector<std::uint32_t> indices(ds.count);
    for (std::uint32_t i = 0; i < ds.count; ++i) {
        indices[i] = i;
    }
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (std::uint32_t c = 0; c < k; ++c) {
        auto src = ds.row(indices[c]);
        std::copy(src.begin(), src.end(), res.centroids.begin() + static_cast<std::size_t>(c) * ds.dim);
    }

    std::vector<float> new_centroids(res.centroids.size(), 0.0f);
    std::vector<std::uint32_t> counts(k, 0);

    for (std::uint32_t iter = 0; iter < max_iters; ++iter) {
        bool changed = false;

        // Assign points to nearest centroid
        for (std::uint32_t i = 0; i < ds.count; ++i) {
            auto vec = ds.row(i);
            float best_dist = std::numeric_limits<float>::max();
            std::uint32_t best_c = 0;

            for (std::uint32_t c = 0; c < k; ++c) {
                std::span<const float> cent(res.centroids.data() + static_cast<std::size_t>(c) * ds.dim, ds.dim);
                float d = metric(vec, cent);
                if (d < best_dist) {
                    best_dist = d;
                    best_c = c;
                }
            }

            if (res.assignments[i] != best_c || iter == 0) {
                res.assignments[i] = best_c;
                changed = true;
            }
        }

        if (!changed) {
            break;
        }

        // Recompute centroids
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (std::uint32_t i = 0; i < ds.count; ++i) {
            std::uint32_t c = res.assignments[i];
            counts[c]++;
            auto vec = ds.row(i);
            float* dst = new_centroids.data() + static_cast<std::size_t>(c) * ds.dim;
            for (std::uint32_t d = 0; d < ds.dim; ++d) {
                dst[d] += vec[d];
            }
        }

        for (std::uint32_t c = 0; c < k; ++c) {
            float* dst = res.centroids.data() + static_cast<std::size_t>(c) * ds.dim;
            if (counts[c] > 0) {
                const float* src = new_centroids.data() + static_cast<std::size_t>(c) * ds.dim;
                float inv = 1.0f / static_cast<float>(counts[c]);
                for (std::uint32_t d = 0; d < ds.dim; ++d) {
                    dst[d] = src[d] * inv;
                }
            }
        }
    }

    return res;
}

} // namespace vs
