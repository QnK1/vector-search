#pragma once

#include <cmath>
#include <concepts>
#include <span>

namespace vs {

template <typename M>
concept DistanceMetric = requires(M m, std::span<const float> a, std::span<const float> b) {
    { m(a, b) } -> std::convertible_to<float>;
};

struct L2Metric {
    float operator()(std::span<const float> a, std::span<const float> b) const noexcept {
        float sum = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            float d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }
};

struct CosineMetric {
    float operator()(std::span<const float> a, std::span<const float> b) const noexcept {
        float dot = 0.0f;
        float na = 0.0f;
        float nb = 0.0f;
        for (std::size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if (na == 0.0f || nb == 0.0f) {
            return 1.0f;
        }
        return 1.0f - (dot / (std::sqrt(na) * std::sqrt(nb)));
    }
};

static_assert(DistanceMetric<L2Metric>);
static_assert(DistanceMetric<CosineMetric>);

} // namespace vs
