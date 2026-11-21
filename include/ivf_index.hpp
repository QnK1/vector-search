#pragma once

#include "dataset.hpp"
#include "metrics.hpp"
#include <algorithm>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace vs {

enum class MetricType : std::uint32_t { L2 = 0, Cosine = 1 };

#pragma pack(push, 1)
struct IndexHeader {
    std::uint32_t magic = 0x49564631; // "IVF1"
    std::uint32_t dim = 0;
    std::uint32_t num_vectors = 0;
    std::uint32_t num_clusters = 0;
    std::uint32_t metric_type = 0;
};

struct ListMeta {
    std::uint64_t offset = 0;
    std::uint32_t count = 0;
};
#pragma pack(pop)

struct SearchResult {
    std::uint32_t id;
    float distance;
};

std::expected<void, std::string> build_ivf_index(const Dataset &ds,
                                                 const std::filesystem::path &index_path,
                                                 std::uint32_t num_clusters,
                                                 MetricType metric_type = MetricType::L2,
                                                 std::uint32_t max_iters = 25);

// Memory-mapped IVF index (read-only)
class MappedIvfIndex {
  public:
    static std::expected<MappedIvfIndex, std::string> open(const std::filesystem::path &path);

    const IndexHeader &header() const noexcept { return *header_; }

    std::span<const float> centroid(std::size_t cluster_id) const noexcept {
        return {centroids_.data() + cluster_id * header_->dim, header_->dim};
    }

    std::span<const std::uint32_t> cluster_ids(std::size_t cluster_id) const noexcept {
        const auto &meta = lists_[cluster_id];
        return {ids_.data() + meta.offset, meta.count};
    }

    std::span<const float> cluster_vector(std::size_t cluster_id,
                                          std::size_t pos_in_cluster) const noexcept {
        const auto &meta = lists_[cluster_id];
        std::size_t vec_idx = meta.offset + pos_in_cluster;
        return {vectors_.data() + vec_idx * header_->dim, header_->dim};
    }

    template <DistanceMetric Metric>
    std::expected<std::vector<SearchResult>, std::string>
    search(std::span<const float> query, std::uint32_t top_k, std::uint32_t nprobe,
           Metric metric = {}) const {
        if (query.size() != header_->dim) {
            return std::unexpected("Query dimension mismatch");
        }
        if (top_k == 0 || header_->num_vectors == 0) {
            return std::vector<SearchResult>{};
        }

        nprobe = std::clamp(nprobe, 1u, header_->num_clusters);

        // Find nprobe nearest centroids
        std::vector<SearchResult> cluster_dists(header_->num_clusters);
        for (std::uint32_t c = 0; c < header_->num_clusters; ++c) {
            cluster_dists[c] = {c, metric(query, centroid(c))};
        }
        std::partial_sort(
            cluster_dists.begin(), cluster_dists.begin() + nprobe, cluster_dists.end(),
            [](const SearchResult &a, const SearchResult &b) { return a.distance < b.distance; });

        // Scan inverted lists of selected clusters
        std::vector<SearchResult> candidates;
        for (std::uint32_t i = 0; i < nprobe; ++i) {
            std::uint32_t c = cluster_dists[i].id;
            auto c_ids = cluster_ids(c);
            for (std::size_t j = 0; j < c_ids.size(); ++j) {
                float dist = metric(query, cluster_vector(c, j));
                candidates.push_back({c_ids[j], dist});
            }
        }

        // Keep top_k results
        std::size_t k = std::min<std::size_t>(top_k, candidates.size());
        std::partial_sort(
            candidates.begin(), candidates.begin() + k, candidates.end(),
            [](const SearchResult &a, const SearchResult &b) { return a.distance < b.distance; });
        candidates.resize(k);
        return candidates;
    }

    std::expected<std::vector<SearchResult>, std::string>
    search_auto(std::span<const float> query, std::uint32_t top_k, std::uint32_t nprobe) const;

  private:
    boost::interprocess::file_mapping mapping_;
    boost::interprocess::mapped_region region_;

    const IndexHeader *header_ = nullptr;
    std::span<const float> centroids_;
    std::span<const ListMeta> lists_;
    std::span<const std::uint32_t> ids_;
    std::span<const float> vectors_;
};

} // namespace vs
