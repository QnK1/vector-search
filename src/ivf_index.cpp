#include "ivf_index.hpp"
#include "kmeans.hpp"
#include <fstream>

namespace vs {

std::expected<void, std::string> build_ivf_index(const Dataset &ds,
                                                 const std::filesystem::path &index_path,
                                                 std::uint32_t num_clusters, MetricType metric_type,
                                                 std::uint32_t max_iters) {
    if (ds.count == 0 || ds.dim == 0) {
        return std::unexpected("Empty dataset");
    }
    if (num_clusters == 0) {
        return std::unexpected("Cluster count must be > 0");
    }

    num_clusters = std::min(num_clusters, ds.count);

    KMeansResult km = (metric_type == MetricType::Cosine)
                          ? fit_kmeans(ds, num_clusters, max_iters, CosineMetric{})
                          : fit_kmeans(ds, num_clusters, max_iters, L2Metric{});

    // Group vector IDs by cluster
    std::vector<std::vector<std::uint32_t>> inverted_lists(num_clusters);
    for (std::uint32_t i = 0; i < ds.count; ++i) {
        inverted_lists[km.assignments[i]].push_back(i);
    }

    std::vector<ListMeta> metas(num_clusters);
    std::vector<std::uint32_t> flat_ids;
    std::vector<float> flat_vectors;
    flat_ids.reserve(ds.count);
    flat_vectors.reserve(static_cast<std::size_t>(ds.count) * ds.dim);

    std::uint64_t current_offset = 0;
    for (std::uint32_t c = 0; c < num_clusters; ++c) {
        metas[c].offset = current_offset;
        metas[c].count = static_cast<std::uint32_t>(inverted_lists[c].size());
        for (std::uint32_t id : inverted_lists[c]) {
            flat_ids.push_back(id);
            auto vec = ds.row(id);
            flat_vectors.insert(flat_vectors.end(), vec.begin(), vec.end());
        }
        current_offset += metas[c].count;
    }

    std::ofstream out(index_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return std::unexpected("Cannot create index file: " + index_path.string());
    }

    IndexHeader header{};
    header.dim = ds.dim;
    header.num_vectors = ds.count;
    header.num_clusters = num_clusters;
    header.metric_type = static_cast<std::uint32_t>(metric_type);

    out.write(reinterpret_cast<const char *>(&header), sizeof(IndexHeader));
    out.write(reinterpret_cast<const char *>(km.centroids.data()),
              km.centroids.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(metas.data()), metas.size() * sizeof(ListMeta));
    out.write(reinterpret_cast<const char *>(flat_ids.data()),
              flat_ids.size() * sizeof(std::uint32_t));
    out.write(reinterpret_cast<const char *>(flat_vectors.data()),
              flat_vectors.size() * sizeof(float));

    if (!out) {
        return std::unexpected("Failed to write index file");
    }

    return {};
}

std::expected<MappedIvfIndex, std::string> MappedIvfIndex::open(const std::filesystem::path &path) {
    if (!std::filesystem::exists(path)) {
        return std::unexpected("Index file does not exist: " + path.string());
    }

    try {
        MappedIvfIndex idx;
        std::string p_str = path.string();
        idx.mapping_ =
            boost::interprocess::file_mapping(p_str.c_str(), boost::interprocess::read_only);
        idx.region_ =
            boost::interprocess::mapped_region(idx.mapping_, boost::interprocess::read_only);

        if (idx.region_.get_size() < sizeof(IndexHeader)) {
            return std::unexpected("Index file is too small");
        }

        const auto *base = static_cast<const std::byte *>(idx.region_.get_address());
        idx.header_ = reinterpret_cast<const IndexHeader *>(base);

        if (idx.header_->magic != 0x49564631) {
            return std::unexpected("Invalid index file header");
        }

        std::size_t centroids_count =
            static_cast<std::size_t>(idx.header_->num_clusters) * idx.header_->dim;
        std::size_t lists_count = idx.header_->num_clusters;
        std::size_t ids_count = idx.header_->num_vectors;
        std::size_t vectors_count =
            static_cast<std::size_t>(idx.header_->num_vectors) * idx.header_->dim;

        std::size_t expected_size =
            sizeof(IndexHeader) + centroids_count * sizeof(float) + lists_count * sizeof(ListMeta) +
            ids_count * sizeof(std::uint32_t) + vectors_count * sizeof(float);

        if (idx.region_.get_size() != expected_size) {
            return std::unexpected("Corrupted index file (size mismatch)");
        }

        std::size_t offset = sizeof(IndexHeader);
        idx.centroids_ =
            std::span<const float>(reinterpret_cast<const float *>(base + offset), centroids_count);
        offset += centroids_count * sizeof(float);

        idx.lists_ = std::span<const ListMeta>(reinterpret_cast<const ListMeta *>(base + offset),
                                               lists_count);
        offset += lists_count * sizeof(ListMeta);

        idx.ids_ = std::span<const std::uint32_t>(
            reinterpret_cast<const std::uint32_t *>(base + offset), ids_count);
        offset += ids_count * sizeof(std::uint32_t);

        idx.vectors_ =
            std::span<const float>(reinterpret_cast<const float *>(base + offset), vectors_count);

        return idx;
    } catch (const boost::interprocess::interprocess_exception &ex) {
        return std::unexpected(std::string("Memory mapping error: ") + ex.what());
    }
}

std::expected<std::vector<SearchResult>, std::string>
MappedIvfIndex::search_auto(std::span<const float> query, std::uint32_t top_k,
                            std::uint32_t nprobe) const {
    if (header_->metric_type == static_cast<std::uint32_t>(MetricType::Cosine)) {
        return search(query, top_k, nprobe, CosineMetric{});
    }
    return search(query, top_k, nprobe, L2Metric{});
}

} // namespace vs
