#include "dataset.hpp"
#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>

namespace vs {

std::expected<Dataset, std::string> load_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        return std::unexpected("Cannot open file: " + path.string());
    }

    Dataset ds;
    std::string line;
    std::size_t line_no = 0;

    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) {
            continue;
        }

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);

        std::vector<float> row;
        float val = 0.0f;
        while (iss >> val) {
            row.push_back(val);
        }

        if (row.empty()) {
            continue;
        }

        if (ds.dim == 0) {
            ds.dim = static_cast<std::uint32_t>(row.size());
        } else if (row.size() != ds.dim) {
            return std::unexpected("Dimension mismatch at line " + std::to_string(line_no));
        }

        ds.data.insert(ds.data.end(), row.begin(), row.end());
        ds.count++;
    }

    if (ds.count == 0) {
        return std::unexpected("File contains no vectors");
    }

    return ds;
}

std::expected<void, std::string> save_csv(const std::filesystem::path& path, const Dataset& ds) {
    std::ofstream out(path);
    if (!out) {
        return std::unexpected("Cannot write file: " + path.string());
    }

    for (std::uint32_t i = 0; i < ds.count; ++i) {
        auto r = ds.row(i);
        for (std::uint32_t d = 0; d < ds.dim; ++d) {
            if (d > 0) out << ',';
            out << r[d];
        }
        out << '\n';
    }
    return {};
}

Dataset generate_synthetic(std::uint32_t count, std::uint32_t dim, std::uint32_t clusters, std::uint32_t seed) {
    Dataset ds;
    ds.count = count;
    ds.dim = dim;
    ds.data.resize(static_cast<std::size_t>(count) * dim);

    if (count == 0 || dim == 0) {
        return ds;
    }

    clusters = std::max(1u, clusters);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> center_dist(-50.0f, 50.0f);
    std::normal_distribution<float> noise_dist(0.0f, 1.5f);

    std::vector<float> centers(static_cast<std::size_t>(clusters) * dim);
    for (float& v : centers) {
        v = center_dist(rng);
    }

    for (std::uint32_t i = 0; i < count; ++i) {
        std::uint32_t c = i % clusters;
        for (std::uint32_t d = 0; d < dim; ++d) {
            ds.data[static_cast<std::size_t>(i) * dim + d] =
                centers[static_cast<std::size_t>(c) * dim + d] + noise_dist(rng);
        }
    }

    return ds;
}

} // namespace vs
