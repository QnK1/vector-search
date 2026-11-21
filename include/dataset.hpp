#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace vs {

struct Dataset {
    std::uint32_t count = 0;
    std::uint32_t dim = 0;
    std::vector<float> data;

    std::span<const float> row(std::size_t idx) const noexcept {
        return std::span<const float>(data.data() + idx * dim, dim);
    }
};

std::expected<Dataset, std::string> load_csv(const std::filesystem::path &path);
std::expected<void, std::string> save_csv(const std::filesystem::path &path, const Dataset &ds);
Dataset generate_synthetic(std::uint32_t count, std::uint32_t dim, std::uint32_t clusters,
                           std::uint32_t seed = 42);

} // namespace vs
