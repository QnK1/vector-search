#include "dataset.hpp"
#include "ivf_index.hpp"
#include "kmeans.hpp"
#include "metrics.hpp"
#include <array>
#include <filesystem>
#include <gtest/gtest.h>

TEST(MetricsTest, L2AndCosineBasic) {
    const std::array<float, 3> a = {1.0f, 0.0f, 0.0f};
    const std::array<float, 3> b = {0.0f, 1.0f, 0.0f};
    const std::array<float, 3> c = {2.0f, 0.0f, 0.0f};

    vs::L2Metric l2;
    vs::CosineMetric cosine;

    EXPECT_NEAR(l2(a, a), 0.0f, 1e-6f);
    EXPECT_NEAR(l2(a, c), 1.0f, 1e-6f);
    EXPECT_NEAR(l2(a, b), std::sqrt(2.0f), 1e-6f);

    EXPECT_NEAR(cosine(a, c), 0.0f, 1e-6f);
    EXPECT_NEAR(cosine(a, b), 1.0f, 1e-6f);
}

TEST(KMeansTest, SeparatesClusters) {
    auto ds = vs::generate_synthetic(120, 4, 3, 123);
    auto res = vs::fit_kmeans(ds, 3, 30, vs::L2Metric{});

    ASSERT_EQ(res.assignments.size(), 120u);
    ASSERT_EQ(res.centroids.size(), 12u);

    // Points from the same generator cluster should end up together
    EXPECT_EQ(res.assignments[0], res.assignments[3]);
    EXPECT_EQ(res.assignments[1], res.assignments[4]);
    EXPECT_EQ(res.assignments[2], res.assignments[5]);
}

TEST(IvfIndexTest, BuildMapAndQueryExactMatch) {
    auto ds = vs::generate_synthetic(200, 6, 5, 77);
    std::filesystem::path idx_path = std::filesystem::temp_directory_path() / "test_ivf_index.bin";

    auto build_res = vs::build_ivf_index(ds, idx_path, 5, vs::MetricType::L2, 25);
    ASSERT_TRUE(build_res.has_value()) << build_res.error();

    {
        auto mapped = vs::MappedIvfIndex::open(idx_path);
        ASSERT_TRUE(mapped.has_value()) << mapped.error();
        EXPECT_EQ(mapped->header().num_vectors, 200u);
        EXPECT_EQ(mapped->header().dim, 6u);
        EXPECT_EQ(mapped->header().num_clusters, 5u);

        auto target = ds.row(42);
        auto results = mapped->search(target, 5, 2, vs::L2Metric{});
        ASSERT_TRUE(results.has_value()) << results.error();
        ASSERT_EQ(results->size(), 5u);
        EXPECT_EQ((*results)[0].id, 42u);
        EXPECT_NEAR((*results)[0].distance, 0.0f, 1e-5f);
    }

    std::filesystem::remove(idx_path);
}

TEST(IvfIndexTest, HandlesInvalidDimensionAndMissingFile) {
    auto missing = vs::MappedIvfIndex::open("non_existent_file_12345.bin");
    EXPECT_FALSE(missing.has_value());

    auto ds = vs::generate_synthetic(30, 4, 2, 11);
    std::filesystem::path idx_path = std::filesystem::temp_directory_path() / "test_ivf_dim.bin";
    ASSERT_TRUE(vs::build_ivf_index(ds, idx_path, 2).has_value());

    {
        auto mapped = vs::MappedIvfIndex::open(idx_path);
        ASSERT_TRUE(mapped.has_value());

        const std::array<float, 2> bad_query = {1.0f, 2.0f};
        auto res = mapped->search(bad_query, 3, 1, vs::L2Metric{});
        EXPECT_FALSE(res.has_value());
    }

    std::filesystem::remove(idx_path);
}
