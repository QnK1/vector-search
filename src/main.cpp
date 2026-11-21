#include "dataset.hpp"
#include "ivf_index.hpp"
#include "metrics.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::expected<std::vector<float>, std::string> parse_vector(std::string s) {
    std::replace(s.begin(), s.end(), ',', ' ');
    std::istringstream iss(s);
    std::vector<float> vec;
    float v = 0.0f;
    while (iss >> v) {
        vec.push_back(v);
    }
    if (vec.empty()) {
        return std::unexpected("Failed to parse query vector");
    }
    return vec;
}

} // namespace

int main(int argc, char **argv) {
    CLI::App app{"vsearch - in-memory ANN vector search (IVF + Boost.Interprocess)"};
    app.require_subcommand(1);

    auto *gen_cmd = app.add_subcommand("generate", "Generate synthetic vectors to a CSV file");
    std::string gen_out = "data.csv";
    std::uint32_t gen_count = 1000;
    std::uint32_t gen_dim = 8;
    std::uint32_t gen_clusters = 10;
    gen_cmd->add_option("-o,--output", gen_out, "Output CSV path")->required();
    gen_cmd->add_option("-n,--count", gen_count, "Number of vectors (default: 1000)");
    gen_cmd->add_option("-d,--dim", gen_dim, "Vector dimension (default: 8)");
    gen_cmd->add_option("-c,--clusters", gen_clusters, "Number of clusters (default: 10)");

    auto *build_cmd = app.add_subcommand("build", "Build IVF index from CSV and save to disk");
    std::string build_in;
    std::string build_out;
    std::uint32_t build_clusters = 16;
    std::uint32_t build_iters = 25;
    std::string build_metric = "l2";
    build_cmd->add_option("-i,--input", build_in, "Input CSV file")->required();
    build_cmd->add_option("-o,--output", build_out, "Output binary index file")->required();
    build_cmd->add_option("-c,--clusters", build_clusters,
                          "Number of k-means clusters (default: 16)");
    build_cmd->add_option("--iters", build_iters, "Max k-means iterations (default: 25)");
    build_cmd
        ->add_option("-m,--metric", build_metric, "Distance metric: l2 or cosine (default: l2)")
        ->check(CLI::IsMember({"l2", "cosine"}));

    auto *query_cmd =
        app.add_subcommand("query", "Search Top-K nearest vectors using memory-mapped index");
    std::string query_idx;
    std::string query_vec_str;
    std::uint32_t query_k = 5;
    std::uint32_t query_nprobe = 3;
    std::string query_metric = "auto";
    query_cmd->add_option("-i,--index", query_idx, "Index file path")->required();
    query_cmd->add_option("-q,--vector", query_vec_str, "Query vector, e.g. \"1.0,2.5,-0.5\"")
        ->required();
    query_cmd->add_option("-k,--top-k", query_k, "Number of neighbors to return (default: 5)");
    query_cmd->add_option("-p,--nprobe", query_nprobe, "Clusters to probe (default: 3)");
    query_cmd->add_option("-m,--metric", query_metric, "Metric: auto, l2 or cosine (default: auto)")
        ->check(CLI::IsMember({"auto", "l2", "cosine"}));

    CLI11_PARSE(app, argc, argv);

    if (*gen_cmd) {
        auto ds = vs::generate_synthetic(gen_count, gen_dim, gen_clusters);
        auto res = vs::save_csv(gen_out, ds);
        if (!res) {
            std::cerr << "Error: " << res.error() << "\n";
            return 1;
        }
        std::cout << "Generated " << ds.count << " vectors (dim=" << ds.dim << ") -> " << gen_out
                  << "\n";
    } else if (*build_cmd) {
        auto t0 = std::chrono::steady_clock::now();
        auto ds_res = vs::load_csv(build_in);
        if (!ds_res) {
            std::cerr << "CSV error: " << ds_res.error() << "\n";
            return 1;
        }

        vs::MetricType mt =
            (build_metric == "cosine") ? vs::MetricType::Cosine : vs::MetricType::L2;
        auto build_res = vs::build_ivf_index(*ds_res, build_out, build_clusters, mt, build_iters);
        if (!build_res) {
            std::cerr << "Build error: " << build_res.error() << "\n";
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::cout << "Index built: " << build_out << "\n"
                  << "  vectors:  " << ds_res->count << " (dim: " << ds_res->dim << ")\n"
                  << "  clusters: " << std::min(build_clusters, ds_res->count) << "\n"
                  << "  metric:   " << build_metric << "\n"
                  << "  time:     " << ms << " ms\n";
    } else if (*query_cmd) {
        auto q_vec = parse_vector(query_vec_str);
        if (!q_vec) {
            std::cerr << "Error: " << q_vec.error() << "\n";
            return 1;
        }

        auto t_open0 = std::chrono::steady_clock::now();
        auto idx_res = vs::MappedIvfIndex::open(query_idx);
        auto t_open1 = std::chrono::steady_clock::now();
        if (!idx_res) {
            std::cerr << "Open error: " << idx_res.error() << "\n";
            return 1;
        }

        auto t_search0 = std::chrono::steady_clock::now();
        std::expected<std::vector<vs::SearchResult>, std::string> search_res;
        if (query_metric == "l2") {
            search_res = idx_res->search(*q_vec, query_k, query_nprobe, vs::L2Metric{});
        } else if (query_metric == "cosine") {
            search_res = idx_res->search(*q_vec, query_k, query_nprobe, vs::CosineMetric{});
        } else {
            search_res = idx_res->search_auto(*q_vec, query_k, query_nprobe);
        }
        auto t_search1 = std::chrono::steady_clock::now();

        if (!search_res) {
            std::cerr << "Search error: " << search_res.error() << "\n";
            return 1;
        }

        auto open_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_open1 - t_open0).count();
        auto search_us =
            std::chrono::duration_cast<std::chrono::microseconds>(t_search1 - t_search0).count();

        std::cout << "Top-" << search_res->size() << " (mmap: " << open_us
                  << " us, search: " << search_us << " us):\n";
        for (std::size_t i = 0; i < search_res->size(); ++i) {
            std::cout << "  #" << (i + 1) << "  id=" << (*search_res)[i].id
                      << "  dist=" << (*search_res)[i].distance << "\n";
        }
    }

    return 0;
}
