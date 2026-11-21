# vsearch

Small in-memory ANN (Approximate Nearest Neighbor) vector search engine using an Inverted File (IVF) index and memory-mapped files.

It follows a write-once, read-many workflow:
1. `build` clusters input vectors from a CSV file with k-means and writes a flat binary index to disk.
2. `query` memory-maps the index file via `Boost.Interprocess` and scans the `nprobe` closest clusters to return Top-K neighbors.

## Stack & C++23 features

- **CLI11** for command-line parsing (`-h/--help`, subcommands)
- **Boost.Interprocess** (`file_mapping`, `mapped_region`) for zero-copy read-only index loading
- **GoogleTest** for unit tests
- **C++23**: `std::span` (vector views over mapped memory), `concepts` (`DistanceMetric` for L2/Cosine), `std::expected` (error handling), `std::chrono` (timing)

## Build

Requires CMake 3.20+ and a C++23 compiler. Dependencies are fetched automatically by CMake.

```powershell
cmake -B build
cmake --build build --config Release
```

Run tests:

```powershell
ctest --test-dir build -C Release --output-on-failure
```

## Usage

```powershell
# 1. Generate sample dataset (1000 vectors, 4D, 8 clusters)
./build/Release/vsearch.exe generate -o data.csv -n 1000 -d 4 -c 8

# 2. Build IVF index
./build/Release/vsearch.exe build -i data.csv -o index.bin -c 8 -m l2

# 3. Query Top-5 nearest vectors (probing 2 nearest clusters)
./build/Release/vsearch.exe query -i index.bin -q "10.5,-3.2,8.1,0.4" -k 5 -p 2
```

Run `./build/Release/vsearch.exe --help` (or `<subcommand> --help`) for all options.

## Note on AI usage

AI tools were used to help withparts of the code, CMake setup, layout boilerplate, and unit tests.
