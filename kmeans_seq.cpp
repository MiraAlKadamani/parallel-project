// kmeans_seq.cpp
// Sequential K-means in k dimensions with CSV loading and simple timing.
//
// Usage:
//   ./kmeans_seq <csv_file> <K> <max_iters> [tolerance]
//
// CSV format: numeric only, comma-separated, no header.
//   - each row = one data point
//   - each column = one feature

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>

// ---------------------------------------------------------------------------
// CSV loader: numeric, comma-separated, no header
// ---------------------------------------------------------------------------
bool load_csv_numeric(const std::string &filename,
                      std::vector<float> &data,
                      int &num_points,
                      int &num_dims) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << filename << "\n";
        return false;
    }

    data.clear();
    num_points = 0;
    num_dims = -1;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string cell;
        int col_count = 0;
        std::vector<float> row_values;

        while (std::getline(ss, cell, ',')) {
            if (cell.empty())
                continue;
            try {
                float val = std::stof(cell);
                row_values.push_back(val);
                col_count++;
            } catch (const std::exception &) {
                std::cerr << "Warning: non-numeric value in CSV, skipping line: "
                          << line << "\n";
                col_count = 0;
                row_values.clear();
                break;
            }
        }

        if (col_count == 0)
            continue;

        if (num_dims == -1) {
            num_dims = col_count;
        } else if (col_count != num_dims) {
            std::cerr << "Error: inconsistent number of columns in CSV.\n";
            return false;
        }

        data.insert(data.end(), row_values.begin(), row_values.end());
        num_points++;
    }

    if (num_points == 0 || num_dims <= 0) {
        std::cerr << "Error: no valid data in CSV.\n";
        return false;
    }

    std::cout << "Loaded CSV: " << num_points << " points, "
              << num_dims << " dimensions.\n";
    return true;
}

// ---------------------------------------------------------------------------
// K-means helpers
// ---------------------------------------------------------------------------

// Random initialization: pick K distinct random points as initial centroids
void init_centroids_random(const std::vector<float> &data,
                           int N, int d, int K,
                           std::vector<float> &centroids) {
    centroids.resize(K * d);
    std::mt19937 rng(12345); // fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> chosen;
    chosen.reserve(K);

    for (int k = 0; k < K; ++k) {
        int idx;
        bool unique;
        do {
            unique = true;
            idx = dist(rng);
            for (int c : chosen) {
                if (c == idx) { unique = false; break; }
            }
        } while (!unique);
        chosen.push_back(idx);

        const float *src = &data[idx * d];
        float *dst = &centroids[k * d];
        for (int j = 0; j < d; ++j) {
            dst[j] = src[j];
        }
    }
}

// Compute squared Euclidean distance between two d-dimensional points
inline float squared_distance(const float *a, const float *b, int d) {
    float dist = 0.0f;
    for (int j = 0; j < d; ++j) {
        float diff = a[j] - b[j];
        dist += diff * diff;
    }
    return dist;
}

// One full K-means run (sequential)
void kmeans_sequential(const std::vector<float> &data,
                       int N, int d, int K,
                       int max_iters, float tol,
                       std::vector<float> &centroids_out,
                       std::vector<int> &assignments_out) {
    // Initialize centroids
    std::vector<float> centroids;
    init_centroids_random(data, N, d, K, centroids);

    std::vector<int> assignments(N, 0);
    std::vector<float> sums(K * d);
    std::vector<int> counts(K);

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1. Assignment step
        for (int i = 0; i < N; ++i) {
            const float *x = &data[i * d];

            int best_k = 0;
            float best_dist = std::numeric_limits<float>::max();

            for (int k = 0; k < K; ++k) {
                const float *c = &centroids[k * d];
                float dist = squared_distance(x, c, d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }

            assignments[i] = best_k;
        }

        // 2. Update step
        // Reset sums and counts
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        // Accumulate
        for (int i = 0; i < N; ++i) {
            int k = assignments[i];
            const float *x = &data[i * d];

            counts[k] += 1;
            float *cluster_sum = &sums[k * d];
            for (int j = 0; j < d; ++j) {
                cluster_sum[j] += x[j];
            }
        }

        // Compute new centroids and track max shift
        float max_shift_sq = 0.0f;

        for (int k = 0; k < K; ++k) {
            int count = counts[k];
            if (count == 0) {
                // No points in this cluster: keep old centroid
                continue;
            }

            float *c = &centroids[k * d];
            float *sum = &sums[k * d];

            for (int j = 0; j < d; ++j) {
                float old_val = c[j];
                float new_val = sum[j] / static_cast<float>(count);
                float diff = new_val - old_val;
                float shift_sq = diff * diff;
                if (shift_sq > max_shift_sq) {
                    max_shift_sq = shift_sq;
                }
                c[j] = new_val;
            }
        }

        float max_shift = std::sqrt(max_shift_sq);
        std::cout << "Iteration " << iter
                  << ", centroid max shift = " << max_shift << "\n";

        if (max_shift < tol) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }
    }

    centroids_out = std::move(centroids);
    assignments_out = std::move(assignments);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <csv_file> <K> <max_iters> [tolerance]\n";
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int max_iters = std::stoi(argv[3]);
    float tol = (argc >= 5) ? std::stof(argv[4]) : 1e-4f;

    // Load CSV
    std::vector<float> data;
    int N = 0, d = 0;
    if (!load_csv_numeric(filename, data, N, d)) {
        return EXIT_FAILURE;
    }

    if (K <= 0 || K > N) {
        std::cerr << "Error: invalid K (number of clusters).\n";
        return EXIT_FAILURE;
    }

    std::cout << "Running SEQUENTIAL K-means with K=" << K
              << ", max_iters=" << max_iters
              << ", tolerance=" << tol << "\n";

    std::vector<float> centroids;
    std::vector<int> assignments;

    auto t_start = std::chrono::high_resolution_clock::now();

    kmeans_sequential(data, N, d, K, max_iters, tol, centroids, assignments);

    auto t_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Total K-means time (sequential): " << seconds << " s\n\n";

    // Print final centroids
    std::cout << "Final centroids:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << "Centroid " << k << ": ";
        for (int j = 0; j < d; ++j) {
            std::cout << centroids[k * d + j];
            if (j + 1 < d) std::cout << ", ";
        }
        std::cout << "\n";
    }

    // If you need per-point assignments for analysis, they are in `assignments`

    return EXIT_SUCCESS;
}
