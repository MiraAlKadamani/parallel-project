// kmeans_seq.cpp
// Sequential K-means in k dimensions with CSV loading and timing.
// Just the basic version of kmeans, running only on the CPU.
//
// Usage example:
//   ./kmeans_seq data.csv 3 50 0.0001

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>

// I use this to load the input dataset from CSV
// It reads everything into one flat vector: [x1,y1,z1,x2,y2,z2,...]
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
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string cell;
        int col_count = 0;
        std::vector<float> row_values;

        // reading each column in the row
        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) continue;
            try {
                row_values.push_back(std::stof(cell));
                col_count++;
            } catch (...) {
                // if there's a non-numeric value, I just skip the line
                std::cerr << "Warning: bad row, skipping: " << line << "\n";
                row_values.clear();
                col_count = 0;
                break;
            }
        }

        if (col_count == 0) continue;

        // first row gives us the dimensionality
        if (num_dims == -1) {
            num_dims = col_count;
        } else if (col_count != num_dims) {
            std::cerr << "Error: inconsistent row dimensions.\n";
            return false;
        }

        // append row to big data array
        data.insert(data.end(), row_values.begin(), row_values.end());
        num_points++;
    }

    if (num_points == 0) return false;

    std::cout << "Loaded " << num_points
              << " points, each with " << num_dims << " features.\n";
    return true;
}

// pick K random initial centroids from the dataset
// nothing fancy, just random sampling
void init_centroids_random(const std::vector<float> &data,
                           int N, int d, int K,
                           std::vector<float> &centroids) {
    centroids.resize(K * d);
    std::mt19937 rng(12345); // so results stay the same each run
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> chosen_indices; // making sure we don't repeat a point

    for (int k = 0; k < K; ++k) {
        int idx;
        bool ok;
        do {
            ok = true;
            idx = dist(rng);
            for (int c : chosen_indices)
                if (c == idx) ok = false;
        } while (!ok);

        chosen_indices.push_back(idx);

        // copy chosen point → centroid list
        const float *src = &data[idx * d];
        float *dst = &centroids[k * d];
        for (int j = 0; j < d; ++j)
            dst[j] = src[j];
    }
}

// just the distance metric we need (no sqrt to save time)
inline float squared_distance(const float *a, const float *b, int d) {
    float dist = 0;
    for (int j = 0; j < d; ++j)
        dist += (a[j] - b[j]) * (a[j] - b[j]);
    return dist;
}

// main K-means loop, done completely in CPU
void kmeans_sequential(const std::vector<float> &data,
                       int N, int d, int K,
                       int max_iters, float tol,
                       std::vector<float> &centroids_out,
                       std::vector<int> &assignments_out) {
    std::vector<float> centroids;
    init_centroids_random(data, N, d, K, centroids);

    std::vector<int> assignments(N);
    std::vector<float> sums(K * d);
    std::vector<int> counts(K);

    // repeat two steps: assignment and update
    for (int iter = 0; iter < max_iters; ++iter) {

        // find nearest centroid for each point
        for (int i = 0; i < N; ++i) {
            const float *x = &data[i * d];
            float best_dist = std::numeric_limits<float>::max();
            int best_k = 0;

            for (int k = 0; k < K; ++k) {
                float dist = squared_distance(x, &centroids[k * d], d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            assignments[i] = best_k;
        }

        // clear sums to recalc new centroid positions
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        // accumulate points inside the same cluster
        for (int i = 0; i < N; ++i) {
            int k = assignments[i];
            counts[k]++;

            float *sum = &sums[k * d];
            const float *x = &data[i * d];
            for (int j = 0; j < d; ++j)
                sum[j] += x[j];
        }

        // update each centroid as mean of its cluster
        float max_shift_sq = 0.0f;

        for (int k = 0; k < K; ++k) {
            if (counts[k] == 0) continue; // cluster empty → no update

            float *c = &centroids[k * d];
            float *sum = &sums[k * d];

            for (int j = 0; j < d; ++j) {
                float new_val = sum[j] / counts[k];
                float diff = new_val - c[j];
                float shift_sq = diff * diff;
                if (shift_sq > max_shift_sq) max_shift_sq = shift_sq;
                c[j] = new_val;
            }
        }

        float max_shift = std::sqrt(max_shift_sq);
        std::cout << "Iteration " << iter
                  << "  shift=" << max_shift << "\n";

        // stop if movement small enough
        if (max_shift < tol) {
            std::cout << "Reached convergence.\n";
            break;
        }
    }

    centroids_out = centroids;
    assignments_out = assignments;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <csv_file> <K> <max_iters> [tolerance]\n";
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int max_iters = std::stoi(argv[3]);
    float tol = (argc >= 5 ? std::stof(argv[4]) : 1e-4f);

    // load input first
    std::vector<float> data;
    int N = 0, d = 0;
    if (!load_csv_numeric(filename, data, N, d))
        return EXIT_FAILURE;

    if (K <= 0 || K > N) {
        std::cerr << "Invalid K parameter.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Running sequential K-means...\n";

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<float> centroids;
    std::vector<int> assignments;

    // execute CPU Kmeans
    kmeans_sequential(data, N, d, K, max_iters, tol, centroids, assignments);
    auto end = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration<double>(end - start).count();
    std::cout << "Time taken: " << secs << " s\n\n";

    // show resulting centroids
    std::cout << "Final centroids:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << "C" << k << ": ";
        for (int j = 0; j < d; ++j) {
            std::cout << centroids[k * d + j];
            if (j + 1 < d) std::cout << ", ";
        }
        std::cout << "\n";
    }

    return EXIT_SUCCESS;
}
