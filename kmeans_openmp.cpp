// kmeans_omp.cpp
// Parallel K-means using OpenMP. Each thread processes part of the dataset to speed up clustering.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>
#include <omp.h>

// load numeric CSV: used to read dataset into a flat vector
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
    // read file row by row
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string cell;
        int col_count = 0;
        std::vector<float> row_values;

        // split the line by commas and convert to float
        while (std::getline(ss, cell, ',')) {
            if (cell.empty())
                continue;
            try {
                row_values.push_back(std::stof(cell));
                col_count++;
            } catch (...) {
                // if anything is not numeric, we drop the row
                col_count = 0;
                row_values.clear();
                break;
            }
        }

        if (col_count == 0)
            continue;

        // detect dimension from first valid row
        if (num_dims == -1) {
            num_dims = col_count;
        }
        // verify consistent dimensionality
        else if (col_count != num_dims) {
            std::cerr << "Error: inconsistent number of columns in CSV.\n";
            return false;
        }

        // append row into flat data vector
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

// choose K random rows as starting centroids
void init_centroids_random(const std::vector<float> &data,
                           int N, int d, int K,
                           std::vector<float> &centroids) {
    centroids.resize(K * d);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> chosen;
    chosen.reserve(K);

    for (int k = 0; k < K; ++k) {
        int idx;
        bool unique;
        do {
            idx = dist(rng);
            unique = (std::find(chosen.begin(), chosen.end(), idx) == chosen.end());
        } while (!unique);
        chosen.push_back(idx);

        // copy selected row into centroid storage
        const float *src = &data[idx * d];
        for (int j = 0; j < d; ++j) {
            centroids[k * d + j] = src[j];
        }
    }
}

// squared Euclidean distance used for comparing cluster closeness
inline float squared_distance(const float *a, const float *b, int d) {
    float dist = 0.0f;
    for (int j = 0; j < d; ++j) {
        float diff = a[j] - b[j];
        dist += diff * diff;
    }
    return dist;
}

// main OpenMP K-means function
void kmeans_openmp(const std::vector<float> &data,
                   int N, int d, int K,
                   int max_iters, float tol,
                   std::vector<float> &centroids_out,
                   std::vector<int> &assignments_out) {

    // initialize centroids first time outside loop
    std::vector<float> centroids;
    init_centroids_random(data, N, d, K, centroids);

    std::vector<int> assignments(N, 0);
    std::vector<float> sums(K * d);
    std::vector<int> counts(K);

    // main K-means loop
    for (int iter = 0; iter < max_iters; ++iter) {

        // assignment step: parallel over data points
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            const float *x = &data[i * d];
            int best_k = 0;
            float best_dist = std::numeric_limits<float>::max();

            // compute distance to each centroid
            for (int k = 0; k < K; ++k) {
                float dist = squared_distance(x, &centroids[k * d], d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            assignments[i] = best_k; // record chosen cluster
        }

        // reset cluster sums/counts before update
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        // parallel accumulation: each thread keeps its own partial sums
        #pragma omp parallel
        {
            std::vector<float> local_sums(K * d, 0.0f);
            std::vector<int> local_counts(K, 0);

            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                int k = assignments[i];
                const float *x = &data[i * d];

                local_counts[k] += 1;
                for (int j = 0; j < d; ++j)
                    local_sums[k * d + j] += x[j];
            }

            // reduce local results into global
            #pragma omp critical
            {
                for (int k = 0; k < K; ++k) {
                    counts[k] += local_counts[k];
                    for (int j = 0; j < d; ++j)
                        sums[k * d + j] += local_sums[k * d + j];
                }
            }
        }

        // update centroids and track how much they moved
        float max_shift_sq = 0.0f;

        for (int k = 0; k < K; ++k) {
            int count = counts[k];
            if (count == 0)
                continue; // leave unchanged if no points went to this cluster

            float *c = &centroids[k * d];
            for (int j = 0; j < d; ++j) {
                float new_val = sums[k * d + j] / count;
                float diff = new_val - c[j];
                float shift_sq = diff * diff;
                if (shift_sq > max_shift_sq)
                    max_shift_sq = shift_sq;
                c[j] = new_val;
            }
        }

        float max_shift = std::sqrt(max_shift_sq);
        std::cout << "Iteration " << iter
                  << ", centroid max shift = " << max_shift << "\n";

        // stopping condition if centroids not moving much
        if (max_shift < tol) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }
    }

    // return final centroids and cluster assignments
    centroids_out = centroids;
    assignments_out = assignments;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <csv_file> <K> <max_iters> [tolerance]\n";
        return EXIT_FAILURE;
    }

    // input args from user
    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int max_iters = std::stoi(argv[3]);
    float tol = (argc >= 5) ? std::stof(argv[4]) : 1e-4f;

    // load dataset into memory
    std::vector<float> data;
    int N = 0, d = 0;
    if (!load_csv_numeric(filename, data, N, d)) {
        return EXIT_FAILURE;
    }

    if (K <= 0 || K > N) {
        std::cerr << "Error: invalid K.\n";
        return EXIT_FAILURE;
    }

    std::cout << "Running OpenMP K-means with K=" << K
              << ", max_iters=" << max_iters << "\n";
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";

    // to store final result
    std::vector<float> centroids;
    std::vector<int> assignments;

    // time measurement for performance
    auto t_start = std::chrono::high_resolution_clock::now();
    kmeans_openmp(data, N, d, K, max_iters, tol, centroids, assignments);
    auto t_end = std::chrono::high_resolution_clock::now();

    double seconds = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Total K-means time (OpenMP): " << seconds << " s\n\n";

    // show final cluster centers
    std::cout << "Final centroids:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << "Centroid " << k << ": ";
        for (int j = 0; j < d; ++j) {
            std::cout << centroids[k * d + j] << (j + 1 < d ? ", " : "");
        }
        std::cout << "\n";
    }

    return EXIT_SUCCESS;
}
