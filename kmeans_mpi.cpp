// kmeans_mpi.cpp
// K-means in k dimensions using MPI (distributed-memory).
//
// Usage:
//   mpirun -np <P> ./kmeans_mpi <csv_file> <K> <max_iters> [tolerance]
//
// Only rank 0 loads the CSV, then scatters the data to all ranks.
// CSV format: numeric only, comma-separated, no header:
//   - each row = one data point
//   - each column = one feature

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// CSV loader (used only on rank 0): numeric, comma-separated, no header
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

// Random initialization on rank 0: pick K distinct random points as initial centroids
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0 && argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <csv_file> <K> <max_iters> [tolerance]\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string filename;
    int K = 0, max_iters = 0;
    float tol = 1e-4f;

    if (rank == 0) {
        filename = argv[1];
        K = std::stoi(argv[2]);
        max_iters = std::stoi(argv[3]);
        if (argc >= 5) {
            tol = std::stof(argv[4]);
        }
    }

    // Broadcast parameters K, max_iters, tol
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Rank 0 loads CSV
    std::vector<float> full_data;
    int N = 0, d = 0;
    if (rank == 0) {
        if (!load_csv_numeric(filename, full_data, N, d)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (K <= 0 || K > N) {
            std::cerr << "Error: invalid K (number of clusters).\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "Running MPI K-means with "
                  << size << " processes, K=" << K
                  << ", max_iters=" << max_iters
                  << ", tolerance=" << tol << "\n";
    }

    // Broadcast N and d to all ranks
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N == 0 || d == 0) {
        MPI_Finalize();
        return 0;
    }

    // Decide how many points each rank gets
    std::vector<int> counts_points(size);
    std::vector<int> displs_points(size);

    if (rank == 0) {
        int base = N / size;
        int rem = N % size;
        int offset = 0;

        for (int r = 0; r < size; ++r) {
            int n_local = base + (r < rem ? 1 : 0);
            counts_points[r] = n_local;
            displs_points[r] = offset;
            offset += n_local;
        }
    }

    // Broadcast counts_points and displs_points so each rank knows its size
    MPI_Bcast(counts_points.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs_points.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_N = counts_points[rank];
    std::vector<float> local_data(local_N * d);

    // Prepare sendcounts and displs for MPI_Scatterv (in floats)
    std::vector<int> sendcounts_f(size), displs_f(size);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            sendcounts_f[r] = counts_points[r] * d;
            displs_f[r] = displs_points[r] * d;
        }
    }

    // Scatter data rows to each rank
    MPI_Scatterv(rank == 0 ? full_data.data() : nullptr,
                 rank == 0 ? sendcounts_f.data() : nullptr,
                 rank == 0 ? displs_f.data() : nullptr,
                 MPI_FLOAT,
                 local_data.data(),
                 local_N * d,
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD);

    // Initialize centroids on rank 0, then broadcast to all
    std::vector<float> centroids(K * d);
    if (rank == 0) {
        init_centroids_random(full_data, N, d, K, centroids);
    }
    MPI_Bcast(centroids.data(), K * d, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Local structures
    std::vector<int> local_assignments(local_N, 0);
    std::vector<float> local_sums(K * d);
    std::vector<int>   local_counts(K);

    // Global sums and counts (same on all ranks after Allreduce)
    std::vector<float> global_sums(K * d);
    std::vector<int>   global_counts(K);

    double t_start = MPI_Wtime();

    for (int iter = 0; iter < max_iters; ++iter) {
        // 1) Assignment + local accumulation
        std::fill(local_sums.begin(), local_sums.end(), 0.0f);
        std::fill(local_counts.begin(), local_counts.end(), 0);

        for (int i = 0; i < local_N; ++i) {
            const float *x = &local_data[i * d];

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

            local_assignments[i] = best_k;

            // accumulate local sums
            local_counts[best_k] += 1;
            float *cluster_sum = &local_sums[best_k * d];
            for (int j = 0; j < d; ++j) {
                cluster_sum[j] += x[j];
            }
        }

        // 2) Allreduce to get global sums and counts
        MPI_Allreduce(local_sums.data(), global_sums.data(),
                      K * d, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(),
                      K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // 3) Update centroids and compute max shift (all ranks do same work)
        float max_shift_sq = 0.0f;

        for (int k = 0; k < K; ++k) {
            int count = global_counts[k];
            if (count == 0) {
                // no points in this cluster, keep old centroid
                continue;
            }

            float *c   = &centroids[k * d];
            float *sum = &global_sums[k * d];

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

        if (rank == 0) {
            std::cout << "Iteration " << iter
                      << ", centroid max shift = " << max_shift << "\n";
        }

        if (max_shift < tol) {
            if (rank == 0) {
                std::cout << "Converged at iteration " << iter << "\n";
            }
            break;
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total K-means time (MPI, max over ranks): "
                  << max_time << " s\n\n";

        std::cout << "Final centroids:\n";
        for (int k = 0; k < K; ++k) {
            std::cout << "Centroid " << k << ": ";
            for (int j = 0; j < d; ++j) {
                std::cout << centroids[k * d + j];
                if (j + 1 < d) std::cout << ", ";
            }
            std::cout << "\n";
        }
    }

    MPI_Finalize();
    return 0;
}
