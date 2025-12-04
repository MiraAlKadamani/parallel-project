// kmeans_mpi.cpp
// K-means clustering using MPI so each process handles part of the dataset.
// Rank 0 reads data and sends to others, then we collaborate to update centroids.

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>

// loading numeric CSV (only rank 0 uses this)
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
    // reading file row by row
    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string cell;
        int col_count = 0;
        std::vector<float> row_values;

        // split by comma and convert to float
        while (std::getline(ss, cell, ',')) {
            if (cell.empty())
                continue;
            try {
                float val = std::stof(cell);
                row_values.push_back(val);
                col_count++;
            } catch (...) {
                // if non-numeric values → skip this row entirely
                col_count = 0;
                row_values.clear();
                break;
            }
        }

        if (col_count == 0)
            continue;

        // set dimension from the first row
        if (num_dims == -1) {
            num_dims = col_count;
        }
        // check all rows have same number of columns
        else if (col_count != num_dims) {
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

// choose K random points as starting centroids (rank 0 does it)
void init_centroids_random(const std::vector<float> &data,
                           int N, int d, int K,
                           std::vector<float> &centroids) {
    centroids.resize(K * d);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> chosen;
    chosen.reserve(K);

    // make sure we select distinct rows
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

// helper for squared Euclidean distance
inline float squared_distance(const float *a, const float *b, int d) {
    float dist = 0.0f;
    for (int j = 0; j < d; ++j) {
        float diff = a[j] - b[j];
        dist += diff * diff;
    }
    return dist;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); // initialize MPI

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // find my rank ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); // number of processes

    // only rank 0 checks arguments
    if (rank == 0 && argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <csv_file> <K> <max_iters> [tolerance]\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string filename;
    int K = 0, max_iters = 0;
    float tol = 1e-4f; // default stopping tolerance

    if (rank == 0) {
        filename = argv[1];
        K = std::stoi(argv[2]);
        max_iters = std::stoi(argv[3]);
        if (argc >= 5) {
            tol = std::stof(argv[4]);
        }
    }

    // broadcast parameters so all ranks know
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&max_iters, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    std::vector<float> full_data;
    int N = 0, d = 0;

    // only rank 0 loads the dataset
    if (rank == 0) {
        if (!load_csv_numeric(filename, full_data, N, d)) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if (K <= 0 || K > N) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::cout << "Running MPI K-means with " << size
                  << " processes, K=" << K
                  << ", max_iters=" << max_iters << "\n";
    }

    // broadcast dataset shape to all ranks
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&d, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (N == 0 || d == 0) {
        MPI_Finalize();
        return 0;
    }

    // split dataset rows across processes as evenly as possible
    std::vector<int> counts_points(size);
    std::vector<int> displs_points(size);

    if (rank == 0) {
        int base = N / size;
        int rem = N % size;
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            counts_points[r] = base + (r < rem ? 1 : 0);
            displs_points[r] = offset;
            offset += counts_points[r];
        }
    }

    // share distribution info with all ranks
    MPI_Bcast(counts_points.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs_points.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    int local_N = counts_points[rank];
    std::vector<float> local_data(local_N * d); // buffer for my portion of data

    // convert counts from "points" to "floats" for scattering
    std::vector<int> sendcounts_f(size), displs_f(size);
    if (rank == 0) {
        for (int r = 0; r < size; ++r) {
            sendcounts_f[r] = counts_points[r] * d;
            displs_f[r] = displs_points[r] * d;
        }
    }

    // scatter subsets of rows to each rank
    MPI_Scatterv(rank == 0 ? full_data.data() : nullptr,
                 rank == 0 ? sendcounts_f.data() : nullptr,
                 rank == 0 ? displs_f.data() : nullptr,
                 MPI_FLOAT,
                 local_data.data(),
                 local_N * d,
                 MPI_FLOAT,
                 0,
                 MPI_COMM_WORLD);

    // initialize and broadcast centroids
    std::vector<float> centroids(K * d);
    if (rank == 0) {
        init_centroids_random(full_data, N, d, K, centroids);
    }
    MPI_Bcast(centroids.data(), K * d, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // local accumulators
    std::vector<int> local_assignments(local_N, 0);
    std::vector<float> local_sums(K * d);
    std::vector<int>   local_counts(K);

    // global accumulators
    std::vector<float> global_sums(K * d);
    std::vector<int>   global_counts(K);

    double t_start = MPI_Wtime(); // start timing K-means

    for (int iter = 0; iter < max_iters; ++iter) {
        // reset local accumulators each iteration
        std::fill(local_sums.begin(), local_sums.end(), 0.0f);
        std::fill(local_counts.begin(), local_counts.end(), 0);

        // assign each local point to nearest centroid
        for (int i = 0; i < local_N; ++i) {
            const float *x = &local_data[i * d];

            int best_k = 0;
            float best_dist = std::numeric_limits<float>::max();

            for (int k = 0; k < K; ++k) {
                float dist = squared_distance(x, &centroids[k * d], d);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }

            local_assignments[i] = best_k;
            local_counts[best_k]++;
            float *sum = &local_sums[best_k * d];
            for (int j = 0; j < d; ++j)
                sum[j] += x[j];
        }

        // combine all local results into global sums/counts
        MPI_Allreduce(local_sums.data(), global_sums.data(),
                      K * d, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_counts.data(), global_counts.data(),
                      K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        // update centroids from global results
        float max_shift_sq = 0.0f;
        for (int k = 0; k < K; ++k) {
            int count = global_counts[k];
            if (count == 0) continue; // leave centroid if no points assigned

            float *c = &centroids[k * d];
            float *sum = &global_sums[k * d];

            for (int j = 0; j < d; ++j) {
                float new_val = sum[j] / count;
                float diff = new_val - c[j];
                float shift_sq = diff * diff;
                if (shift_sq > max_shift_sq)
                    max_shift_sq = shift_sq;
                c[j] = new_val;
            }
        }

        float max_shift = std::sqrt(max_shift_sq);

        // only rank 0 prints progress
        if (rank == 0)
            std::cout << "Iteration " << iter
                      << ", centroid max shift = " << max_shift << "\n";

        if (max_shift < tol) {
            if (rank == 0)
                std::cout << "Converged at iteration " << iter << "\n";
            break;
        }
    }

    double t_end = MPI_Wtime();
    double local_time = t_end - t_start;
    double max_time = 0.0;

    // find the slowest rank time for fairness
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total K-means time (max rank time): "
                  << max_time << " s\n";

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
