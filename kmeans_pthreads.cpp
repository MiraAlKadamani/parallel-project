// kmeans_pthreads.cpp
// K-means in k dimensions using POSIX threads (pthreads).
//
// Usage:
//   ./kmeans_pthreads <csv_file> <K> <max_iters> [tolerance] [num_threads]
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
#include <thread>     // for std::thread::hardware_concurrency
#include <pthread.h>

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

// ---------------------------------------------------------------------------
// Thread data and worker function
// ---------------------------------------------------------------------------
struct ThreadData {
    const float *data;          // [N * d]
    const float *centroids;     // [K * d]
    int *assignments;           // [N]
    int N;
    int d;
    int K;
    int start_idx;
    int end_idx;
    int tid;
    float *local_sums;          // [num_threads * K * d] (thread's slice)
    int   *local_counts;        // [num_threads * K] (thread's slice)
    int   num_threads;
};

void* kmeans_worker(void *arg) {
    ThreadData *td = static_cast<ThreadData*>(arg);

    const float *data = td->data;
    const float *centroids = td->centroids;
    int *assignments = td->assignments;
    int N = td->N;
    int d = td->d;
    int K = td->K;
    int start = td->start_idx;
    int end   = td->end_idx;
    int tid   = td->tid;
    int num_threads = td->num_threads;

    // pointer to this thread's local accumulators
    float *local_sums   = td->local_sums   + tid * K * d;
    int   *local_counts = td->local_counts + tid * K;

    // zero local accumulators
    for (int i = 0; i < K * d; ++i) {
        local_sums[i] = 0.0f;
    }
    for (int k = 0; k < K; ++k) {
        local_counts[k] = 0;
    }

    // Assignment + local accumulation
    for (int i = start; i < end; ++i) {
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

        // accumulate into local sums
        local_counts[best_k] += 1;
        float *cluster_sum = &local_sums[best_k * d];
        for (int j = 0; j < d; ++j) {
            cluster_sum[j] += x[j];
        }
    }

    return nullptr;
}

// ---------------------------------------------------------------------------
// Pthreads K-means (driver)
// ---------------------------------------------------------------------------
void kmeans_pthreads(const std::vector<float> &data,
                     int N, int d, int K,
                     int max_iters, float tol, int num_threads,
                     std::vector<float> &centroids_out,
                     std::vector<int> &assignments_out) {
    // Initialize centroids (shared for all threads)
    std::vector<float> centroids;
    init_centroids_random(data, N, d, K, centroids);

    std::vector<int> assignments(N, 0);
    std::vector<float> global_sums(K * d);
    std::vector<int>   global_counts(K);

    // Thread-related allocations
    if (num_threads > N) num_threads = N; // no reason to have more threads than points
    if (num_threads < 1) num_threads = 1;

    // Per-thread accumulators, laid out as a single contiguous array
    std::vector<float> local_sums(num_threads * K * d);
    std::vector<int>   local_counts(num_threads * K);

    std::cout << "Using " << num_threads << " pthreads.\n";

    for (int iter = 0; iter < max_iters; ++iter) {
        // Reset global sums and counts
        std::fill(global_sums.begin(), global_sums.end(), 0.0f);
        std::fill(global_counts.begin(), global_counts.end(), 0);

        // Spawn threads
        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadData> tdata(num_threads);

        int base = 0;
        int chunk = N / num_threads;
        int remainder = N % num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start = base;
            int size = chunk + (t < remainder ? 1 : 0);
            int end = start + size;
            base = end;

            tdata[t].data         = data.data();
            tdata[t].centroids    = centroids.data();
            tdata[t].assignments  = assignments.data();
            tdata[t].N            = N;
            tdata[t].d            = d;
            tdata[t].K            = K;
            tdata[t].start_idx    = start;
            tdata[t].end_idx      = end;
            tdata[t].tid          = t;
            tdata[t].local_sums   = local_sums.data();
            tdata[t].local_counts = local_counts.data();
            tdata[t].num_threads  = num_threads;

            pthread_create(&threads[t], nullptr, kmeans_worker, &tdata[t]);
        }

        // Join threads
        for (int t = 0; t < num_threads; ++t) {
            pthread_join(threads[t], nullptr);
        }

        // Reduce per-thread local accumulators into global_sums / global_counts
        for (int t = 0; t < num_threads; ++t) {
            float *ls = local_sums.data()   + t * K * d;
            int   *lc = local_counts.data() + t * K;

            for (int k = 0; k < K; ++k) {
                global_counts[k] += lc[k];
                float *gsum = &global_sums[k * d];
                float *lsum = &ls[k * d];
                for (int j = 0; j < d; ++j) {
                    gsum[j] += lsum[j];
                }
            }
        }

        // Update centroids (sequential) and compute max shift
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
                  << " <csv_file> <K> <max_iters> [tolerance] [num_threads]\n";
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int max_iters = std::stoi(argv[3]);
    float tol = 1e-4f;
    int argi = 4;

    if (argi < argc) {
        tol = std::stof(argv[argi++]); // optional tolerance
    }

    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // fallback

    if (argi < argc) {
        num_threads = std::stoi(argv[argi++]); // optional num_threads
    }

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

    std::cout << "Running PTHREADS K-means with K=" << K
              << ", max_iters=" << max_iters
              << ", tolerance=" << tol << "\n";

    std::vector<float> centroids;
    std::vector<int> assignments;

    auto t_start = std::chrono::high_resolution_clock::now();

    kmeans_pthreads(data, N, d, K, max_iters, tol, num_threads,
                    centroids, assignments);

    auto t_end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "Total K-means time (pthreads): " << seconds << " s\n\n";

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

    return EXIT_SUCCESS;
}
