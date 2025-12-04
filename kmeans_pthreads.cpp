// kmeans_pthreads.cpp
// Same K-means logic but this time I'm manually creating threads myself (POSIX threads)

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>
#include <thread>     // mainly to ask system how many threads exist
#include <pthread.h>

// CSV loader kept exactly the same as other implementations
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
                row_values.push_back(std::stof(cell));
                col_count++;
            } catch (...) {
                row_values.clear(); col_count = 0;
                break;
            }
        }

        if (col_count == 0)
            continue;

        if (num_dims == -1) num_dims = col_count;
        else if (col_count != num_dims) return false;

        data.insert(data.end(), row_values.begin(), row_values.end());
        num_points++;
    }

    return (num_points > 0 && num_dims > 0);
}

// random centroid init (same across versions)
void init_centroids_random(const std::vector<float> &data,
                           int N, int d, int K,
                           std::vector<float> &centroids) {
    centroids.resize(K * d);
    std::mt19937 rng(12345);  // fixed seed to debug easier
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> chosen;
    chosen.reserve(K);

    for (int k = 0; k < K; ++k) {
        int idx;
        do idx = dist(rng);
        while (std::find(chosen.begin(), chosen.end(), idx) != chosen.end());
        chosen.push_back(idx);

        // copy chosen point into centroid
        for (int j = 0; j < d; ++j)
            centroids[k*d + j] = data[idx*d + j];
    }
}

// distance calculation simple and reused everywhere
inline float squared_distance(const float *a, const float *b, int d) {
    float dist = 0.0f;
    for (int j = 0; j < d; ++j) {
        float diff = a[j] - b[j];
        dist += diff * diff;
    }
    return dist;
}

// Struct to pass multiple args into pthread worker
struct ThreadData {
    const float *data;
    const float *centroids;
    int *assignments;
    int N, d, K;
    int start_idx, end_idx;
    int tid;
    float *local_sums;  // each thread writes only to its own space
    int   *local_counts;
    int num_threads;
};

// what each thread does in parallel
void* kmeans_worker(void *arg) {
    ThreadData *td = static_cast<ThreadData*>(arg);

    const float *data = td->data;
    const float *centroids = td->centroids;
    int *assignments = td->assignments;

    int start = td->start_idx;
    int end   = td->end_idx;
    int tid   = td->tid;
    int K = td->K;
    int d = td->d;
    int num_threads = td->num_threads;

    // giving each thread its own mini accumulator to avoid race conditions
    float *local_sums = td->local_sums + tid * K * d;
    int   *local_counts = td->local_counts + tid * K;

    // wiping before starting
    std::fill(local_sums, local_sums + K*d, 0.0f);
    std::fill(local_counts, local_counts + K, 0);

    // assignment step for only this thread’s chunk
    for (int i = start; i < end; ++i) {
        const float *x = &data[i*d];

        int best_k = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (int k = 0; k < K; ++k) {
            float dist = squared_distance(x, centroids + k*d, d);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }

        assignments[i] = best_k;

        // update this thread’s partial sums
        local_counts[best_k]++;
        float *sum_ptr = &local_sums[best_k*d];
        for (int j = 0; j < d; ++j)
            sum_ptr[j] += x[j];
    }

    return nullptr;
}

// main pthread-based iteration logic
void kmeans_pthreads(const std::vector<float> &data,
                     int N, int d, int K,
                     int max_iters, float tol, int num_threads,
                     std::vector<float> &centroids_out,
                     std::vector<int> &assignments_out) {

    std::vector<float> centroids;
    init_centroids_random(data, N, d, K, centroids);

    std::vector<int> assignments(N);

    std::vector<float> global_sums(K * d);
    std::vector<int>   global_counts(K);

    // not useful to create more threads than points
    if (num_threads > N) num_threads = N;

    // allocate all per-thread work space in one block
    std::vector<float> local_sums(num_threads * K * d);
    std::vector<int>   local_counts(num_threads * K);

    std::cout << "Using " << num_threads << " pthreads.\n";

    for (int iter = 0; iter < max_iters; ++iter) {

        // clear global combine arrays
        std::fill(global_sums.begin(), global_sums.end(), 0.0f);
        std::fill(global_counts.begin(), global_counts.end(), 0);

        // launch threads manually
        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadData> info(num_threads);

        // splitting points between threads evenly
        int base = 0;
        int chunk = N / num_threads;
        int leftover = N % num_threads; // give one extra point to some threads

        for (int t = 0; t < num_threads; ++t) {
            int size = chunk + (t < leftover ? 1 : 0);
            int start = base;
            int end = base + size;
            base = end;

            info[t] = { data.data(), centroids.data(),
                        assignments.data(),
                        N, d, K,
                        start, end, t,
                        local_sums.data(), local_counts.data(), num_threads };

            pthread_create(&threads[t], nullptr, kmeans_worker, &info[t]);
        }

        for (int t = 0; t < num_threads; ++t)
            pthread_join(threads[t], nullptr);

        // combine all thread results here single-threaded
        for (int t = 0; t < num_threads; ++t) {
            float *ls = &local_sums[t * K * d];
            int *lc = &local_counts[t * K];

            for (int k = 0; k < K; ++k) {
                global_counts[k] += lc[k];
                for (int j = 0; j < d; ++j)
                    global_sums[k*d + j] += ls[k*d + j];
            }
        }

        // update centroids + check convergence
        float max_shift_sq = 0.0f;

        for (int k = 0; k < K; ++k) {
            if (global_counts[k] == 0) continue;

            float *c = &centroids[k*d];
            float *sum = &global_sums[k*d];

            for (int j = 0; j < d; ++j) {
                float new_val = sum[j] / global_counts[k];
                float diff = new_val - c[j];
                max_shift_sq = std::max(max_shift_sq, diff * diff);
                c[j] = new_val;
            }
        }

        float max_shift = std::sqrt(max_shift_sq);
        std::cout << "Iteration " << iter << ", shift = " << max_shift << "\n";

        if (max_shift < tol) break;
    }

    centroids_out = centroids;
    assignments_out = assignments;
}

int main(int argc, char **argv) {

    if (argc < 4) return EXIT_FAILURE;

    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int max_iters = std::stoi(argv[3]);

    float tol = 1e-4f;
    if (argc > 4) tol = std::stof(argv[4]);

    int num_threads = std::thread::hardware_concurrency();
    if (argc > 5) num_threads = std::stoi(argv[5]);

    std::vector<float> data;
    int N = 0, d = 0;
    if (!load_csv_numeric(filename, data, N, d))
        return EXIT_FAILURE;

    std::cout << "Running pthreads K-means...\n";

    std::vector<float> centroids;
    std::vector<int> assignments;

    auto start = std::chrono::high_resolution_clock::now();

    kmeans_pthreads(data, N, d, K,
                    max_iters, tol, num_threads,
                    centroids, assignments);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Total time: "
              << std::chrono::duration<double>(end - start).count()
              << " s\n";

    for (int k = 0; k < K; ++k) {
        std::cout << "Centroid " << k << ": ";
        for (int j = 0; j < d; ++j)
            std::cout << centroids[k*d + j] << (j+1<d?", ":"");
        std::cout << "\n";
    }

    return 0;
}
