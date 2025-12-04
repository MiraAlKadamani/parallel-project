// kmeans_cuda.cu
// Simple CUDA K-means implementation using GPU parallelism.
// This version relies mostly on global memory and atomic operations.
// I kept everything simple so it’s easy to explain.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <limits>

// helper macro to check for errors after CUDA calls
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// reading input CSV into 1D vector: row-major flattened layout
bool load_csv_numeric(const std::string &filename,
                      std::vector<float> &data,
                      int &num_points,
                      int &num_dims) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;

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

        // parsing row cells
        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) continue;
            try {
                row_values.push_back(std::stof(cell));
                col_count++;
            } catch (...) {
                col_count = 0;
                row_values.clear();
                break;
            }
        }

        if (col_count == 0) continue;

        if (num_dims == -1) num_dims = col_count;
        else if (col_count != num_dims) return false;

        // append row to full dataset
        data.insert(data.end(), row_values.begin(), row_values.end());
        num_points++;
    }

    if (num_points == 0) return false;
    std::cout << "Loaded: " << num_points << " points, " << num_dims << " dims\n";
    return true;
}

// GPU kernel: each thread handles 1 point → finds closest centroid
__global__
void assign_points_kernel(const float *data,
                          const float *centroids,
                          int *assignments,
                          int N, int d, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return; // out of bounds threads do nothing

    const float *x = &data[idx * d];
    float best_dist = FLT_MAX;
    int best_k = 0;

    // brute force: compare to all centroids
    for (int k = 0; k < K; ++k) {
        const float *c = &centroids[k * d];
        float dist = 0.0f;
        for (int j = 0; j < d; ++j) {
            float diff = x[j] - c[j];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }

    assignments[idx] = best_k;
}

// GPU kernel: sum values of each cluster using atomic adds
__global__
void accumulate_sums_kernel(const float *data,
                            const int *assignments,
                            float *sums,
                            int *counts,
                            int N, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int k = assignments[idx];
    const float *x = &data[idx * d];

    // atomically update cluster stats
    for (int j = 0; j < d; ++j)
        atomicAdd(&sums[k * d + j], x[j]);
    atomicAdd(&counts[k], 1);
}

// choose K random points from dataset as starting centroids
void init_centroids_random(const std::vector<float> &data,
                           int N, int d, int K,
                           std::vector<float> &centroids) {
    centroids.resize(K * d);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, N - 1);

    std::vector<int> used;
    for (int k = 0; k < K; ++k) {
        int idx;
        bool ok;
        do {
            ok = true;
            idx = dist(rng);
            for (int u : used)
                if (u == idx) ok = false;
        } while (!ok);

        used.push_back(idx);
        const float *src = &data[idx * d];
        float *dst = &centroids[k * d];
        for (int j = 0; j < d; ++j)
            dst[j] = src[j];
    }
}

// updating centroid = average of its assigned points
float update_centroids(std::vector<float> &centroids,
                       const std::vector<float> &sums,
                       const std::vector<int> &counts,
                       int d, int K) {
    float max_shift = 0.0f;
    for (int k = 0; k < K; ++k) {
        if (counts[k] == 0) continue; // cluster empty, keep as is

        for (int j = 0; j < d; ++j) {
            float old_val = centroids[k * d + j];
            float new_val = sums[k * d + j] / counts[k];
            float diff = new_val - old_val;
            float shift = diff * diff;
            if (shift > max_shift) max_shift = shift;
            centroids[k * d + j] = new_val;
        }
    }
    return sqrt(max_shift); // checking convergence magnitude
}

int main(int argc, char **argv) {
    if (argc < 4) return EXIT_FAILURE;

    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int max_iters = std::stoi(argv[3]);
    float tol = (argc >= 5 ? std::stof(argv[4]) : 1e-4f);

    // load dataset into host memory
    std::vector<float> h_data;
    int N = 0, d = 0;
    if (!load_csv_numeric(filename, h_data, N, d))
        return EXIT_FAILURE;

    if (K <= 0 || K > N) return EXIT_FAILURE;

    std::vector<float> h_centroids;
    init_centroids_random(h_data, N, d, K, h_centroids);

    // GPU memory allocation to move work onto device
    float *d_data = nullptr, *d_centroids = nullptr;
    int *d_assignments = nullptr;
    float *d_sums = nullptr;
    int *d_counts = nullptr;

    size_t bytes_data = N * d * sizeof(float);
    size_t bytes_centroids = K * d * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, bytes_data));
    CUDA_CHECK(cudaMalloc(&d_centroids, bytes_centroids));
    CUDA_CHECK(cudaMalloc(&d_assignments, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums, bytes_centroids));
    CUDA_CHECK(cudaMalloc(&d_counts, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes_data, cudaMemcpyHostToDevice));

    // these stay on CPU — we update centroids here
    std::vector<float> h_sums(K * d);
    std::vector<int> h_counts(K);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int iter = 0; iter < max_iters; ++iter) {
        // copy updated centroids to GPU
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), bytes_centroids, cudaMemcpyHostToDevice));

        // phase 1: assign points
        assign_points_kernel<<<gridSize, blockSize>>>(d_data, d_centroids, d_assignments, N, d, K);
        CUDA_CHECK(cudaGetLastError());

        // reset cluster accumulators
        CUDA_CHECK(cudaMemset(d_sums, 0, bytes_centroids));
        CUDA_CHECK(cudaMemset(d_counts, 0, K * sizeof(int)));

        // phase 2: accumulate per cluster
        accumulate_sums_kernel<<<gridSize, blockSize>>>(d_data, d_assignments, d_sums, d_counts, N, d);
        CUDA_CHECK(cudaGetLastError());

        // bring updated totals back to CPU
        CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums, bytes_centroids, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, K * sizeof(int), cudaMemcpyDeviceToHost));

        // compute new centroid positions here on CPU
        float shift = update_centroids(h_centroids, h_sums, h_counts, d, K);
        std::cout << "Iter " << iter << " shift=" << shift << "\n";

        if (shift < tol) {
            std::cout << "Converged!\n";
            break;
        }
    }

    // measure timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "GPU K-means time: " << ms / 1000.0f << " s\n";

    // show centroids found
    std::cout << "Final centroids:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << k << ": ";
        for (int j = 0; j < d; ++j) {
            std::cout << h_centroids[k * d + j] << (j+1<d ? ", " : "");
        }
        std::cout << "\n";
    }

    // cleanup GPU memory
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_sums);
    cudaFree(d_counts);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}
