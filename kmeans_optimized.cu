// kmeans_cuda_shared.cu
// CUDA K-means optimized by using shared memory to reduce global reads
// and block-level partial reductions to reduce atomic contention.

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
#include <cfloat>

// basic runtime error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// same CSV loader used for CPU/MPI/OpenMP versions
bool load_csv_numeric(const std::string &filename,
                      std::vector<float> &data,
                      int &num_points,
                      int &num_dims) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening CSV\n";
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
        std::vector<float> row;
        int cols = 0;

        while (std::getline(ss, cell, ',')) {
            if (cell.empty()) continue;
            try {
                row.push_back(std::stof(cell));
                cols++;
            } catch (...) {
                row.clear();
                cols = 0;
                break;
            }
        }

        if (cols == 0) continue;

        if (num_dims == -1) num_dims = cols;
        else if (cols != num_dims) return false;

        data.insert(data.end(), row.begin(), row.end());
        num_points++;
    }

    return (num_points > 0 && num_dims > 0);
}

// ---------------------------------------------------------
// KERNEL 1: Each block loads full centroids into shared mem
// -> huge reduction in global memory traffic
// -> each thread computes nearest centroid for one point
// ---------------------------------------------------------
__global__
void assign_points_kernel_shared(const float *data,
                                 const float *centroids,
                                 int *assignments,
                                 int N, int d, int K) {

    extern __shared__ float centroids_sh[]; // K*d elements

    // cooperatively load centroids tile into shared mem once per block
    for (int i = threadIdx.x; i < K * d; i += blockDim.x)
        centroids_sh[i] = centroids[i];
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float *x = &data[idx * d];
    float best_dist = FLT_MAX;
    int best_k = 0;

    // distance compute now reading from fast shared memory
    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;
        const float *c = &centroids_sh[k * d];
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

// ---------------------------------------------------------
// KERNEL 2: Block-local partial reduction in shared memory
// -> reduces atomic writes to global memory massively
// ---------------------------------------------------------
__global__
void accumulate_sums_kernel_shared(const float *data,
                                   const int *assignments,
                                   float *sums_global,
                                   int *counts_global,
                                   int N, int d, int K) {

    extern __shared__ float shmem[];
    float *sums_sh = shmem;
    int *counts_sh = (int*)&sums_sh[K * d];

    // initialize shared buffers once per block
    for (int i = threadIdx.x; i < K*d; i += blockDim.x)
        sums_sh[i] = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        counts_sh[i] = 0;
    __syncthreads();

    // grid-stride loop so each thread can process >1 point
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += stride) {
        int k = assignments[idx];
        const float *x = &data[idx * d];

        atomicAdd(&counts_sh[k], 1);
        for (int j = 0; j < d; ++j)
            atomicAdd(&sums_sh[k * d + j], x[j]);
    }
    __syncthreads();

    // flush block results once (fewer atomics in total)
    for (int i = threadIdx.x; i < K*d; i += blockDim.x)
        atomicAdd(&sums_global[i], sums_sh[i]);
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        atomicAdd(&counts_global[i], counts_sh[i]);
}

// random init of K centroids on CPU — unchanged from other versions
void init_centroids_random(const std::vector<float> &data,
                           int N, int d, int K,
                           std::vector<float> &centroids) {
    centroids.resize(K * d);
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, N-1);
    std::vector<int> used;
    used.reserve(K);

    for (int k = 0; k < K; ++k) {
        int idx;
        do idx = dist(rng);
        while (std::find(used.begin(), used.end(), idx) != used.end());
        used.push_back(idx);
        for (int j = 0; j < d; ++j)
            centroids[k*d + j] = data[idx * d + j];
    }
}

// simple host centroid update + convergence tracking
float update_centroids(std::vector<float> &centroids,
                       const std::vector<float> &sums,
                       const std::vector<int> &counts,
                       int d, int K) {
    float max_shift = 0.0f;
    for (int k = 0; k < K; ++k) {
        if (counts[k] == 0) continue; // avoid division by zero
        for (int j = 0; j < d; ++j) {
            int id = k*d + j;
            float new_val = sums[id] / counts[k];
            float diff = new_val - centroids[id];
            max_shift = fmaxf(max_shift, diff * diff);
            centroids[id] = new_val;
        }
    }
    return sqrtf(max_shift);
}

int main(int argc, char **argv) {

    if (argc < 4) return EXIT_FAILURE;

    std::string filename = argv[1];
    int K = std::stoi(argv[2]);
    int max_iters = std::stoi(argv[3]);
    float tol = (argc >= 5 ? std::stof(argv[4]) : 1e-4f);

    // CPU loads all points once
    std::vector<float> h_data;
    int N = 0, d = 0;
    if (!load_csv_numeric(filename, h_data, N, d)) return EXIT_FAILURE;

    // initial random centroids
    std::vector<float> h_centroids;
    init_centroids_random(h_data, N, d, K, h_centroids);

    // device allocations
    float *d_data, *d_centroids, *d_sums;
    int *d_assignments, *d_counts;

    size_t data_bytes = (size_t)N*d*sizeof(float);
    size_t centroid_bytes = (size_t)K*d*sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_centroids, centroid_bytes));
    CUDA_CHECK(cudaMalloc(&d_assignments, N*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums, centroid_bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, K*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_bytes,
                          cudaMemcpyHostToDevice));

    // vectors for pulling reduce results back to CPU side
    std::vector<float> h_sums(K * d);
    std::vector<int>   h_counts(K);

    int block = 256;
    int grid = (N + block - 1) / block; // ensure all points covered

    // shared memory sizes = K*d floats (and +K ints in accumulate kernel)
    size_t shAssign = (size_t)K*d*sizeof(float);
    size_t shUpdate = shAssign + (size_t)K*sizeof(int);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int it = 0; it < max_iters; ++it) {

        // push new centroids down to GPU each iteration
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(),
                              centroid_bytes, cudaMemcpyHostToDevice));

        // Kernel 1: shared-mem assignment
        assign_points_kernel_shared<<<grid, block, shAssign>>>(
            d_data, d_centroids, d_assignments, N, d, K);

        // reset global buffers before reduction pass
        CUDA_CHECK(cudaMemset(d_sums, 0, centroid_bytes));
        CUDA_CHECK(cudaMemset(d_counts, 0, K*sizeof(int)));

        // Kernel 2: shared reduction → global
        accumulate_sums_kernel_shared<<<grid, block, shUpdate>>>(
            d_data, d_assignments, d_sums, d_counts, N, d, K);

        // bring partial results back up to CPU
        CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums,
                              centroid_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                              K*sizeof(int), cudaMemcpyDeviceToHost));

        float shift = update_centroids(h_centroids, h_sums, h_counts, d, K);
        std::cout << "Iteration " << it << " shift=" << shift << "\n";

        if (shift < tol) {
            std::cout << "Converged\n";
            break;
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Total GPU time (shared optimized) = " << ms/1000.0f << " s\n";

    // show centroids for validation
    for (int k = 0; k < K; ++k) {
        std::cout << "Centroid " << k << ": ";
        for (int j = 0; j < d; ++j)
            std::cout << h_centroids[k*d+j] << (j+1<d?", ":"");
        std::cout << "\n";
    }

    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_sums);
    cudaFree(d_counts);

    return 0;
}
