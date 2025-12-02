// kmeans_cuda_shared.cu
// K-means with shared-memory and tiling for centroids & per-block reductions.
//
// Usage:
//   ./main_cuda_kmeans_shared <csv_file> <K> <max_iters> [tolerance]
//
// CSV format: numeric only, comma-separated, no header.
//   - each row = one data point
//   - each column = one feature

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

// ---------------------------------------------------------------------------
// CUDA error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

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
// Device kernels with shared memory + tiling
// ---------------------------------------------------------------------------

// Assignment kernel:
// - Loads all K*d centroid coordinates into shared memory (a tile of centroids).
// - Each thread processes one point at a time.
// data: [N * d] row-major
// centroids: [K * d] row-major
// assignments: [N]
__global__
void assign_points_kernel_shared(const float *data,
                                 const float *centroids,
                                 int *assignments,
                                 int N, int d, int K) {
    extern __shared__ float centroids_sh[]; // size = K * d floats

    // 1) Load centroids into shared memory (all blocks see the full centroid tile)
    int total_cd = K * d;
    for (int i = threadIdx.x; i < total_cd; i += blockDim.x) {
        centroids_sh[i] = centroids[i];
    }
    __syncthreads();

    // 2) Compute nearest centroid for each point assigned to this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float *x = &data[idx * d];
    int best_k = 0;
    float best_dist = FLT_MAX;

    // tile over centroids (here the tile is actually all K centroids,
    // but conceptually this is a centroid-tile used by the block)
    for (int k = 0; k < K; ++k) {
        const float *c = &centroids_sh[k * d];
        float dist = 0.0f;
        // unrolled loop could be used here if d is known
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

// Update kernel with shared-memory tiling across points in a block:
// - Each block has a shared-memory buffer for K*d sums and K counts.
// - Threads accumulate into shared sums, then a subset of threads flush
//   those sums to global memory with a single set of atomics per cluster.
__global__
void accumulate_sums_kernel_shared(const float *data,
                                   const int *assignments,
                                   float *sums_global,
                                   int *counts_global,
                                   int N, int d, int K) {
    extern __shared__ float shmem[]; // dynamic shared memory

    // Layout in shared memory:
    // [0 .. K*d-1]   -> sums_sh (floats)
    // [K*d .. K*d + K-1] -> counts_sh (as float* then cast to int*)
    float *sums_sh = shmem;
    int   *counts_sh = reinterpret_cast<int*>(&sums_sh[K * d]);

    // 1) Initialize shared sums and counts to 0
    int total_sums = K * d;
    for (int i = threadIdx.x; i < total_sums; i += blockDim.x) {
        sums_sh[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        counts_sh[i] = 0;
    }
    __syncthreads();

    // 2) Grid-stride loop over points (tiling across points dimension)
    int global_stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += global_stride) {

        int k = assignments[idx];
        const float *x = &data[idx * d];

        // accumulate into shared memory (per-block tile)
        atomicAdd(&counts_sh[k], 1);
        int base = k * d;
        for (int j = 0; j < d; ++j) {
            atomicAdd(&sums_sh[base + j], x[j]);
        }
    }

    __syncthreads();

    // 3) Flush per-block shared sums to global sums using atomics
    for (int i = threadIdx.x; i < total_sums; i += blockDim.x) {
        atomicAdd(&sums_global[i], sums_sh[i]);
    }
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        atomicAdd(&counts_global[i], counts_sh[i]);
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

// Random initialization: pick K distinct random points as initial centroids
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

// Update centroids on host given sums and counts.
// Returns L2 max shift.
float update_centroids(std::vector<float> &centroids,
                       const std::vector<float> &sums,
                       const std::vector<int> &counts,
                       int d, int K) {
    float max_shift_sq = 0.0f;

    for (int k = 0; k < K; ++k) {
        int count = counts[k];
        if (count == 0) {
            // keep old centroid; could re-init if desired
            continue;
        }

        for (int j = 0; j < d; ++j) {
            float old_val = centroids[k * d + j];
            float new_val = sums[k * d + j] / static_cast<float>(count);
            float diff = new_val - old_val;
            float shift_sq = diff * diff;
            if (shift_sq > max_shift_sq) {
                max_shift_sq = shift_sq;
            }
            centroids[k * d + j] = new_val;
        }
    }

    return std::sqrt(max_shift_sq);
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
    std::vector<float> h_data;
    int N = 0, d = 0;
    if (!load_csv_numeric(filename, h_data, N, d)) {
        return EXIT_FAILURE;
    }

    if (K <= 0 || K > N) {
        std::cerr << "Error: invalid K (number of clusters).\n";
        return EXIT_FAILURE;
    }

    // Check shared memory requirement (rough sanity check)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t maxSharedPerBlock = prop.sharedMemPerBlock;
    size_t neededAssign = static_cast<size_t>(K) * d * sizeof(float);
    size_t neededUpdate = static_cast<size_t>(K) * d * sizeof(float)
                        + static_cast<size_t>(K) * sizeof(int);

    if (neededAssign > maxSharedPerBlock || neededUpdate > maxSharedPerBlock) {
        std::cerr << "Warning: K*d too large for shared memory tiling on this GPU.\n"
                  << " neededAssign = " << neededAssign
                  << " bytes, neededUpdate = " << neededUpdate
                  << " bytes, maxShared = " << maxSharedPerBlock << " bytes.\n"
                  << "Reduce K or d, or implement multi-tile scheme.\n";
    }

    std::cout << "Running K-means (shared memory) with K=" << K
              << ", max_iters=" << max_iters
              << ", tolerance=" << tol << "\n";

    // Initialize centroids on host
    std::vector<float> h_centroids;
    init_centroids_random(h_data, N, d, K, h_centroids);

    // Allocate device memory
    float *d_data = nullptr;
    float *d_centroids = nullptr;
    int   *d_assignments = nullptr;
    float *d_sums = nullptr;
    int   *d_counts = nullptr;

    size_t data_bytes = static_cast<size_t>(N) * d * sizeof(float);
    size_t centroid_bytes = static_cast<size_t>(K) * d * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_centroids, centroid_bytes));
    CUDA_CHECK(cudaMalloc(&d_assignments, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sums, centroid_bytes));
    CUDA_CHECK(cudaMalloc(&d_counts, K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), data_bytes,
                          cudaMemcpyHostToDevice));

    // Host buffers for sums and counts
    std::vector<float> h_sums(K * d);
    std::vector<int>   h_counts(K);

    // Kernel launch configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    if (gridSize == 0) gridSize = 1;

    // Shared memory sizes
    size_t sharedBytesAssign = static_cast<size_t>(K) * d * sizeof(float);
    size_t sharedBytesUpdate = static_cast<size_t>(K) * d * sizeof(float)
                             + static_cast<size_t>(K) * sizeof(int);

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    int iter = 0;
    for (iter = 0; iter < max_iters; ++iter) {
        // Copy centroids to device
        CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), centroid_bytes,
                              cudaMemcpyHostToDevice));

        // 1. Assignment step (centroid tile in shared memory)
        assign_points_kernel_shared<<<gridSize, blockSize, sharedBytesAssign>>>(
            d_data, d_centroids, d_assignments, N, d, K);
        CUDA_CHECK(cudaGetLastError());

        // 2. Reset global sums and counts
        CUDA_CHECK(cudaMemset(d_sums, 0, centroid_bytes));
        CUDA_CHECK(cudaMemset(d_counts, 0, K * sizeof(int)));

        // 3. Accumulate sums with per-block shared-memory reduction
        accumulate_sums_kernel_shared<<<gridSize, blockSize, sharedBytesUpdate>>>(
            d_data, d_assignments, d_sums, d_counts, N, d, K);
        CUDA_CHECK(cudaGetLastError());

        // 4. Copy sums and counts back to host
        CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums, centroid_bytes,
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts, K * sizeof(int),
                              cudaMemcpyDeviceToHost));

        // 5. Update centroids on host
        float shift = update_centroids(h_centroids, h_sums, h_counts, d, K);

        std::cout << "Iteration " << iter
                  << ", centroid max shift = " << shift << "\n";

        if (shift < tol) {
            std::cout << "Converged at iteration " << iter << "\n";
            break;
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::cout << "Total K-means time on GPU (shared mem): "
              << ms / 1000.0f << " s\n";

    // Print final centroids
    std::cout << "\nFinal centroids:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << "Centroid " << k << ": ";
        for (int j = 0; j < d; ++j) {
            std::cout << h_centroids[k * d + j];
            if (j + 1 < d) std::cout << ", ";
        }
        std::cout << "\n";
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_assignments));
    CUDA_CHECK(cudaFree(d_sums));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
