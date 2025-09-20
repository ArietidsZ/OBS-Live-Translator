//! CUDA kernels for GPU-accelerated audio processing and inference

#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// Tensor Core optimized matrix multiplication for Volta/Ampere GPUs
#if __CUDA_ARCH__ >= 700
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_gemm(const half* A, const half* B, float* C,
                                 int M, int N, int K) {
    // Tensor Core fragment size
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}
#endif

// Optimized FFT using cuFFT for audio processing
extern "C" {

struct GpuFFTContext {
    cufftHandle plan;
    float* d_input;
    cufftComplex* d_output;
    size_t fft_size;
};

GpuFFTContext* gpu_fft_create(int fft_size) {
    GpuFFTContext* ctx = new GpuFFTContext;
    ctx->fft_size = fft_size;

    // Allocate device memory
    cudaMalloc(&ctx->d_input, fft_size * sizeof(float));
    cudaMalloc(&ctx->d_output, (fft_size / 2 + 1) * sizeof(cufftComplex));

    // Create FFT plan
    cufftPlan1d(&ctx->plan, fft_size, CUFFT_R2C, 1);

    return ctx;
}

void gpu_fft_destroy(GpuFFTContext* ctx) {
    cufftDestroy(ctx->plan);
    cudaFree(ctx->d_input);
    cudaFree(ctx->d_output);
    delete ctx;
}

void gpu_fft_forward(GpuFFTContext* ctx, const float* input, float* magnitude) {
    // Copy input to device
    cudaMemcpy(ctx->d_input, input, ctx->fft_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // Execute FFT
    cufftExecR2C(ctx->plan, ctx->d_input, ctx->d_output);

    // Compute magnitude on GPU
    int n_bins = ctx->fft_size / 2 + 1;
    dim3 blocks((n_bins + 255) / 256);
    dim3 threads(256);

    compute_magnitude<<<blocks, threads>>>(ctx->d_output, magnitude, n_bins);

    cudaDeviceSynchronize();
}

__global__ void compute_magnitude(cufftComplex* complex_data, float* magnitude, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float real = complex_data[idx].x;
        float imag = complex_data[idx].y;
        magnitude[idx] = sqrtf(real * real + imag * imag);
    }
}

// Mel filterbank computation on GPU
__global__ void apply_mel_filterbank(const float* power_spectrum,
                                    const float* filterbank,
                                    float* mel_energies,
                                    int n_mels, int n_fft) {
    int mel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (mel_idx < n_mels) {
        float sum = 0.0f;
        const float* filter = &filterbank[mel_idx * n_fft];

        // Vectorized accumulation
        for (int i = 0; i < n_fft; i++) {
            sum += power_spectrum[i] * filter[i];
        }

        // Apply log with numerical stability
        mel_energies[mel_idx] = logf(sum + 1e-10f);
    }
}

// Attention mechanism for Transformer models (Whisper)
__global__ void scaled_dot_product_attention(
    const float* Q, const float* K, const float* V,
    float* output, int seq_len, int d_k, int d_v,
    float scale) {

    extern __shared__ float shared_mem[];
    float* scores = shared_mem;

    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int query_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Compute attention scores for this query
    if (tid < seq_len) {
        float score = 0.0f;
        int q_offset = (batch_idx * seq_len + query_idx) * d_k;
        int k_offset = (batch_idx * seq_len + tid) * d_k;

        // Dot product
        for (int i = 0; i < d_k; i++) {
            score += Q[q_offset + i] * K[k_offset + i];
        }

        scores[tid] = score * scale;
    }
    __syncthreads();

    // Softmax normalization
    if (tid == 0) {
        float max_score = scores[0];
        for (int i = 1; i < seq_len; i++) {
            max_score = fmaxf(max_score, scores[i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            scores[i] = expf(scores[i] - max_score);
            sum_exp += scores[i];
        }

        for (int i = 0; i < seq_len; i++) {
            scores[i] /= sum_exp;
        }
    }
    __syncthreads();

    // Weighted sum of values
    if (tid < d_v) {
        float out = 0.0f;
        int out_offset = (batch_idx * seq_len + query_idx) * d_v;

        for (int i = 0; i < seq_len; i++) {
            int v_offset = (batch_idx * seq_len + i) * d_v;
            out += scores[i] * V[v_offset + tid];
        }

        output[out_offset + tid] = out;
    }
}

// Batch normalization for inference
__global__ void batch_norm_inference(float* data, const float* mean,
                                    const float* var, const float* gamma,
                                    const float* beta, int size, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        int ch = idx % channels;
        float normalized = (data[idx] - mean[ch]) / sqrtf(var[ch] + 1e-5f);
        data[idx] = normalized * gamma[ch] + beta[ch];
    }
}

// Voice Activity Detection on GPU
__global__ void compute_frame_energy(const float* audio, float* energy,
                                    int frame_size, int num_frames) {
    int frame_idx = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float partial_sums[];

    float sum = 0.0f;
    int offset = frame_idx * frame_size;

    // Each thread computes partial sum
    for (int i = tid; i < frame_size; i += blockDim.x) {
        float val = audio[offset + i];
        sum += val * val;
    }

    partial_sums[tid] = sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        energy[frame_idx] = partial_sums[0] / frame_size;
    }
}

// Optimized convolution for CNN-based models
__global__ void depthwise_conv2d(const float* input, const float* kernel,
                                float* output, int H, int W, int C,
                                int kernel_size, int stride, int pad) {
    int out_h = blockIdx.y;
    int out_w = blockIdx.x;
    int c = threadIdx.x;

    if (c < C) {
        float sum = 0.0f;

        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int in_h = out_h * stride - pad + kh;
                int in_w = out_w * stride - pad + kw;

                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                    int in_idx = (in_h * W + in_w) * C + c;
                    int k_idx = (kh * kernel_size + kw) * C + c;
                    sum += input[in_idx] * kernel[k_idx];
                }
            }
        }

        int out_idx = (out_h * ((W + 2*pad - kernel_size)/stride + 1) + out_w) * C + c;
        output[out_idx] = sum;
    }
}

// GPU memory pool for efficient allocation
class GpuMemoryPool {
private:
    void* pool_ptr;
    size_t pool_size;
    size_t current_offset;

public:
    GpuMemoryPool(size_t size) : pool_size(size), current_offset(0) {
        cudaMalloc(&pool_ptr, size);
    }

    ~GpuMemoryPool() {
        cudaFree(pool_ptr);
    }

    void* allocate(size_t size) {
        size_t aligned_size = (size + 255) & ~255; // 256-byte alignment
        if (current_offset + aligned_size > pool_size) {
            return nullptr;
        }

        void* ptr = (char*)pool_ptr + current_offset;
        current_offset += aligned_size;
        return ptr;
    }

    void reset() {
        current_offset = 0;
    }
};

// C API exports
void* gpu_memory_pool_create(size_t size) {
    return new GpuMemoryPool(size);
}

void gpu_memory_pool_destroy(void* pool) {
    delete static_cast<GpuMemoryPool*>(pool);
}

void* gpu_memory_pool_allocate(void* pool, size_t size) {
    return static_cast<GpuMemoryPool*>(pool)->allocate(size);
}

void gpu_memory_pool_reset(void* pool) {
    static_cast<GpuMemoryPool*>(pool)->reset();
}

// GPU info functions
int gpu_get_device_count() {
    int count;
    cudaGetDeviceCount(&count);
    return count;
}

int gpu_get_compute_capability(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}

size_t gpu_get_memory_info() {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    return free;
}

} // extern "C"