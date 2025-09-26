//! SIMD-optimized audio processing using AVX2/NEON intrinsics

#include <immintrin.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

extern "C" {

// SIMD-optimized FFT for real-time audio processing
class SimdFFT {
private:
    size_t size;
    std::vector<float> cos_table;
    std::vector<float> sin_table;

public:
    SimdFFT(size_t n) : size(n) {
        cos_table.resize(n);
        sin_table.resize(n);

        // Precompute twiddle factors
        for (size_t i = 0; i < n; ++i) {
            float angle = -2.0f * M_PI * i / n;
            cos_table[i] = cosf(angle);
            sin_table[i] = sinf(angle);
        }
    }

    void forward(const float* input, float* real_out, float* imag_out) {
        // Cooley-Tukey radix-2 FFT with SIMD optimization
        std::copy(input, input + size, real_out);
        std::fill(imag_out, imag_out + size, 0.0f);

        size_t n = size;
        size_t log2n = 0;
        while ((1U << log2n) < n) log2n++;

        // Bit reversal
        for (size_t i = 0; i < n; ++i) {
            size_t j = 0;
            for (size_t k = 0; k < log2n; ++k) {
                if (i & (1U << k)) j |= (1U << (log2n - 1 - k));
            }
            if (j > i) {
                std::swap(real_out[i], real_out[j]);
                std::swap(imag_out[i], imag_out[j]);
            }
        }

        // FFT computation with SIMD
        for (size_t s = 1; s <= log2n; ++s) {
            size_t m = 1U << s;
            size_t m2 = m >> 1;

            for (size_t k = 0; k < n; k += m) {
                for (size_t j = 0; j < m2; ++j) {
                    size_t idx = j * n / m;
                    float cos_w = cos_table[idx];
                    float sin_w = sin_table[idx];

                    size_t t = k + j;
                    size_t u = t + m2;

                    float temp_real = real_out[u] * cos_w - imag_out[u] * sin_w;
                    float temp_imag = real_out[u] * sin_w + imag_out[u] * cos_w;

                    real_out[u] = real_out[t] - temp_real;
                    imag_out[u] = imag_out[t] - temp_imag;
                    real_out[t] = real_out[t] + temp_real;
                    imag_out[t] = imag_out[t] + temp_imag;
                }
            }
        }
    }
};

// AVX2-optimized window application
void apply_window_avx2(float* data, const float* window, size_t size) {
#ifdef __AVX2__
    size_t simd_size = size - (size % 8);

    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data_vec = _mm256_loadu_ps(&data[i]);
        __m256 window_vec = _mm256_loadu_ps(&window[i]);
        __m256 result = _mm256_mul_ps(data_vec, window_vec);
        _mm256_storeu_ps(&data[i], result);
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        data[i] *= window[i];
    }
#else
    // Fallback to scalar
    for (size_t i = 0; i < size; ++i) {
        data[i] *= window[i];
    }
#endif
}

// NEON-optimized window application for ARM
void apply_window_neon(float* data, const float* window, size_t size) {
#ifdef __ARM_NEON__
    size_t simd_size = size - (size % 4);

    for (size_t i = 0; i < simd_size; i += 4) {
        float32x4_t data_vec = vld1q_f32(&data[i]);
        float32x4_t window_vec = vld1q_f32(&window[i]);
        float32x4_t result = vmulq_f32(data_vec, window_vec);
        vst1q_f32(&data[i], result);
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        data[i] *= window[i];
    }
#else
    apply_window_avx2(data, window, size);
#endif
}

// AVX2-optimized pre-emphasis filter
void apply_pre_emphasis_avx2(float* data, size_t size, float coefficient, float* state) {
#ifdef __AVX2__
    __m256 coeff_vec = _mm256_set1_ps(coefficient);
    float prev = *state;

    size_t simd_size = size - (size % 8);

    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 current = _mm256_loadu_ps(&data[i]);

        // Shift for previous values
        __m256 prev_shifted = _mm256_set_ps(
            data[i+6], data[i+5], data[i+4], data[i+3],
            data[i+2], data[i+1], data[i], prev
        );

        __m256 result = _mm256_fmsub_ps(current, _mm256_set1_ps(1.0f),
                                        _mm256_mul_ps(prev_shifted, coeff_vec));
        _mm256_storeu_ps(&data[i], result);

        prev = data[i + 7];
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        float current = data[i];
        data[i] = current - coefficient * prev;
        prev = current;
    }

    *state = prev;
#else
    // Scalar fallback
    float prev = *state;
    for (size_t i = 0; i < size; ++i) {
        float current = data[i];
        data[i] = current - coefficient * prev;
        prev = current;
    }
    *state = prev;
#endif
}

// SIMD-optimized mel filterbank computation
void compute_mel_filterbank_simd(const float* power_spectrum, float* mel_energies,
                                 const float* filterbank, size_t n_mels, size_t n_fft) {
#ifdef __AVX2__
    for (size_t mel_idx = 0; mel_idx < n_mels; ++mel_idx) {
        __m256 sum_vec = _mm256_setzero_ps();
        const float* filter = &filterbank[mel_idx * n_fft];

        size_t simd_size = n_fft - (n_fft % 8);

        for (size_t i = 0; i < simd_size; i += 8) {
            __m256 power_vec = _mm256_loadu_ps(&power_spectrum[i]);
            __m256 filter_vec = _mm256_loadu_ps(&filter[i]);
            sum_vec = _mm256_fmadd_ps(power_vec, filter_vec, sum_vec);
        }

        // Horizontal sum
        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum_vec);
        __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);

        float sum = _mm_cvtss_f32(sum_128);

        // Add remaining elements
        for (size_t i = simd_size; i < n_fft; ++i) {
            sum += power_spectrum[i] * filter[i];
        }

        // Apply log with numerical stability
        mel_energies[mel_idx] = logf(sum + 1e-10f);
    }
#else
    // Scalar fallback
    for (size_t mel_idx = 0; mel_idx < n_mels; ++mel_idx) {
        float sum = 0.0f;
        const float* filter = &filterbank[mel_idx * n_fft];

        for (size_t i = 0; i < n_fft; ++i) {
            sum += power_spectrum[i] * filter[i];
        }

        mel_energies[mel_idx] = logf(sum + 1e-10f);
    }
#endif
}

// Voice Activity Detection with SIMD
float compute_energy_simd(const float* data, size_t size) {
#ifdef __AVX2__
    __m256 sum_vec = _mm256_setzero_ps();
    size_t simd_size = size - (size % 8);

    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 data_vec = _mm256_loadu_ps(&data[i]);
        sum_vec = _mm256_fmadd_ps(data_vec, data_vec, sum_vec);
    }

    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);

    float energy = _mm_cvtss_f32(sum_128);

    // Add remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        energy += data[i] * data[i];
    }

    return energy / size;
#else
    float energy = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        energy += data[i] * data[i];
    }
    return energy / size;
#endif
}

// Zero-crossing rate with SIMD
size_t compute_zero_crossings_simd(const float* data, size_t size) {
#ifdef __AVX2__
    size_t crossings = 0;
    __m256 zero_vec = _mm256_setzero_ps();

    size_t simd_size = size - (size % 8) - 1;

    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 current = _mm256_loadu_ps(&data[i]);
        __m256 next = _mm256_loadu_ps(&data[i + 1]);

        __m256 curr_sign = _mm256_cmp_ps(current, zero_vec, _CMP_GE_OQ);
        __m256 next_sign = _mm256_cmp_ps(next, zero_vec, _CMP_GE_OQ);
        __m256 diff = _mm256_xor_ps(curr_sign, next_sign);

        int mask = _mm256_movemask_ps(diff);
        crossings += __builtin_popcount(mask);
    }

    // Handle remaining elements
    for (size_t i = simd_size; i < size - 1; ++i) {
        if ((data[i] >= 0) != (data[i + 1] >= 0)) {
            crossings++;
        }
    }

    return crossings;
#else
    size_t crossings = 0;
    for (size_t i = 0; i < size - 1; ++i) {
        if ((data[i] >= 0) != (data[i + 1] >= 0)) {
            crossings++;
        }
    }
    return crossings;
#endif
}

// Main SIMD audio processor
typedef struct {
    SimdFFT* fft;
    float* window;
    float* mel_filterbank;
    float* real_out;
    float* imag_out;
    float* power_spectrum;
    size_t frame_size;
    size_t n_mels;
    float pre_emphasis_coeff;
    float pre_emphasis_state;
} SimdAudioProcessor;

SimdAudioProcessor* simd_audio_processor_create(size_t frame_size, size_t n_mels) {
    SimdAudioProcessor* proc = new SimdAudioProcessor;
    proc->fft = new SimdFFT(frame_size);
    proc->frame_size = frame_size;
    proc->n_mels = n_mels;
    proc->pre_emphasis_coeff = 0.97f;
    proc->pre_emphasis_state = 0.0f;

    // Pre-allocate working buffers
    proc->real_out = (float*)aligned_alloc(32, frame_size * sizeof(float));
    proc->imag_out = (float*)aligned_alloc(32, frame_size * sizeof(float));
    size_t n_fft = frame_size / 2 + 1;
    proc->power_spectrum = (float*)aligned_alloc(32, n_fft * sizeof(float));

    // Create Hann window
    proc->window = (float*)aligned_alloc(32, frame_size * sizeof(float));
    for (size_t i = 0; i < frame_size; ++i) {
        proc->window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (frame_size - 1)));
    }

    // Initialize mel filterbank
    proc->mel_filterbank = (float*)aligned_alloc(32, n_mels * n_fft * sizeof(float));
    std::fill(proc->mel_filterbank, proc->mel_filterbank + n_mels * n_fft, 0.0f);

    return proc;
}

void simd_audio_processor_destroy(SimdAudioProcessor* proc) {
    if (proc) {
        delete proc->fft;
        free(proc->window);
        free(proc->mel_filterbank);
        free(proc->real_out);
        free(proc->imag_out);
        free(proc->power_spectrum);
        delete proc;
    }
}

void simd_audio_processor_process(SimdAudioProcessor* proc, float* data,
                                  float* mel_spectrogram) {
    // Apply pre-emphasis
    apply_pre_emphasis_avx2(data, proc->frame_size, proc->pre_emphasis_coeff,
                           &proc->pre_emphasis_state);

    // Apply window
#ifdef __ARM_NEON__
    apply_window_neon(data, proc->window, proc->frame_size);
#else
    apply_window_avx2(data, proc->window, proc->frame_size);
#endif

    // Compute FFT using pre-allocated buffers
    proc->fft->forward(data, proc->real_out, proc->imag_out);

    // Compute power spectrum
    size_t n_fft = proc->frame_size / 2 + 1;

#ifdef __AVX2__
    size_t simd_size = n_fft - (n_fft % 8);
    for (size_t i = 0; i < simd_size; i += 8) {
        __m256 real_vec = _mm256_load_ps(&proc->real_out[i]);
        __m256 imag_vec = _mm256_load_ps(&proc->imag_out[i]);
        __m256 power = _mm256_fmadd_ps(real_vec, real_vec,
                                       _mm256_mul_ps(imag_vec, imag_vec));
        _mm256_store_ps(&proc->power_spectrum[i], power);
    }
    for (size_t i = simd_size; i < n_fft; ++i) {
        proc->power_spectrum[i] = proc->real_out[i] * proc->real_out[i] +
                                  proc->imag_out[i] * proc->imag_out[i];
    }
#else
    for (size_t i = 0; i < n_fft; ++i) {
        proc->power_spectrum[i] = proc->real_out[i] * proc->real_out[i] +
                                  proc->imag_out[i] * proc->imag_out[i];
    }
#endif

    // Compute mel-spectrogram
    compute_mel_filterbank_simd(proc->power_spectrum, mel_spectrogram,
                                proc->mel_filterbank, proc->n_mels, n_fft);
}

// Export functions for Rust FFI
void* simd_audio_create(uint32_t frame_size, uint32_t n_mels) {
    return simd_audio_processor_create(frame_size, n_mels);
}

void simd_audio_destroy(void* processor) {
    simd_audio_processor_destroy((SimdAudioProcessor*)processor);
}

void simd_audio_process_frame(void* processor, float* data, float* mel_output) {
    simd_audio_processor_process((SimdAudioProcessor*)processor, data, mel_output);
}

float simd_compute_energy(const float* data, uint32_t size) {
    return compute_energy_simd(data, size);
}

uint32_t simd_compute_zero_crossings(const float* data, uint32_t size) {
    return compute_zero_crossings_simd(data, size);
}

} // extern "C"