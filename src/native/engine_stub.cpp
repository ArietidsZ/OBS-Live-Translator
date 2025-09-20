/**
 * Stub implementation of optimized engine for testing
 */

#include <cstring>
#include <cstdlib>

extern "C" {

// Stub implementations for testing
void* engine_create(const char* model_path, const char* device) {
    (void)model_path;  // Unused
    (void)device;      // Unused
    return malloc(1);  // Return non-null pointer
}

void engine_destroy(void* engine) {
    free(engine);
}

int engine_transcribe(void* engine, const float* audio, int audio_size,
                        char* result_buffer, int buffer_size, float* confidence) {
    (void)engine; (void)audio; (void)audio_size; (void)buffer_size;
    strcpy(result_buffer, "");  // Stub - no real implementation
    *confidence = 0.95f;
    return 0;
}

int engine_transcribe_batch(void* engine, const float** audio_batch,
                             const int* sizes, int batch_size,
                             char** results, float* confidences) {
    (void)engine; (void)audio_batch; (void)sizes;
    for (int i = 0; i < batch_size; i++) {
        strcpy(results[i], "");  // Stub - no real implementation
        confidences[i] = 0.95f;
    }
    return 0;
}

int engine_stream_start(void* engine) { (void)engine; return 0; }
int engine_stream_feed(void* engine, const float* audio, int size) {
    (void)engine; (void)audio; (void)size;
    return 0;
}
int engine_stream_get_result(void* engine, char* result_buffer,
                               int buffer_size, bool* is_final) {
    (void)engine; (void)buffer_size;
    strcpy(result_buffer, "");  // Stub - no real implementation
    *is_final = false;
    return 0;
}
int engine_stream_stop(void* engine) { (void)engine; return 0; }
int engine_set_language(void* engine, const char* language) {
    (void)engine; (void)language;
    return 0;
}
int engine_enable_vad(void* engine, bool enable) {
    (void)engine; (void)enable;
    return 0;
}
int engine_set_beam_size(void* engine, int beam_size) {
    (void)engine; (void)beam_size;
    return 0;
}
int engine_enable_word_timestamps(void* engine, bool enable) {
    (void)engine; (void)enable;
    return 0;
}
float engine_get_latency_ms(void* engine) { (void)engine; return 0.0f; }  // No real implementation
float engine_get_throughput_fps(void* engine) { (void)engine; return 0.0f; }  // No real implementation
float engine_get_memory_mb(void* engine) { (void)engine; return 0.0f; }  // No real implementation

// GPU stub implementations (CUDA not available)
void* create_gpu_context() { return nullptr; }
void destroy_gpu_context(void* context) { (void)context; }
int process_mel_spectrogram_gpu(void* context, const float* audio, int audio_size,
                                float* mel_output, int mel_rows, int mel_cols) {
    (void)context; (void)audio; (void)audio_size;
    (void)mel_output; (void)mel_rows; (void)mel_cols;
    return -1;  // Not implemented
}
int gpu_conv1d(void* context, const float* input, const float* weights,
               float* output, int batch_size, int in_channels,
               int out_channels, int input_length, int kernel_size, int stride) {
    (void)context; (void)input; (void)weights; (void)output;
    (void)batch_size; (void)in_channels; (void)out_channels;
    (void)input_length; (void)kernel_size; (void)stride;
    return -1;  // Not implemented
}

} // extern "C"