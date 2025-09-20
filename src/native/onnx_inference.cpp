//! High-performance ONNX Runtime C++ implementation

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <cstring>

#ifdef USE_CUDA
#include <cuda_provider_factory.h>
#endif

#ifdef USE_TENSORRT
#include <tensorrt_provider_factory.h>
#endif

extern "C" {

class OnnxInferenceEngine {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::vector<int64_t>> output_shapes;

    // Performance metrics
    std::chrono::microseconds last_inference_time;
    size_t total_inferences;
    double cumulative_time_ms;

public:
    OnnxInferenceEngine() : env(ORT_LOGGING_LEVEL_WARNING, "OnnxEngine"),
                            total_inferences(0), cumulative_time_ms(0.0) {
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Enable memory pattern optimization
        session_options.EnableMemPattern();
        session_options.EnableCpuMemArena();
    }

    bool initialize(const char* model_path, const char* device_type) {
        try {
            // Configure execution provider based on device type
            if (strcmp(device_type, "cuda") == 0) {
#ifdef USE_CUDA
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                cuda_options.arena_extend_strategy = 0;
                cuda_options.gpu_mem_limit = SIZE_MAX;
                cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
                cuda_options.do_copy_in_default_stream = 1;

                session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif
            } else if (strcmp(device_type, "tensorrt") == 0) {
#ifdef USE_TENSORRT
                OrtTensorRTProviderOptions trt_options;
                trt_options.device_id = 0;
                trt_options.trt_max_workspace_size = 2147483648; // 2GB
                trt_options.trt_fp16_enable = 1;
                trt_options.trt_int8_enable = 0;
                trt_options.trt_engine_cache_enable = 1;
                trt_options.trt_engine_cache_path = "/tmp/trt_cache";

                session_options.AppendExecutionProvider_TensorRT(trt_options);
#endif
            } else if (strcmp(device_type, "coreml") == 0) {
#ifdef __APPLE__
                // CoreML provider requires additional setup
                // Disabled for now due to API changes
#endif
            } else if (strcmp(device_type, "directml") == 0) {
#ifdef _WIN32
                session_options.AppendExecutionProvider_DML(0);
#endif
            }

            // Create session
            session = std::make_unique<Ort::Session>(env, model_path, session_options);

            // Get input/output metadata
            size_t num_inputs = session->GetInputCount();
            size_t num_outputs = session->GetOutputCount();

            for (size_t i = 0; i < num_inputs; i++) {
                auto input_name = session->GetInputNameAllocated(i, allocator);
                input_names.push_back(input_name.get());

                auto type_info = session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_shapes.push_back(tensor_info.GetShape());
            }

            for (size_t i = 0; i < num_outputs; i++) {
                auto output_name = session->GetOutputNameAllocated(i, allocator);
                output_names.push_back(output_name.get());

                auto type_info = session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_shapes.push_back(tensor_info.GetShape());
            }

            return true;
        } catch (const Ort::Exception& e) {
            return false;
        }
    }

    bool run_inference(const float* input_data, size_t /*input_size*/,
                      float* output_data, size_t* output_size) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Create input tensor
            std::vector<int64_t> input_shape = input_shapes[0];
            if (input_shape[0] == -1) {
                // Dynamic batch size
                input_shape[0] = 1;
            }

            size_t input_tensor_size = 1;
            for (auto& dim : input_shape) {
                input_tensor_size *= dim;
            }

            auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(input_data), input_tensor_size,
                input_shape.data(), input_shape.size());

            // Run inference
            std::vector<const char*> input_node_names = {input_names[0].c_str()};
            std::vector<const char*> output_node_names = {output_names[0].c_str()};

            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_node_names.data(),
                &input_tensor,
                1,
                output_node_names.data(),
                output_node_names.size());

            // Get output
            float* floatarr = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t output_count = output_shape_info.GetElementCount();

            if (*output_size < output_count) {
                *output_size = output_count;
                return false; // Buffer too small
            }

            memcpy(output_data, floatarr, output_count * sizeof(float));
            *output_size = output_count;

            // Update metrics
            auto end_time = std::chrono::high_resolution_clock::now();
            last_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);

            total_inferences++;
            cumulative_time_ms += last_inference_time.count() / 1000.0;

            return true;
        } catch (const Ort::Exception& e) {
            return false;
        }
    }

    // Batch inference for better throughput
    bool run_batch_inference(const float* batch_data, size_t batch_size,
                            size_t sample_size, float* output_data,
                            size_t* output_sizes) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();

            // Create batched input tensor
            std::vector<int64_t> batch_shape = input_shapes[0];
            batch_shape[0] = batch_size;

            size_t total_input_size = batch_size * sample_size;

            auto memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, const_cast<float*>(batch_data), total_input_size,
                batch_shape.data(), batch_shape.size());

            // Run batch inference
            std::vector<const char*> input_node_names = {input_names[0].c_str()};
            std::vector<const char*> output_node_names = {output_names[0].c_str()};

            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_node_names.data(),
                &input_tensor,
                1,
                output_node_names.data(),
                output_node_names.size());

            // Get batched output
            float* floatarr = output_tensors[0].GetTensorMutableData<float>();
            auto output_shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            auto output_shape = output_shape_info.GetShape();
            size_t output_per_sample = output_shape_info.GetElementCount() / batch_size;

            // Copy outputs for each sample
            for (size_t i = 0; i < batch_size; i++) {
                memcpy(&output_data[i * output_per_sample],
                       &floatarr[i * output_per_sample],
                       output_per_sample * sizeof(float));
                output_sizes[i] = output_per_sample;
            }

            // Update metrics
            auto end_time = std::chrono::high_resolution_clock::now();
            last_inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time);

            total_inferences += batch_size;
            cumulative_time_ms += last_inference_time.count() / 1000.0;

            return true;
        } catch (const Ort::Exception& e) {
            return false;
        }
    }

    double get_average_latency_ms() const {
        if (total_inferences == 0) return 0.0;
        return cumulative_time_ms / total_inferences;
    }

    double get_last_latency_ms() const {
        return last_inference_time.count() / 1000.0;
    }

    size_t get_total_inferences() const {
        return total_inferences;
    }
};

// Whisper-specific ONNX implementation
class WhisperOnnxModel {
private:
    OnnxInferenceEngine encoder;
    OnnxInferenceEngine decoder;

    // Model configuration
    static constexpr size_t n_audio_ctx = 1500;
    static constexpr size_t n_text_ctx = 448;

public:
    WhisperOnnxModel() {}

    bool initialize(const char* encoder_path, const char* decoder_path,
                   const char* device_type) {
        if (!encoder.initialize(encoder_path, device_type)) {
            return false;
        }

        if (!decoder.initialize(decoder_path, device_type)) {
            return false;
        }

        return true;
    }

    bool transcribe(const float* mel_spectrogram, size_t mel_size,
                   int32_t* token_ids, size_t* token_count) {
        // Run encoder
        std::vector<float> encoder_output(n_audio_ctx * 512); // Assuming 512-dim embeddings
        size_t encoder_output_size = encoder_output.size();

        if (!encoder.run_inference(mel_spectrogram, mel_size,
                                  encoder_output.data(), &encoder_output_size)) {
            return false;
        }

        // Autoregressive decoding
        std::vector<int32_t> generated_tokens;
        int32_t prev_token = 50258; // <|startoftranscript|>

        for (size_t i = 0; i < n_text_ctx; i++) {
            // Prepare decoder input
            std::vector<float> decoder_input(n_audio_ctx * 512 + 1);
            memcpy(decoder_input.data(), encoder_output.data(), encoder_output_size * sizeof(float));
            decoder_input[encoder_output_size] = static_cast<float>(prev_token);

            // Run decoder
            float logits[51865]; // Whisper vocab size
            size_t logits_size = sizeof(logits) / sizeof(float);

            if (!decoder.run_inference(decoder_input.data(), encoder_output_size + 1,
                                      logits, &logits_size)) {
                break;
            }

            // Get next token (argmax)
            int32_t next_token = 0;
            float max_logit = logits[0];
            for (size_t j = 1; j < logits_size; j++) {
                if (logits[j] > max_logit) {
                    max_logit = logits[j];
                    next_token = j;
                }
            }

            generated_tokens.push_back(next_token);

            // Check for end token
            if (next_token == 50257) { // <|endoftext|>
                break;
            }

            prev_token = next_token;
        }

        // Copy tokens to output
        if (*token_count < generated_tokens.size()) {
            *token_count = generated_tokens.size();
            return false; // Buffer too small
        }

        memcpy(token_ids, generated_tokens.data(),
               generated_tokens.size() * sizeof(int32_t));
        *token_count = generated_tokens.size();

        return true;
    }

    double get_encoder_latency_ms() const {
        return encoder.get_average_latency_ms();
    }

    double get_decoder_latency_ms() const {
        return decoder.get_average_latency_ms();
    }
};

// C API exports
void* onnx_engine_create() {
    return new OnnxInferenceEngine();
}

void onnx_engine_destroy(void* engine) {
    delete static_cast<OnnxInferenceEngine*>(engine);
}

int onnx_engine_initialize(void* engine, const char* model_path,
                          const char* device_type) {
    return static_cast<OnnxInferenceEngine*>(engine)->initialize(
        model_path, device_type) ? 1 : 0;
}

int onnx_engine_run(void* engine, const float* input, uint32_t input_size,
                   float* output, uint32_t* output_size) {
    size_t out_size = *output_size;
    bool success = static_cast<OnnxInferenceEngine*>(engine)->run_inference(
        input, input_size, output, &out_size);
    *output_size = out_size;
    return success ? 1 : 0;
}

int onnx_engine_run_batch(void* engine, const float* batch_data,
                         uint32_t batch_size, uint32_t sample_size,
                         float* output_data, uint32_t* output_sizes) {
    std::vector<size_t> sizes(batch_size);
    bool success = static_cast<OnnxInferenceEngine*>(engine)->run_batch_inference(
        batch_data, batch_size, sample_size, output_data, sizes.data());

    for (size_t i = 0; i < batch_size; i++) {
        output_sizes[i] = sizes[i];
    }

    return success ? 1 : 0;
}

float onnx_engine_get_latency_ms(void* engine) {
    return static_cast<OnnxInferenceEngine*>(engine)->get_average_latency_ms();
}

// Whisper-specific API
void* whisper_onnx_create() {
    return new WhisperOnnxModel();
}

void whisper_onnx_destroy(void* model) {
    delete static_cast<WhisperOnnxModel*>(model);
}

int whisper_onnx_initialize(void* model, const char* encoder_path,
                           const char* decoder_path, const char* device) {
    return static_cast<WhisperOnnxModel*>(model)->initialize(
        encoder_path, decoder_path, device) ? 1 : 0;
}

int whisper_onnx_transcribe(void* model, const float* mel_spectrogram,
                           uint32_t mel_size, int32_t* tokens,
                           uint32_t* token_count) {
    size_t count = *token_count;
    bool success = static_cast<WhisperOnnxModel*>(model)->transcribe(
        mel_spectrogram, mel_size, tokens, &count);
    *token_count = count;
    return success ? 1 : 0;
}

} // extern "C"