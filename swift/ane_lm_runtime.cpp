#include "include/ane_lm_runtime.h"

#include "../core/ane_runtime.h"
#include "../core/model_loader.h"
#include "../core/sampling.h"
#include "../models/llm/qwen3.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void* objc_autoreleasePoolPush(void);
extern "C" void objc_autoreleasePoolPop(void*);

struct ane_lm_runtime {
    std::unique_ptr<ane_lm::Qwen3Model> model;
};

namespace {

void set_error(char **error_out, const std::string& message) {
    if (!error_out) return;
    *error_out = static_cast<char*>(malloc(message.size() + 1));
    if (*error_out) memcpy(*error_out, message.c_str(), message.size() + 1);
}

bool continue_generation(ane_lm_token_callback_t callback, int32_t token, void *context) {
    return !callback || callback(token, context) != 0;
}

void seed_sampler_once() {
    static std::once_flag once;
    std::call_once(once, [] { srand48(0x414E454C4D); });
}

int64_t required_positive_integer(
    const nlohmann::json& object,
    const char *key,
    int64_t maximum) {
    if (!object.contains(key) || !object[key].is_number_integer()) {
        throw std::runtime_error(std::string("Missing or invalid Qwen3 config value: ") + key);
    }
    int64_t value = object[key].get<int64_t>();
    if (value <= 0 || value > maximum) {
        throw std::runtime_error(std::string("Unsupported Qwen3 config value: ") + key);
    }
    return value;
}

void require_bf16_tensor(
    const ane_lm::ModelWeights& weights,
    const std::string& name,
    std::initializer_list<int64_t> expected_shape) {
    const ane_lm::SFTensor *tensor = weights.find(name.c_str());
    if (!tensor) throw std::runtime_error("Missing Qwen3 tensor: " + name);
    if (tensor->dtype != ane_lm::SFDtype::BF16) {
        throw std::runtime_error("Qwen3 tensor must use BF16: " + name);
    }
    if (tensor->ndims != static_cast<int>(expected_shape.size())) {
        throw std::runtime_error("Invalid Qwen3 tensor rank: " + name);
    }
    int index = 0;
    for (int64_t dimension : expected_shape) {
        if (tensor->shape[index++] != dimension) {
            throw std::runtime_error("Invalid Qwen3 tensor shape: " + name);
        }
    }
}

void validate_model_directory(const std::string& model_directory) {
    std::ifstream config(model_directory + "/config.json");
    if (!config.is_open()) throw std::runtime_error("Missing config.json");
    nlohmann::json json = nlohmann::json::parse(config, nullptr, false);
    if (json.is_discarded() || !json.is_object() || json.value("model_type", "") != "qwen3") {
        throw std::runtime_error("Only Qwen3 models are supported");
    }

    const nlohmann::json& text = json.contains("text_config") ? json["text_config"] : json;
    if (!text.is_object()) throw std::runtime_error("Invalid Qwen3 text_config");
    const int64_t hidden = required_positive_integer(text, "hidden_size", 65536);
    const int64_t intermediate = required_positive_integer(text, "intermediate_size", 262144);
    const int64_t layers = required_positive_integer(text, "num_hidden_layers", 256);
    const int64_t q_heads = required_positive_integer(text, "num_attention_heads", 256);
    const int64_t kv_heads = required_positive_integer(text, "num_key_value_heads", 256);
    const int64_t vocab = required_positive_integer(text, "vocab_size", 10000000);
    required_positive_integer(text, "max_position_embeddings", 10000000);
    const int64_t head_dim = text.contains("head_dim")
        ? required_positive_integer(text, "head_dim", 4096)
        : hidden / q_heads;
    if ((!text.contains("head_dim") && hidden % q_heads != 0) || kv_heads > q_heads
        || head_dim % 2 != 0) {
        throw std::runtime_error("Unsupported Qwen3 attention dimensions");
    }

    bool tied = json.value("tie_word_embeddings", text.value("tie_word_embeddings", true));
    std::unique_ptr<ane_lm::ModelWeights> weights = ane_lm::ModelWeights::open(model_directory);
    if (!weights) throw std::runtime_error("Invalid or incomplete Qwen3 safetensors weights");

    require_bf16_tensor(*weights, "model.embed_tokens.weight", {vocab, hidden});
    if (!tied) require_bf16_tensor(*weights, "lm_head.weight", {vocab, hidden});
    require_bf16_tensor(*weights, "model.norm.weight", {hidden});

    const int64_t q_projection = q_heads * head_dim;
    const int64_t kv_projection = kv_heads * head_dim;
    for (int64_t layer = 0; layer < layers; ++layer) {
        const std::string prefix = "model.layers." + std::to_string(layer);
        require_bf16_tensor(*weights, prefix + ".input_layernorm.weight", {hidden});
        require_bf16_tensor(*weights, prefix + ".post_attention_layernorm.weight", {hidden});
        require_bf16_tensor(*weights, prefix + ".self_attn.q_norm.weight", {head_dim});
        require_bf16_tensor(*weights, prefix + ".self_attn.k_norm.weight", {head_dim});
        require_bf16_tensor(*weights, prefix + ".self_attn.q_proj.weight", {q_projection, hidden});
        require_bf16_tensor(*weights, prefix + ".self_attn.k_proj.weight", {kv_projection, hidden});
        require_bf16_tensor(*weights, prefix + ".self_attn.v_proj.weight", {kv_projection, hidden});
        require_bf16_tensor(*weights, prefix + ".self_attn.o_proj.weight", {hidden, q_projection});
        require_bf16_tensor(*weights, prefix + ".mlp.gate_proj.weight", {intermediate, hidden});
        require_bf16_tensor(*weights, prefix + ".mlp.up_proj.weight", {intermediate, hidden});
        require_bf16_tensor(*weights, prefix + ".mlp.down_proj.weight", {hidden, intermediate});
    }
}

} // namespace

extern "C" int32_t ane_lm_validate_model(
    const char *model_directory,
    char **error_out) {
    if (error_out) *error_out = nullptr;
    if (!model_directory || model_directory[0] == '\0') {
        set_error(error_out, "Model directory is required");
        return ANE_LM_STATUS_ERROR;
    }
    try {
        validate_model_directory(model_directory);
        return ANE_LM_STATUS_OK;
    } catch (const std::exception& exception) {
        set_error(error_out, exception.what());
    } catch (...) {
        set_error(error_out, "Unknown ANE-LM validation error");
    }
    return ANE_LM_STATUS_ERROR;
}

extern "C" ane_lm_runtime_t *ane_lm_create(
    const char *model_directory,
    char **error_out) {
    if (error_out) *error_out = nullptr;
    if (!model_directory || model_directory[0] == '\0') {
        set_error(error_out, "Model directory is required");
        return nullptr;
    }

    void *pool = objc_autoreleasePoolPush();
    try {
        validate_model_directory(model_directory);

        ane_lm::ane_set_persist_cache(false);
        auto runtime = std::make_unique<ane_lm_runtime>();
        runtime->model = std::make_unique<ane_lm::Qwen3Model>();
        if (!runtime->model->load(model_directory)) {
            throw std::runtime_error("Failed to load the Qwen3 ANE model");
        }
        seed_sampler_once();
        ane_lm_runtime_t *result = runtime.release();
        objc_autoreleasePoolPop(pool);
        return result;
    } catch (const std::exception& exception) {
        set_error(error_out, exception.what());
    } catch (...) {
        set_error(error_out, "Unknown ANE-LM load error");
    }
    objc_autoreleasePoolPop(pool);
    return nullptr;
}

extern "C" void ane_lm_destroy(ane_lm_runtime_t *runtime) {
    if (!runtime) return;
    void *pool = objc_autoreleasePoolPush();
    delete runtime;
    objc_autoreleasePoolPop(pool);
}

extern "C" void ane_lm_reset(ane_lm_runtime_t *runtime) {
    if (runtime && runtime->model) runtime->model->reset();
}

extern "C" int32_t ane_lm_vocab_size(const ane_lm_runtime_t *runtime) {
    return runtime && runtime->model ? runtime->model->vocab_size() : 0;
}

extern "C" int32_t ane_lm_generate(
    ane_lm_runtime_t *runtime,
    const int32_t *prompt_tokens,
    size_t prompt_token_count,
    int32_t max_tokens,
    float temperature,
    float repetition_penalty,
    int32_t sampler_vocab_size,
    int32_t eos_token,
    int32_t stop_token,
    ane_lm_token_callback_t callback,
    void *context,
    char **error_out) {
    if (error_out) *error_out = nullptr;
    if (!runtime || !runtime->model) {
        set_error(error_out, "ANE-LM runtime is not loaded");
        return ANE_LM_STATUS_ERROR;
    }
    if (!prompt_tokens || prompt_token_count == 0 || max_tokens <= 0) {
        set_error(error_out, "Prompt tokens and a positive max token count are required");
        return ANE_LM_STATUS_ERROR;
    }

    void *pool = objc_autoreleasePoolPush();
    try {
        auto& model = *runtime->model;
        model.reset();
        float *logits = nullptr;
        int32_t model_vocab = model.vocab_size();
        for (size_t index = 0; index < prompt_token_count; ++index) {
            int32_t token = prompt_tokens[index];
            if (token < 0 || token >= model_vocab) {
                throw std::runtime_error("Prompt contains a token outside the model vocabulary");
            }
            if (!continue_generation(callback, -1, context)) {
                objc_autoreleasePoolPop(pool);
                return ANE_LM_STATUS_CANCELLED;
            }
            logits = model.forward(token, static_cast<int>(index));
            if (!logits) throw std::runtime_error("ANE forward pass failed during prefill");
        }

        int32_t sample_vocab = sampler_vocab_size > 0
            ? std::min(model_vocab, sampler_vocab_size)
            : model_vocab;
        if (sample_vocab <= 0) throw std::runtime_error("Invalid sampler vocabulary size");

        ane_lm::SamplingParams sampling;
        sampling.temperature = temperature;
        sampling.repetition_penalty = repetition_penalty;
        std::vector<int> generated;
        generated.reserve(static_cast<size_t>(max_tokens));

        for (int32_t index = 0; index < max_tokens; ++index) {
            int32_t token = ane_lm::sample_token(logits, sample_vocab, sampling, generated);
            if (token == eos_token || token == stop_token) break;
            generated.push_back(token);
            if (!continue_generation(callback, token, context)) {
                objc_autoreleasePoolPop(pool);
                return ANE_LM_STATUS_CANCELLED;
            }
            if (index + 1 == max_tokens) break;
            logits = model.forward(token, static_cast<int>(prompt_token_count) + index);
            if (!logits) throw std::runtime_error("ANE forward pass failed during generation");
        }
        objc_autoreleasePoolPop(pool);
        return ANE_LM_STATUS_OK;
    } catch (const std::exception& exception) {
        set_error(error_out, exception.what());
    } catch (...) {
        set_error(error_out, "Unknown ANE-LM generation error");
    }
    objc_autoreleasePoolPop(pool);
    return ANE_LM_STATUS_ERROR;
}

extern "C" void ane_lm_free_string(char *string) {
    free(string);
}
