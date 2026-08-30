#include "include/ane_lm_runtime.h"

#include "../core/ane_runtime.h"
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

} // namespace

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
        std::ifstream config(std::string(model_directory) + "/config.json");
        if (!config.is_open()) throw std::runtime_error("Missing config.json");
        nlohmann::json json = nlohmann::json::parse(config, nullptr, false);
        if (json.is_discarded() || json.value("model_type", "") != "qwen3") {
            throw std::runtime_error("Only Qwen3 models are supported");
        }

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
