#ifndef ANE_LM_RUNTIME_H
#define ANE_LM_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ane_lm_runtime ane_lm_runtime_t;

typedef int32_t (*ane_lm_token_callback_t)(int32_t token, void *context);

enum {
    ANE_LM_STATUS_OK = 0,
    ANE_LM_STATUS_CANCELLED = 1,
    ANE_LM_STATUS_ERROR = -1
};

ane_lm_runtime_t *ane_lm_create(const char *model_directory, char **error_out);
int32_t ane_lm_validate_model(const char *model_directory, char **error_out);
void ane_lm_destroy(ane_lm_runtime_t *runtime);
void ane_lm_reset(ane_lm_runtime_t *runtime);
int32_t ane_lm_vocab_size(const ane_lm_runtime_t *runtime);

// The callback receives -1 while pre-filling and a non-negative generated token
// afterwards. Returning zero stops generation with ANE_LM_STATUS_CANCELLED.
int32_t ane_lm_generate(
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
    char **error_out);

void ane_lm_free_string(char *string);

#ifdef __cplusplus
}
#endif

#endif
