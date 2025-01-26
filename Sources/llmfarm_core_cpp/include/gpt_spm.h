#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
// #include "llama_dadbed9.h"
#include "llama.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef int gpt_token;
typedef int llama_token;

const char * print_system_info(void);


typedef void (*gpt_progress_callback)(float progress, void *ctx);

typedef struct llama_sampling_params_spm {
    int32_t     n_prev                ;       // number of previous tokens to remember
    int32_t     n_probs               ;        // if greater than 0, output the probabilities of top n_probs tokens.
    int32_t     min_keep              ;        // 0 = disabled, otherwise samplers should return at least min_keep tokens
    int32_t     top_k                 ;       // <= 0 to use vocab size
    float       top_p                 ;    // 1.0 = disabled
    float       min_p                 ;    // 0.0 = disabled
    float       tfs_z                 ;    // 1.0 = disabled
    float       typical_p             ;    // 1.0 = disabled
    float       temp                  ;    // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range        ;    // 0.0 = disabled
    float       dynatemp_exponent     ;    // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t     penalty_last_n        ;       // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat        ;    // 1.0 = disabled
    float       penalty_freq          ;    // 0.0 = disabled
    float       penalty_present       ;    // 0.0 = disabled
    int32_t     mirostat              ;        // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau          ;    // target entropy
    float       mirostat_eta          ;    // learning rate
    bool        penalize_nl           ;     // consider newlines as a repeatable token
} llama_sampling_params_spm;

char * get_tensor_name(struct ggml_tensor * t);
int check_tensor_name(struct ggml_tensor * t);

// struct llama_sampling_context *
struct common_sampler* init_sampling(struct llama_model* model,
                                                int32_t     n_prev,                 // number of previous tokens to remember
                                                int32_t     top_k,                 // <= 0 to use vocab size
                                                float       top_p,              // 1.0 = disabled
                                                float       min_p,              // 0.0 = disabled
                                                float       tfs_z,              // 1.0 = disabled
                                                float       typical_p,              // 1.0 = disabled
                                                float       temp,              // <= 0.0 to sample greedily, 0.0 to not output probabilities
                                                float       dynatemp_range,              // 0.0 = disabled
                                                float       dynatemp_exponent,              // controls how entropy maps to temperature in dynamic temperature sampler
                                                int32_t     penalty_last_n,                 // last n tokens to penalize (0 = disable penalty, -1 = context size)
                                                float       penalty_repeat,              // 1.0 = disabled
                                                float       penalty_freq,              // 0.0 = disabled
                                                float       penalty_present,              // 0.0 = disabled
                                                int32_t     mirostat,                 // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
                                                float       mirostat_tau,              // target entropy
                                                float       mirostat_eta,              // learning rate
                                                bool        penalize_nl,              // consider newlines as a repeatable token
                                                uint32_t    seed,
                                                const char * grammar_path);


llama_token spm_llama_sampling_sample(
        // struct llama_sampling_context * ctx_sampling,
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        // struct llama_context * ctx_cfg,
        int idx,        
        bool grammar_first);

void spm_llama_sampling_accept(
        // struct llama_sampling_context * ctx_sampling,
        struct common_sampler * ctx_sampling,
        struct llama_context * ctx_main,
        llama_token id,
        bool apply_grammar);

#ifdef __cplusplus
}
#endif
