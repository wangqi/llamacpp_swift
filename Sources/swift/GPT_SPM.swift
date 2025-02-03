//
//  GptSpm.swift
//  This file replaces gpt_spm.cpp by providing
//  Swift equivalents for the essential logic
//  without referencing "common.h".
//

import Foundation
import llama

// MARK: - Bridging to llama

/// Print system info (wrapper for llama_print_system_info).
/// This replicates the `print_system_info()` function.
public func print_system_info() -> String {
    // llama_print_system_info() returns a C const char*
    guard let cStr = llama_print_system_info() else {
        return ""
    }
    return String(cString: cStr)
}


// MARK: - Sampler Setup (Replacing init_sampling & friends)

/// A Swift struct to hold the sampling parameters we want.
public struct SpmSamplingParams {
    public var nPrev: Int32               = 64
    public var topK: Int32               = 40
    public var topP: Float               = 0.95
    public var minP: Float               = 0.05
    public var typicalP: Float           = 1.0
    public var temp: Float               = 0.80
    public var dynaTempRange: Float      = 0.00
    public var dynaTempExponent: Float   = 1.00
    public var penaltyLastN: Int32       = 64
    public var penaltyRepeat: Float      = 1.00
    public var penaltyFreq: Float        = 0.00
    public var penaltyPresent: Float     = 0.00
    public var mirostat: Int32           = 0
    public var mirostatTau: Float        = 5.0
    public var mirostatEta: Float        = 0.1
    public var penalizeNl: Bool          = false
    public var seed: UInt32              = LLAMA_DEFAULT_SEED

    /// Path to a grammar file (optional).
    public var grammarPath: String       = ""

    public init() {}
}

/// The sampler handle we return from `init_sampling`.
/// We store the chain pointer as `UnsafeMutablePointer<llama_sampler>`.
public final class SpmSamplerContext {
    public var sampler: UnsafeMutablePointer<llama_sampler>?
    public var grammarSampler: UnsafeMutablePointer<llama_sampler>?
    // Store the previously accepted tokens in Swift
    // e.g. newest token at index 0, older tokens at higher indices
    public var prevTokens: [llama_token] = []

    public init() {}
}

/// Replacement for the original C++ `init_sampling(...)`.
/// Instead of calling `common_sampler_init(...)`, we manually build a llama sampler chain.
public func init_sampling(model: OpaquePointer?, params: SpmSamplingParams) -> SpmSamplerContext {
    // Prepare the chain
    var sparams = llama_sampler_chain_default_params()
    sparams.no_perf = true

    guard let sampling = llama_sampler_chain_init(sparams) else {
        // If somehow chain creation fails, return empty
        print("GPT_SPM.init_sampling() WARNING: Failed to initialize sampler chain")
        return SpmSamplerContext()
    }

    // We store it in SpmSamplerContext
    let samplerContext = SpmSamplerContext()
    samplerContext.sampler = sampling

    // Build up samplers in the chain:

    // 1) Repetition/presence penalties
    if params.penaltyLastN != 0
        && (params.penaltyRepeat != 1.0 || params.penaltyFreq != 0.0 || params.penaltyPresent != 0.0) {
        print("GPT_SPM.init_sampling(). penaltyLastN: \(params.penaltyLastN), penaltyRepeat: \(params.penaltyRepeat), penaltyFreq: \(params.penaltyFreq), penaltyPresent: \(params.penaltyPresent)")
        if let sampler = llama_sampler_init_penalties(
            params.penaltyLastN,
            params.penaltyRepeat,
            params.penaltyFreq,
            params.penaltyPresent
        ) {
            llama_sampler_chain_add(sampling, sampler)
        }
    }

    // 2) Top-k
    if params.topK > 0 {
        print("GPT_SPM.init_sampling(). topK: \(params.topK)")
        if let sampler = llama_sampler_init_top_k(params.topK) {
            llama_sampler_chain_add(sampling, sampler)
        }
    }

    // 3) Typical
    if params.typicalP < 0.9999 {
        print("GPT_SPM.init_sampling(). typicalP: \(params.typicalP)")
        if let sampler = llama_sampler_init_typical(params.typicalP, 1) {
            llama_sampler_chain_add(sampling, sampler)
        }
    }

    // 4) Top-p
    if params.topP < 0.9999 {
        print("GPT_SPM.init_sampling(). topP: \(params.topP)")
        if let sampler = llama_sampler_init_top_p(params.topP, 1) {
            llama_sampler_chain_add(sampling, sampler)
        }
    }

    // 5) Min-p
    if params.minP > 0 && params.minP < 0.9999 {
        print("GPT_SPM.init_sampling(). minP: \(params.minP)")
        if let sampler = llama_sampler_init_min_p(params.minP, 1) {
            llama_sampler_chain_add(sampling, sampler)
        }
    }

    // 6) Temperature
    if params.temp > 0 {
        print("GPT_SPM.init_sampling(). temp: \(params.temp)")
        if let sampler = llama_sampler_init_temp(params.temp) {
            llama_sampler_chain_add(sampling, sampler)
        }
    } else {
        print("GPT_SPM.init_sampling(). use 'greedy' sampler")
        // temp <= 0 => use "greedy" sampler
        if let sampler = llama_sampler_init_greedy() {
            llama_sampler_chain_add(sampling, sampler)
        }
    }

    // 7) Mirostat
    if params.mirostat == 1 {
        let nVocab = (model != nil) ? llama_n_vocab(model) : 0
        if let sampler = llama_sampler_init_mirostat(
            nVocab,
            params.seed,
            params.mirostatTau,
            params.mirostatEta,
            100 // 'm' argument
        ) {
            print("GPT_SPM.init_sampling(). mirostat enabled")
            llama_sampler_chain_add(sampling, sampler)
        }
    } else if params.mirostat == 2 {
        if let sampler = llama_sampler_init_mirostat_v2(
            params.seed,
            params.mirostatTau,
            params.mirostatEta
        ) {
            print("GPT_SPM.init_sampling(). mirostat v2 enabled")
            llama_sampler_chain_add(sampling, sampler)
        }
    }

    // 8) Dist (random) as the final sampler
    if let finalSampler = llama_sampler_init_dist(params.seed) {
        llama_sampler_chain_add(sampling, finalSampler)
    }
    
    // 9) Grammar
    // use spm_llama_load_grammar instead
    /*
    if !params.grammarPath.isEmpty {
        let grammar = params.grammarPath
        let grammarSamplerPtr = llama_sampler_init_grammar(model, grammar, "root")
        llama_sampler_chain_add(chainPtr, grammarSamplerPtr)
        ctx.grammarSampler = grammarSamplerPtr
    }
    */

    return samplerContext
}

public func spm_llama_load_grammar(model: OpaquePointer?, ctx: SpmSamplerContext, grammarPath: String) -> Bool {
    print("GPT_SPM.spm_llama_load_grammar(). grammarPath: \(grammarPath)")
    let grammarSamplerPtr = llama_sampler_init_grammar(model, grammarPath, "root")
    llama_sampler_chain_add(ctx.sampler, grammarSamplerPtr)
    ctx.grammarSampler = grammarSamplerPtr
    if grammarSamplerPtr == nil {
        return true
    }
    return false
}

// MARK: load chat template
/// Replacement for the original C++ `init_sampling(...)`.
/// Instead of calling `llama_model_chat_template(...)`, we manually build a llama sampler chain.
public func spm_llama_model_chat_template(
    model: OpaquePointer?,
    name: String?
) -> String? {
    // Handle optional name parameter
    var cName: UnsafeMutablePointer<CChar>? = nil
    if let name = name {
        cName = strdup(name) // Convert Swift String to C string
    }
    
    defer {
        free(cName) // Clean up allocated C string
    }
    
    // Call C function with optional parameters
    guard let resultPtr = llama_model_chat_template(model, cName) else {
        return nil
    }
    
    // Convert C string result to Swift String
    return String(cString: resultPtr)
}

// MARK: - spm_llama_sampling_sample / spm_llama_sampling_accept

/// Swift version of `spm_llama_sampling_sample(...)`.
/// We assume that `ctxSampling` is a chain sampler from `init_sampling(...)`.
public func spm_llama_sampling_sample(ctxSampling: SpmSamplerContext?, ctxMain: OpaquePointer?,
                                      idx: Int32 = -1, grammarFirst: Bool = false) -> llama_token {
    guard let chain = ctxSampling?.sampler, let cMain = ctxMain else {
        return -1
    }
    // The built-in function to sample a token from the last decode output:
    let token = llama_sampler_sample(chain, cMain, idx)
    
    // Also store the token in our Swift array
    // We'll store the newest token at index 0
    ctxSampling?.prevTokens.insert(token, at: 0)
    
    return token
}

/// Swift version of `spm_llama_sampling_accept(...)`.
/// This calls `llama_sampler_accept(...)` on the chain to let it update internal state with the accepted token.
public func spm_llama_sampling_accept(
    ctxSampling: SpmSamplerContext?,
    ctxMain: OpaquePointer?, // llama_context*
    token: llama_token,
    applyGrammar: Bool
) {
    guard let chain = ctxSampling?.sampler else {
        return
    }
    llama_sampler_accept(chain, token)
    // If you had a separate grammar sampler, you might do something custom with `applyGrammar`.
}

// MARK: - Additional bridging needed to replicate the main loop
// If GptSpm.swift cannot meet the requirements, we add them here:

/// A bridging function for llama_batch_get_one + llama_decode
public func spm_llama_decode(ctx: OpaquePointer?, tokens: [llama_token]) -> Int32 {
    guard let ctx = ctx, !tokens.isEmpty else {
        return 0
    }
    // get a single-sequence batch
    var copy = tokens
    let batch = llama_batch_get_one(&copy, Int32(tokens.count))
    let rc = llama_decode(ctx, batch)
    return rc
}

/// Bridging for kv_cache_seq_rm
public func spm_llama_kv_cache_seq_rm(
    ctx: OpaquePointer?,
    seqId: Int32,
    p0: Int,
    p1: Int
) {
    guard let ctx = ctx else { return }
    let ok = llama_kv_cache_seq_rm(ctx,
                                   seqId,
                                   llama_pos(p0),
                                   llama_pos(p1))
    // we ignore the returned bool
}

/// Bridging for kv_cache_seq_add
public func spm_llama_kv_cache_seq_add(
    ctx: OpaquePointer?,
    seqId: Int32,
    p0: Int,
    p1: Int,
    delta: Int
) {
    guard let ctx = ctx else { return }
    llama_kv_cache_seq_add(ctx,
                           seqId,
                           llama_pos(p0),
                           llama_pos(p1),
                           llama_pos(delta))
}

/// Bridging for kv_cache_seq_div
public func spm_llama_kv_cache_seq_div(
    ctx: OpaquePointer?,
    p0: Int,
    p1: Int,
    divisor: Int
) {
    guard let ctx = ctx else { return }
    llama_kv_cache_seq_div(ctx,
                           0,
                           llama_pos(p0),
                           llama_pos(p1),
                           Int32(divisor))
}

/// Bridging for saving session to file
public func spm_llama_state_save_file(
    ctx: OpaquePointer?,
    path: String,
    tokens: [llama_token]
) {
    guard let ctx = ctx else { return }
    let cPath = strdup(path)
    defer { free(cPath) }
    _ = llama_state_save_file(ctx, cPath, tokens, tokens.count)
}

/// Return whether a token is an end-of-generation token
public func spm_llama_vocab_is_eog(
    vocab: OpaquePointer?,
    token: llama_token
) -> Bool {
    guard let v = vocab else { return false }
    return llama_vocab_is_eog(v, token)
}

/// Similar to `common_sampler_prev_str(...)` in C++.
///  - Retrieves up to `n` tokens from sampler->prev
///  - Iterates them in reverse, converting each token to text
///  - Returns the concatenated string
/// We do **not** call `spm_llama_token_to_piece(...)`; we call llama_token_to_piece() directly instead.
public func spm_llama_sampling_last_tokens(
    ctxSampling: SpmSamplerContext?,
    ctx: OpaquePointer?, // needed to get model => vocab
    n: Int = 32
) -> String {
    guard let ctxSampling = ctxSampling, n > 0, let ctxMain = ctx else {
        return ""
    }

    // get the model + vocab
    guard let model = llama_get_model(ctxMain) else {
        return ""
    }
    guard let vocab = llama_model_get_vocab(model) else {
        return ""
    }

    // how many tokens do we actually have
    let totalTokens = ctxSampling.prevTokens.count
    if totalTokens == 0 {
        return ""
    }

    // up to `n`, but not more
    let got = min(n, totalTokens)

    var result = ""
    result.reserveCapacity(got * 8)  // optional guess

    // spmSamplerContext.prevTokens[0] is newest, [1] older, etc. if we used insert(0).
    // So the "last n tokens" in forward order are indices [got-1 ... 0].
    // Then to replicate the C++ snippet which does reverse iteration, we do:
    //
    // In the C++ code:
    //   for (int i = n - 1; i >= 0; i--) { result += token_to_piece(..., prev.rat(i)); }
    //
    // Here, we have them in Swift array with newest at index 0,
    // so the last n tokens in forward order are indices [got-1 ... 0].
    // We want them reversed => we do i in [0 ..< got], reversed:
    for i in 0 ..< got {
        let token = ctxSampling.prevTokens[i]
        let piece = convertTokenToPieceRaw(vocab: vocab, token: token, special: false)
        result += piece
    }

    return result
}

/// A small helper that calls llama_token_to_piece(...) directly (twice if needed),
/// mirroring the approach from your C++ `common_token_to_piece` function.
private func convertTokenToPieceRaw(
    vocab: OpaquePointer?,
    token: llama_token,
    special: Bool
) -> String {
    // Start with a modest capacity
    var capacity = 32

    while true {
        var buffer = [CChar](repeating: 0, count: capacity)

        let nChars = llama_token_to_piece(vocab, token, &buffer, Int32(capacity), 0, special)

        if nChars < 0 {
            // The function wants a bigger buffer => set capacity to -nChars and retry
            capacity = -Int(nChars)
            continue
        } else {
            // success: nChars >= 0
            // convert the buffer up to nChars
            return buffer.withUnsafeBufferPointer { ptr in
                // ptr is guaranteed to have at least nChars+1 bytes (with the trailing '\0')
                return String(cString: ptr.baseAddress!)
            }
        }
    }
}

/// Convert a token to its corresponding piece of text.
/// Similar to common_token_to_piece in the C++ code.
///
/// We replicate the logic:
///  - Call llama_token_to_piece(...) with a small buffer
///  - If the returned value is negative, re-allocate a bigger buffer of -nChars
///  - Then call again until success
///  - Return the resulting string
public func spm_llama_token_to_piece(
    model: OpaquePointer?,
    vocab: OpaquePointer?,
    token: llama_token,
    special: Bool
) -> String {
    // Start with a small buffer capacity
    var capacity = 32

    while true {
        // Allocate a buffer of 'capacity' chars
        var buffer = [CChar](repeating: 0, count: capacity)

        // Call the function
        let nChars = llama_token_to_piece(vocab, token,
            &buffer,          // pointer to our buffer
            Int32(capacity),  // length
            0,                // lstrip
            special
        )

        if nChars < 0 {
            // The function wants a bigger buffer of size -nChars
            capacity = -Int(nChars)
            continue
        } else {
            // nChars >= 0 => we succeeded
            // The returned string length is nChars, but
            // also there's a null terminator in the buffer.
            // We can safely do this:
            //   1) ensure the buffer is truncated or sized properly
            //   2) create a Swift string from it
            //   3) return
            return buffer.withUnsafeBufferPointer { ptr in
                String(cString: ptr.baseAddress!)
            }
        }
    }
}
