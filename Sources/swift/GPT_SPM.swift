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
    public var penalizeNl: Bool          = false
    public var penaltyLastN: Int32       = 64
    public var penaltyRepeat: Float      = 1.00
    public var penaltyFreq: Float        = 0.00
    public var penaltyPresent: Float     = 0.00
    // This parameter selects the version or mode of the Mirostat algorithm to use
    // 0: Mirostat is disabled (i.e., standard sampling techniques like top-k or top-p are used instead).
    // 1: Use Mirostat version 1, which focuses on real-time feedback to adjust the sampling process using a simple feedback loop.
    //    It dynamically adjusts the temperature of the next token selection by comparing the current token’s entropy to a target entropy.
    // 2: Use Mirostat version 2, an enhanced version with additional stability and convergence features.
    //    It introduces additional mathematical refinements to improve convergence speed and prediction stability.
    public var mirostat: Int32           = 0
    // This parameter sets the target entropy level that Mirostat tries to maintain during generation. 
    // The goal is to ensure that the text generation is neither too chaotic (high entropy) nor too repetitive/deterministic (low entropy).
    // * Low values (e.g., 2.0): More focused, deterministic output (less diverse generation).
    // * Higher values (e.g., 5.0 or more): More diverse and creative output.
    public var mirostatTau: Float        = 5.0
    // This parameter defines the magnitude of the adjustment step applied to the token probabilities during each iteration of Mirostat sampling.
    // * Larger mirostat_eta: Faster adjustments, but may overshoot the target entropy.
    // * Smaller mirostat_eta: Slower but more stable convergence toward the target entropy.
    // mirostat_eta works in conjunction with mirostat_lr to fine-tune the feedback loop
    public var mirostatEta: Float        = 0.1
    /*
    Parameter	    Values          Description	
      mirostat	    0, 1, or 2  Mode of Mirostat (0 = disabled, 1 = version 1, 2 = version 2)	
      mirostat_lr	0.1 to 1.0  Learning rate for the feedback mechanism, controlling how fast entropy corrections are applied	
      mirostat_ent	Dynamic     Current entropy estimate (dynamic, often calculated internally)	
      mirostat_tau	2.0 to 5.0  Target entropy level for balancing diversity and coherence	
      mirostat_eta	0.1 to 0.5  Step size for entropy correction	
    */
    public var seed: UInt32              = LLAMA_DEFAULT_SEED

    /// Path to a grammar file (optional).
    public var grammarPath: String       = ""
    
    /*
     The dry run mechanism in token sampling is a technique used to penalize certain undesirable tokens during text
     generation before making a final token selection. Its goal is to dynamically discourage or suppress the likelihood
     of generating tokens that violate constraints, such as repetitions, low-probability tokens, or contextually
     inappropriate words.

     The “dry run” itself means that the model first evaluates the token probabilities (logits) and applies penalties
     or constraints before finalizing the token selection. Think of it as a pre-check phase where the model decides
     whether certain tokens should be penalized or excluded.
     */
    
    /// Controls the severity of the penalty applied when a token violates constraints during sampling.
    /// Higher values encourage the model to be more creative by exploring less frequently chosen tokens.
    /// - Values like 1.0 (mild penalty) to 2.0 or higher (aggressive penalty) are used.
    /// - Default: 1.0
    public var dryMultiplier: Float       = 1.0

    /// Specifies how many tokens can be generated without triggering a penalty based on the dry run constraints.
    /// If the token violates some penalty rules (e.g., frequency, repetition), dry_base sets a baseline penalty before applying dry_multiplier.
    /// - Range: Values around 0.5 to 1.0 are common
    /// - Default: 1.0
    public var dryBase: Float             = 1.0

    /// Maximum allowed token sequence length for applying dry sampling adjustments.
    /// If dry_allowed_length is too low, penalties will trigger too early, possibly interrupting coherent sequences.
    /// Setting it too high may allow too much repetition or off-topic output before corrections occur.
    /// - Range: Usually depends on the application, but could range from 5 tokens to 100 tokens.
    /// - Default: 0
    public var dryAllowedLength: Int32    = 0

    /// Specifies the number of previously generated tokens to consider when applying the dry run penalty for repetition.
    /// - Range: Common values range between 10 and 100 tokens.
    /// - Default: 64
    public var dryPenaltyLastN: Int32     = 64
    
    /// List of token sequences that, if encountered, will break the dry sampling chain.
    /// - Default: ["\n", ":", "\"", "*"]
    public var drySequenceBreakers: [String] = ["\n", ":", "\"", "*"]

    /// Used in extended token constraints (XTC) sampling to limit certain tokens based on predefined conditions.
    /// It defines the probability threshold for sampling from a constrained set of tokens.
    /// Lower xtc_probability values allow more diversity, while higher values enforce stricter control over token selection based on constraints.
    /// - Range:  Values between 0.5 and 1.0 are common.
    /// - Default: 0.0 (disabled)
    public var xtcProbability: Float      = 0.0

    /// Represents the logit threshold for a token to be considered valid under extended token constraints (XTC).
    /// Values vary depending on how aggressively you want to filter tokens:
    /// - Range: Low values (e.g., -10): Loosely penalize tokens, allowing most to pass.
    ///         Higher values (e.g., -1): Strongly restrict token choices.
    /// - Default: 0.0 (disabled)
    public var xtcThreshold: Float        = 0.0

    /// Minimum number of tokens to keep in sampling operations (e.g. top-p, typical).
    /// - Range: positive integer
    /// - Default: 1
    public var minKeep: Int32             = 1
    
    /// An array of (token, bias). If empty, no logit-bias is applied.
    public var logitBias: [(token: llama_token, bias: Float)] = []

    public init() {}
    
    public func toString() -> String {
        var result = "Sampling Parameters:\n"
        
        // Basic parameters
        result += "Basic:\n"
        result += "  Temperature: \(temp)\n"
        result += "  Top-K: \(topK)\n"
        result += "  Top-P: \(topP)\n"
        result += "  Min-P: \(minP)\n"
        result += "  Typical-P: \(typicalP)\n"
        
        // Dynamic temperature
        if dynaTempRange > 0 {
            result += "Dynamic Temperature:\n"
            result += "  Range: \(dynaTempRange)\n"
            result += "  Exponent: \(dynaTempExponent)\n"
        }
        
        // Penalties
        result += "Penalties:\n"
        result += "  Last N: \(penaltyLastN)\n"
        result += "  Repeat: \(penaltyRepeat)\n"
        result += "  Frequency: \(penaltyFreq)\n"
        result += "  Present: \(penaltyPresent)\n"
        result += "  Penalize Newline: \(penalizeNl)\n"
        
        // Mirostat
        result += "Mirostat:\n"
        result += "  mirostat:     \(mirostat)\n"
        if mirostat > 0 {
            result += "  mirostat_tau: \(mirostatTau)\n"
            result += "  mirostat_eta: \(mirostatEta)\n"
        }
        
        // Dry run parameters
        if dryAllowedLength > 0 {
            result += "Dry Run:\n"
            result += "  Multiplier: \(dryMultiplier)\n"
            result += "  Base: \(dryBase)\n"
            result += "  Allowed Length: \(dryAllowedLength)\n"
            result += "  Penalty Last N: \(dryPenaltyLastN)\n"
        }
        
        // XTC parameters
        if xtcProbability > 0 || xtcThreshold > 0 {
            result += "XTC:\n"
            result += "  Probability: \(xtcProbability)\n"
            result += "  Threshold: \(xtcThreshold)\n"
        }
        
        // Other parameters
        result += "Other:\n"
        result += "  Previous tokens: \(nPrev)\n"
        result += "  Min Keep: \(minKeep)\n"
        result += "  Seed: \(seed)\n"
        if !grammarPath.isEmpty {
            result += "  Grammar Path: \(grammarPath)\n"
        }
        
        return result
    }
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
public func init_sampling(model: OpaquePointer?, vocab: OpaquePointer?, params: SpmSamplingParams) -> SpmSamplerContext {
    // Prepare the chain
    var sparams = llama_sampler_chain_default_params()
    sparams.no_perf = true
    
    // We store it in SpmSamplerContext
    let samplerContext = SpmSamplerContext()

    // Build up samplers in the chain:
    
    // Track sampler chain
    var samplerChain = ["logits"]
    
    // 1) Create the sampler chain
    guard let sampling = llama_sampler_chain_init(sparams) else {
        // If somehow chain creation fails, return empty
        print("GPT_SPM.init_sampling() WARNING: Failed to initialize sampler chain")
        return SpmSamplerContext()
    }
    samplerContext.sampler = sampling
    
    // 2) Add logit-bias sampler ---
    // Convert Swift array to [llama_logit_bias]
    print("GPT_SPM.init_sampling() Adding logit bias sampler")
    // Convert Swift array of bias pairs into C-compatible llama_logit_bias structures.
    let logitBiasArray = params.logitBias.map { pair in
        llama_logit_bias(token: pair.token, bias: pair.bias)
    }
    let nBias = logitBiasArray.count

    // Allocate memory for the C array.
    let pointer = UnsafeMutablePointer<llama_logit_bias>.allocate(capacity: nBias)
    // Ensure memory is deinitialized and deallocated when done.
    defer {
        pointer.deinitialize(count: nBias)
        pointer.deallocate()
    }
    // Copy the bias values into the allocated memory.
    logitBiasArray.withUnsafeBufferPointer { buffer in
        // Assumes that buffer.baseAddress is non-nil because the array is non-empty.
        pointer.initialize(from: buffer.baseAddress!, count: nBias)
    }
    
    // Get the vocabulary count.
    let vocabCount = llama_vocab_n_tokens(vocab)

    // Initialize and chain the logit-bias sampler if available.
    if let logitBiasSampler = llama_sampler_init_logit_bias(vocabCount, Int32(nBias), pointer) {
        print("GPT_SPM.init_sampling(): Adding logit-bias sampler with \(nBias) entries.")
        llama_sampler_chain_add(sampling, logitBiasSampler)
        samplerChain.append("logit-bias")
    }

    // 8) DRY Handle sampler chain differently based on the mirostat setting.
    if params.mirostat == 0 {
        // (A) --- Non-mirostat branch: add optional dry and xtc samplers.
        
        // 3) Repetition/presence penalties
        print("GPT_SPM.init_sampling(). penaltyLastN: \(params.penaltyLastN), penaltyRepeat: \(params.penaltyRepeat), " +
                "penaltyFreq: \(params.penaltyFreq), penaltyPresent: \(params.penaltyPresent)")
        if let sampler = llama_sampler_init_penalties(
            //penalty_last_n,
            params.penaltyLastN,
            // penalty_repeat
            params.penaltyRepeat,
            // penalty_freq
            params.penaltyFreq,
            // penalty_present
            params.penaltyPresent
        ) {
            llama_sampler_chain_add(sampling, sampler)
            samplerChain.append("penalties")
        }

        // 4) Top-k
        let topK = params.topK <= 0 ? 1 : Int32(params.topK)
        print("GPT_SPM.init_sampling(). topK: \(topK)")
        if let sampler = llama_sampler_init_top_k(topK) {
            llama_sampler_chain_add(sampling, sampler)
            samplerChain.append("top-k")
        }

        // 5) Typical
        let typicalP = params.typicalP > 0.9999 ? 0.9999 : params.typicalP
        print("GPT_SPM.init_sampling(). typicalP: \(typicalP)")
        if let sampler = llama_sampler_init_typical(typicalP, 1) {
            llama_sampler_chain_add(sampling, sampler)
            samplerChain.append("typicalP")
        }

        // 6) Top-p
        let topP = params.topP > 0.9999 ? 0.9999 : params.topP
        print("GPT_SPM.init_sampling(). topP: \(topP)")
        if let sampler = llama_sampler_init_top_p(topP, 1) {
            llama_sampler_chain_add(sampling, sampler)
            samplerChain.append("top-p")
        }

        // 7) Min-p
        let minP = params.minP > 0.9999 || params.minP < 0 ? 0.9 : params.minP
        print("GPT_SPM.init_sampling(). minP: \(minP)")
        if let sampler = llama_sampler_init_min_p(minP, 1) {
            llama_sampler_chain_add(sampling, sampler)
            samplerChain.append("min-p")
        }

        // Dry sampler: enable if dryAllowedLength > 0 (i.e. nonzero means enabled).
        if params.dryAllowedLength > 0 {
            print("GPT_SPM.init_sampling(). dry sampler: multiplier: \(params.dryMultiplier), base: \(params.dryBase), " +
                  "allowed_length: \(params.dryAllowedLength), penalty_last_n: \(params.dryPenaltyLastN), breakers: \(params.drySequenceBreakers)")
            let ctxTrain = (model != nil) ? llama_model_n_ctx_train(model) : 0
            // Convert array of C strings to array of UnsafePointer<CChar>
            let breakersCStrings = params.drySequenceBreakers.map { strdup($0) }
            let breakersArray = breakersCStrings.map { ptr -> UnsafePointer<CChar>? in
                guard let ptr = ptr else { return nil }
                return UnsafePointer(ptr)
            }
            // Allocate buffer and copy pointers
            let breakersPtr = UnsafeMutablePointer<UnsafePointer<CChar>?>.allocate(capacity: breakersArray.count + 1)
            breakersArray.enumerated().forEach { breakersPtr[$0.0] = $0.1 }
            breakersPtr[breakersArray.count] = nil  // Null terminate
            
            if let sampler = llama_sampler_init_dry(model, ctxTrain, params.dryMultiplier, params.dryBase,
                                                    params.dryAllowedLength, params.dryPenaltyLastN,
                                                    breakersPtr, breakersArray.count) {
                llama_sampler_chain_add(sampling, sampler)
                samplerChain.append("dry")
            }
            
            // Clean up allocated memory after sampler is initialized
            breakersPtr.deallocate()
            // Free allocated C strings only once
            breakersCStrings.forEach { ptr in
                if let p = ptr { free(UnsafeMutableRawPointer(mutating: p)) }
            }
        }

        // XTC sampler: enable if xtcProbability > 0.
        if params.xtcProbability > 0 {
            print("GPT_SPM.init_sampling(). xtc sampler: probability: \(params.xtcProbability), threshold: \(params.xtcThreshold), min_keep: \(params.minKeep)")
            if let sampler = llama_sampler_init_xtc(params.xtcProbability, params.xtcThreshold, Int(params.minKeep), params.seed) {
                llama_sampler_chain_add(sampling, sampler)
                samplerChain.append("xtc")
            }
        }

        // Temperature sampler using dynamic temperature parameters.
        if params.temp > 0 {
            print("GPT_SPM.init_sampling(). temp: \(params.temp) with dyn range: \(params.dynaTempRange) and exponent: \(params.dynaTempExponent)")
            if let sampler = llama_sampler_init_temp_ext(params.temp, params.dynaTempRange, params.dynaTempExponent) {
                llama_sampler_chain_add(sampling, sampler)
                samplerChain.append("temp-ext")
            }
        } else {
            print("GPT_SPM.init_sampling(). using 'greedy' sampler")
            if let sampler = llama_sampler_init_greedy() {
                llama_sampler_chain_add(sampling, sampler)
                samplerChain.append("greedy")
            }
        }
        
        // Finally, add the dist (random) sampler.
        if let finalSampler = llama_sampler_init_dist(params.seed) {
            llama_sampler_chain_add(sampling, finalSampler)
            samplerChain.append("dist")
        }
    } else if params.mirostat == 1 {
        // (B) --- Mirostat v1: add temperature first, then mirostat.
        if params.temp > 0 {
            print("GPT_SPM.init_sampling(). temp (for mirostat): \(params.temp)")
            if let sampler = llama_sampler_init_temp(params.temp) {
                llama_sampler_chain_add(sampling, sampler)
                samplerChain.append("temp-ext")
            }
        }
        let nVocab = (model != nil) ? llama_n_vocab(model) : 0
        if let sampler = llama_sampler_init_mirostat(nVocab, params.seed, params.mirostatTau, params.mirostatEta, 100) {
            print("GPT_SPM.init_sampling(). mirostat enabled (v1)")
            llama_sampler_chain_add(sampling, sampler)
            samplerChain.append("mirostat-v1")
        }
    } else if params.mirostat == 2 {
        // (C) --- Mirostat v2: add temperature then mirostat v2.
        if params.temp > 0 {
            print("GPT_SPM.init_sampling(). temp (for mirostat v2): \(params.temp)")
            if let sampler = llama_sampler_init_temp(params.temp) {
                llama_sampler_chain_add(sampling, sampler)
                samplerChain.append("temp-ext")
            }
        }
        if let sampler = llama_sampler_init_mirostat_v2(params.seed, params.mirostatTau, params.mirostatEta) {
            print("GPT_SPM.init_sampling(). mirostat v2 enabled")
            llama_sampler_chain_add(sampling, sampler)
            samplerChain.append("mirostat-v2")
        }
    }

    print(params.toString())
    
    // Print the final sampler chain
    print("GPT_SPM.init_sampling() Sampler chain: " + samplerChain.joined(separator: " -> "))
    
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
