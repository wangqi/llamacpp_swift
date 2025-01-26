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
    public var chain: UnsafeMutablePointer<llama_sampler>?
    public var grammarSampler: UnsafeMutablePointer<llama_sampler>?  // optional if you want grammar usage

    public init() {}
}

/// Replacement for the original C++ `init_sampling(...)`.
/// Instead of calling `common_sampler_init(...)`, we manually build a llama sampler chain.
public func init_sampling(
    model: OpaquePointer?,
    params: SpmSamplingParams
) -> SpmSamplerContext {
    // Prepare the chain
    var chainParams = llama_sampler_chain_default_params()
    chainParams.no_perf = true

    guard let chainPtr = llama_sampler_chain_init(chainParams) else {
        // If somehow chain creation fails, return empty
        return SpmSamplerContext()
    }

    // We store it in SpmSamplerContext
    let ctx = SpmSamplerContext()
    ctx.chain = chainPtr

    // Build up samplers in the chain:

    // 1) Repetition/presence penalties
    if params.penaltyLastN != 0
        && (params.penaltyRepeat != 1.0 || params.penaltyFreq != 0.0 || params.penaltyPresent != 0.0) {
        if let sampler = llama_sampler_init_penalties(
            params.penaltyLastN,
            params.penaltyRepeat,
            params.penaltyFreq,
            params.penaltyPresent
        ) {
            llama_sampler_chain_add(chainPtr, sampler)
        }
    }

    // 2) Top-k
    if params.topK > 0 {
        if let sampler = llama_sampler_init_top_k(params.topK) {
            llama_sampler_chain_add(chainPtr, sampler)
        }
    }

    // 3) Typical
    if params.typicalP < 0.9999 {
        if let sampler = llama_sampler_init_typical(params.typicalP, 1) {
            llama_sampler_chain_add(chainPtr, sampler)
        }
    }

    // 4) Top-p
    if params.topP < 0.9999 {
        if let sampler = llama_sampler_init_top_p(params.topP, 1) {
            llama_sampler_chain_add(chainPtr, sampler)
        }
    }

    // 5) Min-p
    if params.minP > 0 && params.minP < 0.9999 {
        if let sampler = llama_sampler_init_min_p(params.minP, 1) {
            llama_sampler_chain_add(chainPtr, sampler)
        }
    }

    // 6) Temperature
    if params.temp > 0 {
        if let sampler = llama_sampler_init_temp(params.temp) {
            llama_sampler_chain_add(chainPtr, sampler)
        }
    } else {
        // temp <= 0 => use "greedy" sampler
        if let sampler = llama_sampler_init_greedy() {
            llama_sampler_chain_add(chainPtr, sampler)
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
            llama_sampler_chain_add(chainPtr, sampler)
        }
    } else if params.mirostat == 2 {
        if let sampler = llama_sampler_init_mirostat_v2(
            params.seed,
            params.mirostatTau,
            params.mirostatEta
        ) {
            llama_sampler_chain_add(chainPtr, sampler)
        }
    }

    // 8) Dist (random) as the final sampler
    if let finalSampler = llama_sampler_init_dist(params.seed) {
        llama_sampler_chain_add(chainPtr, finalSampler)
    }

    // 9) Optional grammar usage
    // If we wanted to load grammar from file, do it here:
    // let grammar = ...
    // let grammarSamplerPtr = llama_sampler_init_grammar(model, grammar, "root")
    // llama_sampler_chain_add(chainPtr, grammarSamplerPtr)
    // ctx.grammarSampler = grammarSamplerPtr

    return ctx
}

// MARK: - spm_llama_sampling_sample / spm_llama_sampling_accept

/// Swift version of `spm_llama_sampling_sample(...)`.
/// We assume that `ctxSampling` is a chain sampler from `init_sampling(...)`.
public func spm_llama_sampling_sample(
    ctxSampling: SpmSamplerContext?,
    ctxMain: OpaquePointer?,    // llama_context*
    idx: Int32 = -1,
    grammarFirst: Bool = false
) -> llama_token {
    guard let chain = ctxSampling?.chain,
          let cMain = ctxMain else {
        return -1
    }
    // The built-in function to sample a token from the last decode output:
    let token = llama_sampler_sample(chain, cMain, idx)
    return token
}

/// Swift version of `spm_llama_sampling_accept(...)`.
/// This calls `llama_sampler_accept(...)` on the chain to let it update internal state with the accepted token.
public func spm_llama_sampling_accept(
    ctxSampling: SpmSamplerContext?,
    ctxMain: OpaquePointer?,    // llama_context*
    token: llama_token,
    applyGrammar: Bool
) {
    guard let chain = ctxSampling?.chain else {
        return
    }
    llama_sampler_accept(chain, token)
    // If you had a separate grammar sampler, you might do something custom with `applyGrammar`.
}

// End of GptSpm.swift

// MARK: Original package_helper.m functions


/// Returns the resource path of the SwiftPM module’s bundle.
/// (Equivalent to `SWIFTPM_MODULE_BUNDLE.resourcePath`.)
/*
func get_core_bundle_path() -> String {
    // If for some reason `resourcePath` is nil, we fall back to ""
    return Bundle.module.resourcePath ?? ""
}
*/

/// Returns the hardware machine name via the C `uname` call.
/// (Equivalent to `[NSString stringWithUTF8String:sysinfo.machine]` in Objective-C.)
func Get_Machine_Hardware_Name() -> String? {
    var sysinfo = utsname()
    // uname(...) returns 0 on success
    guard uname(&sysinfo) == 0 else {
        return nil
    }
    // Convert the machine field (CChar array) to a Swift string
    let machineMirror = Mirror(reflecting: sysinfo.machine)
    var machineString = ""
    for child in machineMirror.children {
        if let value = child.value as? Int8, value != 0 {
            machineString.append(Character(UnicodeScalar(UInt8(value))))
        }
    }
    return machineString
}
