//
//  LlamaProxy.swift
//
//  Example Swift “proxy” to the llama C API
//

import Foundation
import llama

/// Swift wrapper for llama_model_params
public struct LlamaModelParams {
    var cParams: llama_model_params
    
    public init() {
        // Call the default initializer from the C API:
        self.cParams = llama_model_default_params()
    }
    
    /// Example: set the number of GPU layers
    public mutating func setGpuLayers(_ n: Int32) {
        cParams.n_gpu_layers = n
    }
    
    /// Example: set whether to only load vocabulary
    public mutating func setVocabOnly(_ enable: Bool) {
        cParams.vocab_only = enable
    }
    
    // -- Add more Swift-friendly setters/getters as desired --
}

/// Swift wrapper for llama_context_params
public struct LlamaContextParams {
    var cParams: llama_context_params
    
    public init() {
        // Call the default initializer from the C API:
        self.cParams = llama_context_default_params()
    }
    
    /// Example: set context size
    public mutating func setContextSize(_ size: UInt32) {
        cParams.n_ctx = size
    }
    
    /// Example: set thread count for generation
    public mutating func setThreadCount(_ n: Int32) {
        cParams.n_threads = n
    }
    
    // -- Add more Swift-friendly setters/getters as desired --
}

/// A Swift class for the llama_model* pointer
public final class LlamaModel {
    private var pointer: OpaquePointer?
    
    /// Create a llama_model by loading from file
    /// - Parameters:
    ///   - path: file path to the model
    ///   - params: Swift wrapper of llama_model_params
    public init?(path: String, params: LlamaModelParams) {
        guard let cStrPath = path.cString(using: .utf8) else {
            return nil
        }
        
        let modelPtr = llama_load_model_from_file(cStrPath, params.cParams)
        guard modelPtr != nil else {
            return nil
        }
        self.pointer = modelPtr
    }
    
    /// Explicitly free the model
    public func free() {
        if let ptr = pointer {
            llama_free_model(ptr)
            pointer = nil
        }
    }
    
    deinit {
        // Ensure it gets freed
        free()
    }
    
    /// Raw pointer for advanced usage (avoid exposing if you want maximum safety).
    public var cPointer: OpaquePointer? {
        pointer
    }
}

/// A Swift class for llama_context*
public final class LlamaContext {
    private var pointer: OpaquePointer?
    /// Keep a strong reference to the model so it remains valid
    public let model: LlamaModel
    
    /// Create a llama_context from a model
    /// - Parameters:
    ///   - model: LlamaModel object
    ///   - params: Swift wrapper of llama_context_params
    public init?(model: LlamaModel, params: LlamaContextParams) {
        guard let modelPtr = model.cPointer else {
            return nil
        }
        let ctxPtr = llama_new_context_with_model(modelPtr, params.cParams)
        guard ctxPtr != nil else {
            return nil
        }
        self.pointer = ctxPtr
        self.model = model
    }
    
    /// Explicitly free the context
    public func free() {
        if let ptr = pointer {
            llama_free(ptr)
            pointer = nil
        }
    }
    
    deinit {
        // Ensure it gets freed
        free()
    }
    
    /// Expose the raw pointer for advanced usage if needed
    public var cPointer: OpaquePointer? {
        pointer
    }
    
    /// Decode a batch of tokens
    /// - Returns: 0 on success, or a non-zero code on error
    public func decode(batch: LlamaBatch) -> Int32 {
        guard let ctxPtr = pointer else {
            return -999  // Arbitrary error code indicating context is invalid
        }
        return llama_decode(ctxPtr, batch.cBatch)
    }
}

/// Swift struct wrapper for llama_batch
public struct LlamaBatch {
    public var cBatch: llama_batch
    
    /// Create a single-sequence batch from an array of token IDs
    public init(tokens: [llama_token]) {
        // Copy Swift array to a newly allocated C buffer
        var copy = tokens
        let ptr = UnsafeMutablePointer<llama_token>.allocate(capacity: tokens.count)
        ptr.initialize(from: &copy, count: tokens.count)
        
        // Use the llama_batch_get_one helper
        let b = llama_batch_get_one(ptr, Int32(tokens.count))
        self.cBatch = b
    }
    
    /// For advanced usage, you can initialize your own `llama_batch` with custom data. This is just an example.
}

// MARK: - Usage Example (Comment or remove this section if unneeded)
/*
func exampleUsage() {
    // 1) Initialize a model
    var modelParams = LlamaModelParams()
    modelParams.setGpuLayers(10)
    modelParams.setVocabOnly(false)
    
    guard let llamaModel = LlamaModel(path: "/path/to/your/model.bin", params: modelParams) else {
        print("Failed to load the model.")
        return
    }
    
    // 2) Create a context
    var ctxParams = LlamaContextParams()
    ctxParams.setContextSize(2048)
    ctxParams.setThreadCount(4)
    
    guard let llamaCtx = LlamaContext(model: llamaModel, params: ctxParams) else {
        print("Failed to create a context.")
        return
    }
    
    // 3) Create a batch from tokens
    let tokens: [llama_token] = [100, 101, 102] // Example token IDs
    let batch = LlamaBatch(tokens: tokens)
    
    // 4) Decode the batch
    let result = llamaCtx.decode(batch: batch)
    if result != 0 {
        print("Decode returned error code \(result)")
    } else {
        print("Decode succeeded.")
    }
    
    // 5) Cleanup
    llamaCtx.free()
    llamaModel.free()
}
*/
