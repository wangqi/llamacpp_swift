//
//    that.swift
//    llamacpp_swift
//
//    Created by Qi Wang on 2025-02-03.
//


import Foundation
import Logging
import llama

/// An actor that handles inference using the LLama language model.
public actor LLama {
    private let logger = Logger.llama
    private let model: Model
    private var chat_tempate: ChatTemplate?
    private let sampling: UnsafeMutablePointer<llama_sampler>

    // MARK: - Init & Teardown

    /// Initializes a new instance of `LLama` with the specified model.
    ///
    /// - Parameter model: The language model to use for inference.
    public init(model: Model) {
        self.model = model

        // Initialize sampling
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.8))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_softmax())
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
    }
    

    deinit {
        llama_sampler_free(self.sampling)
    }
    
    // MARK: - load
    
    public func load_chat_template() {
        //Cache bos & eos tokens
//        var llm_bos_token_str = String(cString: tokenToPieceArray(token: llama_vocab_bos(self.model.vocab)))
//        var llm_eos_token_str = String(cString: tokenToPieceArray(token: llama_vocab_eos(self.model.vocab)))

        // Get model's chat template
        var cName: UnsafeMutablePointer<CChar>? = nil
        let modelChatTemplate = String(cString: llama_model_chat_template(self.model.model, cName))
        
        // Initialize chat template
        self.chat_tempate = ChatTemplate(
            source: modelChatTemplate,
            bosToken: "<s>",
            eosToken: "</s>"
        )
    }

    // MARK: - Inference

    /// Generates an asynchronous stream of tokens as strings based on the given prompt.
    ///
    /// - Parameters:
    ///     - prompt: The input text prompt to generate completions for.
    ///     - maxTokens: The maximum number of tokens to generate. Defaults to 128.
    ///
    /// - Returns: An `AsyncThrowingStream` emitting generated tokens as strings.
    public func infer(prompt: String, maxTokens: Int32 = 128) -> AsyncThrowingStream<String, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.infer(prompt: prompt, maxTokens: maxTokens, continuation: continuation)
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Initiates the inference process and manages the lifecycle of variables.
    ///
    /// - Parameters:
    ///     - prompt: The input text prompt to generate completions for.
    ///     - maxTokens: The maximum number of tokens to generate.
    ///     - continuation: The stream continuation to yield tokens to.
    private func infer(
        prompt: String,
        maxTokens: Int32,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async throws {
        var isDone = false
        let nLen: Int32 = 1024
        var nCur: Int32 = 0
        var nDecode: Int32 = 0
        var batch = llama_batch_init(512, 0, 1)
        var temporaryInvalidCChars: [CChar] = []
        defer {
            llama_batch_free(batch)
        }

        try self.initializeInference(
            prompt: prompt,
            batch: &batch,
            nLen: nLen,
            nCur: &nCur
        )

        try await self.runInferenceLoop(
            batch: &batch,
            temporaryInvalidCChars: &temporaryInvalidCChars,
            isDone: &isDone,
            nLen: nLen,
            nCur: &nCur,
            nDecode: &nDecode,
            maxTokens: maxTokens,
            continuation: continuation
        )
    }

    // MARK: - Private Helpers

    /// Initializes the inference process by tokenizing the input and preparing the batch.
    ///
    /// - Parameters:
    ///     - prompt: The input text prompt.
    ///     - batch: The batch to initialize.
    ///     - nLen: The maximum sequence length.
    ///     - nCur: The current position in the sequence.
    ///
    /// - Throws: An `InferError` if the KV cache is insufficient or decoding fails.
    private func initializeInference(
        prompt: String,
        batch: inout llama_batch,
        nLen: Int32,
        nCur: inout Int32
    ) throws {
        logger.debug("Attempting to complete \"\(prompt)\"")

        // Build the messages array for chat template
        let messages: [[String: Any]] = [
            ["role": "system", "content": ""],
            ["role": "user", "content": prompt],
        ]
        
        var query = prompt
        if let template = self.chat_tempate {
            // Use ChatTemplate to format the messages
            query = template.apply(
                messages: messages,
                tools: [:],
                addGenerationPrompt: true,
                extraContext: ["bos_token": "<s>"]
            )
        }
        print("Final query: \(query)")
        
        let tmpTokensList = tokenize(text: query, add_bos: true)
        let tokensList = tmpTokensList[0..<Int(tmpTokensList.count-1)]
        print("Input tokens: \(tokensList)")

        let nCtx = llama_n_ctx(model.context)
        let nKvReq = tokensList.count + Int(nLen - Int32(tokensList.count))

        logger.debug("\nn_len = \(nLen), n_ctx = \(nCtx), n_kv_req = \(nKvReq)")

        if nKvReq > nCtx {
            logger.error("Error: n_kv_req > n_ctx, the required KV cache size is not big enough")
            throw InferError(message: "KV cache too small", code: .kvCacheFailure)
        }

        batch.clear()

        for (i, token) in tokensList.enumerated() {
            llamaBatchAdd(&batch, token, Int32(i), [0], false)
        }
        if batch.n_tokens > 0 {
            batch.logits[Int(batch.n_tokens) - 1] = 1    // true
        }

        if llama_decode(model.context, batch) != 0 {
            throw InferError(message: "llama_decode failed", code: .decodingFailure)
        }

        nCur = batch.n_tokens
    }

    /// Runs the main inference loop, generating tokens and yielding them to the continuation.
    ///
    /// - Parameters:
    ///     - batch: The batch used for decoding.
    ///     - temporaryInvalidCChars: Buffer for building partial UTF8 strings.
    ///     - isDone: A flag indicating whether inference is complete.
    ///     - nLen: The maximum sequence length.
    ///     - nCur: The current position in the sequence.
    ///     - nDecode: The number of tokens decoded so far.
    ///     - maxTokens: The maximum number of tokens to generate.
    ///     - continuation: The stream continuation to yield tokens to.
    private func runInferenceLoop(
        batch: inout llama_batch,
        temporaryInvalidCChars: inout [CChar],
        isDone: inout Bool,
        nLen: Int32,
        nCur: inout Int32,
        nDecode: inout Int32,
        maxTokens: Int32,
        continuation: AsyncThrowingStream<String, Error>.Continuation
    ) async throws {
        while !isDone && nCur < nLen && nCur - batch.n_tokens < maxTokens {
            guard !Task.isCancelled else {
                continuation.finish()
                return
            }
            let newTokenStr = self.generateNextToken(
                batch: &batch,
                temporaryInvalidCChars: &temporaryInvalidCChars,
                isDone: &isDone,
                nLen: nLen,
                nCur: &nCur,
                nDecode: &nDecode
            )
            continuation.yield(newTokenStr)
        }
        continuation.finish()
    }

    /// Generates the next token and updates necessary states.
    ///
    /// - Parameters:
    ///     - batch: The batch used for decoding.
    ///     - temporaryInvalidCChars: Buffer for building partial UTF8 strings.
    ///     - isDone: A flag indicating whether inference is complete.
    ///     - nLen: The maximum sequence length.
    ///     - nCur: The current position in the sequence.
    ///     - nDecode: The number of tokens decoded so far.
    ///
    /// - Returns: The newly generated token as a string.
    private func generateNextToken(
        batch: inout llama_batch,
        temporaryInvalidCChars: inout [CChar],
        isDone: inout Bool,
        nLen: Int32,
        nCur: inout Int32,
        nDecode: inout Int32
    ) -> String {
        var newTokenID: llama_token = 0
        newTokenID = llama_sampler_sample(sampling, model.context, batch.n_tokens - 1)

        //change model.model to model.vocab wangqi
        if llama_token_is_eog(model.vocab, newTokenID) || nCur == nLen {
            isDone = true
            let newTokenStr = String(
                decoding: Data(temporaryInvalidCChars.map { UInt8(bitPattern: $0) }), as: UTF8.self)
            temporaryInvalidCChars.removeAll()
            return newTokenStr
        }

        let newTokenCChars = tokenToPieceArray(token: newTokenID)
        temporaryInvalidCChars.append(contentsOf: newTokenCChars)
        let newTokenStr: String

        if let string = String(validatingUTF8: temporaryInvalidCChars) {
            temporaryInvalidCChars.removeAll()
            newTokenStr = string
        } else if let partialStr = attemptPartialString(from: temporaryInvalidCChars) {
            temporaryInvalidCChars.removeAll()
            newTokenStr = partialStr
        } else {
            newTokenStr = ""
        }

        batch.clear()
        llamaBatchAdd(&batch, newTokenID, nCur, [0], true)

        nDecode += 1
        nCur += 1

        if llama_decode(model.context, batch) != 0 {
            logger.error("Failed to evaluate llama!")
        }

        return newTokenStr
    }

    /// Adds a token to the batch.
    ///
    /// - Parameters:
    ///     - batch: The batch to add the token to.
    ///     - id: The token ID to add.
    ///     - pos: The position of the token in the sequence.
    ///     - seq_ids: The sequence IDs associated with the token.
    ///     - logits: A flag indicating whether to compute logits for this token.
    private func llamaBatchAdd(
        _ batch: inout llama_batch,
        _ id: llama_token,
        _ pos: llama_pos,
        _ seq_ids: [llama_seq_id],
        _ logits: Bool
    ) {
        batch.token[Int(batch.n_tokens)] = id
        batch.pos[Int(batch.n_tokens)] = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits[Int(batch.n_tokens)] = logits ? 1 : 0

        batch.n_tokens += 1
    }

    /// Tokenizes the given text using the model's tokenizer.
    ///
    /// - Parameters:
    ///     - text: The text to tokenize.
    ///     - add_bos: A flag indicating whether to add a beginning-of-sequence token.
    ///
    /// - Returns: An array of token IDs.
    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Data = text.utf8CString
        let nTokens = Int32(utf8Data.count) + (add_bos ? 1 : 0)
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: Int(nTokens))
        defer { tokens.deallocate() }

        //change model.model to model.vocab wangqi
        let tokenCount = llama_tokenize(
            model.vocab, text, Int32(utf8Data.count), tokens, Int32(nTokens), add_bos, true)
        guard tokenCount > 0 else {
            return []
        }

        return Array(UnsafeBufferPointer(start: tokens, count: Int(tokenCount)))
    }

    /// Converts a token ID to an array of CChars representing the token piece.
    ///
    /// - Parameter token: The token ID to convert.
    ///
    /// - Returns: An array of CChars representing the token piece.
    private func tokenToPieceArray(token: llama_token) -> [CChar] {
        var buffer = [CChar](repeating: 0, count: 8)
        //change model.model to model.vocab wangqi
        var nTokens = llama_token_to_piece(model.vocab, token, &buffer, 8, 0, true)

        if nTokens < 0 {
            let requiredSize = -nTokens
            buffer = [CChar](repeating: 0, count: Int(requiredSize))
            //change model.model to model.vocab wangqi
            nTokens = llama_token_to_piece(model.vocab, token, &buffer, requiredSize, 0, true)
        }

        return Array(buffer.prefix(Int(nTokens)))
    }

    /// Attempts to create a partial string from an array of CChars if the full string is invalid.
    ///
    /// - Parameter cchars: The array of CChars to attempt to convert.
    ///
    /// - Returns: A valid string if possible; otherwise, `nil`.
    private func attemptPartialString(from cchars: [CChar]) -> String? {
        for i in (1..<cchars.count).reversed() {
            let subArray = Array(cchars.prefix(i))
            if let str = String(validatingUTF8: subArray) {
                return str
            }
        }
        return nil
    }
}

extension llama_batch {
    /// Clears the batch by resetting the token count.
    fileprivate mutating func clear() {
        n_tokens = 0
    }
}

extension String {
    /// Initializes a string from a sequence of CChars, validating UTF8 encoding.
    ///
    /// - Parameter validatingUTF8: The array of CChars to initialize the string from.
    fileprivate init?(validatingUTF8 cchars: [CChar]) {
        if #available(macOS 15.0, iOS 18.0, *) {
            self.init(decoding: Data(cchars.map { UInt8(bitPattern: $0) }), as: UTF8.self)
        } else {
            self.init(cString: cchars)
        }
    }
}


// MARK: Model

public final class Model {
    private let logger = Logger.llama
    let model: OpaquePointer
    let vocab: OpaquePointer
    let context: OpaquePointer

    public init(modelPath: String, contextSize: UInt32 = 2048) throws {
        llama_backend_init()
        var modelParams = llama_model_default_params()

        #if targetEnvironment(simulator)
            modelParams.n_gpu_layers = 0
            logger.debug("Running on simulator, force use n_gpu_layers = 0")
        #endif

        guard let model = llama_load_model_from_file(modelPath, modelParams) else {
            llama_backend_free()
            throw InitializationError(message: "Failed to load model", code: .failedToLoadModel)
        }
        self.model = model
        guard let vocab = llama_model_get_vocab(model) else {
        llama_free_model(model)
            llama_backend_free()
            throw InitializationError(message: "Failed to get vocabulary", code: .failedToLoadVocab)
        }
        self.vocab = vocab

        // Initialize context parameters
        let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        logger.debug("Using \(nThreads) threads")

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = contextSize
        ctxParams.n_threads = Int32(nThreads)
        ctxParams.n_threads_batch = Int32(nThreads)

        guard let context = llama_new_context_with_model(model, ctxParams) else {
            llama_free_model(model)
            llama_backend_free()
            throw InitializationError(
                message: "Failed to initialize context", code: .failedToInitializeContext)
        }
        self.context = context
    }

    deinit {
        llama_free(context)
        llama_free_model(model)
        llama_backend_free()
    }
}


// MARK: logging

extension Logger {
        static let llama = Logger(label: "llama-cpp-swift")
}

// MARK: Error

public struct InitializationError: Error, Sendable {
        public let message: String
        public let code: Code
        
        public enum Code: Int, Sendable {
                case failedToLoadModel = 1
                case failedToLoadVocab = 2
                case failedToInitializeContext
        }
}

public struct InferError: Error, Sendable {
        public let message: String
        public let code: Code
        
        public enum Code: Int, Sendable {
                case cancelled = 1
                case kvCacheFailure
                case decodingFailure
        }
}
