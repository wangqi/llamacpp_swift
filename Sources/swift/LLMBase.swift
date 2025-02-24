//
//  LLMBase.swift
//  Created by Guinmoon.
//

import Foundation
import Combine
import llama
import llamacpp_swift_cpp

public enum ModelLoadError: Error {
    case modelLoadError
    case contextLoadError
    case grammarLoadError
}

/// A base class for loading and using a Large Language Model (LLM).
public class LLMBase {
    
    // MARK: - Constants
    
    /// A special marker used when the context limit is reached.
    public static let CONTEXT_LIMIT_MARKER = " `C_LIMIT` "
    
    // MARK: - Properties
    
    public var context: OpaquePointer?
    public var grammar: OpaquePointer?
    public var contextParams: ModelAndContextParams
    public var sampleParams: ModelSampleParams = .default
    public var session_tokens: [Int32] = []
    
    public var modelLoadProgressCallback: ((Float) -> (Bool))? = nil
    public var modelLoadCompleteCallback: ((String) -> ())? = nil
    public var evalCallback: ((Int) -> (Bool))? = nil
    public var evalDebugCallback: ((String) -> (Bool))? = nil
    
    public var modelPath: String
    public var outputRepeatTokens: [ModelToken] = []
    
    /// Holds the entire "past" token sequences (not always used).
    /// public typealias ModelToken = Int32
    var past: [[ModelToken]] = []
    
    /// Number of tokens that have been fed (prompt + generated).
    public var nPast: Int32 = 0
    
    var llm_bos_token: ModelToken?
    var llm_eos_token: ModelToken?
    var llm_nl_token: ModelToken?
    var llm_bos_token_str: String?
    var llm_eos_token_str: String?
    
    // MARK: - Initialization
    
    public init(path: String, contextParams: ModelAndContextParams = .default) throws {
        self.modelPath = path
        self.contextParams = contextParams
        
        // Ensure the model file exists at the specified path
        if !FileManager.default.fileExists(atPath: self.modelPath) {
            throw ModelError.modelNotFound(self.modelPath)
        }
    }

    // MARK: - Initialize the model and context.
    
    public func load_model() throws {
        var loadResult: Bool? = false
        do {
            try ExceptionCather.catchException {
                loadResult = try? self.internal_load_model(path: self.modelPath, contextParams: contextParams)
            }
            
            if loadResult != true {
                throw ModelLoadError.modelLoadError
            }
            
            // Load grammar if specified
            if let grammarPath = self.contextParams.grammar_path, !grammarPath.isEmpty {
                print("LLMBase.load_model(): Loading grammar from \(grammarPath)...")
                try? self.load_grammar(grammarPath)
            }
            
            print(String(cString: print_system_info()))
            try ExceptionCather.catchException {
                _ = try? self.llm_init_logits()
            }
            print("Logits inited.")
            
        } catch {
            print("load_model() error: \(error)")
            throw error
        }
    }
    
    /// Use specific engine to load the model and create a context
    public func internal_load_model(path: String = "", contextParams: ModelAndContextParams = .default) throws -> Bool {
        // Load the model and create a context
        // llama_model* model = llama_model_load_from_file(model_path, model_params);
        // llama_context* ctx = llama_new_context_with_model(model, context_params);
        return false
    }
    
    // MARK: - High level APIs
    
    /// Tokenize the User Input using llama_tokenize()
    ///
    public func InputToTokens(_ input: String, system_prompt: String? = nil,
                                  img_path: String? = nil) throws -> [llama_token] {
        let contextLength = Int32(self.contextParams.context)
        
        // (1) Evaluate system prompt once, if any
        var system_prompt = system_prompt ?? self.contextParams.system_prompt
        var systemPromptTokens = [ModelToken]()
        if self.nPast == 0 {
            systemPromptTokens = try self._eval_system_prompt(system_prompt: system_prompt)
            print("System prompt tokens: \(systemPromptTokens)")
        }
        print("Past token count: \(self.nPast)/\(contextLength) (\(self.past.count))")
        // (2) Evaluate image embeddings if any
        try self._eval_img(img_path: img_path)
        
        // print("Past token count: \(self.nPast)/\(contextLength) (\(self.past.count))")

        // (3) Tokenize the user input (with system prompt if needed)
        var inputTokens = try self.LLMTokenize(input)
        if inputTokens.isEmpty && (img_path ?? "").isEmpty {
            return []
        }
        // Merge systemPromptTokens and inputTokens
        inputTokens = systemPromptTokens + inputTokens
        
        let inputTokensCount = inputTokens.count
        print("Input tokens: \(inputTokens)")
        print("[BEGIN AI]")
        
        if inputTokensCount > contextLength {
            throw ModelError.inputTooLong
        }
        return inputTokens
    }

    /**
     A base method to tokenize user inputs with optional system prompts and custom chat templates.
     Subclasses (like LLaMa) often override with specialized logic.
     */
    public func LLMTokenize(_ input: String, add_bos: Bool? = nil,
                            parse_special: Bool? = nil,
                            use_template: Bool = true
    ) -> [ModelToken] {
        return []
    }
    
    /**
     Converts a single token to a string (partial or complete).
     - Overridden in a subclass to implement partial UTF-8 logic.
     */
    public func LLMTokenToStr(outputToken: Int32) -> String? {
        return nil
    }
    
    public func KVShift() throws {
    }
    
    public func CheckSkipTokens(_ token: Int32) -> Bool {
        for skip in self.contextParams.skip_tokens {
            if skip == token {
                return false
            }
        }
        return true
    }
    
    /// Evaluate tokens in the context using llama_decode().
    /// - Parameters
    ///
    private func EvalInputTokensBatched(inputTokens: inout [ModelToken],
                                       callback: ((String, Double) -> Bool)) throws {
        var inputBatch: [ModelToken] = []
        
        while !inputTokens.isEmpty {
            inputBatch.removeAll()
            let evalCount = min(inputTokens.count, Int(sampleParams.n_batch))
            inputBatch.append(contentsOf: inputTokens[0..<evalCount])
            inputTokens.removeFirst(evalCount)

            // If adding these tokens would exceed the context limit, do a KVShift
            if self.nPast + Int32(inputBatch.count) >= self.contextParams.context {
                try self.KVShift()
                // Optionally notify the callback about the context limit
                // _ = callback(LLMBase.CONTEXT_LIMIT_MARKER, 0)
                print("LLMBase.EvalInputTokensBatched: Context limit reached, KVShifted.")
            }
            
            var evalResult: Bool? = nil
            try ExceptionCather.catchException {
                // Evaluate the initial tokens in the context
                evalResult = try? self.llm_decode(inputBatch: &inputBatch)
            }
            if evalResult == false {
                throw ModelError.failedToEval
            }
            
            // Update nPast
            self.nPast += Int32(evalCount)
        }
    }
    
    /**
     Evaluates a single pass of the system prompt tokens if present.
     */
    private func _eval_system_prompt(system_prompt: String? = nil) throws -> [ModelToken] {
        if let sp = system_prompt, !sp.isEmpty {
            print("LLMBase: system prompt: \(sp)")
            var systemPromptTokens: [ModelToken] = try LLMTokenize(sp + "\n", use_template: false)
            var evalResult: Bool? = nil
            try ExceptionCather.catchException {
                evalResult = try? self.llm_decode(inputBatch: &systemPromptTokens)
            }
            if evalResult == false {
                throw ModelError.failedToEval
            }
            self.nPast += Int32(systemPromptTokens.count)
            return systemPromptTokens
        }
        return []
    }

    /**
     Evaluates an image embed if present (CLIP or other).
     */
    private func _eval_img(img_path: String? = nil) throws {
        if let path = img_path {
            do {
                try ExceptionCather.catchException {
                    _ = self.load_clip_model()
                    _ = self.make_image_embed(path)
                    _ = try? self.llm_eval_clip()
                    self.deinit_clip_model()
                }
            } catch {
                print("_eval_img() error: \(error)")
                throw error
            }
        }
    }

    public func parse_skip_tokens(){
        self.contextParams.skip_tokens.append(self.llm_token_bos())
        let splitSkipTokens = self.contextParams.skip_tokens_str.components(separatedBy: [","])
        for word in splitSkipTokens {
            let tokenizedSkip = self.LLMTokenize(word, add_bos: false, parse_special: true)
            if tokenizedSkip.count == 1 {
                self.contextParams.skip_tokens.append(tokenizedSkip[0])
            }
        }
    }

    
    // MARK: - Prediction
    
    /**
     Generates predictions from the model given an input text.
     
     - Parameters:
       - input: The user input text.
       - evalCallback: A closure that receives generated tokens as strings, plus timing.
                      Return `true` to stop generation early.
       - system_prompt: (Optional) A system prompt for the context, usually run once at the start.
       - img_path: (Optional) An image path for generating embeddings.
       - infoCallback: A closure to receive debug or info metrics, e.g. token counts.
     
     - Throws:
       - `ModelError.inputTooLong` if tokens exceed the context size.
       - `ModelError.failedToEval` if the internal model evaluation fails.
     
     - Returns: The full generated string after final token is generated or generation ends.
     */
    public func Predict(
        _ input: String,
        _ evalCallback: @escaping ((String, Double) -> Bool),
        system_prompt: String? = nil,
        img_path: String? = nil,
        infoCallback: ((String, Any) -> Void)? = nil
    ) throws -> String {
        var finalOutput = ""
        let semaphore = DispatchSemaphore(value: 0)
        
        chatsStream(
            input: input,
            system_prompt: system_prompt,
            img_path: img_path,
            onPartialResult: { result in
                switch result {
                case .success(let modelResult):
                    let cont = evalCallback(modelResult.choices, modelResult.time)
                    if !cont { semaphore.signal() }
                case .failure(_):
                    semaphore.signal()
                }
            },
            shouldContinue: { result in
                switch result {
                case .success(let modelResult):
                    return evalCallback(modelResult.choices, modelResult.time)
                case .failure(_):
                    return false
                }
            },
            infoCallback: infoCallback,
            completion: { output, time, error in
                finalOutput = output
                semaphore.signal()
            }
        )
        
        semaphore.wait()
        return finalOutput
    }
    
    public enum ChatStreamResult {
        case success(ModelResult)
        case failure(Error)
    }

    public struct ModelResult {
        public let choices: String
        public let time: Double
    }

    // Closure-based streaming
    /**
     A streaming generation function that uses single-token decode batches (similar to the reference code’s `infer`).
     
     **Key changes from your original approach**:
     1. We initialize the prompt (system prompt, image embeddings).
     2. We "batch-evaluate" the input tokens via `EvalInputTokensBatched`.
     3. **Then** we do a loop:
        - `llm_sample()` -> single token
        - Optionally skip if in `skip_tokens`.
        - Convert token to partial UTF-8 string.
        - Accumulate partial results -> call `onPartialResult` once we have a chunk.
        - Build a single-token `llama_batch` (if your subclass supports it) -> decode -> `nPast++`.
     This matches the "sample -> decode -> sample -> decode" pattern from the reference code.
     */
    public func chatsStream(input: String, system_prompt: String? = nil, img_path: String? = nil,
                            onPartialResult: @escaping (ChatStreamResult) -> Void,
                            shouldContinue: ((ChatStreamResult) -> Bool)? = nil,
                            infoCallback: ((String, Any) -> Void)? = nil,
                            completion: @escaping (String, Double, Error?) -> Void) {
        do {
            let startTime = Date()
            // Step 1: Tokenize the User Input using llama_tokenize()
            // Tokenize the input string into a sequence of tokens that the model understands.
            var inputTokens = try InputToTokens(input, system_prompt: system_prompt, img_path: img_path)
            let inputTokensCount = inputTokens.count
            if inputTokens.isEmpty && (img_path ?? "").isEmpty {
                completion("", 0, ModelError.emptyInput)
                return
            }
            
            // Step 2: Evaluate the Context (Prompt Tokens) using llama_decode()
            // Feed the tokenized input to the model, starting with the prompt tokens, and evaluate the context.
            // And evaluate the initial tokens in the context
            try self.EvalInputTokensBatched(inputTokens: &inputTokens) { _, _ in true }
            
            // Prepare for sampling loop
            self.outputRepeatTokens = []
            var outputCache = [String]()
            var fullOutput = [String]()
            var completion_loop = true
            
            // We may call infoCallback if you want to pass debug info at any point
            infoCallback?("promptTokensUsed", inputTokensCount)
            
            // Step 3: Sampling Process (llama_sampler_sample())
            // Now that the context is evaluated, start generating tokens one-by-one using autoregressive decoding:
            // • Use the logits from the last token.
            // • Apply temperature scaling, top-k sampling, top-p sampling, or repetition penalties as necessary.
            // • Sample the next token.
            var callback_message: String?
            while completion_loop {
                var outputToken: Int32 = -1
                try ExceptionCather.catchException {
                    // llama_sampler_sample() selects the next token based on the probability distribution derived from the logits.
                    outputToken = self.llm_sample()
                }
                
                // Keep track of used tokens for repetition penalty, etc.
                self.outputRepeatTokens.append(outputToken)
                if self.outputRepeatTokens.count > self.sampleParams.repeat_last_n {
                    self.outputRepeatTokens.removeFirst()
                }
                
                // If token is EOG/EOS, break
                if self.llm_token_is_eog(token: outputToken) {
                    completion_loop = false
                    break
                }
                
                // Potentially skip the token
                var skipCallback = false
                if !self.CheckSkipTokens(outputToken) {
                    print("LLaMa.CheckSkipTokens() Skip token: \(outputToken)")
                    skipCallback = true
                }
                
                // Step 4: Convert Tokens Back to Output Text (llama_token_to_str())
                if !skipCallback, let str = self.LLMTokenToStr(outputToken: outputToken) {
                    outputCache.append(str)
                    fullOutput.append(str)
                    
                    // If we have enough data, flush partial
                    if outputCache.count >= self.contextParams.predict_cache_length {
                        let chunk = outputCache.joined()
                        outputCache.removeAll()
                        let result = ModelResult(choices: chunk, time: 0)
                        onPartialResult(.success(result))
                        
                        // Check if we should continue after sending partial result
                        if let shouldContinue = shouldContinue, !shouldContinue(.success(result)) {
                            completion_loop = false
                            break
                        }
                    }
                } else {
                    print("LLMBase. Skip token: \(outputToken) because it failed to convert to string or skipCallback is \(skipCallback)")
                }
                
                // Check if we've reached user-requested max tokens
                let outputCount = fullOutput.count
                if self.contextParams.n_predict != 0 && outputCount > self.contextParams.n_predict {
                    print(" * n_predict reached *")
                    callback_message = "n_predict reached(\(self.contextParams.n_predict))"
                    completion_loop = false
                    break
                }
                
                // Check if we should continue after processing this token
                if let shouldContinue = shouldContinue {
                    let currentResult = ModelResult(choices: fullOutput.joined(), time: Date().timeIntervalSince(startTime))
                    if !shouldContinue(.success(currentResult)) {
                        completion_loop = false
                        break
                    }
                }
                
                // If still going, handle context limit
                if completion_loop {
                    if self.nPast >= self.contextParams.context - 2 {
                        // Attempt context shifting if feasible
                        try self.KVShift()
                        // onPartialResult(.success(ModelResult(choices: LLMBase.CONTEXT_LIMIT_MARKER, time: 0)))
                    }
                    
                    // Single-token decode
                    // Instead of building an array [outputToken], we can use the batch approach
                    // if the subclass supports it. But here's the simpler fallback:
                    var singleTokenBatch = [outputToken]
                    var evalResult: Bool? = nil
                    try ExceptionCather.catchException {
                        evalResult = try? self.llm_decode(inputBatch: &singleTokenBatch)
                    }
                    if evalResult == false {
                        throw ModelError.failedToEval
                    }
                    self.nPast += 1
                }
            }
            
            // If anything remains in outputCache, flush it
            if !outputCache.isEmpty {
                let chunk = outputCache.joined()
                onPartialResult(.success(ModelResult(choices: chunk, time: 0)))
            }
            
            // (6) Done
            let endTime = Date()
            let processingTime = endTime.timeIntervalSince(startTime) * 1000
            
            print("\n[END AI]\nTotal tokens: \(inputTokensCount + fullOutput.count) (\(inputTokensCount) -> \(fullOutput.count))")
            completion(fullOutput.joined(), processingTime, callback_message)
            
        } catch {
            completion("", 0, error)
        }
    }
    
    // Combine-based streaming
    
    public func chatsStream(
        input: String,
        system_prompt: String? = nil,
        img_path: String? = nil
    ) -> AnyPublisher<ModelResult, Error> {
        let subject = PassthroughSubject<ModelResult, Error>()
        
        self.chatsStream(
            input: input,
            system_prompt: system_prompt,
            img_path: img_path,
            onPartialResult: { partialResult in
                switch partialResult {
                case .success(let modelResult):
                    subject.send(modelResult)
                case .failure(let error):
                    subject.send(completion: .failure(error))
                }
            },
            completion: { _, _, error in
                if let error = error {
                    subject.send(completion: .failure(error))
                } else {
                    subject.send(completion: .finished)
                }
            }
        )
        
        return subject.eraseToAnyPublisher()
    }
    
    // AsyncSequence-based streaming
    
    public func chatsStream(
        input: String,
        system_prompt: String? = nil,
        img_path: String? = nil
    ) -> AsyncThrowingStream<ModelResult, Error> {
        
        return AsyncThrowingStream { continuation in
            self.chatsStream(input: input, system_prompt: system_prompt, img_path: img_path) { partialResult in
                switch partialResult {
                case .success(let modelResult):
                    continuation.yield(modelResult)
                case .failure(let error):
                    continuation.finish(throwing: error)
                }
            } completion: { _, _, error in
                if let error = error {
                    continuation.finish(throwing: error)
                } else {
                    continuation.finish()
                }
            }
        }
    }
    

    // MARK: - CLIP / Image Embeddings
    
    public func load_clip_model() -> Bool {
        return true
    }
    
    public func make_image_embed(_ image_path: String) -> Bool {
        return true
    }
    
    // MARK: - Session / State
    
    public func load_state(){
    }
    
    public func delete_state() {
    }
    
    public func save_state(){
    }

    
    // MARK: - Low level API
    
    public func load_grammar(_ path: String) throws {
    }
    
    public func load_chat_template(name: String? = nil) -> String? {
        return nil
    }
    
    public func llm_token_is_eog(token: ModelToken) -> Bool {
        return token == llm_token_eos()
    }
    
    public func llm_token_nl() -> ModelToken {
        return llm_nl_token ?? 13
    }
    
    public func llm_token_bos() -> ModelToken {
        return llm_bos_token ?? 0
    }
    
    public func llm_token_eos() -> ModelToken {
        return llm_eos_token ?? 0
    }
    
    func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32 {
        return 0
    }
    
    func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>? {
        return nil
    }
    
    func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32 {
        return 0
    }
    
    /**
     - Note: This is the function that calls a bridging `llama_decode` or similar
       to evaluate a batch of tokens. By default, it just returns false
       unless you override it in a subclass (like LLaMa).
     */
    public func llm_decode(inputBatch: inout [ModelToken]) throws -> Bool {
        return false
    }

    public func llm_eval_clip() throws -> Bool {
        return true
    }
    
    /**
     Samples the next token from the model. If not overridden, returns 0.
     */
    public func llm_sample() -> ModelToken {
        return 0
    }
    
    func llm_init_logits() throws -> Bool {
        do {
            var inputs = [llm_token_bos(), llm_token_eos()]
            try ExceptionCather.catchException {
                _ = try? llm_decode(inputBatch: &inputs)
            }
            return true
        } catch {
            print("llm_init_logits() error: \(error)")
            throw error
        }
    }
    
    public func get_gpu_layers() -> Int32 {
        var n_gpu_layers: Int32 = 0
        let hardware_arch = Get_Machine_Hardware_Name()
        if hardware_arch == "x86_64" {
            n_gpu_layers = 0
        } else if contextParams.use_metal {
            #if targetEnvironment(simulator)
            n_gpu_layers = 0
            print("Running on simulator, force use n_gpu_layers = 0")
            #else
            n_gpu_layers = 100
            #endif
        }
        print("LLMBase n_gpu_layers: \(n_gpu_layers)")
        return n_gpu_layers
    }
    
    
    // MARK: deinit
    
    public func deinit_clip_model() {
    }
    
    public func destroy_objects() {
    }
    
    deinit {
        print("deinit LLMBase")
    }
}
