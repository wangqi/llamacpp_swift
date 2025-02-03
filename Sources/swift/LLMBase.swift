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
    
    var past: [[ModelToken]] = []
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

    // MARK: - Model Loading
    
    public func load_model() throws {
        var loadResult: Bool? = false
        do {
            try ExceptionCather.catchException {
                loadResult = try? self.llm_load_model(path: self.modelPath, contextParams: contextParams)
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

    // MARK: - CLIP placeholders
    
    public func load_clip_model() -> Bool {
        return true
    }
    
    public func deinit_clip_model() {
    }

    public func destroy_objects() {
    }
    
    deinit {
        print("deinit LLMBase")
    }
    
    // MARK: - GPU Layers
    
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
    
    // MARK: - Grammar
    
    public func load_grammar(_ path: String) throws {
    }
    
    // MARK: - Core LLM Calls
    
    public func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default) throws -> Bool {
        return false
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
    
    // MARK: - CLIP / Image Embeddings
    
    public func make_image_embed(_ image_path: String) -> Bool {
        return true
    }
    
    // MARK: - Session / State
    
    public func ForgotLastNTokens(_ N: Int32) {
    }
    
    public func load_state(){
    }
    
    public func delete_state() {
    }
    
    public func save_state(){
    }
    
    // MARK: - Evaluation
    
    public func llm_decode(inputBatch: inout [ModelToken]) throws -> Bool {
        return false
    }

    public func llm_eval_clip() throws -> Bool {
        return true
    }
    
    public func llm_sample() -> ModelToken {
        return 0
    }
    
    // MARK: - Logits Initialization
    
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
    
    public func LLMTokenToStr(outputToken: Int32) -> String? {
        return nil
    }
    
    // MARK: - System Prompt & Image Evaluation
    
    public func _eval_system_prompt(system_prompt: String? = nil) throws {
        if let sp = system_prompt, !sp.isEmpty {
            var systemPromptTokens: [ModelToken] = try TokenizePrompt(sp, .None)
            var evalResult: Bool? = nil
            try ExceptionCather.catchException {
                evalResult = try? self.llm_decode(inputBatch: &systemPromptTokens)
            }
            if evalResult == false {
                throw ModelError.failedToEval
            }
            self.nPast += Int32(systemPromptTokens.count)
        }
    }

    public func _eval_img(img_path: String? = nil) throws {
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
    
    // MARK: - Context Rotation
    
    public func KVShift() throws {
        self.nPast = self.nPast / 2
        try ExceptionCather.catchException {
            var in_batch = [self.llm_token_eos()]
            _ = try? self.llm_decode(inputBatch: &in_batch)
        }
    }
    
    // MARK: - Token Skipping
    
    public func CheckSkipTokens(_ token: Int32) -> Bool {
        for skip in self.contextParams.skip_tokens {
            if skip == token {
                return false
            }
        }
        return true
    }
    
    // MARK: - Batched Evaluation
    
    public func EvalInputTokensBatched(inputTokens: inout [ModelToken],
                                       callback: ((String, Double) -> Bool)) throws {
        var inputBatch: [ModelToken] = []
        
        while !inputTokens.isEmpty {
            inputBatch.removeAll()
            let evalCount = min(inputTokens.count, Int(sampleParams.n_batch))
            inputBatch.append(contentsOf: inputTokens[0..<evalCount])
            inputTokens.removeFirst(evalCount)

            if self.nPast + Int32(inputBatch.count) >= self.contextParams.context {
                try self.KVShift()
                // Optionally notify the callback about the context limit
                _ = callback(LLMBase.CONTEXT_LIMIT_MARKER, 0)
            }
            
            var evalResult: Bool? = nil
            try ExceptionCather.catchException {
                evalResult = try? self.llm_decode(inputBatch: &inputBatch)
            }
            if evalResult == false {
                throw ModelError.failedToEval
            }
            self.nPast += Int32(evalCount)
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
       - stopWhenContextLimitReach: Whether to stop generation if the context is forced to shift.
                                    Defaults to `true`.
     
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
        infoCallback: ((String, Any) -> Void)? = nil,
        stopWhenContextLimitReach: Bool = true
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
            stopWhenContextLimitReach: stopWhenContextLimitReach,
            completion: { output, time, error in
                finalOutput = output
                semaphore.signal()
            }
        )
        
        semaphore.wait()
        return finalOutput
    }
    
    // MARK: - Prompt Tokenization
    
    public func TokenizePrompt(_ input: String, _ style: ModelPromptStyle) throws -> [ModelToken] {
        switch style {
        case .None:
            print("LLMBase.TokenizePrompt: use model default Jinja2 Chat Template")
            return LLMTokenize(input)
        case .Custom:
            print("LLMBase.TokenizePrompt: use custom Jinja2 Chat Template")
            return LLMTokenize(input, chatTemplate: self.contextParams.custom_prompt_format)
        }
    }

    public func TokenizePromptWithSystem(_ input: String,
                                         _ systemPrompt: String,
                                         _ style: ModelPromptStyle) throws -> [ModelToken] {
        switch style {
        case .None:
            print("LLMBase.TokenizePromptWithSystem: use model default Jinja2 Chat Template")
            return LLMTokenize(input, systemPrompt: systemPrompt)
        case .Custom:
            print("LLMBase.TokenizePromptWithSystem: use custom Jinja2 Chat Template")
            return LLMTokenize(input,
                               chatTemplate: self.contextParams.custom_prompt_format,
                               systemPrompt: systemPrompt)
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
    
    public func LLMTokenize(
        _ input: String,
        chatTemplate: String? = nil,
        systemPrompt: String? = nil,
        add_bos: Bool? = nil,
        parse_special: Bool? = nil
    ) -> [ModelToken] {
        return []
    }
    
    // MARK: - Chats Streaming
    
    public enum ChatStreamResult {
        case success(ModelResult)
        case failure(Error)
    }

    public struct ModelResult {
        public let choices: String
        public let time: Double
    }

    // MARK: - Closure-based streaming

    public func chatsStream(input: String, system_prompt: String? = nil, img_path: String? = nil,
                            onPartialResult: @escaping (ChatStreamResult) -> Void,
                            shouldContinue: ((ChatStreamResult) -> Bool)? = nil,
                            infoCallback: ((String, Any) -> Void)? = nil,
                            stopWhenContextLimitReach: Bool = true,
                            completion: @escaping (String, Double, Error?) -> Void) {
        do {
            let startTime = Date()
            let contextLength = Int32(self.contextParams.context)
            
            if self.nPast == 0 {
                try self._eval_system_prompt(system_prompt: system_prompt)
            }
            try self._eval_img(img_path: img_path)
            
            print("Past token count: \(self.nPast)/\(contextLength) (\(self.past.count))")
            
            var inputTokens = try self.TokenizePromptWithSystem(input, system_prompt ?? "", self.contextParams.promptFormat)
            if inputTokens.isEmpty && (img_path ?? "").isEmpty {
                completion("", 0, ModelError.emptyInput)
                return
            }
            
            let inputTokensCount = inputTokens.count
            print("Input tokens: \(inputTokens)")
            
            if inputTokensCount > contextLength {
                throw ModelError.inputTooLong
            }
            
            try self.EvalInputTokensBatched(inputTokens: &inputTokens) { _, _ in true }
            
            self.outputRepeatTokens = []
            var outputCache = [String]()
            var fullOutput = [String]()
            var completion_loop = true
            
            while completion_loop {
                var outputToken: Int32 = -1
                try ExceptionCather.catchException {
                    outputToken = self.llm_sample()
                }
                
                self.outputRepeatTokens.append(outputToken)
                if self.outputRepeatTokens.count > self.sampleParams.repeat_last_n {
                    self.outputRepeatTokens.removeFirst()
                }
                
                if self.llm_token_is_eog(token: outputToken) {
                    completion_loop = false
                    print("[EOG]")
                    break
                }
                
                var skipCallback = false
                if !self.CheckSkipTokens(outputToken) {
                    print("Skip token: \(outputToken)")
                    skipCallback = true
                }
                
                if !skipCallback, let str = self.LLMTokenToStr(outputToken: outputToken) {
                    outputCache.append(str)
                    fullOutput.append(str)
                    
                    if outputCache.count >= self.contextParams.predict_cache_length {
                        let chunk = outputCache.joined()
                        outputCache.removeAll()
                        onPartialResult(.success(ModelResult(choices: chunk, time: 0)))
                    }
                }
                
                let outputCount = fullOutput.count
                if self.contextParams.n_predict != 0 && outputCount > self.contextParams.n_predict {
                    print(" * n_predict reached *")
                    completion_loop = false
                    break
                }
                
                if completion_loop {
                    if self.nPast >= self.contextParams.context - 2 {
                        try self.KVShift()
                        onPartialResult(.success(ModelResult(choices: LLMBase.CONTEXT_LIMIT_MARKER, time: 0)))
                        
                        // If the user wants to stop when the context limit is reached
                        if stopWhenContextLimitReach {
                            print(" * context limit reached, stop generation *")
                            completion_loop = false
                            break
                        }
                    }
                    
                    var evalResult: Bool? = nil
                    try ExceptionCather.catchException {
                        var inputBatch = [outputToken]
                        evalResult = try? self.llm_decode(inputBatch: &inputBatch)
                    }
                    if evalResult == false {
                        throw ModelError.failedToEval
                    }
                    self.nPast += 1
                }
            }
            
            if !outputCache.isEmpty {
                let chunk = outputCache.joined()
                fullOutput.append(chunk)
                onPartialResult(.success(ModelResult(choices: chunk, time: 0)))
            }
            
            let endTime = Date()
            let processingTime = endTime.timeIntervalSince(startTime) * 1000
            
            print("Total tokens: \(inputTokensCount + fullOutput.count) (\(inputTokensCount) -> \(fullOutput.count))")
            completion(fullOutput.joined(), processingTime, nil)
            
        } catch {
            completion("", 0, error)
        }
    }
    
    // MARK: - Combine-based streaming
    
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
    
    // MARK: - AsyncSequence-based streaming
    
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
}
