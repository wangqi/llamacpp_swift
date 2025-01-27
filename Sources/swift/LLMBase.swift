//
//  LLMBase.swift
//  Created by Guinmoon.

import Foundation
import llama
import llamacpp_swift_cpp

public enum ModelLoadError: Error {
    case modelLoadError
    case contextLoadError
    case grammarLoadError
}

public class LLMBase {
    
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
    
    // Used to keep old context until it needs to be rotated or purged for new tokens
    var past: [[ModelToken]] = []
    public var nPast: Int32 = 0
    
    // Initializes the model with the given file path and parameters
    public init(path: String, contextParams: ModelAndContextParams = .default) throws {
        self.modelPath = path
        self.contextParams = contextParams
        
        // Ensure the model file exists at the specified path
        if !FileManager.default.fileExists(atPath: self.modelPath) {
            throw ModelError.modelNotFound(self.modelPath)
        }
    }

    // Loads the model into memory, sets up grammar if specified, and initializes logits
    public func load_model() throws {
        var load_res: Bool? = false
        do {
            try ExceptionCather.catchException {
                load_res = try? self.llm_load_model(path: self.modelPath, contextParams: contextParams)
            }
        
            if load_res != true {
                throw ModelLoadError.modelLoadError
            }
            
            // Load grammar if specified in context parameters
            if let grammarPath = self.contextParams.grammar_path, !grammarPath.isEmpty {
                try? self.load_grammar(grammarPath)
            }
            
            print(String(cString: print_system_info()))
            try ExceptionCather.catchException {
                _ = try? self.llm_init_logits()
            }

            print("Logits inited.")
        } catch {
            print(error)
            throw error
        }
    }

    // Placeholder function for loading a CLIP model
    public func load_clip_model() -> Bool {
        return true
    }
    
    // Placeholder function for deinitializing a CLIP model
    public func deinit_clip_model() {}

    // Placeholder function for destroying all resources or objects
    public func destroy_objects() {}

    deinit {
        print("deinit LLMBase")
    }
    
    // Retrieves the number of GPU layers based on hardware and context parameters
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
        return n_gpu_layers
    }
    
    // Loads grammar from the specified file path
    public func load_grammar(_ path: String) throws {
        do {
            // Implement grammar loading logic here if required
        } catch {
            print(error)
            throw error
        }
    }
    
    // Loads the model with the specified path and parameters
    public func llm_load_model(path: String = "", contextParams: ModelAndContextParams = .default) throws -> Bool {
        return false
    }
    
    // Checks if a token represents the end of generation
    public func llm_token_is_eog(token: ModelToken) -> Bool {
        return token == llm_token_eos()
    }
    
    // Returns the newline token
    public func llm_token_nl() -> ModelToken {
        return 13
    }
    
    // Returns the beginning-of-sequence (BOS) token
    public func llm_token_bos() -> ModelToken {
        print("llm_token_bos base")
        return 0
    }
    
    // Returns the end-of-sequence (EOS) token
    public func llm_token_eos() -> ModelToken {
        print("llm_token_eos base")
        return 0
    }
    
    // Retrieves the vocabulary size for the given context
    func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32 {
        print("llm_n_vocab base")
        return 0
    }
    
    // Retrieves the logits for the given context
    func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>? {
        print("llm_get_logits base")
        return nil
    }
    
    // Retrieves the context length for the given context
    func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32 {
        print("llm_get_n_ctx base")
        return 0
    }
    
    // Placeholder function for creating an image embedding
    public func make_image_embed(_ image_path: String) -> Bool {
        return true
    }
    
    public func load_state(){
        
    }
    
    public func save_state(){
        
    }
    
    // Evaluates the input tokens and updates the logits
    public func llm_eval(inputBatch: inout [ModelToken]) throws -> Bool {
        return false
    }

    // Evaluates the CLIP model
    public func llm_eval_clip() throws -> Bool {
        return true
    }
    
    // Simple topK, topP, temp sampling, with repeat penalty
    public func llm_sample() -> ModelToken {
        return 0;
    }
    
    // Initializes the logits for the model
    func llm_init_logits() throws -> Bool {
        do {
            var inputs = [llm_token_bos(), llm_token_eos()]
            try ExceptionCather.catchException {
                _ = try? llm_eval(inputBatch: &inputs)
            }
            return true
        } catch {
            print(error)
            throw error
        }
    }
    
    // Converts a token to its string representation
    public func llm_token_to_str(outputToken: Int32) -> String? {
        return nil
    }
    
    // Evaluates a system prompt
    public func _eval_system_prompt(system_prompt: String? = nil) throws {
        if let prompt = system_prompt {
            var systemPromptTokens = try tokenizePrompt(prompt, .None)
            var eval_res: Bool? = nil
            try ExceptionCather.catchException {
                eval_res = try? self.llm_eval(inputBatch: &systemPromptTokens)
            }
            if eval_res == false {
                throw ModelError.failedToEval
            }
            self.nPast += Int32(systemPromptTokens.count)
        }
    }

    // Evaluates an image and generates embeddings
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
                print(error)
                throw error
            }
        }
    }
    
    // Performs context rotation to handle context size limits
    public func kv_shift() throws {
        self.nPast = self.nPast / 2
        try ExceptionCather.catchException {
            var in_batch = [self.llm_token_eos()]
            _ = try? self.llm_eval(inputBatch: &in_batch)
        }
        print("Context Limit!")
    }

    // Checks if a token should be skipped
    public func chekc_skip_tokens(_ token: Int32) -> Bool {
        for skip in self.contextParams.skip_tokens {
            if skip == token {
                return false
            }
        }
        return true
    }

    // Evaluates input tokens in batches and calls the callback function after each batch
    public func eval_input_tokens_batched(inputTokens: inout [ModelToken], callback: ((String, Double) -> Bool)) throws {
        var inputBatch: [ModelToken] = []
        while inputTokens.count > 0 {
            inputBatch.removeAll()
            // See how many to eval (up to batch size??? or can we feed the entire input)
            // Move tokens to batch
            let evalCount = min(inputTokens.count, Int(sampleParams.n_batch))
            inputBatch.append(contentsOf: inputTokens[0 ..< evalCount])
            inputTokens.removeFirst(evalCount)

            if self.nPast + Int32(inputBatch.count) >= self.contextParams.context{
                try self.kv_shift()
                _ = callback(" `C_LIMIT` ",0)
            }
            var eval_res:Bool? = nil
            try ExceptionCather.catchException {
                eval_res = try? self.llm_eval(inputBatch: &inputBatch)
            }
            if eval_res == false{
                throw ModelError.failedToEval
            }
            self.nPast += Int32(evalCount)
        }
    }

    // Generates predictions based on input text and optional image/system prompts
    public func predict(_ input: String, _ callback: ((String, Double) -> Bool), system_prompt: String? = nil, img_path: String? = nil) throws -> String {
        //Eval system prompt then image if it's not nil
        if self.nPast == 0{
            try _eval_system_prompt(system_prompt:system_prompt)
        }
        try _eval_img(img_path:img_path)
        
        let contextLength = Int32(contextParams.context)
        print("Past token count: \(nPast)/\(contextLength) (\(past.count))")
        // Tokenize with prompt format
        do {
            var inputTokens = try tokenizePromptWithSystem(input, system_prompt ?? "", self.contextParams.promptFormat)
            if inputTokens.count == 0 && img_path == nil{
                return "Empty input."
            }
            let inputTokensCount = inputTokens.count
            print("Input tokens: \(inputTokens)")

            if inputTokensCount > contextLength {
                throw ModelError.inputTooLong
            }

            var inputBatch: [ModelToken] = []
        
            //Batched Eval all input tokens
            try eval_input_tokens_batched(inputTokens: &inputTokens,callback:callback)
            // Output
            outputRepeatTokens = []
            var output = [String]()
            //The output_cache is used to cumulate the predicted string
            var output_cache = [String]()
            // Loop until target count is reached
            var completion_loop = true
            // let eos_token = llm_token_eos()
            while completion_loop {
                // Pull a generation from context
                var outputToken:Int32 = -1
                try ExceptionCather.catchException {
                    outputToken = self.llm_sample()
                }
                // Repeat tokens update
                outputRepeatTokens.append(outputToken)
                if outputRepeatTokens.count > sampleParams.repeat_last_n {
                    outputRepeatTokens.removeFirst()
                }
                // Check for eos - end early - check eos before bos in case they are the same
                // if outputToken == eos_token {
                //     completion_loop = false
                //     print("[EOS]")
                //     break
                // }
                if llm_token_is_eog(token:outputToken) {
                    completion_loop = true
                    print("[EOG]")
                    break
                }

                // Check for BOS and tokens in skip list
                var skipCallback = false
                if !self.chekc_skip_tokens(outputToken){
                    print("Skip token: \(outputToken)")
                    skipCallback = true
                }
                // Convert token to string and callback
                if !skipCallback, let str = llm_token_to_str(outputToken: outputToken){
                    //output.append(str)
                    output_cache.append(str)
                    if output_cache.count >= self.contextParams.predict_cache_length {
                        let cached_content = output_cache.joined()
                        output.append(contentsOf: output_cache)
                        output_cache = []
                        // Per token callback with accumulated cache
                        let (_, time) = Utils.time {
                            return cached_content
                        }
                        if callback(cached_content, time) {
                            // Early exit if requested by callback
                            print(" * exit requested by callback *")
                            completion_loop = false
                            break
                        }
                    }
                }
                // Max output tokens count reached
                let output_count = output.count + output_cache.count
                if (self.contextParams.n_predict != 0 && output_count>self.contextParams.n_predict){
                    print(" * n_predict reached *")
                    completion_loop = false
                    output.append(contentsOf: output_cache)
                    output_cache = []
                    break
                }
                // Check if we need to run another response eval
                if completion_loop {
                    // Send generated token back into model for next generation
                    var eval_res:Bool? = nil
                    if self.nPast >= self.contextParams.context - 2{
                        try self.kv_shift()
                        _ = callback(" `C_LIMIT` ",0)
                    }
                    try ExceptionCather.catchException {
                        inputBatch = [outputToken]
                        eval_res = try? self.llm_eval(inputBatch: &inputBatch)
                    }
                    if eval_res == false{
                        print("Eval res false")
                        throw ModelError.failedToEval
                    }
                    nPast += 1
                }
            }
            if output_cache.count > 0 {
                output.append(contentsOf: output_cache)
                output_cache = []
            }
            print("Total tokens: \(inputTokensCount + output.count) (\(inputTokensCount) -> \(output.count))")
            // print("Past token count: \(nPast)/\(contextLength) (\(past.count))")
            // Return full string for case without callback
            return output.joined()
        }catch{
            print(error)
            throw error
        }
    }

    // Tokenizes the input string based on the given style
    public func tokenizePrompt(_ input: String, _ style: ModelPromptStyle) throws -> [ModelToken] {
        switch style {
        case .None:
            return llm_tokenize(input)
        case .Custom:
            var formated_input = self.contextParams.custom_prompt_format.replacingOccurrences(of: "{{prompt}}", with: input)
            formated_input = formated_input.replacingOccurrences(of: "{prompt}", with: input)
            formated_input = formated_input.replacingOccurrences(of: "\\n", with: "\n")
            var tokenized:[ModelToken] = []
            try ExceptionCather.catchException {
                tokenized = llm_tokenize(formated_input)
            }
            return tokenized
         }
    }

    // Tokenizes input with both system and prompt strings
    public func tokenizePromptWithSystem(_ input: String, _ systemPrompt: String, _ style: ModelPromptStyle) throws -> [ModelToken] {
        switch style {
        case .None:
            return llm_tokenize(input)
        case .Custom:
            var formated_input = self.contextParams.custom_prompt_format.replacingOccurrences(of: "{{system}}", with: systemPrompt)
            formated_input = formated_input.replacingOccurrences(of: "{system}", with: systemPrompt)
            formated_input = formated_input.replacingOccurrences(of: "{{prompt}}", with: input)
            formated_input = formated_input.replacingOccurrences(of: "{prompt}", with: input)
            formated_input = formated_input.replacingOccurrences(of: "\\n", with: "\n")
            print("LLMBase.tokenizePromptWithSystem: Input text '\(formated_input)'")
            var tokenized:[ModelToken] = []
            try ExceptionCather.catchException {
                tokenized = llm_tokenize(formated_input)
            }
            return tokenized
         }
    }

    // Parses and adds tokens to be skipped during processing
    public func parse_skip_tokens(){
        // This function must be called after model loaded
        // Add BOS token to skip
        self.contextParams.skip_tokens.append(self.llm_token_bos())

        let splited_skip_tokens = self.contextParams.skip_tokens_str.components(separatedBy: [","])
        for word in splited_skip_tokens{
            let tokenized_skip = self.llm_tokenize(word,add_bos: false,parse_special: true)
            // Add only if tokenized text is one token
            if tokenized_skip.count == 1{
                self.contextParams.skip_tokens.append(tokenized_skip[0])
            }
        }
        
    }
    
    public func llm_tokenize(_ input: String, add_bos: Bool? = nil, parse_special:Bool? = nil) -> [ModelToken] {
        return []
    }
}
