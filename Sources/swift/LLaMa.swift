//
//  LLaMa.swift
//  Created by Guinmoon.
//

import Foundation
import llama
import llamacpp_swift
import llamacpp_swift_cpp
import Jinja

// Global reference (potentially a strong reference) that can cause retain cycles or memory leaks.
// Take care that you release or nil out LLaMa_obj if you want the LLaMa instance to deinit.
var LLaMa_obj: LLaMa? = nil

public class LLaMa: LLMBase {

    // MARK: - Public Properties
    
    public var model: OpaquePointer?
    public var ctx_sampling: SpmSamplerContext?
    public var vocab: OpaquePointer?
    public var batch: llama_batch?
    public var hardware_arch: String = ""
    public private(set) var chatTemplate: ChatTemplate?
    
    // Temporarily accumulates CChar arrays when partial UTF-8 decoding is happening
    public var temporary_invalid_cchars: [CChar] = []

    // MARK: - init_sampling_param()
    
    /// Initialize sampling parameters for the new sampling context.
    /// This sets up the SpmSamplingParams struct from older sampleParams.
    public func init_sampling_param() {
        // Create a new SpmSamplingParams instance
        var spmParams = SpmSamplingParams()

        // Map all your old parameters into the new struct
        spmParams.penaltyLastN        = sampleParams.repeat_last_n
        spmParams.topK                = sampleParams.top_k
        spmParams.topP                = sampleParams.top_p
        spmParams.minP                = sampleParams.min_p
        // If tfs_z was effectively typical p (sometimes TFS is used in place of typicalP),
        // set that here IF you no longer have a separate "tfs_z" in SpmSamplingParams.
        spmParams.typicalP            = sampleParams.typical_p
        spmParams.temp                = sampleParams.temp

        // The old call was passing 0.0 for dynamic temperature range and 1.0 for exponent
        spmParams.dynaTempRange       = 0.0
        spmParams.dynaTempExponent    = 1.0

        spmParams.penaltyRepeat       = sampleParams.repeat_penalty
        spmParams.penaltyFreq         = sampleParams.frequence_penalty
        spmParams.penaltyPresent      = sampleParams.presence_penalty
        spmParams.mirostat            = sampleParams.mirostat
        spmParams.mirostatTau         = sampleParams.mirostat_tau
        spmParams.mirostatEta         = sampleParams.mirostat_eta
        spmParams.penalizeNl          = sampleParams.penalize_nl

        // If you want a specific seed, set it here.
        // The old code was just passing 0, so:
        spmParams.seed                = 0

        // Grammar path
        spmParams.grammarPath         = self.contextParams.grammar_path ?? ""

        // Now call the updated function with the struct
        self.ctx_sampling = init_sampling(model: model, params: spmParams)
    }
    
    // MARK: - llm_load_model()
    
    /**
     Loads the model using llama_load_model_from_file and sets up the context.
     
     - Parameters:
       - path: Path to the model file.
       - contextParams: ModelAndContextParams with context settings.
     - Returns: True if loading succeeded, false otherwise.
     - Throws: If `llama_load_model_from_file` or subsequent calls throw, we propagate the error.
     */
    public override func llm_load_model(path: String = "",
                                        contextParams: ModelAndContextParams = .default) throws -> Bool
    {
        var context_params = llama_context_default_params()
        var model_params   = llama_model_default_params()
        
        // Setup context parameters
        context_params.n_ctx       = UInt32(contextParams.context)
        // context_params.seed     = UInt32(contextParams.seed) // commented out
        context_params.n_threads   = contextParams.n_threads
        context_params.logits_all  = contextParams.logitsAll
        context_params.flash_attn  = contextParams.flash_attn
        // context_params.flash_attn = false
        
        // Setup model parameters
        model_params.vocab_only = contextParams.vocabOnly
        model_params.use_mlock  = contextParams.useMlock
        model_params.use_mmap   = contextParams.useMMap
        
        // Retain a global reference to self. This can cause memory leaks if never released.
        // If you only need a global pointer without incrementing the reference count, consider
        // passUnretained. If you do need to keep the object around, ensure you set LLaMa_obj to nil
        // when you want to let it deinit.
        //todo: Potential memory leak from passRetained. If LLaMa_obj is never nil-ed, might cause a retain cycle.
        self.retain_new_self_ptr()
        
        // Provide a model-load callback for progress
        model_params.progress_callback = { progress, b in
            if (LLaMa_obj?.modelLoadProgressCallback != nil) {
                let res = LLaMa_obj?.modelLoadProgressCallback!(progress)
                return res ?? false
            }
            return true
        }
        
        // Possibly retrieve GPU layer count from device or user setting
        model_params.n_gpu_layers = get_gpu_layers()
        
        #if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
        #endif
        
        // If using LoRA adapters, user forced use_mmap = false
        if contextParams.lora_adapters.count > 0 {
            model_params.use_mmap = false
        }

        // Model load progress callback
        _ = self.modelLoadProgressCallback?(0)

        // Initialize the llama backend
        llama_backend_init()

        // Attempt model load with error capture
        try ExceptionCather.catchException {
            self.model = llama_load_model_from_file(path, model_params)
            self.vocab = llama_model_get_vocab(model) // self.model is the same pointer if not nil
        }
        if self.model == nil {
            return false
        }

        // Initialize sampling context
        init_sampling_param()

        // Attempt context creation
        try ExceptionCather.catchException {
            self.context = llama_new_context_with_model(self.model, context_params)
        }
        if self.context == nil {
            return false
        }
        
        // load_clip_model() is called here, but the function is not actually provided
        // in this snippet. If load_clip_model() always returns true, it's fine.
        // Otherwise, you need to implement or link that function so it loads and returns properly.
        //todo: 'load_clip_model' is a stub. If it returns false, everything fails. Implement or remove?
        if !load_clip_model() {
            return false
        }
        
        // Attempt to restore previously saved state if needed
        self.load_state()
        
        // Create a new batch
        self.batch = llama_batch_init(sampleParams.n_batch, 0, 1)
        
        // Get model's chat template
        let modelChatTemplate = self.load_chat_template() ?? """
{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
"""
        
        // Initialize chat template
        self.chatTemplate = ChatTemplate(
            source: modelChatTemplate,
            bosToken: LLMTokenToStr(outputToken: llama_vocab_bos(self.vocab)) ?? "<s>",
            eosToken: LLMTokenToStr(outputToken: llama_vocab_eos(self.vocab)) ?? "</s>"
        )
        
        return true
    }
    
    // MARK: - load_chat_template()
    
    /**
     Loads a default chat template from model metadata, if available.
     
     - Parameter name: Template name if using multiple templates.
     - Returns: A string-based template or nil if not found.
     */
    public override func load_chat_template(name: String? = nil) -> String? {
        return spm_llama_model_chat_template(model: self.model, name: name)
    }
    
    // MARK: - llm_sample()
    
    /**
     Samples the next token from the model using spm_llama_sampling.
     
     - Returns: The sampled token as a ModelToken.
     */
    public override func llm_sample() -> ModelToken {
        // spm_llama_sampling_sample picks a token
        let id = spm_llama_sampling_sample(ctxSampling: self.ctx_sampling,
                                           ctxMain: self.context,
                                           idx: -1,
                                           grammarFirst: false)
        // Then accept that token so grammar or rep-penalty states are updated
        spm_llama_sampling_accept(ctxSampling: self.ctx_sampling,
                                  ctxMain: self.context,
                                  token: id,
                                  applyGrammar: true)
        return id
    }
    
    // MARK: - load_state()
    
    /**
     Attempts to load state from a file if `save_load_state` is true and a path is given.
     It also loads tokens and sets nPast from the last item in that token array.
     */
    public override func load_state() {
        if self.contextParams.save_load_state &&
           self.contextParams.state_dump_path != "" &&
           FileManager.default.fileExists(atPath: self.contextParams.state_dump_path)
        {
            var tokens_tmp: [llama_token] = [Int32](repeating: 0, count: 4096)
            var tokens_count: Int = 0
            
            llama_state_load_file(self.context,
                                  self.contextParams.state_dump_path,
                                  tokens_tmp.mutPtr,
                                  4096,
                                  &tokens_count)
            // If tokens_count > 0, we treat the last token as nPast
            // This is a bit unusual: it means we stored the nPast as the last item in the token array.
            //todo: This approach lumps nPast in the last token. Confirm no off-by-one issues.
            if tokens_count > 0 {
                self.outputRepeatTokens.append(contentsOf: tokens_tmp[0 ..< tokens_count-1])
                self.nPast = tokens_tmp[tokens_count - 1]
            }
        }
    }
    
    // MARK: - delete_state()
    
    /**
     Deletes the serialized state file if `save_load_state` is true and the file exists.
     */
    public override func delete_state() {
        if self.contextParams.save_load_state &&
           self.contextParams.state_dump_path != "" &&
           FileManager.default.fileExists(atPath: self.contextParams.state_dump_path)
        {
            do {
                try FileManager.default.removeItem(atPath: self.contextParams.state_dump_path)
                print("State file deleted successfully: \(self.contextParams.state_dump_path)")
            } catch {
                print("Error deleting state file: \(error.localizedDescription)")
            }
        } else {
            print("No state file found to delete.")
        }
    }
    
    // MARK: - save_state()
    
    /**
     Appends the current nPast to outputRepeatTokens, then calls llama_state_save_file
     to persist them if `save_load_state` is set.
     */
    public override func save_state() {
        if self.contextParams.save_load_state &&
           self.contextParams.state_dump_path != "" &&
           self.context != nil
        {
            // We store nPast as the last item.
            //todo: We store 'nPast' as the last token. Confirm it's always handled in 'load_state()'.
            self.outputRepeatTokens.append(self.nPast)
            
            llama_state_save_file(self.context,
                                  self.contextParams.state_dump_path,
                                  self.outputRepeatTokens,
                                  self.outputRepeatTokens.count)
        }
    }
    
    // MARK: - retain_new_self_ptr()
    
    /**
     Creates a retained reference to self and stores it in the global LLaMa_obj variable.
     Potentially leads to strong reference cycles if never nil-ed out.
     */
    //todo: Potential memory leak from passRetained if not freed.
    private func retain_new_self_ptr() {
        LLaMa_obj = Unmanaged<LLaMa>
            .fromOpaque(Unmanaged.passRetained(self).toOpaque())
            .takeRetainedValue()
    }
    
    // MARK: - destroy_objects()
    
    /**
     Frees the batch, context, and model. destroy_clip is also called, but llama_backend_free
     is commented out (not used).
     */
    public override func destroy_objects() {
        print("destroy LLaMa")
        
        // Freed if not nil
        if batch != nil {
            llama_batch_free(batch!)
        }
        if context != nil {
            llama_free(context)
        }
        if model != nil {
            llama_free_model(model)
        }
        
        // If you have a separate clip model, ensure that is freed too
        self.destroy_clip()
        
        // llama_backend_free() is commented out in your snippet, so the entire backend might remain
        // if you need to completely shut down the backend, consider calling it
        // llama_backend_free()
    }
    
    // MARK: - destroy_clip()
    
    /**
     Destroys clip-related allocations. Currently empty placeholder.
     */
    public func destroy_clip() {
        // Not implemented in the snippet
    }
    
    // MARK: - deinit
    
    deinit {
        // Saves the state upon destruction if needed
        self.save_state()
        print("deinit LLaMa")
        self.destroy_objects()
        print("LLaMa deinited")
    }
    
    // MARK: - Low-Level Overrides
    
    /// Returns context size
    override func llm_get_n_ctx(ctx: OpaquePointer!) -> Int32 {
        return Int32(llama_n_ctx(self.context))
    }
    
    /// Returns size of vocab
    override func llm_n_vocab(_ ctx: OpaquePointer!) -> Int32 {
        return llama_n_vocab(self.model)
    }
    
    /// Returns pointer to logits
    override func llm_get_logits(_ ctx: OpaquePointer!) -> UnsafeMutablePointer<Float>? {
        return llama_get_logits(self.context)
    }
    
    // load_grammar is empty here—check if you really need to do something
    public override func load_grammar(_ path:String) throws -> Void { }

    // MARK: - llm_decode()
    
    /**
     Evaluates tokens using llama_decode. In official llama.cpp, the standard function is `llama_eval()`,
     but your bridging library might name it `llama_decode()`. Double-check that you are calling
     the correct function for forward evaluation.

     - Parameter inputBatch: The tokens to be evaluated (decoded).
     - Returns: True on success, false otherwise.
     - Throws: Rethrows any bridging errors encountered.
     */
    public override func llm_decode(inputBatch: inout [ModelToken]) throws -> Bool {
        
        // Potential mismatch: "llm_eval" typically calls something like llama_eval(...),
        // but the bridging layer might have "llama_decode" do the same job.
        if llama_decode(context, llama_batch_get_one(&inputBatch, Int32(inputBatch.count))) != 0
        {
            print("failed to evaluate llama!")
            return false
        }
        return true
    }
    
    // MARK: - ForgotLastNTokens(_ N:)

    /**
     This function claims to "forget" the last N tokens, but currently ignores `N` and
     calls `llama_kv_cache_seq_rm(self.context, -1, 0, -1)`, which likely removes
     everything in the KV cache. If the intent is to remove exactly N tokens from the end,
     you'll need to adjust your call to pass the correct offsets.

     - Parameter N: Number of tokens to forget, but is currently ignored.
     */
    public override func ForgotLastNTokens(_ N: Int32) {
        // BUG: Ignores N. The call below typically means "rm from seq_id = -1, pos = 0, end = -1",
        // which could remove the entire context or be an undefined range. Adjust accordingly.
        llama_kv_cache_seq_rm(self.context, -1, 0, -1)
    }

    // MARK: - KVShift()
    
    /**
     Shifts the KV cache by discarding half of nPast starting after repeat_last_n.
     Then sets nPast = nPast - n_discard + 1.
     This could be correct for your partial context approach, but watch for boundary cases
     (e.g., if nPast < 2 or if n_discard is bigger than the valid range).
     */
    public override func KVShift() throws {
        let n_discard = self.nPast / 2
        
        // Removes tokens from repeat_last_n to repeat_last_n + n_discard
        llama_kv_cache_seq_rm(context,
                              0,
                              self.sampleParams.repeat_last_n,
                              self.sampleParams.repeat_last_n + n_discard)
        
        // Adds them in with an offset of -n_discard
        llama_kv_cache_seq_add(context,
                               0,
                               self.sampleParams.repeat_last_n + n_discard,
                               self.nPast,
                               -n_discard)
        
        self.nPast -= n_discard
        self.nPast += 1
        // print("Context Limit!")
    }
    
    // MARK: - llm_init_logits()
    
    /**
     Placeholder. Currently returns true without doing anything.
     Implement any required logic to ensure logits are valid
     or state is ready before you sample or decode further.
     */
    override func llm_init_logits() throws -> Bool {
        return true
    }
    
    // MARK: - LLaMa Batch Helpers
    
    /**
     Clears the batch token count (n_tokens = 0).
     */
    func llama_batch_clear(_ batch: inout llama_batch) {
        batch.n_tokens = 0
    }
    
    /**
     Adds a token and its positional / seq-id metadata into the batch.
     - Parameter logits: If true, indicates we want to compute logits for this token.
     */
    func llama_batch_add(_ batch: inout llama_batch,
                         _ id: llama_token,
                         _ pos: llama_pos,
                         _ seq_ids: [llama_seq_id],
                         _ logits: Bool)
    {
        batch.token[Int(batch.n_tokens)]    = id
        batch.pos[Int(batch.n_tokens)]      = pos
        batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
        
        for i in 0..<seq_ids.count {
            batch.seq_id[Int(batch.n_tokens)]![Int(i)] = seq_ids[i]
        }
        batch.logits[Int(batch.n_tokens)] = logits ? 1 : 0
        
        batch.n_tokens += 1
    }
    
    // MARK: - model_info()
    
    /**
     Grabs the model description string (if any) from llama_model_desc.
     Possibly returns short info about the model architecture or version.
     */
    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }
        
        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))
        
        var swiftString = ""
        for char in bufferPointer {
            swiftString.append(Character(UnicodeScalar(UInt8(char))))
        }
        
        return swiftString
    }
    
    // MARK: - token_to_piece(token:)

    /**
     Converts a single token to its piece (UTF-8 data) by calling llama_token_to_piece.
     If the returned size is negative, we re-allocate to the needed size.
     */
    private func token_to_piece(token: Int32) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }

        let nTokens = llama_token_to_piece(self.vocab,
                                           token,
                                           result,
                                           8,
                                           0,
                                           /* parse_special_tokens: */ self.contextParams.parse_special_tokens)
        
        if nTokens < 0 {
            // Negative means we need to allocate -nTokens capacity
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(self.vocab,
                                                  token,
                                                  newResult,
                                                  -nTokens,
                                                  0,
                                                  self.contextParams.parse_special_tokens)
            
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
    
    // MARK: - LLMTokenToStr(outputToken:)
    
    /**
     Converts a token to its string representation. Accumulates partial UTF-8 bytes
     until a valid string can be formed. If the partial data cannot form a valid UTF-8
     string, attempts to parse suffixes. If you often have partial tokens from streaming,
     you can refine this logic or track partial merges carefully.
     //todo: This partial UTF-8 accumulation logic may discard or lose data if not tested thoroughly.
     
     - Returns: A valid UTF-8 string or an empty string if still partial/invalid.
     */
    public override func LLMTokenToStr(outputToken: Int32) -> String? {
        let new_token_cchars = token_to_piece(token: outputToken)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            // If the entire set is now valid
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: { idx in
            // Attempt partial suffix parse
            idx != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix(idx)) + [0]) != nil
        }) {
            // In this scenario, a suffix of the array is valid. The code as written forcibly uses
            // the entire array again. This logic is fairly tricky, so ensure it does what you want.
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        return new_token_str
    }

    // MARK: - Token/ID checks
    
    /// Checks if token is "End of Generation" according to llama_vocab_is_eog.
    /// This is not standard in official llama.cpp (only EOS is standard).
    public override func llm_token_is_eog(token: ModelToken) -> Bool {
        return llama_vocab_is_eog(self.vocab, token)
    }
    
    /// Returns the "newline" token if your bridging library has it.
    /// Not standard in upstream llama.cpp. Possibly a custom bridging function.
    public override func llm_token_nl() -> ModelToken {
        return llama_vocab_nl(self.vocab)
    }
    
    /// Returns the "BOS" token from the bridging library if available.
    public override func llm_token_bos() -> ModelToken {
        return llama_vocab_bos(self.vocab)
    }
    
    /// Returns the "EOS" token from the bridging library if available.
    public override func llm_token_eos() -> ModelToken {
        return llama_vocab_eos(self.vocab)
    }
    
    // MARK: - LLMTokenize()
    
    /**
     Tokenizes input using a Jinja template approach by default. It inserts systemPrompt
     and user text, plus special tokens as needed.
     
     - Parameters:
       - input: The user string to tokenize.
       - chatTemplate: A custom template, if you want to override the default model template.
       - systemPrompt: A system prompt to insert.
       - add_bos: Override for whether the BOS token is included. If nil, use contextParams.add_bos_token.
       - parse_special: If true, tries to interpret  and similar special tokens.
     
     - Returns: Array of tokens for the entire combined prompt.
     */
    public override func LLMTokenize(_ input: String,
                                     chatTemplate: String? = nil,
                                     systemPrompt: String? = nil,
                                     add_bos: Bool? = nil,
                                     parse_special: Bool? = nil) -> [ModelToken]
    {
        let final_add_bos       = add_bos ?? self.contextParams.add_bos_token
        let final_parse_special = parse_special ?? self.contextParams.parse_special_tokens
        
        print("LLaMa tokenize: input: \"\(input)\", add_bos: \(final_add_bos), parse_special: \(final_parse_special)")
        
        let systemQuery = systemPrompt ?? ""
        let userQuery   = input
        
        // Build the messages array for chat template
        let messages: [[String: Any]] = [
            ["role": "system", "content": systemQuery],
            ["role": "user", "content": userQuery],
            /*
            [
                "role": "assistant",
                "content": NSNull(),
                "tool_calls": []
            ]
             */
        ]
        
        var query = input
        if let template = self.chatTemplate {
            // Use ChatTemplate to format the messages
            query = template.apply(
                messages: messages,
                tools: [:],
                addGenerationPrompt: true,
                extraContext: final_add_bos ? ["bos_token": "<s>"] : [:]
            )
        } else {
            print("LLaMa.chatTemplate is nil. Using input directly as query.")
        }
        
        if query.count == 0 {
            return []
        }
        print("Final query: \(query)")
        
        // Prepare space for tokens
        let utf8_count = query.utf8.count
        let n_tokens   = Int32(utf8_count) + (final_add_bos ? 1 : 0)
        var embeddings = [llama_token](repeating: llama_token(), count: Int(utf8_count))
        
        // llama_tokenize typically returns the number of tokens found (n).
        // The 6th parameter is whether we want to add the BOS token, the 7th is parse special tokens.
        let n: Int32 = llama_tokenize(self.vocab,
                                      query,
                                      Int32(utf8_count),
                                      &embeddings,
                                      n_tokens,
                                      final_add_bos,
                                      final_parse_special)
        if n <= 0 {
            return []
        }
        
        // Truncate the buffer to the actual number of tokens
        if Int(n) <= embeddings.count {
            embeddings.removeSubrange(Int(n) ..< embeddings.count)
        }
        
        // Optionally append EOS token if requested
        if self.contextParams.add_eos_token {
            embeddings.append(llm_token_eos())
        }
        
        // If the model has an encoder (like a seq2seq architecture?), you might need an encode call
        // followed by adding a decoder start token. This is not standard for normal LLaMA, so confirm
        // that llama_model_has_encoder is truly needed.
        if llama_model_has_encoder(model) {
            // This bridging function may differ from standard llama.cpp, which usually uses llama_eval
            if llama_encode(context,
                            llama_batch_get_one(&embeddings, Int32(embeddings.count))) != 0
            {
                print("failed to eval encode.")
                return []
            }
            
            // If the model has a separate decoder start token, fetch it. If -1, fallback to BOS.
            var decoder_start_token_id = llama_model_decoder_start_token(self.model)
            if decoder_start_token_id == -1 {
                decoder_start_token_id = llama_vocab_bos(self.vocab)
            }
            // Reset embeddings to [decoder_start_token]
            embeddings = [decoder_start_token_id]
        }
        
        return embeddings
    }
}
