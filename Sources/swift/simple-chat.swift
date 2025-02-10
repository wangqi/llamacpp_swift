//
//  simple-chat.swift
//  A Swift implementation of the simple-chat example from llama.cpp
//

import Foundation
import llama

/// Simple chat implementation using llama.cpp
public class SimpleChat {
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var vocab: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private let nCtx: Int32
    private let ngl: Int32
    
    // In the C++ code, we keep track of all conversation messages
    // so that each user query builds on previous messages.
    private var messages: [llama_chat_message] = []
    
    // Buffer to hold the "formatted" conversation after applying the chat template.
    // We then feed only the newly appended text to `generate(...)` in each user turn.
    private var formatted: [Int8]
    
    // Tracks where the last user turn ended so we can extract only the new prompt text.
    private var prevLen: Int32 = 0

    /// Initialize SimpleChat with model path and optional parameters
    /// - Parameters:
    ///   - modelPath: Path to the model file (.gguf)
    ///   - contextSize: Size of the context window (default: 2048)
    ///   - nGpuLayers: Number of GPU layers to use (default: 99)
    public init?(modelPath: String, contextSize: Int32 = 2048, nGpuLayers: Int32 = 99) {
        self.nCtx = contextSize
        self.ngl = nGpuLayers
        
        // We will allocate a buffer used for applying the chat template (same as in C++)
        self.formatted = [Int8](repeating: 0, count: Int(contextSize))
        
        // only print errors (mirroring llama_log_set(...) in C++ to reduce verbosity)
        llama_log_set({ level, text, _ in
            if level.rawValue >= GGML_LOG_LEVEL_ERROR.rawValue {
                fputs(String(cString: text!), stderr)
            }
        }, nil)

        // Load all available GGML backends (CPU, GPU, etc.)
        ggml_backend_load_all()
        
        // Initialize the model with specified parameters
        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = ngl  // number of layers to offload to GPU

        guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
            print("Error: Unable to load model")
            return nil
        }
        self.model = loadedModel
        
        // Get the model's vocabulary
        guard let vocabPtr = llama_model_get_vocab(model) else {
            print("Error: Unable to get vocabulary")
            return nil
        }
        self.vocab = vocabPtr
        
        // Initialize the inference context
        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx   = UInt32(nCtx)
        ctxParams.n_batch = UInt32(nCtx)
        print("Context: \(ctxParams.n_ctx), \(ctxParams.n_batch) batches")
        
        guard let context = llama_init_from_model(model, ctxParams) else {
            print("Error: Failed to create llama context")
            return nil
        }
        self.ctx = context
        
        // Initialize the sampling chain
        let samplerParams = llama_sampler_chain_default_params()
        guard let sampler = llama_sampler_chain_init(samplerParams) else {
            print("Error: Failed to initialize sampler chain")
            return nil
        }
        self.sampler = sampler
        
        // Add sampling strategies to the chain
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(0.05, 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.6))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
    }
    
    /// Generate a response for the given prompt
    /// - Parameter prompt: The input prompt (new portion of the conversation)
    /// - Returns: Generated response string
    public func generate(prompt: String) -> String {
        var response = ""
        
        guard let ctx = self.ctx, let vocab = self.vocab, let sampler = self.sampler else {
            print("Error initializing llama context or sampler first")
            return response
        }
        
        // isFirst check (same as in C++: llama_get_kv_cache_used_cells(ctx) == 0)
        let isFirst = llama_get_kv_cache_used_cells(ctx) == 0
        
        // Tokenize the prompt

        let nPromptTokens = -llama_tokenize(vocab, prompt, Int32(prompt.utf8.count), nil, 0, isFirst, true)
        if nPromptTokens < 0 {
            print("Error: Failed to compute token count")
            return response
        }
        var promptTokens = [llama_token](repeating: 0, count: Int(nPromptTokens))
        
        let tokenizedCount = llama_tokenize(vocab, prompt, Int32(prompt.utf8.count),
                                           &promptTokens, Int32(promptTokens.count),
                                           isFirst, true)
        if tokenizedCount < 0 {
            print("Error: Failed to tokenize prompt")
            return response
        }
        print("Input query : \(prompt)")
        print("Input tokens: \(promptTokens)")

        // Prepare the batch for the prompt
        var batch = llama_batch_get_one(&promptTokens, Int32(promptTokens.count))
        
        // Repeatedly decode and generate tokens until EOG
        while true {
            let nCtxUsed = llama_get_kv_cache_used_cells(ctx)
            if nCtxUsed + batch.n_tokens > self.nCtx {
                print("Error: context size exceeded")
                break
            }
            
            // Evaluate model on the batch of tokens
            if llama_decode(ctx, batch) != 0 {
                print("Error: Failed to decode")
                break
            }
            
            // Sample next token
            var newTokenId = llama_sampler_sample(sampler, ctx, -1)
            
            // Check for end of generation
            if llama_vocab_is_eog(vocab, newTokenId) {
                break
            }
            
            // Convert token to a text piece
            var buf = [Int8](repeating: 0, count: 256)
            let n = llama_token_to_piece(vocab, newTokenId, &buf, Int32(buf.count), 0, true)
            if n < 0 {
                print("Error: Failed to convert token to piece")
                break
            }
            
            // We rely on the function to have appended a null terminator
            let piece = String(cString: buf)
            print(piece, terminator: "")  // Print incrementally (like the C++ version)
            fflush(stdout)
            
            response += piece
            
            // Prepare the next batch with the single new token
            batch = llama_batch_get_one(&newTokenId, 1)
        }
        
        return response
    }
    
    /// NOTE: This original function was single-turn,
    /// but the C++ code is multi-turn (loop until user hits Enter).
    /// We keep the function and comments but add a new multi-turn version below.
    /// This function does a single round of user -> assistant.
    public func startChat(userInput: String) {
        var messages = [llama_chat_message]()
        var formatted = [Int8](repeating: 0, count: Int(nCtx))
        
        // Get chat template
        let template = """
        {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{{'<｜Assistant｜>'}}
        """
        
        // Add user message and format
        let message = llama_chat_message(role: strdup("user"), content: strdup(userInput))
        messages.append(message)
        
        let newLen = llama_chat_apply_template(template, &messages, messages.count, true, &formatted, Int32(formatted.count))
        if newLen > formatted.count {
            print("Error: Formatted message too long")
            return
        }
        if newLen < 0 {
            print("Error: Failed to apply the chat template")
            return
        }
        
        // Because we are applying the entire conversation at once, just convert the entire buffer
        let prompt = String(cString: formatted)
        let response = generate(prompt: prompt)
        
        // Add assistant message and apply chat template
        let assistantMessage = llama_chat_message(role: strdup("assistant"), content: strdup(response))
        messages.append(assistantMessage)
        
        let _ = llama_chat_apply_template(template, &messages, messages.count, false, nil, 0)
        // Not storing prevLen here because this is a single-turn demonstration.
    }
    
    /// This new function replicates the multi-turn logic from simple_chat.cpp
    /// Reads user inputs in a loop until the user presses Enter on an empty line.
    public func startInteractiveChat() {
        
        // The same chat template used in the C++ code
        // We only define it here once; we'll reuse it for each user message.
        let tmpl = """
        {% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{{'<｜Assistant｜>'}}
        """
        
        print("Welcome to SimpleChat (Swift). Type your prompt and press Enter.\nEmpty line to exit.\n")
        
        while true {
            // Prompt user for input
            print("> ", terminator: "")
            guard let userInput = readLine() else {
                // EOF or read error
                break
            }
            
            // Break on empty line
            if userInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                print("Exiting chat loop.")
                break
            }
            
            // Print out the chat template for debugging (like the C++ version)
            // This can be commented out if you don't want to see the template every turn.
            print("\nChat Template:\n\(tmpl)\n")
            
            // Push the user message into the conversation
            let userMessage = llama_chat_message(role: strdup("user"), content: strdup(userInput))
            messages.append(userMessage)
            
            // Apply the chat template with partial = true (like C++ does)
            var newLen = llama_chat_apply_template(tmpl, &messages, messages.count,
                                                   true, &formatted, Int32(formatted.count))
            
            // If the template output was bigger than our current buffer, resize and retry
            if newLen > formatted.count {
                formatted = [Int8](repeating: 0, count: Int(newLen))
                newLen = llama_chat_apply_template(tmpl, &messages, messages.count,
                                                   true, &formatted, Int32(formatted.count))
            }
            
            if newLen < 0 {
                print("Failed to apply the chat template.")
                return
            }
            
            // We only want the newly appended text from [prevLen ..< newLen].
            // Insert a null terminator to be safe.
            if newLen < formatted.count {
                formatted[Int(newLen)] = 0
            }
            
            // Construct a Swift string from the substring (pointer) of `formatted`.
            // This mirrors C++: std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len)
            let prompt: String = formatted.withUnsafeBufferPointer { ptr in
                // Start address
                let start = ptr.baseAddress!.advanced(by: Int(prevLen))
                return String(cString: start)
            }
            
            print("\nprompt: \(prompt)\n")
            
            // Generate the assistant reply for just the new prompt portion
            print("Assistant:", terminator: " ")
            let response = generate(prompt: prompt)
            print("\n") // finish line
            
            // Add the assistant's message to the conversation
            let assistantMsg = llama_chat_message(role: strdup("assistant"), content: strdup(response))
            messages.append(assistantMsg)
            
            // Finally, recalculate prevLen with partial=false
            let calcLen = llama_chat_apply_template(tmpl, &messages, messages.count,
                                                    false, nil, 0)
            if calcLen < 0 {
                print("Failed to apply the chat template (post-generation).")
                return
            }
            
            // Save this for next iteration, so next time we only generate new text
            prevLen = calcLen
        }
        
        // After user exits, free the message contents
        // (mirroring the final cleanup in the C++ code)
        /*
        for msg in messages {
            if let rolePtr = msg.role {
                free(rolePtr)
            }
            if let contentPtr = msg.content {
                free(contentPtr)
            }
        }
         */
        messages.removeAll()
    }
    
    deinit {
        // Free the sampler
        if let sampler = sampler {
            llama_sampler_free(sampler)
        }
        // Free the context
        if let ctx = ctx {
            llama_free(ctx)
        }
        // Free the model (the correct call is llama_model_free(...))
        if let model = model {
            llama_model_free(model)
        }
    }
}

// Example of how a Swift "main.swift" might call this class:
// (In real usage, you'd put this in a separate main.swift or top-level scope)
/*
import Foundation

// A simple argument check, e.g. swift run ... -m /path/to/model.gguf
var modelPath: String = ""
var nCtx: Int32 = 2048
var nGpuLayers: Int32 = 99

var i = 1
while i < CommandLine.arguments.count {
    let arg = CommandLine.arguments[i]
    if arg == "-m" && i+1 < CommandLine.arguments.count {
        i += 1
        modelPath = CommandLine.arguments[i]
    } else if arg == "-c" && i+1 < CommandLine.arguments.count {
        i += 1
        nCtx = Int32(CommandLine.arguments[i]) ?? 2048
    } else if arg == "-ngl" && i+1 < CommandLine.arguments.count {
        i += 1
        nGpuLayers = Int32(CommandLine.arguments[i]) ?? 99
    } else {
        // Show usage, etc.
        print("Usage: simple_chat -m model.gguf [-c context_size] [-ngl n_gpu_layers]")
        exit(1)
    }
    i += 1
}

guard !modelPath.isEmpty else {
    print("No model path specified.")
    exit(1)
}

// Create and run chat
if let chat = SimpleChat(modelPath: modelPath, contextSize: nCtx, nGpuLayers: nGpuLayers) {
    chat.startInteractiveChat()
} else {
    print("Could not initialize chat.")
}
*/
