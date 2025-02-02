//
//  MainLoopState.swift
//  DemoProject
//
//  Created by Qi Wang on 2025-02-02.
//


import Foundation
import llama
import llamacpp_swift
import llamacpp_swift_cpp


/// Data structure to hold relevant state from your C++ code
/// (like n_remain, n_past, is_interacting, etc.)
public struct MainLoopState {
    // Common model-related context
    public var ctx: OpaquePointer?            // the llama_context*
    public var samplerCtx: SpmSamplerContext? // your chain sampler
    public var vocab: OpaquePointer?          // optional: the llama_vocab*

    // Key loop variables
    public var nRemain: Int                   // n_remain in C++ code
    public var isAntiprompt: Bool             // is_antiprompt
    public var isInteracting: Bool            // is_interacting
    public var needInsertEot: Bool            // need_insert_eot
    public var display: Bool                  // controlling whether to display tokens

    // These track tokens in the KV context
    public var nPast: Int                     // n_past
    public var gaN: Int                       // ga_n
    public var gaW: Int                       // ga_w
    public var gaI: Int                       // ga_i

    // The prompt tokens or user input tokens we have not processed yet
    public var embd: [llama_token]            // C++ "embd"
    public var embdInp: [llama_token]         // C++ "embd_inp"
    public var nConsumed: Int                 // n_consumed
    public var pathSession: String            // path to session file if any

    // session tokens, for partial reuse
    public var sessionTokens: [llama_token]
    public var nSessionConsumed: Int

    // The user I/O results
    public var outputTokens: [llama_token]    // store output tokens for logging
    public var inputTokens: [llama_token]     // store input tokens for logging
    public var outputBuffer: String           // analogous to output_ss in C++
    public var assistantBuffer: String        // storing assistant message (assistant_ss)

    // Some config booleans
    public var ctxShift: Bool                 // params.ctx_shift
    public var interactive: Bool              // params.interactive
    public var nPredict: Int                  // params.n_predict
    public var nCtx: Int                      // n_ctx
    public var nBatch: Int                    // params.n_batch
    public var nKeep: Int                     // params.n_keep
    public var promptCacheReadOnly: Bool      // params.prompt_cache_ro
    public var promptCacheAll: Bool           // params.prompt_cache_all
    public var needToSaveSession: Bool
    public var conversationMode: Bool         // params.conversation_mode
    public var enableChatTemplate: Bool       // params.enable_chat_template
    public var inputPrefixBos: Bool           // params.input_prefix_bos
    public var multilineInput: Bool           // params.multiline_input
    public var escape: Bool                   // params.escape
    public var useColor: Bool                 // params.use_color, controlling color

    // "antiprompt" strings to check, plus tokenized forms
    public var antiprompts: [String]          // params.antiprompt
    public var antipromptIds: [[llama_token]] // tokenized variants
}

/// This function replicates the "while" loop from your main.cpp, but in Swift.
/// It references bridging calls from GptSpm.swift for sampling, plus some new bridging we appended.
public func runMainLoop(state: inout MainLoopState) {
    let model: OpaquePointer = llama_get_model(state.ctx)
    let vocab: OpaquePointer = llama_model_get_vocab(model)
    
    while ((state.nRemain != 0 && !state.isAntiprompt) || state.interactive) {
        // 1) Evaluate or decode `embd` if not empty
        if !state.embd.isEmpty {
            // ensure we don't exceed the context
            let maxEmbdSize = state.nCtx - 4
            if state.embd.count > maxEmbdSize {
                let skipped = state.embd.count - maxEmbdSize
                state.embd.removeLast(skipped)
                print("<<input too long: skipped \(skipped) tokens>>")
            }

            // handle context extension or shifting
            if state.gaN == 1 {
                // standard infinite text generation logic
                // if we run out of context:
                if state.nPast + state.embd.count >= state.nCtx {
                    if !state.ctxShift {
                        print("context full and context shift is disabled => stopping")
                        break
                    }
                    if state.nPredict == -2 {
                        print("context full and n_predict == -2 => stopping")
                        break
                    }
                    let nLeft = state.nPast - state.nKeep
                    let nDiscard = nLeft/2
                    print("context full, swapping: n_past = \(state.nPast), n_keep = \(state.nKeep), n_discard = \(nDiscard)")
                    // bridging calls to remove tokens from the KV
                    spm_llama_kv_cache_seq_rm(ctx: state.ctx, seqId: 0, p0: state.nKeep, p1: state.nKeep + nDiscard)
                    spm_llama_kv_cache_seq_add(ctx: state.ctx, seqId: 0, p0: state.nKeep + nDiscard, p1: state.nPast, delta: -nDiscard)

                    state.nPast -= nDiscard
                    print("after swap: n_past = \(state.nPast)")
                    state.pathSession = ""  // clear session path
                }
            } else {
                // group-attention logic
                while state.nPast >= state.gaI + state.gaW {
                    let ib = (state.gaN * state.gaI) / state.gaW
                    let bd = (state.gaW/state.gaN) * (state.gaN - 1)
                    let dd = (state.gaW/state.gaN) - ib*bd - state.gaW
                    print("context extension shifting stuff ...")
                    // replicate the llama_kv_cache_seq_add / seq_div calls
                    spm_llama_kv_cache_seq_add(ctx: state.ctx, seqId: 0, p0: state.gaI, p1: state.nPast, delta: ib*bd)
                    spm_llama_kv_cache_seq_div(ctx: state.ctx, p0: state.gaI + ib*bd, p1: state.gaI + ib*bd + state.gaW, divisor: state.gaN)
                    spm_llama_kv_cache_seq_add(ctx: state.ctx, seqId: 0, p0: state.gaI + ib*bd + state.gaW, p1: state.nPast + ib*bd, delta: dd)
                    state.nPast -= bd
                    state.gaI += state.gaW/state.gaN
                }
            }

            // try to reuse a matching prefix from session
            if state.nSessionConsumed < state.sessionTokens.count {
                var i = 0
                while i < state.embd.count {
                    if state.embd[i] != state.sessionTokens[state.nSessionConsumed] {
                        // mismatch
                        state.sessionTokens.removeSubrange(state.nSessionConsumed..<state.sessionTokens.count)
                        break
                    }
                    state.nPast += 1
                    state.nSessionConsumed += 1
                    i += 1
                    if state.nSessionConsumed >= state.sessionTokens.count {
                        i += 1
                        break
                    }
                }
                if i > 0 {
                    state.embd.removeFirst(i)
                }
            }

            // Evaluate the tokens in batches
            var i = 0
            while i < state.embd.count {
                let chunkSize = min(state.nBatch, state.embd.count - i)
                let chunk = Array(state.embd[i..<(i+chunkSize)])
                // bridging call: spm_llama_decode
                let rc = spm_llama_decode(ctx: state.ctx, tokens: chunk)
                if rc != 0 {
                    print("failed to eval => rc = \(rc)")
                    break
                }
                state.nPast += chunkSize
                i += chunkSize
            }

            // Possibly save these tokens to session
            if !state.embd.isEmpty && !state.pathSession.isEmpty {
                state.sessionTokens.append(contentsOf: state.embd)
                state.nSessionConsumed = state.sessionTokens.count
            }

            // done processing embd for now
            state.embd.removeAll()
        }

        // 2) If we have no queued input left, sample new token from the model
        if state.embdInp.count <= state.nConsumed && !state.isInteracting {
            // if we have never saved session yet, do it now
            if !state.pathSession.isEmpty && state.needToSaveSession && !state.promptCacheReadOnly {
                state.needToSaveSession = false
                spm_llama_state_save_file(ctx: state.ctx,
                                          path: state.pathSession,
                                          tokens: state.sessionTokens)
                print("saved session to \(state.pathSession)")
            }

            // sample token
            let newToken = spm_llama_sampling_sample(ctxSampling: state.samplerCtx,
                                                     ctxMain: state.ctx,
                                                     idx: -1)
            spm_llama_sampling_accept(ctxSampling: state.samplerCtx,
                                      ctxMain: state.ctx,
                                      token: newToken,
                                      applyGrammar: true)

            state.embd.append(newToken)
            // echo to console
            // decrement remaining tokens
            state.nRemain -= 1
            print("n_remain: \(state.nRemain)")
        } else {
            // we still have leftover prompt or user input
            while state.embdInp.count > state.nConsumed {
                let tk = state.embdInp[state.nConsumed]
                state.embd.append(tk)
                // push the prompt token in the sampler
                spm_llama_sampling_accept(ctxSampling: state.samplerCtx,
                                          ctxMain: state.ctx,
                                          token: tk,
                                          applyGrammar: false)
                state.nConsumed += 1
                if state.embd.count >= state.nBatch {
                    break
                }
            }
        }

        // 3) Display newly added tokens in `embd`
        // (Equivalent to "if (input_echo && display)" etc.)
        // We'll assume we want to always display for simplicity:
        for tk in state.embd {
            let tokenStr = spm_llama_token_to_piece(model: model, vocab: vocab, token: tk, special: false)
            // Print
            print(tokenStr, terminator: "")
            // Distinguish user vs model tokens?  Original code uses "if embd.size()>1"
            state.outputTokens.append(tk)
            state.outputBuffer += tokenStr
        }
        // flush line if needed
        if !state.embd.isEmpty {
            print("")
        }

        // 4) Check if we have no more leftover input
        if state.embdInp.count <= state.nConsumed {

            // 1) Check for antiprompt in, say, the last 32 tokens
            //    We'll retrieve them as a single string in reverse order.
            let last32 = spm_llama_sampling_last_tokens(ctxSampling: state.samplerCtx, ctx: state.ctx, n: 32)
            // see if any antiprompt substring appears
            for ap in state.antiprompts {
                if last32.contains(ap) {
                    print("found antiprompt: \(ap)")
                    // optionally set state.isInteracting, isAntiprompt, etc.
                    state.isAntiprompt = true
                    if state.interactive {
                        state.isInteracting = true
                    }
                    break
                }
            }

            // 2) Attempt to check EOG by looking at the last 1 token's text
            //    (In the original code, you do spm_llama_vocab_is_eog with a token,
            //     but now we do a purely text-based guess.)
            let lastOneStr = spm_llama_sampling_last_tokens(
                ctxSampling: state.samplerCtx,
                ctx: state.ctx,
                n: 1
            )
            if lastOneStr == "<EOG>" {
                // or some known representation if your EOG token has a unique text
                print("found EOG token (by text)")
                if state.interactive {
                    if state.enableChatTemplate {
                        // ...
                    }
                    state.isInteracting = true
                    print("")
                }
            }

            // 3) If conversation mode, append the last token's text to assistant buffer
            if state.conversationMode {
                state.assistantBuffer += lastOneStr
            }

            // 4) If we are in interactive mode and is_interacting, wait for user input
            if state.nPast > 0 && state.isInteracting {
                print("waiting for user input")
                // ... read from console, tokenize, push to embdInp ...
                // then reset isInteracting when done
                state.isInteracting = false
            }
        }

        // 5) End-of-generation check if not interactive
        if !state.embd.isEmpty &&
            spm_llama_vocab_is_eog(vocab: state.vocab, token: state.embd.last!) &&
            !state.interactive {
            print(" [end of text]")
            break
        }

        // 6) If in interactive mode and we've used up n_remain
        if state.interactive && state.nRemain <= 0 && state.nPredict >= 0 {
            state.nRemain = state.nPredict
            state.isInteracting = true
        }
    }
    // end of the while loop
}

struct MainCPP {
    
    static func main(path: String) {
        var context_params = llama_context_default_params()
        var model_params   = llama_model_default_params()
        
        var spmParams = SpmSamplingParams()
        // Initialize the llama backend
        llama_backend_init()
        let model = llama_model_load_from_file(path, model_params)
        let vocab = llama_model_get_vocab(model) // self.model is the same pointer if not nil
        let ctx_sampling = init_sampling(model: model, params: spmParams)
        let context = llama_init_from_model(model, context_params)
        //logits init
        var inputs = [llama_vocab_bos(vocab), llama_vocab_eos(vocab)]
        llama_decode(context, llama_batch_get_one(&inputs, Int32(inputs.count)))
        
        var state = MainLoopState(
            ctx: context,
            samplerCtx: ctx_sampling,
            vocab: vocab,
            nRemain: 128,
            isAntiprompt: false,
            isInteracting: false,
            needInsertEot: false,
            display: true,

            nPast: 0,
            gaN: 1,
            gaW: 512,
            gaI: 0,

            embd: [],
            embdInp: [],
            nConsumed: 0,
            pathSession: "",

            sessionTokens: [],
            nSessionConsumed: 0,
            outputTokens: [],
            inputTokens: [],
            outputBuffer: "",
            assistantBuffer: "",

            ctxShift: true,
            interactive: true,
            nPredict: 128,
            nCtx: 2048,
            nBatch: 32,
            nKeep: 0,
            promptCacheReadOnly: false,
            promptCacheAll: false,
            needToSaveSession: true,
            conversationMode: false,
            enableChatTemplate: false,
            inputPrefixBos: false,
            multilineInput: false,
            escape: true,
            useColor: false,

            antiprompts: ["STOP"],
            antipromptIds: [] // you might fill in with tokenized forms
        )
        llama_batch_init(5, 0, 1)

        // Example: fill embdInp with some initial tokens

        runMainLoop(state: &state)

        print("\nFinal output buffer:\n\(state.outputBuffer)")
    }
}
