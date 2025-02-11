//
//  main.swift
//  DemoProject
//
//  Created by guinmoon on 30.01.2024.
//

import Foundation
import llamacpp_swift
import Combine
import Jinja

func load_ai(modelPath: String) -> AI {
    let ai = AI(_modelPath: modelPath, _chatName: "chat")
    print("Load model: \(modelPath)")

    // Set up model parameters
    var params: ModelAndContextParams = .default
    //params.promptFormat = .Custom
    //params.custom_prompt_format = """
    //<|User|>{prompt}<|Assistant|>
    //"""
    params.promptFormat = .Default
    params.system_prompt = "You are a helpful assistant."
    params.custom_prompt_format = ""
    params.use_metal = true
    params.logitsAll = true
    print("Model Params: \(params)")

    // Initialize the model
    // Create a LLaMa object and return
    ai.initModel(ModelInference.LLama_gguf, contextParams: params)
    if ai.model == nil {
        print("Model load error.")
        exit(2)
    }

    // Configure sampling parameters
    ai.model?.sampleParams.temp = 0.8
    ai.model?.sampleParams.mirostat = 2
    ai.model?.sampleParams.mirostat_eta = 0.1
    ai.model?.sampleParams.mirostat_tau = 5.0
    ai.model?.sampleParams.dryAllowedLength = 2
    ai.model?.sampleParams.dryPenaltyLastN = 4096
    ai.model?.contextParams.context = 4096
    ai.model?.contextParams.add_bos_token = false
    ai.model?.contextParams.add_eos_token = false

    try? ai.loadModel_sync()
    return ai
}

// Initialize AI with model path
let modelPath = "/Users/wangqi/disk/projects/ai/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
//let modelPath = "/Users/wangqi/disk/projects/ai/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
//let modelPath = "/Users/wangqi/disk/projects/ai/models/qwen2.5-3b-instruct-q4_k_m.gguf"
// Get chat template
let template = """
{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{{'<｜Assistant｜>'}}
"""

// Select test mode
print("Select test mode:")
print("1. Traditional Predict")
print("2. Structured Concurrency Stream")
print("3. Callback Stream")
print("4. Combine Stream")
print("5. MainCPP")
print("6. llama-cpp-swift")
print("7. simple-chat")
print("8. libllama")


var ai: AI?
var model: Model?
var llama: LLama?
var simpleChat: SimpleChat?
var llamaContext: LlamaLlamaContext?

var testMode = 1  // Default to traditional predict
if let input = readLine(), let mode = Int(input) {
    testMode = mode
}

if testMode == 6 {
    model = try Model(modelPath: modelPath)
    llama = LLama(model: model!)
    await llama?.load_chat_template()
} else if testMode == 7 {
    simpleChat = SimpleChat(modelPath: modelPath)
    guard simpleChat != nil else {
        print("Failed to initialize SimpleChat")
        exit(1)
    }
} else if testMode == 8 {
    do {
        llamaContext = try LlamaLlamaContext.create_context(path: modelPath)
        print("\n=== LibLlama Model Info ===")
        print(llamaContext!.llama_model_info())
    } catch {
        print("Error initializing LibLlama context: \(error)")
        exit(1)
    }
} else {
    ai = load_ai(modelPath: modelPath)
}

// Main query loop
while true {
    print("\nEnter your query (type 'exit' to quit):")
    guard let query = readLine() else { continue }
    
    if query.lowercased() == "exit" {
        print("Goodbye!")
        break
    }
    
    switch testMode {
    case 1:
        testPredict(ai: ai!, query: query)
    case 2:
        Task {
            await testStructuredStream(ai: ai!, query: query)
        }
    case 3:
        testCallbackStream(ai: ai!, query: query)
    case 4:
        testCombineStream(ai: ai!, query: query)
    case 5:
        MainCPP.main(path: modelPath)
    case 6:
        try await testLlamacppswift(model: model!, llama: llama!, query: query)
    case 7:
        if let chat = simpleChat {
            testSimpleChat(chat: chat, query: query)
        } else {
            print("Error: SimpleChat not initialized")
        }
    case 8:
        if let context = llamaContext {
            var chatTemplate = ChatTemplate(source: template, bosToken: "", eosToken: "</s>")
            let prompt = chatTemplate.apply(messages: [
                ChatTemplate.createMessage(role: .user, content: query)
            ], tools: [:], addGenerationPrompt: true)
            await testLibllama(context: context, query: prompt)
        } else {
            print("Error: LibLlama context not initialized")
        }
    default:
        testPredict(ai: ai!, query: query)
    }
}
