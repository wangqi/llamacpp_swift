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

// Initialize AI with model path
let modelPath = "/Users/wangqi/disk/projects/ai/models/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"
let ai = AI(_modelPath: modelPath, _chatName: "chat")
print("Load model: \(modelPath)")

// Set up model parameters
var params: ModelAndContextParams = .default
//params.promptFormat = .Custom
//params.custom_prompt_format = """
//<|User|>{prompt}<|Assistant|>
//"""
params.promptFormat = .None
params.custom_prompt_format = "{prompt}"
params.use_metal = true
print("Model Params: \(params)")

// Initialize the model
ai.initModel(ModelInference.LLama_gguf, contextParams: params)
if ai.model == nil {
    print("Model load error.")
    exit(2)
}

// Configure sampling parameters
ai.model?.sampleParams.mirostat = 2
ai.model?.sampleParams.mirostat_eta = 0.1
ai.model?.sampleParams.mirostat_tau = 5.0

try? ai.loadModel_sync()

// load chat template
var chatTemplate = ai.loadChatTempate()
if let chatTemplate = chatTemplate {
    print("chat template: \(chatTemplate)")
} else {
    print("no chat template")
}

// Test query
var context: [String: Any] = [
    "messages": [
            ["role": "system", "content": "You are a helpful assistant."],
            ["role": "user", "content": "What is the meaning of life?"
        ],
        [
            "role": "assistant", "content": nil,
            "tool_calls": [
//                ["type": "function", "function": ["name": "get_weather", "arguments": "{\"city\": \"Paris\"}"]]
            ]
        ],
//        ["role": "tool", "content": "Sunny, 22°C"]
    ],
        "add_generation_prompt": true,
        "bos_token": "<s>"
]
let query = try Template(chatTemplate!).render(context)
print("Final query: \(query)")

// Select test mode
print("Select test mode:")
print("1. Traditional Predict")
print("2. Structured Concurrency Stream")
print("3. Callback Stream")
print("4. Combine Stream")

if let input = readLine(), let testMode = Int(input) {
    switch testMode {
    case 1:
        testPredict(ai: ai, query: query)
    case 2:
        Task {
            await testStructuredStream(ai: ai, query: query)
        }
        // Add a small delay to see the results
        Thread.sleep(forTimeInterval: 2)
    case 3:
        testCallbackStream(ai: ai, query: query)
        // Add a small delay to see the results
        Thread.sleep(forTimeInterval: 2)
    case 4:
        testCombineStream(ai: ai, query: query)
        // Add a small delay to see the results
        Thread.sleep(forTimeInterval: 2)
    default:
        print("Invalid test mode selected")
    }
}
