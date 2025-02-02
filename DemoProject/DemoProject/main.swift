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
//let modelPath = "/Users/wangqi/disk/projects/ai/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
//let modelPath = "/Users/wangqi/disk/projects/ai/models/Dolphin3.0-Qwen2.5-3b-Q4_K_M.gguf"
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
ai.model?.sampleParams.mirostat = 2
ai.model?.sampleParams.mirostat_eta = 0.1
ai.model?.sampleParams.mirostat_tau = 5.0
ai.model?.contextParams.context = 4096

try? ai.loadModel_sync()

// Select test mode
print("Select test mode:")
print("1. Traditional Predict")
print("2. Structured Concurrency Stream")
print("3. Callback Stream")
print("4. Combine Stream")
print("5. MainCPP")

var testMode = 1  // Default to traditional predict
if let input = readLine(), let mode = Int(input) {
    testMode = mode
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
        testPredict(ai: ai, query: query)
    case 2:
        Task {
            await testStructuredStream(ai: ai, query: query)
        }
        Thread.sleep(forTimeInterval: 2)
    case 3:
        testCallbackStream(ai: ai, query: query)
        Thread.sleep(forTimeInterval: 2)
    case 4:
        testCombineStream(ai: ai, query: query)
        Thread.sleep(forTimeInterval: 2)
    case 5:
        MainCPP.main(path: modelPath)
    default:
        testPredict(ai: ai, query: query)
    }
}
