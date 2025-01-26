//
//  main.swift
//  DemoProject
//
//  Created by guinmoon on 30.01.2024.
//

import Foundation
import llmfarm_core

let maxOutputLength = 256
var total_output = 0

func mainCallback(_ str: String, _ time: Double) -> Bool {
    print("\(str)",terminator: "")
    total_output += str.count
    if(total_output>maxOutputLength){
        return true
    }
    return false
}


//load model
let modelPath = "/Users/wangqi/disk/projects/ai/models/Dolphin3.0-Qwen2.5-3b-Q4_K_M.gguf"
let ai = AI(_modelPath: modelPath,_chatName: "chat")
print("Load model: \(modelPath)")
var params: ModelAndContextParams = .default

//set custom prompt format
params.promptFormat = .Custom
params.custom_prompt_format = """
SYSTEM: You are a helpful, respectful and honest assistant.
USER: {prompt}
ASSISTANT:
"""
var input_text = "State the meaning of life"
params.use_metal = true
print("Model Params: \(params)")

//Uncomment this line to add lora adapter
//params.lora_adapters.append(("lora-open-llama-3b-v2-q8_0-shakespeare-LATEST.bin",1.0 ))

//_ = try? ai.loadModel_sync(ModelInference.LLama_gguf,contextParams: params)
ai.initModel(ModelInference.LLama_gguf, contextParams: params)
if ai.model == nil{
    print( "Model load eror.")
    exit(2)
}
// to use other inference like RWKV set ModelInference.RWKV
// to use old ggjt_v3 llama models use ModelInference.LLama_bin

// Set mirostat_v2 sampling method
ai.model?.sampleParams.mirostat = 2
ai.model?.sampleParams.mirostat_eta = 0.1
ai.model?.sampleParams.mirostat_tau = 5.0

try ai.loadModel_sync()
//eval with callback
let output = try? ai.model?.predict(input_text, mainCallback)

