# LLMFarm_core.swift Fork
This is a modified fork of [LLMFarm_core.swift](https://github.com/guinmoon/llmfarm_core.swift) with a key architectural change: it uses llama.cpp as an external package dependency instead of including it in the source directory. This decoupling makes the codebase cleaner and easier to maintain.

The library allows you to work with large language models (LLM) on Apple platforms. It is based on [ggml](https://github.com/ggerganov/ggml) and [llama.cpp](https://github.com/ggerganov/llama.cpp) by [Georgi Gerganov](https://github.com/ggerganov).

# Features

- [x] MacOS (13+)
- [x] iOS (16+)
- [x] Various inferences
- [x] Various sampling methods
- [x] Metal ([dont work](https://github.com/ggerganov/llama.cpp/issues/2407#issuecomment-1699544808) on intel Mac)
- [x] Model setting templates
- [ ] LoRA adapters support ([read more](https://github.com/guinmoon/LLMFarm/blob/main/lora.md))
- [ ] LoRA train support
- [ ] LoRA export as model support
- [ ] Restore context state (now only chat history) 

# Inferences

See full list [here](https://github.com/ggerganov/llama.cpp).
  
# Sampling methods
- [x] Temperature (temp, tok-k, top-p)
- [x] [Tail Free Sampling (TFS)](https://www.trentonbricken.com/Tail-Free-Sampling/)
- [x] [Locally Typical Sampling](https://arxiv.org/abs/2202.00666)
- [x] [Mirostat](https://arxiv.org/abs/2007.14966)
- [x] Greedy
- [x] Grammar 
- [ ] Classifier-Free Guidance


# Installation

```bash
git clone https://github.com/wangqi/llamacpp_swift/
```

## Swift Package Manager

Add `llamacpp_swift` to your project using Xcode (File > Add Packages...) or by adding it to your project's `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/wangqi/llamacpp_swift", branch: "main")
]
```

The package automatically manages the llama.cpp dependency, so you don't need to include it separately.

## Build and Debug 

To Debug `llamacpp_swift` package, do not forget to comment `.unsafeFlags(["-Ofast"])` in `Package.swift`.
Don't forget that the debug version is slower than the release version.

# Usage

## [See examples in the Demo Project](/DemoProject)

## Streaming Methods

The library provides three different streaming methods to suit various use cases:

### 1. Closure-based Streaming

Traditional callback-based approach using closures:

```swift
llm.chatsStream(
    input: "Your prompt here",
    system_prompt: "Optional system prompt",
    onPartialResult: { result in
        switch result {
        case .success(let modelResult):
            // Handle each chunk of the response
            print(modelResult.choices[0])
        case .failure(let error):
            // Handle error
            print(error)
        }
    },
    completion: { finalOutput, processingTime, error in
        if let error = error {
            // Handle error
            print(error)
            return
        }
        // Handle completion
        print("Final output: \(finalOutput)")
        print("Processing time: \(processingTime)ms")
    }
)
```

### 2. AsyncSequence-based Streaming

Modern async/await pattern using AsyncSequence:

```swift
do {
    for try await result in llm.chatsStream(
        input: "Your prompt here",
        system_prompt: "Optional system prompt"
    ) {
        // Handle each chunk of the response
        print(result.choices[0])
    }
} catch {
    // Handle error
    print(error)
}
```

### 3. Combine-based Streaming

Integration with Apple's Combine framework:

```swift
var cancellables = Set<AnyCancellable>()

llm.chatsStream(
    input: "Your prompt here",
    system_prompt: "Optional system prompt"
)
.sink { completion in
    switch completion {
    case .finished:
        print("Stream finished")
    case .failure(let error):
        print("Stream failed with error: \(error)")
    }
} receiveValue: { partialResult in
    // partialResult is your ModelResult
    let chunk = partialResult.choices.joined()
    print("Partial chunk: \(chunk)")
}
.store(in: &cancellables)
```

Choose the streaming method that best fits your application's architecture:
- Use closure-based streaming for traditional callback patterns
- Use AsyncSequence-based streaming for modern async/await code
- Use Combine-based streaming for reactive programming with Combine framework
