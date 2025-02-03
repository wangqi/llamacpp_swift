import Foundation
import llamacpp_swift

func testLlamacppswift(model: Model, llama: LLama, query: String) async throws {
    let prompt = "<s><｜User｜>{{query}}<｜Assistant｜>".replacingOccurrences(of: "{{query}}", with: query)
    
    print("\n=== Testing Structured Concurrency Streaming ===")
    for try await token in await llama.infer(prompt: prompt, maxTokens: 1024) {
        print(token, terminator: "")
    }
}
