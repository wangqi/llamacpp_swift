import Foundation
import llamacpp_swift

func testLlamacppswift(model: Model, llama: LLama, query: String) async throws {
    
    print("\n=== Testing Structured Concurrency Streaming ===")
    for try await token in await llama.infer(prompt: query, maxTokens: 1024) {
        print(token, terminator: "")
    }
}
