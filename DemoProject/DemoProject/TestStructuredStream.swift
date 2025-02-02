import Foundation
import llamacpp_swift

func testStructuredStream(ai: AI, query: String) async {
    print("\n=== Testing Structured Concurrency Streaming ===")
    if let model = ai.model {
        do {
            for try await result in model.chatsStream(input: query) {
                print(result.choices, terminator: "")
            }
        } catch {
            print("Error: \(error)")
        }
    }
    print("\n=== Structured Stream Test Complete ===\n")
}
