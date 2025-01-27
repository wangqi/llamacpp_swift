import Foundation
import llamacpp_swift

func testCallbackStream(ai: AI, query: String) {
    print("\n=== Testing Callback-based Streaming ===")
    if let model = ai.model {
        model.chatsStream(input: query) { partialResult in
            switch partialResult {
            case .success(let result):
                print(result.choices.joined(), terminator: "")
            case .failure(let error):
                print("ERROR: \(error)")
            }
        } completion: { final_string, processing_time, error in
            if let error = error {
                print("Error: \(error)")
            } else {
                print("\nFinal Answer in \(processing_time) ms")
            }
        }
    }
    print("\n=== Callback Stream Test Complete ===\n")
}
