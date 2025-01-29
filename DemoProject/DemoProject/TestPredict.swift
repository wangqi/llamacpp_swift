import Foundation
import llamacpp_swift

func testPredict(ai: AI, query: String) {
    print("\n=== Testing Traditional Predict ===")
    let output = try? ai.model?.Predict(query, { str, time in
        print(str, terminator: "")
        return false
    })
    print("\n=== Predict Test Complete ===\n")
}
