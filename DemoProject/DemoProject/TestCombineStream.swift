import Foundation
import llamacpp_swift
import Combine

var cancellables = Set<AnyCancellable>()

func testCombineStream(ai: AI, query: String) {
    print("\n=== Testing Combine-based Streaming ===")
    if let model = ai.model {
        model.chatsStream(input: query)
            .sink { completion in
                switch completion {
                case .finished:
                    print("\nStream finished")
                case .failure(let error):
                    print("\nStream failed with error: \(error)")
                }
            } receiveValue: { partialResult in
                let chunk = partialResult.choices.joined()
                print(chunk, terminator: "")
            }
            .store(in: &cancellables)
    }
    print("\n=== Combine Stream Test Complete ===\n")
}
