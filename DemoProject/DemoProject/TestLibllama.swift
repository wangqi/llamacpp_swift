import Foundation
import llamacpp_swift

func testLibllama(context: LlamaLlamaContext, query: String) async {
    print("\n=== Testing LibLlama ===")
    print("\nPrompt: \(query)")
    
    // Process the prompt
    context.completion_init(text: query)
    var result = ""
    while !context.is_done {
        let response = context.completion_loop()
        result += response
    }
    print(result)
    print("[END]")
    context.clear()
    
    print("\n=== LibLlama Test Complete ===\n")
}
