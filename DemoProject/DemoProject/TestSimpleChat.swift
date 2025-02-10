import Foundation
import llamacpp_swift

func testSimpleChat(chat: SimpleChat, query: String) {
    print("\n=== Testing Simple Chat ===")

    // Optionally start interactive chat mode
    chat.startChat(userInput: query)
//    chat.startInteractiveChat()
}
