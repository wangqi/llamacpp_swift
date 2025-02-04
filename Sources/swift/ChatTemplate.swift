import Foundation
import Jinja

/// Enum representing the message roles (excluding tool messages).
public enum MessageRole: String {
    case user = "user"
    case system = "system"
    case assistant = "assistant"
}

/// A structure representing a single tool call.
public struct ToolCall {
    public var id: String?
    public var type: String
    public var functionName: String
    public var arguments: Any?  // This can be a String or a Dictionary
    
    public init(id: String? = nil, type: String, functionName: String, arguments: Any? = nil) {
        self.id = id
        self.type = type
        self.functionName = functionName
        self.arguments = arguments
    }
    
    /// Converts the ToolCall instance into a dictionary.
    public func toDictionary() -> [String: Any] {
        var functionDict: [String: Any] = ["name": functionName]
        if let arguments = arguments {
            functionDict["arguments"] = arguments
        }
        var dict: [String: Any] = [
            "type": type,
            "function": functionDict
        ]
        if let id = id {
            dict["id"] = id
        }
        return dict
    }
    
    // CHANGED: Added a failable initializer to convert a dictionary back to a ToolCall.
    public init?(dictionary: [String: Any]) {
        guard let type = dictionary["type"] as? String,
              let functionDict = dictionary["function"] as? [String: Any],
              let name = functionDict["name"] as? String
        else {
            return nil
        }
        self.type = type
        self.functionName = name
        self.arguments = functionDict["arguments"]
        self.id = dictionary["id"] as? String
    }
}

/// Enum representing tool message types.
public enum ToolMessageType {
    /// Represents a tool call message.
    /// When using this case, the role is set to "assistant", content is set to "<none>",
    /// and the provided tool calls are included.
    case call(toolCalls: [ToolCall])
    /// Represents a tool output message.
    /// When using this case, the role is set to "tool" and the provided content is used.
    case output(content: String)
}

/// A Swift approximation of your C++ minja::chat_template.
/// The idea: store the template source, parse it, track booleans like supports_tools_,
/// and supply an `apply(...)` method that "renders".
public class ChatTemplate {
    // MARK: - Corresponding Fields

    public var supportsTools: Bool
    public var requiresObjectArguments: Bool
    public var requiresTypedContent: Bool
    public var supportsSystemRole: Bool
    public var supportsParallelToolCalls: Bool

    public var source: String
    public var bosToken: String
    public var eosToken: String

    /// The Swift analog of `template_root_ = minja::Parser::parse(...)`.
    /// In real usage, you'd parse `source` into an AST or structured template.
    /// We'll store a placeholder here.
    private var jinjaTemplate: Template?

    // MARK: - Initialization

    /// Swift analog to chat_template(...) constructor in C++.
    /// It tries to detect whether the template uses "tools", "system", typed content, etc.
    public init(source: String,
                bosToken: String,
                eosToken: String) {
        // CHANGED: Modify source template as needed.
        self.source = source
            .replacingOccurrences(of: "is none", with: " == '<none>'")
            .replacingOccurrences(of: "is not none", with: " != '<none>'")
        self.bosToken = bosToken
        self.eosToken = eosToken

        // For simplicity, assume default booleans:
        self.supportsTools = false
        self.requiresObjectArguments = false
        self.requiresTypedContent = false
        self.supportsSystemRole = false
        self.supportsParallelToolCalls = false

        // parse the template
        do {
            // Attempt to render the Jinja template
            self.jinjaTemplate = try Template(self.source)
            print("ChatTemplate successfully parsed Jinja template")
            print(source)
        } catch {
            print("Failed to render jinja template: \(error)")
            print("Source: \(self.source)")
        }

        // check if the source references "tool_calls"
        if source.range(of: "tool_calls", options: .caseInsensitive) != nil {
            self.supportsTools = true
        }

        // check if the source references "tool_call_id" for parallel calls
        if source.range(of: "tool_call_id", options: .caseInsensitive) != nil {
            self.supportsParallelToolCalls = true
        }

        // Try the "tryRawRender" approach to see if it generates certain text:
        let testStringArgs = tryRawRender(
            messages: [
                ChatTemplate.createMessage(role: .user, content: "Hey"),
                ChatTemplate.createToolMessage(type: .call(toolCalls: [
                    ToolCall(id: "call_1___",
                             type: "function",
                             functionName: "ipython",
                             arguments: "{\"code\": \"print('Hello, World!')\"}")
                ]))
            ],
            tools: [:],
            addGenerationPrompt: false,
            extraContext: [:]
        )
        let rendersStringArgs = testStringArgs.contains("{\"code\": \"print")

        if !rendersStringArgs {
            // check the object-arguments approach
            let testObjectArgs = tryRawRender(
                messages: [
                    ChatTemplate.createMessage(role: .user, content: "Hey"),
                    ChatTemplate.createToolMessage(type: .call(toolCalls: [
                        ToolCall(
                            id: "call_1___",
                            type: "function",
                            functionName: "ipython",
                            arguments: ["code": "print('Hello, World!')"]
                        )
                    ]))
                ],
                tools: [:],
                addGenerationPrompt: false,
                extraContext: [:]
            )
            let rendersObjectArgs = testObjectArgs.contains("{\"code\": \"print")
            self.requiresObjectArguments = rendersObjectArgs
        } else {
            self.requiresObjectArguments = false
        }

        // check if system role is supported
        let testSystem = tryRawRender(
            messages: [
                ChatTemplate.createMessage(role: .system, content: "<System Needle>"),
                ChatTemplate.createMessage(role: .user, content: "Hey")
            ],
            tools: [:],
            addGenerationPrompt: false,
            extraContext: [:]
        )
        if testSystem.contains("<System Needle>") {
            self.supportsSystemRole = true
        }

        // check typed content
        let testTyped1 = tryRawRender(
            messages: [
                ChatTemplate.createMessage(role: .user, content: "Hey")
            ],
            tools: [:],
            addGenerationPrompt: false,
            extraContext: [:]
        )
        let testTyped2 = tryRawRender(
            messages: [
                ChatTemplate.createMessage(role: .user, content: [
                    ["type": "text", "text": "Hey"]
                ])
            ],
            tools: [:],
            addGenerationPrompt: false,
            extraContext: [:]
        )
        let hidesIn1 = !testTyped1.contains("Hey")
        let showsIn2 = testTyped2.contains("Hey")
        self.requiresTypedContent = (hidesIn1 && showsIn2)
        
        print("ChatTemplate initialized")
        print("SupportsTools: \(supportsTools)")
        print("RequiresObjectArguments: \(requiresObjectArguments)")
        print("RequiresTypedContent: \(requiresTypedContent)")
        print("SupportsSystemRole: \(supportsSystemRole)")
        print("SupportsParallelToolCalls: \(supportsParallelToolCalls)")
    }

    // MARK: - apply()

    /// Swift analog of `apply(...)` from the original code.
    /// The `messages`, `tools`, `extraContext` are Swift dictionaries/arrays,
    /// typically parsed from JSON or constructed by the caller.
    ///
    /// - parameter messages: Array-of-dictionaries describing user, system, or assistant messages.
    /// - parameter tools: Extra tool info.
    /// - parameter addGenerationPrompt: If true, we do some special logic (like adding an EOS?).
    /// - parameter extraContext: Further dictionary data for the template.
    /// - parameter adjustInputs: Whether to fix messages if, for example, we do not support system role.
    /// - returns: The final rendered prompt string.
    public func apply(
        messages finalMessages: [[String: Any]],
        tools: [String: Any],
        addGenerationPrompt: Bool,
        extraContext: [String: Any] = [:],
        adjustInputs: Bool = true,
        dryRun: Bool = false
    ) -> String {
        let finalMessages = adjustInputs ? fixMessagesIfNeeded(finalMessages) : finalMessages
        
        // Build the base context using the high-level API.
        var context = createJinjaContext(addGenerationPrompt: addGenerationPrompt)
        
        // CHANGED: Instead of operating on context["messages"], operate on a separate messages variable.
        var messages: [[String: Any]] = context["messages"] as? [[String: Any]] ?? [[String: Any]]()
        
        for msg in finalMessages {
            guard let roleStr = msg["role"] as? String else { continue }
            if roleStr == MessageRole.system.rawValue {
                let content = msg["content"] as? String
                messages = addMessage(to: &messages, role: .system, content: content)
            } else if roleStr == MessageRole.user.rawValue {
                let content = msg["content"] as? String
                messages = addMessage(to: &messages, role: .user, content: content)
            } else if roleStr == MessageRole.assistant.rawValue {
                if let toolCalls = msg["tool_calls"] as? [ToolCall] {
                    messages = addToolMessage(to: &messages, type: .call(toolCalls: toolCalls))
                } else if let toolCallsArray = msg["tool_calls"] as? [[String: Any]] {
                    let toolCalls = toolCallsArray.compactMap { ToolCall(dictionary: $0) }
                    messages = addToolMessage(to: &messages, type: .call(toolCalls: toolCalls))
                } else {
                    let content = msg["content"] as? String
                    messages = addMessage(to: &messages, role: .assistant, content: content)
                }
            } else if roleStr == "tool" {
                let content = msg["content"] as? String ?? ""
                messages = addToolMessage(to: &messages, type: .output(content: content))
            } else {
                let content = msg["content"] as? String
                messages = addMessage(to: &messages, role: .user, content: content)
            }
        }
        
        // Put the updated messages array back into context.
        context["messages"] = messages
        
        if !tools.isEmpty {
            context["tools"] = tools
        }
        for (k, v) in extraContext {
            context[k] = v
        }
        if !dryRun {
            print("ChatTemplate.apply. Final Context\n[BEGIN CONTEXT]")
            print(context)
            print("[END CONTEXT]\n")
        }
        
        do {
            let prompt = try self.jinjaTemplate?.render(context) ?? "JINJA ERROR"
            return prompt
        } catch {
            if !dryRun {
                print("ChatTemplate.apply error: \(error)")
                print("context: \(context)")
            }
        }
        return "JINJA PARSE ERROR"
    }

    // MARK: - tryRawRender Helper

    /// Replicates your original "try_raw_render" logic:
    /// We do an apply(...) with adjustInputs=false, and if something fails, we return "".
    private func tryRawRender(
        messages: [[String: Any]],
        tools: [String: Any],
        addGenerationPrompt: Bool,
        extraContext: [String: Any]
    ) -> String {
        do {
            let rendered = apply(
                messages: messages,
                tools: tools,
                addGenerationPrompt: addGenerationPrompt,
                extraContext: extraContext,
                adjustInputs: false,
                dryRun: true
            )
            return rendered
        } catch {
            return ""
        }
    }

    // MARK: - fixMessagesIfNeeded

    /// Similar to your original logic that merges system messages or rewrites them if we do not
    /// support system roles, or changes "tool" roles if we do not support them, etc.
    private func fixMessagesIfNeeded(_ messages: [[String: Any]]) -> [[String: Any]] {
        var newMessages = [[String: Any]]()
        var pendingSystem = ""

        for msg in messages {
            guard let role = msg["role"] as? String else { continue }
            var updated = msg

            if role == "system" && !supportsSystemRole {
                if let content = msg["content"] as? String {
                    if !pendingSystem.isEmpty { pendingSystem += "\n" }
                    pendingSystem += content
                }
                continue
            }

            if !supportsTools && msg.keys.contains("tool_calls") {
                var copy = updated
                copy.removeValue(forKey: "tool_calls")
                updated = copy
            }

            if role == "tool" && !supportsTools {
                updated["role"] = MessageRole.user.rawValue
            }

            if role == MessageRole.user.rawValue && !pendingSystem.isEmpty {
                let oldContent = (updated["content"] as? String) ?? ""
                updated["content"] = pendingSystem + (oldContent.isEmpty ? "" : "\n" + oldContent)
                pendingSystem = ""
            }

            newMessages.append(updated)
        }

        if !pendingSystem.isEmpty {
            newMessages.append([
                "role": MessageRole.user.rawValue,
                "content": pendingSystem
            ])
            pendingSystem = ""
        }

        return newMessages
    }
    
    // MARK: - High-Level Jinja2 Context Management

    /// Creates the base Jinja2 context with basic settings.
    /// Uses the ChatTemplate's bosToken property.
    ///
    /// - parameter addGenerationPrompt: A Boolean flag indicating whether to add a generation prompt.
    /// - returns: A dictionary representing the initial context.
    public func createJinjaContext(addGenerationPrompt: Bool) -> [String: Any] {
        return [
            "bos_token": self.bosToken,
            "add_generation_prompt": addGenerationPrompt,
            "messages": [[String: Any]]()
        ]
    }

    /// Adds a regular message to the messages array.
    ///
    /// - Parameters:
    ///   - messages: The messages array to update.
    ///   - role: The role of the message.
    ///   - content: The content of the message.
    /// - Returns: The updated messages array.
    public func addMessage(to messages: inout [[String: Any]], role: MessageRole, content: Any?) -> [[String: Any]] {
        let message: [String: Any] = [
            "role": role.rawValue,
            "content": content ?? NSNull()
        ]
        messages.append(message)
        return messages
    }

    /// Adds a tool-related message to the messages array.
    ///
    /// - Parameters:
    ///   - messages: The messages array to update.
    ///   - type: The type of tool message (call or output).
    /// - Returns: The updated messages array.
    public func addToolMessage(to messages: inout [[String: Any]], type: ToolMessageType) -> [[String: Any]] {
        var message: [String: Any] = [:]
        switch type {
        case .call(let toolCalls):
            message["role"] = MessageRole.assistant.rawValue
            message["content"] = "<none>"
            message["tool_calls"] = toolCalls.map { $0.toDictionary() }
        case .output(let content):
            message["role"] = "tool"
            message["content"] = content
        }
        messages.append(message)
        return messages
    }

    /// Creates and returns a regular message dictionary.
    ///
    /// - Parameters:
    ///   - role: The message role.
    ///   - content: The content of the message.
    /// - Returns: A dictionary representing the message.
    public static func createMessage(role: MessageRole, content: Any?) -> [String: Any] {
        return [
            "role": role.rawValue,
            "content": content ?? "<none>"
        ]
    }

    /// Updated createToolMessage() function that accepts a ToolMessageType.
    /// For the call case, it converts a list of ToolCall instances into dictionaries.
    public static func createToolMessage(type: ToolMessageType) -> [String: Any] {
        var message: [String: Any] = [:]
        switch type {
        case .call(let toolCalls):
            message["role"] = MessageRole.assistant.rawValue
            message["content"] = "<none>"
            message["tool_calls"] = toolCalls.map { $0.toDictionary() }
        case .output(let content):
            message["role"] = "tool"
            message["content"] = content
        }
        return message
    }
}
