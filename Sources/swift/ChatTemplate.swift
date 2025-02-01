//
//  ChatTemplate.swift
//  llamacpp_swift
//
//  Created by Qi Wang on 2025-01-31.
//


import Foundation
import Jinja

/// A Swift approximation of your C++ minja::chat_template
/// The idea: store the template source, parse it, track booleans like
/// supports_tools_, and supply an `apply(...)` method that "renders".
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
        self.source = source
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
            self.jinjaTemplate = try Template(source)
            print("ChatTemplate successfully parsed Jinja template")
            print(source)
        } catch {
            print("Failed to render jinja template: \(error)")
            print("Source: \(self.source)")
        }

        // check if the source references "tools"
        if source.range(of: "tools", options: .caseInsensitive) != nil {
            self.supportsTools = true
        }

        // check if the source references "tool_call_id" for parallel calls
        if source.range(of: "tool_call_id", options: .caseInsensitive) != nil {
            self.supportsParallelToolCalls = true
        }

        // Try the "tryRawRender" approach to see if it generates certain text:
        let testStringArgs = tryRawRender(
            messages: [
                [
                    "role": "user",
                    "content": "Hey"
                ],
                [
                    "role": "assistant",
                    "tool_calls": [[
                        "id": "call_1___",
                        "type": "function",
                        "function": [
                            "arguments": "{\"code\": \"print('Hello, World!')\"}",
                            "name": "ipython"
                        ]
                    ]]
                ]
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
                    [
                        "role": "user",
                        "content": "Hey"
                    ],
                    [
                        "role": "assistant",
                        "tool_calls": [[
                            "id": "call_1___",
                            "type": "function",
                            "function": [
                                "arguments": [
                                    "code": "print('Hello, World!')"
                                ],
                                "name": "ipython"
                            ]
                        ]]
                    ]
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
                [
                    "role": "system",
                    "content": "<System Needle>"
                ],
                [
                    "role": "user",
                    "content": "Hey"
                ]
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
                [
                    "role": "user",
                    "content": "Hey"
                ]
            ],
            tools: [:],
            addGenerationPrompt: false,
            extraContext: [:]
        )
        let testTyped2 = tryRawRender(
            messages: [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": "Hey"
                        ]
                    ]
                ]
            ],
            tools: [:],
            addGenerationPrompt: false,
            extraContext: [:]
        )
        let hidesIn1 = !testTyped1.contains("Hey")
        let showsIn2 = testTyped2.contains("Hey")
        self.requiresTypedContent = (hidesIn1 && showsIn2)
    }

    // MARK: - apply()

    /// Swift analog of `apply(...)` from the original code.
    /// The `messages`, `tools`, `extraContext` are Swift dictionaries/arrays,
    /// typically parsed from JSON or constructed by the caller.
    ///
    /// - parameter messages: array-of-dictionaries describing user, system, or assistant messages.
    /// - parameter tools: extra tool info.
    /// - parameter addGenerationPrompt: if true, we do some special logic (like adding an EOS?)
    /// - parameter extraContext: further dictionary data for the template
    /// - parameter adjustInputs: whether to fix messages if e.g. we do not support system role, etc.
    /// - returns: the final rendered prompt string
    public func apply(
        messages: [[String: Any]],
        tools: [String: Any],
        addGenerationPrompt: Bool,
        extraContext: [String: Any] = [:],
        adjustInputs: Bool = true
    ) -> String {
        // optional rewriting of messages
        let finalMessages: [[String: Any]]
        if adjustInputs {
            finalMessages = fixMessagesIfNeeded(messages)
        } else {
            finalMessages = messages
        }

        // build up the dictionary that will be given to the template
        var context: [String: Any] = [
            "messages": finalMessages,
            "add_generation_prompt": addGenerationPrompt,
            "bos_token": bosToken,
            "eos_token": eosToken
        ]
        if !tools.isEmpty {
            context["tools"] = tools
        }
        for (k, v) in extraContext {
            context[k] = v
        }
        print("ChatTemplate.apply. Final Context\n[BEGIN CONTEXT]")
        print(context)
        print("[END CONTEXT]\n")

        // now do the "render"
        do {
            let prompt = try self.jinjaTemplate?.render(context) ?? "JINJA ERROR"
            return prompt
        } catch {
            print("ChatTemplate.apply error: \(error)")
            print("context: \(context)")
        }
        return "JINJA PARSE ERROR"
    }

    // MARK: - “tryRawRender” helper

    /// Replicates your original "try_raw_render" logic:
    /// We do an apply(...) with adjustInputs=false,
    /// and if something fails, we return "".
    private func tryRawRender(
        messages: [[String: Any]],
        tools: [String: Any],
        addGenerationPrompt: Bool,
        extraContext: [String: Any]
    ) -> String {
        // In Swift, we rarely use exceptions for normal control, but let's illustrate
        do {
            let rendered = apply(
                messages: messages,
                tools: tools,
                addGenerationPrompt: addGenerationPrompt,
                extraContext: extraContext,
                adjustInputs: false
            )
            return rendered
        } catch {
            // If we had an error
            return ""
        }
    }

    // MARK: - fixMessagesIfNeeded

    /// Similar to your original logic that merges system messages or rewrites them if we do not
    /// support system roles, or changes “tool” roles if we do not support them, etc.
    private func fixMessagesIfNeeded(_ messages: [[String: Any]]) -> [[String: Any]] {
        var newMessages = [[String: Any]]()
        var pendingSystem = ""

        for msg in messages {
            guard let role = msg["role"] as? String else {
                // ignoring or throw an error
                continue
            }
            var updated = msg

            // If we do not support system role, accumulate into pendingSystem
            if role == "system" && !supportsSystemRole {
                if let content = msg["content"] as? String {
                    if !pendingSystem.isEmpty { pendingSystem += "\n" }
                    pendingSystem += content
                }
                // skip adding to newMessages
                continue
            }

            // If we do not support tools but user added "tool_calls", remove them
            if !supportsTools && msg.keys.contains("tool_calls") {
                // remove "tool_calls"
                var copy = updated
                copy.removeValue(forKey: "tool_calls")
                updated = copy
            }

            // If the role was "tool" but we do not support tools, rewrite the role to "user"
            if role == "tool" && !supportsTools {
                updated["role"] = "user"
            }

            // merge any pendingSystem text if we see a user role
            if role == "user" && !pendingSystem.isEmpty {
                let oldContent = (updated["content"] as? String) ?? ""
                updated["content"] = pendingSystem + (oldContent.isEmpty ? "" : "\n" + oldContent)
                pendingSystem = ""
            }

            newMessages.append(updated)
        }

        // If we still have leftover system text, optionally append as user
        if !pendingSystem.isEmpty {
            newMessages.append([
                "role": "user",
                "content": pendingSystem
            ])
            pendingSystem = ""
        }

        return newMessages
    }

}
