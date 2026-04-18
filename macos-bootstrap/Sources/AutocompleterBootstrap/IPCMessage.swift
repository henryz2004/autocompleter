import Foundation

/// JSON-line IPC contract with the Python worker.
///
/// Every message is a single UTF-8 JSON object terminated by `\n`. Stdout
/// from Python carries commands *to* Swift; stdin to Python carries triggers
/// *from* Swift. Stderr is the diagnostic log channel and is not structured.
///
/// See `autocompleter/worker.py` for the Python-side handler.
///
/// Phase A keeps the payload minimal. `ax_subtree`, `conversation_turns`,
/// and `cross_app_snapshots` are declared here but stay empty until Phase B
/// needs them — the shape is locked now to avoid a protocol churn later.
enum IPCMessage {
    // MARK: - Swift → Python

    struct Trigger: Codable {
        let seq: Int
        let type: String           // "trigger"
        let triggerType: String    // "manual" | "auto" (Phase A is always manual)
        let timestamp: Double
        let appName: String
        let appPID: Int32
        let windowTitle: String?
        let sourceURL: String?
        let focusedElement: FocusedElement
        let caretPosition: CaretPosition?
        let axSubtree: AXSubtree
        let conversationTurns: [ConversationTurn]
        let crossAppSnapshots: [CrossAppSnapshot]

        enum CodingKeys: String, CodingKey {
            case seq
            case type
            case triggerType = "trigger_type"
            case timestamp
            case appName = "app_name"
            case appPID = "app_pid"
            case windowTitle = "window_title"
            case sourceURL = "source_url"
            case focusedElement = "focused_element"
            case caretPosition = "caret_position"
            case axSubtree = "ax_subtree"
            case conversationTurns = "conversation_turns"
            case crossAppSnapshots = "cross_app_snapshots"
        }
    }

    struct FocusedElement: Codable {
        let role: String
        let value: String
        let insertionPoint: Int?
        let selectionLength: Int
        let placeholderDetected: Bool
        let beforeCursor: String
        let afterCursor: String
        let position: [Double]?       // [x, y]
        let size: [Double]?           // [w, h]

        enum CodingKeys: String, CodingKey {
            case role
            case value
            case insertionPoint = "insertion_point"
            case selectionLength = "selection_length"
            case placeholderDetected = "placeholder_detected"
            case beforeCursor = "before_cursor"
            case afterCursor = "after_cursor"
            case position
            case size
        }
    }

    struct CaretPosition: Codable {
        let x: Double
        let y: Double
        let height: Double
    }

    struct AXSubtree: Codable {
        let bottomUp: String
        let topDown: String

        enum CodingKeys: String, CodingKey {
            case bottomUp = "bottom_up"
            case topDown = "top_down"
        }
    }

    struct ConversationTurn: Codable {
        let speaker: String
        let text: String
        let timestamp: Double?
    }

    struct CrossAppSnapshot: Codable {
        let appName: String
        let windowTitle: String?
        let sourceURL: String?
        let visibleText: String
        let timestamp: Double

        enum CodingKeys: String, CodingKey {
            case appName = "app_name"
            case windowTitle = "window_title"
            case sourceURL = "source_url"
            case visibleText = "visible_text"
            case timestamp
        }
    }

    /// Reply to a Python `inject` telling it whether the injection landed.
    struct InjectResult: Codable {
        let seq: Int
        var type: String = "inject_result"
        let inReplyTo: Int
        let ok: Bool
        let reason: String?

        enum CodingKeys: String, CodingKey {
            case seq
            case type
            case inReplyTo = "in_reply_to"
            case ok
            case reason
        }
    }

    /// Liveness check from Swift to Python. Python replies with Pong.
    struct Ping: Codable {
        let seq: Int
        var type: String = "ping"
    }

    // MARK: - Python → Swift

    /// Discriminator-first lightweight envelope used for initial decoding.
    /// We peek at `type` and then re-decode into the concrete shape.
    struct Envelope: Codable {
        let seq: Int?
        let type: String
    }

    struct Inject: Codable {
        let seq: Int
        let type: String           // "inject"
        let text: String
        let strategy: String       // "ax_value" | "clipboard" | "keystrokes" | "cdp" | "auto"
        let requiresFocusedElement: Bool
        let inReplyTo: Int?

        enum CodingKeys: String, CodingKey {
            case seq
            case type
            case text
            case strategy
            case requiresFocusedElement = "requires_focused_element"
            case inReplyTo = "in_reply_to"
        }
    }

    struct LogMessage: Codable {
        let seq: Int
        let type: String           // "log"
        let level: String
        let message: String
    }

    struct Pong: Codable {
        let seq: Int
        let type: String           // "pong"
        let inReplyTo: Int

        enum CodingKeys: String, CodingKey {
            case seq
            case type
            case inReplyTo = "in_reply_to"
        }
    }

    struct ErrorMessage: Codable {
        let seq: Int
        let type: String           // "error"
        let message: String
        let inReplyTo: Int?

        enum CodingKeys: String, CodingKey {
            case seq
            case type
            case message
            case inReplyTo = "in_reply_to"
        }
    }
}

/// Strategy a Python `inject` message requests from Swift.
enum InjectStrategy: String {
    case axValue = "ax_value"
    case clipboard = "clipboard"
    case keystrokes = "keystrokes"
    case cdp = "cdp"
    case auto = "auto"
}

enum IPCDecodeError: Error, CustomStringConvertible {
    case invalidJSON(String)
    case unexpectedType(String)

    var description: String {
        switch self {
        case .invalidJSON(let s):
            return "Invalid JSON from worker: \(s)"
        case .unexpectedType(let t):
            return "Unexpected message type from worker: \(t)"
        }
    }
}

/// Small convenience for encoding outbound messages into `\n`-terminated UTF-8.
enum IPCCodec {
    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.outputFormatting = [.withoutEscapingSlashes]
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        return d
    }()

    /// Encode a Codable message and append `\n`. Throws on encoding failure.
    static func encodeLine<T: Encodable>(_ msg: T) throws -> Data {
        var data = try encoder.encode(msg)
        data.append(0x0A)  // '\n'
        return data
    }

    /// Decode the envelope first, then re-decode based on the discriminator.
    /// Returns one of the incoming concrete types, or throws.
    static func decodeInbound(_ line: Data) throws -> Any {
        guard !line.isEmpty else {
            throw IPCDecodeError.invalidJSON("empty")
        }
        let envelope: IPCMessage.Envelope
        do {
            envelope = try decoder.decode(IPCMessage.Envelope.self, from: line)
        } catch {
            let preview = String(data: line.prefix(120), encoding: .utf8) ?? "<non-utf8>"
            throw IPCDecodeError.invalidJSON(preview)
        }
        switch envelope.type {
        case "inject":
            return try decoder.decode(IPCMessage.Inject.self, from: line)
        case "log":
            return try decoder.decode(IPCMessage.LogMessage.self, from: line)
        case "pong":
            return try decoder.decode(IPCMessage.Pong.self, from: line)
        case "error":
            return try decoder.decode(IPCMessage.ErrorMessage.self, from: line)
        default:
            throw IPCDecodeError.unexpectedType(envelope.type)
        }
    }
}
