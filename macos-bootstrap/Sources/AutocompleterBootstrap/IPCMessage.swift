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

    /// Experimental: ship a ring-buffer trajectory to Python and ask the
    /// fast LLM what the user is doing. Python replies with a
    /// `TrajectoryGuess`. Fire-and-forget if the worker is busy.
    struct TrajectoryQuery: Codable {
        let seq: Int
        var type: String = "trajectory_query"
        /// Chronological list of recent (app, window, focus) snapshots. Shape
        /// matches `TrajectoryStack.Snapshot`.
        let snapshots: [TrajectorySnapshotPayload]
    }

    struct TrajectorySnapshotPayload: Codable {
        let timestamp: Double
        let lastSeen: Double
        let trigger: String
        let appName: String
        let appPID: Int32
        let bundleID: String?
        let windowTitle: String?
        let url: String?
        let focusedRole: String?
        let focusedValuePreview: String?
        let focusedValueLength: Int?
        let contentSimilarity: Double?

        enum CodingKeys: String, CodingKey {
            case timestamp
            case lastSeen = "last_seen"
            case trigger
            case appName = "app_name"
            case appPID = "app_pid"
            case bundleID = "bundle_id"
            case windowTitle = "window_title"
            case url
            case focusedRole = "focused_role"
            case focusedValuePreview = "focused_value_preview"
            case focusedValueLength = "focused_value_length"
            case contentSimilarity = "content_similarity"
        }
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

    // MARK: - Intelligent paste

    /// Swift → Python: user pressed paste hotkey. Carries the current
    /// field context + recent OCR captures so Python can ask the LLM
    /// for paste suggestions.
    struct PasteTrigger: Codable {
        let seq: Int
        var type: String = "paste_trigger"
        let timestamp: Double
        let appName: String
        let appPID: Int32
        let windowTitle: String?
        let focusedRole: String?
        let focusedValue: String?
        let focusedLabel: String?    // AXTitle / AXDescription / AXPlaceholderValue
        /// OCR text from a small screenshot around the caret/focused element.
        /// Includes bounding boxes. Helps the LLM identify which field the
        /// user is in when AX labels are missing (common in web forms).
        let caretContextOCR: String?
        let recentCaptures: [CapturePayload]

        enum CodingKeys: String, CodingKey {
            case seq, type, timestamp
            case appName = "app_name"
            case appPID = "app_pid"
            case windowTitle = "window_title"
            case focusedRole = "focused_role"
            case focusedValue = "focused_value"
            case focusedLabel = "focused_label"
            case caretContextOCR = "caret_context_ocr"
            case recentCaptures = "recent_captures"
        }
    }

    struct CapturePayload: Codable {
        let timestamp: Double
        let appName: String
        let windowTitle: String?
        let url: String?
        let ocrText: String

        enum CodingKeys: String, CodingKey {
            case timestamp
            case appName = "app_name"
            case windowTitle = "window_title"
            case url
            case ocrText = "ocr_text"
        }
    }

    /// Python → Swift: ranked paste suggestions from the LLM.
    struct PasteSuggestions: Codable {
        let seq: Int
        let type: String  // "paste_suggestions"
        let inReplyTo: Int
        let suggestions: [PasteSuggestion]

        enum CodingKeys: String, CodingKey {
            case seq, type
            case inReplyTo = "in_reply_to"
            case suggestions
        }
    }

    struct PasteSuggestion: Codable {
        let text: String
        let rank: Int
        let source: String?
    }

    /// Python's answer to a `TrajectoryQuery`. The experiment cares about
    /// both the sentence and the latency, so model/latency live in-message
    /// rather than in a side-channel log.
    struct TrajectoryGuess: Codable {
        let seq: Int
        let type: String           // "trajectory_guess"
        let inReplyTo: Int?
        /// Free-form English sentence: "You're drafting a Marketplace listing
        /// for a pair of Sony headphones."
        let intentSentence: String
        /// Self-reported 0..1 confidence. The model isn't calibrated; treat
        /// this as a rough tie-break signal rather than ground truth.
        let confidence: Double?
        /// Which model produced the guess — useful for offline eval when we
        /// swap providers.
        let modelUsed: String?
        /// Wall-clock time from query receipt to response emission on the
        /// Python side, milliseconds.
        let latencyMs: Double?

        enum CodingKeys: String, CodingKey {
            case seq
            case type
            case inReplyTo = "in_reply_to"
            case intentSentence = "intent_sentence"
            case confidence
            case modelUsed = "model_used"
            case latencyMs = "latency_ms"
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
        case "trajectory_guess":
            return try decoder.decode(IPCMessage.TrajectoryGuess.self, from: line)
        case "paste_suggestions":
            return try decoder.decode(IPCMessage.PasteSuggestions.self, from: line)
        default:
            throw IPCDecodeError.unexpectedType(envelope.type)
        }
    }
}
