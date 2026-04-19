import Foundation

/// Bridge between TrajectoryStack and the SwiftUI Trajectory Inspector window.
/// Separate from WorkerStatusModel (which is a flat string log) because the
/// inspector needs typed access to snapshots, a live prompt preview, the latest
/// guess, and bidirectional numeric bindings for tuning.
@MainActor
final class TrajectoryInspectorModel: ObservableObject {
    // MARK: - Live data

    @Published private(set) var snapshots: [TrajectoryStack.Snapshot] = []
    @Published private(set) var promptPreview: String = ""
    @Published var selectedSnapshotIndex: Int? = nil
    @Published private(set) var latestGuess: IPCMessage.TrajectoryGuess? = nil

    // MARK: - Tunable parameters

    @Published var previewCharBudget: Int = 2000
    @Published var cooldownInterval: Double = 4.0
    @Published var heartbeatInterval: Double = 30.0
    @Published var bufferCapacity: Int = 200

    // MARK: - Backing references

    private weak var stack: TrajectoryStack?
    private weak var intentController: TrajectoryIntentController?

    init(stack: TrajectoryStack, intentController: TrajectoryIntentController?) {
        self.stack = stack
        self.intentController = intentController

        // Seed with current state.
        self.snapshots = stack.currentSnapshots()
        self.promptPreview = Self.formatPromptBlock(from: self.snapshots)

        // Read current param values.
        self.previewCharBudget = stack.previewCharLimit
        self.heartbeatInterval = stack.currentHeartbeatInterval
        self.bufferCapacity = stack.currentCapacity
        if let ctrl = intentController {
            self.cooldownInterval = ctrl.currentMinInterval
        }

        // Subscribe to live changes.
        stack.addObserver { [weak self] _ in
            self?.onStackChange()
        }

        // Subscribe to guess updates.
        intentController?.onGuessForInspector = { [weak self] guess in
            self?.latestGuess = guess
        }
    }

    // MARK: - Stack observer

    private func onStackChange() {
        snapshots = stack?.currentSnapshots() ?? []
        promptPreview = Self.formatPromptBlock(from: snapshots)
        // Clamp selection if buffer rotated old entries out.
        if let idx = selectedSnapshotIndex, idx >= snapshots.count {
            selectedSnapshotIndex = snapshots.isEmpty ? nil : snapshots.count - 1
        }
    }

    // MARK: - Apply tuning values

    func applyCooldownInterval(_ value: Double) {
        intentController?.setMinInterval(value)
    }

    func applyPreviewCharBudget(_ value: Int) {
        stack?.setPreviewCharLimit(value)
    }

    func applyHeartbeatInterval(_ value: Double) {
        stack?.setHeartbeatInterval(value)
    }

    func applyBufferCapacity(_ value: Int) {
        stack?.setCapacity(value)
    }

    // MARK: - Prompt formatter

    /// Swift port of `_format_snapshots()` from `trajectory_intent.py`.
    /// Must produce identical output so the inspector shows exactly what
    /// the LLM sees.
    static func formatPromptBlock(from snapshots: [TrajectoryStack.Snapshot]) -> String {
        guard !snapshots.isEmpty else { return "(no activity recorded)" }
        let baseTS = snapshots[0].timestamp
        return snapshots.map { s in
            let dt = Int(max(0, s.timestamp - baseTS))
            let dwell = Int(max(0, s.lastSeen - s.timestamp))
            let trigger = s.trigger.padding(toLength: 13, withPad: " ", startingAt: 0)
            var parts: [String] = [
                String(format: "[+%4ds]", dt),
                trigger,
                s.appName,
            ]
            if dwell > 0 {
                parts.append("(dwell \(dwell)s)")
            }
            if let title = s.windowTitle, !title.isEmpty {
                parts.append("| \(title)")
            }
            if let url = s.url, !url.isEmpty {
                parts.append("| url=\(url)")
            }
            if let role = s.focusedRole {
                if let preview = s.focusedValuePreview, !preview.isEmpty {
                    parts.append("| \(role)=\(preview)")
                } else {
                    parts.append("| role=\(role)")
                }
            }
            return parts.joined(separator: " ")
        }.joined(separator: "\n")
    }
}
