import Foundation

/// Tiny observable bridging the worker coordinator's status callbacks into a
/// form the SwiftUI diagnostic panel can subscribe to. Kept deliberately dumb
/// — it's a log buffer + three counters, not an orchestration layer.
@MainActor
final class WorkerStatusModel: ObservableObject {
    @Published private(set) var lines: [String] = []
    @Published private(set) var lastTriggerSeq: Int = 0
    @Published private(set) var lastInjectSeq: Int = 0
    @Published private(set) var workerAlive: Bool = false
    @Published private(set) var workerExitedAbnormally: Bool = false

    private let maxLines = 200

    func apply(_ status: WorkerCoordinator.Status) {
        switch status {
        case .workerLaunched(let pid):
            workerAlive = true
            workerExitedAbnormally = false
            append("worker launched pid=\(pid)")
        case .workerExited(let status, let intentional):
            workerAlive = false
            workerExitedAbnormally = !intentional && status != 0
            append("worker exited status=\(status) intentional=\(intentional)")
        case .workerGaveUp(let status):
            workerAlive = false
            workerExitedAbnormally = true
            append("worker gave up after retries (last status=\(status))")
        case .triggerSent(let seq, let appName):
            lastTriggerSeq = seq
            append("→ trigger seq=\(seq) app=\(appName)")
        case .triggerFailed(let reason):
            append("trigger failed: \(reason)")
        case .injectApplied(let text, let strategy):
            lastInjectSeq += 1
            append("← inject applied strategy=\(strategy) text=\(debugPreview(text))")
        case .injectFailed(let reason, let strategy):
            append("← inject FAILED strategy=\(strategy) reason=\(reason)")
        case .workerLog(let level, let message):
            append("[\(level)] \(message)")
        case .workerError(let message):
            append("[error] \(message)")
        case .stderr(let line):
            append("[stderr] \(line)")
        }
    }

    func append(_ line: String) {
        let ts = Self.timestampFormatter.string(from: Date())
        lines.append("\(ts) \(line)")
        if lines.count > maxLines {
            lines.removeFirst(lines.count - maxLines)
        }
    }

    private func debugPreview(_ s: String) -> String {
        if s.count <= 60 { return "\"\(s)\"" }
        let head = s.prefix(60)
        return "\"\(head)…\" (\(s.count) chars)"
    }

    private static let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()
}
