import AppKit
import Foundation

/// Glues the experimental bits together:
///
///   TrajectoryStack  ──snapshot──▶  TrajectoryIntentController ──query──▶  Python worker
///                                                                ◀──guess──
///                                                       │
///                                                       └──▶ IntentOverlay
///
/// Responsibility split:
///   * `TrajectoryStack` decides *when a snapshot is worth recording*
///     (event debounce, dedup, heartbeat).
///   * This controller decides *when a recorded snapshot is worth asking
///     the LLM about*. Two triggers right now:
///        1. App-switch / focus-change / title-change snapshots — these
///           actually move the trajectory, so they're worth inferring on.
///        2. Heartbeat snapshots — they let the LLM see dwell time.
///     Initial snapshots (the one on `start()`) don't trigger a query.
///   * A 4-second cooldown keeps us from firing a query on every title
///     change while, e.g., a page loads and Chrome rewrites the title 5
///     times. The TrajectoryStack debounce already smooths rapid-fire
///     events, but Chrome occasionally sneaks through multiple.
///   * If the previous query is still in flight, we drop the new one. The
///     next snapshot will pick it up.
@MainActor
final class TrajectoryIntentController {
    private weak var coordinator: WorkerCoordinator?
    private weak var stack: TrajectoryStack?
    private let overlay: IntentOverlay

    /// Track seq numbers of outstanding trajectory_queries so we can match
    /// the reply back. Phase-A-level simple: one slot, since we drop
    /// overlapping requests.
    private var inflightSeq: Int?

    /// Wall-clock of the last query we fired. Used for the 4s cooldown.
    private var lastQueryAt: Date?
    private var minInterval: TimeInterval = 4.0

    /// If Python never replies (worker crash, handler early-return, LLM
    /// provider hang), the inflight slot would stay occupied forever and
    /// the probe would silently stop firing. Clear the slot after this
    /// many seconds even if no guess arrives. Matches the Python
    /// `request_timeout_s=2.0` plus a safety margin.
    private let inflightTimeout: TimeInterval = 5.0

    /// Diagnostic-log hook. Fired on every interesting event — query sent,
    /// query dropped (cooldown / inflight), guess received, timeout fired.
    /// Wired into the Worker-pane log so Henry can see the pipeline moving
    /// in real time without opening a file.
    var onLog: ((String) -> Void)?

    /// Second observer for the Trajectory Inspector. Fires the full guess
    /// struct so the inspector can display sentence + model + latency.
    var onGuessForInspector: ((IPCMessage.TrajectoryGuess) -> Void)?

    // MARK: - Tunable parameters (for Trajectory Inspector)

    var currentMinInterval: TimeInterval { minInterval }

    func setMinInterval(_ interval: TimeInterval) {
        minInterval = max(0.5, min(interval, 30.0))
    }

    init(coordinator: WorkerCoordinator, stack: TrajectoryStack, overlay: IntentOverlay) {
        self.coordinator = coordinator
        self.stack = stack
        self.overlay = overlay
    }

    func start() {
        overlay.show()
        overlay.updateStatus("(waiting for activity…)")

        stack?.addObserver { [weak self] change in
            Task { @MainActor in
                switch change {
                case .appended(let snap):
                    self?.onSnapshot(snap)
                case .updated:
                    break  // dwell coalesce — no query needed
                }
            }
        }

        // Piggyback on the worker's existing inbound stream: WorkerCoordinator
        // forwards trajectory_guess via a new status case.
        coordinator?.onTrajectoryGuess = { [weak self] guess in
            Task { @MainActor in
                self?.onGuess(guess)
            }
        }
    }

    func stop() {
        overlay.hide()
    }

    // MARK: - Event handling

    private func onSnapshot(_ snap: TrajectoryStack.Snapshot) {
        // Skip the bootstrap snapshot — nothing has happened yet.
        if snap.trigger == "initial" { return }

        // Cooldown: don't hammer the LLM on bursty events.
        if let last = lastQueryAt, Date().timeIntervalSince(last) < minInterval {
            let dt = Date().timeIntervalSince(last)
            onLog?(String(format: "intent: cooldown (%.1fs since last query)", dt))
            return
        }
        // If a previous query hasn't come back, drop this one.
        if let s = inflightSeq {
            onLog?("intent: dropped (inflight seq=\(s))")
            return
        }

        fireQuery()
    }

    private func fireQuery() {
        guard let coordinator, let stack else { return }
        let snaps = stack.currentSnapshots()
        guard !snaps.isEmpty else { return }

        let payload = snaps.map { s in
            IPCMessage.TrajectorySnapshotPayload(
                timestamp: s.timestamp,
                lastSeen: s.lastSeen,
                trigger: s.trigger,
                appName: s.appName,
                appPID: s.appPID,
                bundleID: s.bundleID,
                windowTitle: s.windowTitle,
                url: s.url,
                focusedRole: s.focusedRole,
                focusedValuePreview: s.focusedValuePreview,
                focusedValueLength: s.focusedValueLength,
                contentSimilarity: s.contentSimilarity
            )
        }
        let seq = coordinator.sendTrajectoryQuery(snapshots: payload)
        inflightSeq = seq
        lastQueryAt = Date()
        scheduleInflightTimeout(seq: seq)
        onLog?("intent: → query seq=\(seq) snapshots=\(snaps.count)")
    }

    /// Safety net for inflight tracking: if the reply never shows up we
    /// clear the slot after `inflightTimeout` so subsequent snapshots can
    /// fire queries again.
    private func scheduleInflightTimeout(seq: Int) {
        DispatchQueue.main.asyncAfter(deadline: .now() + inflightTimeout) { [weak self] in
            guard let self else { return }
            // Only clear if we're still tracking the same query — the reply
            // may have landed and a new query may have been fired in the
            // intervening window.
            if self.inflightSeq == seq {
                self.inflightSeq = nil
                self.onLog?("intent: timeout seq=\(seq) (no reply in \(Int(self.inflightTimeout))s)")
            }
        }
    }

    private func onGuess(_ guess: IPCMessage.TrajectoryGuess) {
        inflightSeq = nil
        let model = guess.modelUsed ?? "?"
        let ms = guess.latencyMs.map { String(format: "%.0fms", $0) } ?? "?ms"
        onLog?("intent: ← guess \(model) \(ms) — \(guess.intentSentence)")
        overlay.update(
            sentence: guess.intentSentence,
            latencyMs: guess.latencyMs,
            model: guess.modelUsed
        )
        onGuessForInspector?(guess)
    }
}
