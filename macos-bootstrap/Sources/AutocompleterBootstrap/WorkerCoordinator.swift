import AppKit
import Foundation

/// Glue between the hotkey, the Python worker, and the injector. Owns the
/// trigger→inject round-trip.
///
/// Phase A scope:
///   * Install a CGEvent hotkey tap (default `ctrl+space`).
///   * On hotkey, build a Trigger from the current AX snapshot and send to
///     the worker.
///   * When the worker sends back an `inject` message, apply it via
///     `Injector` and report the outcome back as `inject_result`.
///
/// Phase B will extend this to support regenerate, follow-up triggers, and
/// overlay-driven accept (where Python decides *when* to inject).
@MainActor
final class WorkerCoordinator {
    private let worker: WorkerProcess
    private let hotkey: HotkeyTap
    /// PID of the app focused when the most recent trigger fired. Used so the
    /// Injector can refuse to mutate the wrong app if the user switched away.
    private(set) var lastTriggerPID: pid_t = 0

    /// Optional consumer for status + log events, used by the diagnostics UI.
    var onStatus: ((Status) -> Void)?

    /// Experimental: delivered when Python replies to a `trajectory_query`.
    /// Lives on the `trajectory-intent-probe` branch.
    var onTrajectoryGuess: ((IPCMessage.TrajectoryGuess) -> Void)?

    enum Status {
        case workerLaunched(pid: Int32)
        case workerExited(status: Int32, intentional: Bool)
        case workerGaveUp(status: Int32)
        case triggerSent(seq: Int, appName: String)
        case triggerFailed(reason: String)
        case injectApplied(text: String, strategy: String)
        case injectFailed(reason: String, strategy: String)
        case workerLog(level: String, message: String)
        case workerError(message: String)
        case stderr(line: String)
    }

    init(workerConfig: WorkerProcess.Config, hotkey: HotkeyTap.Hotkey) {
        self.worker = WorkerProcess(config: workerConfig)
        self.hotkey = HotkeyTap(hotkey: hotkey)
        wire()
    }

    func start() {
        worker.start()
        hotkey.start()
    }

    func stop() {
        hotkey.stop()
        worker.stop()
    }

    // MARK: - Wiring

    private func wire() {
        worker.onInbound = { [weak self] message in
            self?.handleInbound(message)
        }
        worker.onLaunched = { [weak self] pid in
            self?.onStatus?(.workerLaunched(pid: pid))
        }
        worker.onExit = { [weak self] status, intentional in
            self?.onStatus?(.workerExited(status: status, intentional: intentional))
        }
        worker.onGaveUp = { [weak self] status in
            self?.onStatus?(.workerGaveUp(status: status))
        }
        worker.onStderr = { [weak self] line in
            self?.onStatus?(.stderr(line: line))
        }
        hotkey.onFire = { [weak self] in
            self?.handleHotkeyFire()
        }
        hotkey.onTapCreateFailed = { [weak self] in
            self?.onStatus?(.triggerFailed(
                reason: "CGEvent.tapCreate failed — accessibility permission missing?"
            ))
        }
    }

    // MARK: - Hotkey → Trigger

    // MARK: - Trajectory intent probe

    /// Ship the current trajectory ring buffer to Python. Returns the seq
    /// we used so the caller can correlate replies.
    func sendTrajectoryQuery(snapshots: [IPCMessage.TrajectorySnapshotPayload]) -> Int {
        let seq = worker.nextSeq()
        let query = IPCMessage.TrajectoryQuery(seq: seq, snapshots: snapshots)
        worker.send(query)
        return seq
    }

    private func handleHotkeyFire() {
        let seq = worker.nextSeq()
        guard let trigger = TriggerBuilder.build(seq: seq) else {
            onStatus?(.triggerFailed(reason: "AXFocusedUIElement returned nil"))
            return
        }
        lastTriggerPID = trigger.appPID
        worker.send(trigger)
        onStatus?(.triggerSent(seq: seq, appName: trigger.appName))
    }

    // MARK: - Worker messages

    private func handleInbound(_ message: Any) {
        switch message {
        case let inject as IPCMessage.Inject:
            applyInject(inject)
        case let log as IPCMessage.LogMessage:
            onStatus?(.workerLog(level: log.level, message: log.message))
        case let err as IPCMessage.ErrorMessage:
            onStatus?(.workerError(message: err.message))
        case let guess as IPCMessage.TrajectoryGuess:
            onTrajectoryGuess?(guess)
        case _ as IPCMessage.Pong:
            // No-op: pong is a liveness signal. If we start sending pings we
            // can measure round-trip here.
            break
        default:
            onStatus?(.workerError(message: "unhandled inbound: \(type(of: message))"))
        }
    }

    private func applyInject(_ msg: IPCMessage.Inject) {
        let strategy = InjectStrategy(rawValue: msg.strategy) ?? .auto
        let pidGuard: pid_t? = msg.requiresFocusedElement ? lastTriggerPID : nil

        let outcome = Injector.inject(
            text: msg.text,
            strategy: strategy,
            requireMatchingPID: pidGuard
        )

        if outcome.ok {
            onStatus?(.injectApplied(text: msg.text, strategy: outcome.strategyUsed))
        } else {
            onStatus?(.injectFailed(
                reason: outcome.reason ?? "unknown",
                strategy: outcome.strategyUsed
            ))
        }

        let result = IPCMessage.InjectResult(
            seq: worker.nextSeq(),
            inReplyTo: msg.seq,
            ok: outcome.ok,
            reason: outcome.reason
        )
        worker.send(result)
    }
}

// MARK: - Default config

extension WorkerProcess.Config {
    /// Phase A dev defaults. Reads two environment overrides so Henry can
    /// point at a different venv or repo without rebuilding:
    ///   * `AUTOCOMPLETER_WORKER_BIN` — absolute path to the python interpreter
    ///   * `AUTOCOMPLETER_REPO_ROOT` — absolute path to the repo root
    ///
    /// Fallbacks match Henry's dev layout on this machine.
    static func dev() -> WorkerProcess.Config {
        let env = ProcessInfo.processInfo.environment
        let defaultRepo = "/Users/henryz2004/code/orchestrator/autocompleter"
        let repoRoot = env["AUTOCOMPLETER_REPO_ROOT"] ?? defaultRepo
        let interpreter = env["AUTOCOMPLETER_WORKER_BIN"]
            ?? "\(repoRoot)/venv/bin/python"
        return WorkerProcess.Config(
            interpreter: interpreter,
            args: ["-m", "autocompleter", "--worker-mode"],
            workingDirectory: repoRoot,
            environment: [:]
        )
    }
}
