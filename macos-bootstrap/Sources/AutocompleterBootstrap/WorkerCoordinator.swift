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

    /// Delivered when Python replies to a `paste_trigger` with suggestions.
    var onPasteSuggestions: ((IPCMessage.PasteSuggestions) -> Void)?

    /// Reference to capture store for building paste triggers.
    weak var captureStore: CaptureStore?

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
        case let paste as IPCMessage.PasteSuggestions:
            onPasteSuggestions?(paste)
        case _ as IPCMessage.Pong:
            // No-op: pong is a liveness signal. If we start sending pings we
            // can measure round-trip here.
            break
        default:
            onStatus?(.workerError(message: "unhandled inbound: \(type(of: message))"))
        }
    }

    // MARK: - Paste hotkey

    /// Called by the paste HotkeyTap. Builds a paste_trigger from the
    /// current focused element + recent capture store entries and sends
    /// to Python.
    func handlePasteHotkey() {
        guard let store = captureStore else {
            onStatus?(.triggerFailed(reason: "No capture store available"))
            return
        }
        let entries = store.entries
        guard !entries.isEmpty else {
            onStatus?(.triggerFailed(reason: "No captures available for paste"))
            return
        }

        // Capture current focused element metadata.
        let frontmost = NSWorkspace.shared.frontmostApplication
        let appName = frontmost?.localizedName ?? "Unknown"
        let appPID = frontmost?.processIdentifier ?? 0

        var windowTitle: String? = nil
        var focusedRole: String? = nil
        var focusedValue: String? = nil
        var focusedLabel: String? = nil
        if appPID > 0 {
            let app = AXUIElementCreateApplication(appPID)
            // Window title
            var winRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(app, kAXFocusedWindowAttribute as CFString, &winRef) == .success,
               let winRef, CFGetTypeID(winRef) == AXUIElementGetTypeID() {
                var titleRef: CFTypeRef?
                if AXUIElementCopyAttributeValue(winRef as! AXUIElement, kAXTitleAttribute as CFString, &titleRef) == .success {
                    windowTitle = titleRef as? String
                }
            }
            // Focused element
            var focRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(app, kAXFocusedUIElementAttribute as CFString, &focRef) == .success,
               let focRef, CFGetTypeID(focRef) == AXUIElementGetTypeID() {
                let el = focRef as! AXUIElement
                var roleRef: CFTypeRef?
                if AXUIElementCopyAttributeValue(el, kAXRoleAttribute as CFString, &roleRef) == .success {
                    focusedRole = roleRef as? String
                }
                var valRef: CFTypeRef?
                if AXUIElementCopyAttributeValue(el, kAXValueAttribute as CFString, &valRef) == .success {
                    focusedValue = valRef as? String
                }
                // Try to get field label — AXTitle, AXDescription, or AXPlaceholderValue.
                for attr in [kAXTitleAttribute, kAXDescriptionAttribute, kAXPlaceholderValueAttribute] as [CFString] {
                    var labelRef: CFTypeRef?
                    if AXUIElementCopyAttributeValue(el, attr, &labelRef) == .success,
                       let str = labelRef as? String, !str.isEmpty {
                        focusedLabel = str
                        break
                    }
                }
            }
        }

        // Build capture payloads — last 10 entries, most recent first.
        let captures = entries.suffix(10).reversed().map { entry in
            IPCMessage.CapturePayload(
                timestamp: entry.timestamp.timeIntervalSince1970,
                appName: entry.appName,
                windowTitle: entry.windowTitle,
                url: entry.url,
                ocrText: entry.ocrText
            )
        }

        // Capture caret-localized screenshot in the background, then send.
        let capturedPID = appPID
        let capturedAppName = appName
        let capturedWindowTitle = windowTitle
        let capturedRole = focusedRole
        let capturedValue = focusedValue
        let capturedLabel = focusedLabel

        Task {
            let caretOCR = await ScreenshotCapture.captureAroundFocusedElement(
                pid: capturedPID
            )

            let seq = self.worker.nextSeq()
            let trigger = IPCMessage.PasteTrigger(
                seq: seq,
                timestamp: Date().timeIntervalSince1970,
                appName: capturedAppName,
                appPID: capturedPID,
                windowTitle: capturedWindowTitle,
                focusedRole: capturedRole,
                focusedValue: capturedValue,
                focusedLabel: capturedLabel,
                caretContextOCR: caretOCR,
                recentCaptures: captures
            )
            self.lastTriggerPID = capturedPID
            self.worker.send(trigger)
            self.onStatus?(.triggerSent(seq: seq, appName: "paste:\(capturedAppName) role=\(capturedRole ?? "nil") label=\(capturedLabel ?? "nil") caretOCR=\(caretOCR?.prefix(80) ?? "nil")"))
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
