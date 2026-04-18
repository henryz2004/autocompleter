import Foundation

/// Owns the Python subprocess: spawns it, feeds JSON-line messages to stdin,
/// reads JSON-line replies from stdout, and tees stderr into a log.
///
/// Lifecycle guarantees (Phase A):
///   * `start()` spawns the worker. If the binary is missing we fail visibly.
///   * If the worker exits, `start()` will re-spawn it with 1s / 2s / 4s
///     backoff, capped at 3 retries per rolling 60-second window.
///   * `stop()` closes stdin (Python's loop treats EOF as clean shutdown) and
///     waits briefly for the process to exit.
///
/// Threading:
///   * All callbacks (`onInbound`, `onExit`, `onLog`) fire on the main
///     DispatchQueue so the UI/AX code can consume them without locking.
///   * The stdout reader runs on a background `DispatchQueue`.
final class WorkerProcess {
    struct Config {
        /// Absolute path to the Python interpreter. In Phase A we use the
        /// repo venv via `AUTOCOMPLETER_WORKER_BIN`, falling back to
        /// `/usr/bin/env python3`.
        var interpreter: String
        /// Args passed to the interpreter. Phase A: `["-m", "autocompleter", "--worker-mode"]`.
        var args: [String]
        /// Working directory. Defaults to the repo root so the package is importable.
        var workingDirectory: String
        /// Extra environment overlays on top of the parent process.
        var environment: [String: String]
    }

    /// Called on main queue for every decoded inbound message.
    var onInbound: ((Any) -> Void)?
    /// Called on main queue right after the child is spawned with its PID.
    var onLaunched: ((Int32) -> Void)?
    /// Called on main queue when the child exits. Value is the raw exit
    /// status (0 means clean). Restart logic runs before this fires.
    var onExit: ((Int32, Bool) -> Void)?
    /// Called on main queue for each stderr line (diagnostic tail).
    var onStderr: ((String) -> Void)?
    /// Called on main queue when `start()` gives up after exceeding the
    /// retry budget. The parameter is the last exit code observed.
    var onGaveUp: ((Int32) -> Void)?

    private let config: Config
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?
    private var stdoutBuffer = Data()
    private var stderrBuffer = Data()
    private var intentionalStop = false
    /// Serializes stdin writes so MainActor `send` calls don't block while the
    /// pipe drains. Writes are still ordered (single queue, single handle).
    private let writeQueue = DispatchQueue(label: "autocompleter.worker.stdin")

    /// Timestamps (seconds since ref) of the last few crashes. Used to cap
    /// restart attempts.
    private var recentCrashes: [TimeInterval] = []
    private let maxRetriesPerWindow = 3
    private let retryWindow: TimeInterval = 60

    private(set) var restartCount: Int = 0
    private(set) var currentPID: Int32 = 0

    private var sendSeq: Int = 0
    private let seqQueue = DispatchQueue(label: "autocompleter.worker.seq")

    init(config: Config) {
        self.config = config
    }

    /// Generate a monotonic sequence number for outbound messages.
    func nextSeq() -> Int {
        seqQueue.sync {
            sendSeq += 1
            return sendSeq
        }
    }

    // MARK: - Public lifecycle

    func start() {
        guard process == nil else { return }
        intentionalStop = false
        spawn()
    }

    func stop() {
        intentionalStop = true
        // Closing stdin tells the Python loop to exit cleanly on EOF.
        try? stdinPipe?.fileHandleForWriting.close()
        guard let p = process, p.isRunning else { return }

        // Poll for up to ~1.5s; if still running, escalate SIGTERM → SIGKILL.
        // This avoids hanging the main thread forever if the Python child is
        // wedged on a blocked I/O or a stuck handler.
        let deadline = Date().addingTimeInterval(1.5)
        while p.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if p.isRunning {
            emitStderr("[worker] stop: SIGTERM pid=\(p.processIdentifier)")
            p.terminate()
            let hardDeadline = Date().addingTimeInterval(1.0)
            while p.isRunning && Date() < hardDeadline {
                Thread.sleep(forTimeInterval: 0.05)
            }
            if p.isRunning {
                emitStderr("[worker] stop: SIGKILL pid=\(p.processIdentifier)")
                kill(p.processIdentifier, SIGKILL)
            }
        }
    }

    /// Encode and send a message. Non-throwing; errors are surfaced via
    /// stderr tail so the caller doesn't have to handle them inline. The
    /// actual `write` happens on a background queue so a full pipe buffer
    /// can't block the MainActor caller.
    func send<T: Encodable>(_ msg: T) {
        guard let handle = stdinPipe?.fileHandleForWriting else {
            emitStderr("[worker] send ignored: worker not running")
            return
        }
        let data: Data
        do {
            data = try IPCCodec.encodeLine(msg)
        } catch {
            emitStderr("[worker] encode failed: \(error)")
            return
        }
        writeQueue.async { [weak self] in
            do {
                try handle.write(contentsOf: data)
            } catch {
                self?.emitStderr("[worker] send failed: \(error)")
            }
        }
    }

    // MARK: - Spawn

    private func spawn() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: config.interpreter)
        p.arguments = config.args
        p.currentDirectoryURL = URL(fileURLWithPath: config.workingDirectory)

        var env = ProcessInfo.processInfo.environment
        for (k, v) in config.environment { env[k] = v }
        // Ensure Python doesn't buffer stdout — we need line-by-line flushes.
        env["PYTHONUNBUFFERED"] = "1"
        p.environment = env

        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe

        self.stdinPipe = inPipe
        self.stdoutPipe = outPipe
        self.stderrPipe = errPipe
        self.stdoutBuffer = Data()
        self.stderrBuffer = Data()

        // Install stdout reader on a background queue.
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                // EOF on stdout means the child is done. The termination
                // handler will pick it up.
                handle.readabilityHandler = nil
                return
            }
            self?.consumeStdout(chunk)
        }

        errPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            if chunk.isEmpty {
                handle.readabilityHandler = nil
                return
            }
            self?.consumeStderr(chunk)
        }

        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            DispatchQueue.main.async {
                self?.handleTermination(status: status)
            }
        }

        do {
            try p.run()
        } catch {
            emitStderr("[worker] failed to launch \(config.interpreter): \(error)")
            DispatchQueue.main.async { [weak self] in
                self?.onGaveUp?(-1)
            }
            return
        }

        self.process = p
        self.currentPID = p.processIdentifier
        emitStderr("[worker] launched pid=\(p.processIdentifier)")
        let pid = p.processIdentifier
        DispatchQueue.main.async { [weak self] in
            self?.onLaunched?(pid)
        }
    }

    // MARK: - Stream handling

    private func consumeStdout(_ chunk: Data) {
        stdoutBuffer.append(chunk)
        while let newlineIdx = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = stdoutBuffer.subdata(in: stdoutBuffer.startIndex..<newlineIdx)
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newlineIdx)
            guard !lineData.isEmpty else { continue }
            do {
                let decoded = try IPCCodec.decodeInbound(lineData)
                DispatchQueue.main.async { [weak self] in
                    self?.onInbound?(decoded)
                }
            } catch {
                let preview = String(data: lineData.prefix(200), encoding: .utf8) ?? "<non-utf8>"
                emitStderr("[worker] decode error: \(error). line=\(preview)")
            }
        }
    }

    private func consumeStderr(_ chunk: Data) {
        stderrBuffer.append(chunk)
        while let newlineIdx = stderrBuffer.firstIndex(of: 0x0A) {
            let lineData = stderrBuffer.subdata(in: stderrBuffer.startIndex..<newlineIdx)
            stderrBuffer.removeSubrange(stderrBuffer.startIndex...newlineIdx)
            if let line = String(data: lineData, encoding: .utf8) {
                DispatchQueue.main.async { [weak self] in
                    self?.onStderr?(line)
                }
            }
        }
    }

    private func emitStderr(_ text: String) {
        DispatchQueue.main.async { [weak self] in
            self?.onStderr?(text)
        }
    }

    // MARK: - Termination

    private func handleTermination(status: Int32) {
        let wasRunning = (process != nil)
        process = nil
        currentPID = 0
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil

        onExit?(status, intentionalStop)

        guard !intentionalStop, wasRunning else { return }

        // Record crash timestamp and prune the window.
        let now = Date().timeIntervalSinceReferenceDate
        recentCrashes.append(now)
        recentCrashes = recentCrashes.filter { now - $0 <= retryWindow }

        guard recentCrashes.count <= maxRetriesPerWindow else {
            emitStderr("[worker] giving up after \(recentCrashes.count) crashes in \(Int(retryWindow))s")
            onGaveUp?(status)
            return
        }

        // Exponential backoff: 1s, 2s, 4s.
        let attempt = recentCrashes.count
        let delay = pow(2.0, Double(attempt - 1))
        emitStderr("[worker] restart #\(attempt) in \(Int(delay))s (exit=\(status))")
        restartCount += 1
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self = self else { return }
            // If stop() ran during the backoff, don't re-spawn. Also skip if
            // some other code path already brought the worker back.
            guard !self.intentionalStop, self.process == nil else { return }
            self.spawn()
        }
    }
}
