import SwiftUI
import AppKit

@main
struct AutocompleterBootstrapApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        // A hidden WindowGroup wouldn't normally be needed, but SwiftUI's App
        // protocol requires at least one Scene. The actual diagnostics window
        // is managed via AppDelegate.showDiagnostics() so we can open it from
        // the menu bar item reliably.
        Settings {
            EmptyView()
        }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var diagnosticsWindow: NSWindow?
    private var trajectoryInspectorWindow: NSWindow?
    private var trajectoryInspectorModel: TrajectoryInspectorModel?
    /// Phase A substrate: subprocess worker + hotkey → inject round-trip.
    /// Lives alongside the diagnostic panel — diagnostics still work.
    private(set) var workerCoordinator: WorkerCoordinator?
    /// Keeps Chromium-family apps' AX engines awake on macOS 15. See
    /// ChromiumWaker.swift for the why.
    private var chromiumWaker: ChromiumWaker?
    /// Experimental trajectory capture — ring buffer of window / focus
    /// events feeding the intent-probe spike. Lives on the
    /// `trajectory-intent-probe` branch; not part of the core Phase A path.
    private(set) var trajectoryStack: TrajectoryStack?
    /// Experimental floating panel that shows the most recent intent guess.
    private var intentOverlay: IntentOverlay?
    /// Glue between stack → worker query → overlay.
    private var intentController: TrajectoryIntentController?
    /// Screenshot + OCR capture pipeline.
    private var screenshotCapture: ScreenshotCapture?
    /// Paste hotkey (ctrl+shift+v) — separate from autocomplete hotkey.
    private var pasteHotkey: HotkeyTap?
    /// Explicit capture hotkey (ctrl+shift+c) — "remember what I'm looking at."
    private var captureHotkey: HotkeyTap?
    private let captureStore = CaptureStore()
    private var captureInspectorWindow: NSWindow?
    /// Published state for the diagnostics UI to observe.
    let workerStatusModel = WorkerStatusModel()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Behave as a menu-bar accessory even if Info.plist's LSUIElement
        // was missed for some reason (e.g. running the raw binary during dev).
        NSApp.setActivationPolicy(.accessory)

        installStatusItem()
        startWorkerIfConfigured()
        startChromiumWaker()
        startTrajectoryStack()
        startTrajectoryIntent()
        startScreenshotCapture()
        // Show diagnostics immediately on first launch so the user knows
        // where to go.
        showDiagnostics()
    }

    func applicationWillTerminate(_ notification: Notification) {
        workerCoordinator?.stop()
        chromiumWaker?.stop()
        trajectoryStack?.stop()
        intentController?.stop()
        screenshotCapture?.stop()
        pasteHotkey?.stop()
        captureHotkey?.stop()
    }

    private func startTrajectoryIntent() {
        guard let coord = workerCoordinator, let stack = trajectoryStack else {
            return
        }
        let overlay = IntentOverlay()
        let controller = TrajectoryIntentController(
            coordinator: coord,
            stack: stack,
            overlay: overlay
        )
        controller.onLog = { [weak self] line in
            self?.workerStatusModel.append(line)
        }
        intentOverlay = overlay
        intentController = controller
        controller.start()
    }

    private func startScreenshotCapture() {
        let capture = ScreenshotCapture(store: captureStore)
        capture.onLog = { [weak self] line in
            self?.workerStatusModel.append(line)
        }
        capture.start()
        screenshotCapture = capture

        // Piggyback on TrajectoryStack's AXObserver for title/window changes.
        trajectoryStack?.addObserver { [weak capture] change in
            switch change {
            case .appended(let snap):
                if snap.trigger == "title_change" || snap.trigger == "focus_change" {
                    capture?.onAXEvent(trigger: snap.trigger)
                }
            case .updated:
                break
            }
        }
    }

    private func startTrajectoryStack() {
        let stack = TrajectoryStack()
        stack.onProfile = { [weak self] line in
            self?.workerStatusModel.append(line)
        }
        stack.addObserver { [weak self] change in
            switch change {
            case .appended(let snap):
                let title = snap.windowTitle ?? "(no title)"
                let role = snap.focusedRole ?? "-"
                self?.workerStatusModel.append(
                    "traj[\(snap.trigger)] \(snap.appName) — \(title) role=\(role)"
                )
            case .updated(let snap):
                let dwell = Int(snap.lastSeen - snap.timestamp)
                self?.workerStatusModel.append(
                    "traj[dwell] \(snap.appName) — \(dwell)s"
                )
            }
        }
        stack.start()
        trajectoryStack = stack
    }

    private func startChromiumWaker() {
        let waker = ChromiumWaker()
        waker.onEvent = { [weak self] line in
            self?.workerStatusModel.append(line)
        }
        waker.start()
        chromiumWaker = waker
    }

    private func startWorkerIfConfigured() {
        // Phase A: always start the worker. If the interpreter path is wrong,
        // the subprocess will exit immediately and the diagnostics UI will
        // surface the crash log. Skipping launch behind a flag would hide the
        // error at exactly the moment we need to see it.
        let config = WorkerProcess.Config.dev()
        let hotkey: HotkeyTap.Hotkey
        do {
            hotkey = try HotkeyTap.Hotkey.parse("ctrl+space")
        } catch {
            workerStatusModel.append("Hotkey parse failed: \(error)")
            return
        }

        let coord = WorkerCoordinator(workerConfig: config, hotkey: hotkey)
        coord.onStatus = { [weak self] status in
            self?.workerStatusModel.apply(status)
        }
        coord.captureStore = captureStore
        coord.onPasteSuggestions = { [weak self] paste in
            guard let self else { return }
            if paste.suggestions.isEmpty {
                self.workerStatusModel.append("paste: no suggestions returned")
                self.intentOverlay?.updateStatus("(no paste suggestions)")
                return
            }
            // Show all suggestions in the overlay as a numbered list.
            let display = paste.suggestions
                .sorted { $0.rank < $1.rank }
                .enumerated()
                .map { "\($0.offset + 1). \($0.element.text)" }
                .joined(separator: "\n")
            self.intentOverlay?.update(
                sentence: display,
                latencyMs: nil,
                model: "paste"
            )
            // Log each suggestion.
            for s in paste.suggestions {
                self.workerStatusModel.append(
                    "paste[\(s.rank)]: \(s.text)"
                )
            }
            // Auto-inject the top-ranked suggestion for POC.
            if let top = paste.suggestions.sorted(by: { $0.rank < $1.rank }).first {
                let outcome = Injector.inject(
                    text: top.text,
                    strategy: .clipboard,
                    requireMatchingPID: coord.lastTriggerPID
                )
                self.workerStatusModel.append(
                    outcome.ok
                        ? "paste: injected '\(top.text.prefix(60))'"
                        : "paste: inject failed — \(outcome.reason ?? "unknown")"
                )
            }
        }
        workerCoordinator = coord
        coord.start()

        // Paste hotkey — separate CGEvent tap.
        do {
            let pasteKey = try HotkeyTap.Hotkey.parse("ctrl+shift+v")
            let tap = HotkeyTap(hotkey: pasteKey)
            tap.onFire = { [weak coord] in
                coord?.handlePasteHotkey()
            }
            tap.start()
            pasteHotkey = tap
        } catch {
            workerStatusModel.append("Paste hotkey parse failed: \(error)")
        }

        // Capture hotkey — "remember what I'm looking at."
        do {
            let capKey = try HotkeyTap.Hotkey.parse("ctrl+shift+c")
            let tap = HotkeyTap(hotkey: capKey)
            tap.onFire = { [weak self] in
                self?.screenshotCapture?.captureNow()
            }
            tap.start()
            captureHotkey = tap
        } catch {
            workerStatusModel.append("Capture hotkey parse failed: \(error)")
        }
    }

    private func installStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = item.button {
            button.title = "AC"
            button.toolTip = "AutocompleterBootstrap diagnostics"
        }
        let menu = NSMenu()
        menu.addItem(NSMenuItem(
            title: "Open Diagnostics…",
            action: #selector(showDiagnostics),
            keyEquivalent: ""
        ))
        menu.addItem(NSMenuItem(
            title: "Open Trajectory Inspector…",
            action: #selector(showTrajectoryInspector),
            keyEquivalent: ""
        ))
        menu.addItem(NSMenuItem(
            title: "Open Capture Inspector…",
            action: #selector(showCaptureInspector),
            keyEquivalent: ""
        ))
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(
            title: "Quit",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        ))
        for mi in menu.items where mi.action == #selector(showDiagnostics)
            || mi.action == #selector(showTrajectoryInspector)
            || mi.action == #selector(showCaptureInspector) {
            mi.target = self
        }
        item.menu = menu
        statusItem = item
    }

    @objc func showTrajectoryInspector() {
        if let window = trajectoryInspectorWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        if trajectoryInspectorModel == nil, let stack = trajectoryStack {
            trajectoryInspectorModel = TrajectoryInspectorModel(
                stack: stack,
                intentController: intentController
            )
        }
        guard let model = trajectoryInspectorModel else { return }

        let hosting = NSHostingController(
            rootView: TrajectoryInspectorView(model: model)
        )
        let window = NSWindow(contentViewController: hosting)
        window.title = "Trajectory Inspector"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.isReleasedWhenClosed = false
        window.setContentSize(NSSize(width: 780, height: 600))
        window.minSize = NSSize(width: 600, height: 400)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        trajectoryInspectorWindow = window
    }

    @objc func showCaptureInspector() {
        if let window = captureInspectorWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let hosting = NSHostingController(
            rootView: CaptureInspectorView(store: captureStore)
        )
        let window = NSWindow(contentViewController: hosting)
        window.title = "Capture Inspector"
        window.styleMask = [.titled, .closable, .miniaturizable, .resizable]
        window.isReleasedWhenClosed = false
        window.setContentSize(NSSize(width: 900, height: 650))
        window.minSize = NSSize(width: 700, height: 450)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        captureInspectorWindow = window
    }

    @objc func showDiagnostics() {
        if let window = diagnosticsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let hosting = NSHostingController(rootView: DiagnosticsView(workerStatus: workerStatusModel))
        let window = NSWindow(contentViewController: hosting)
        window.title = "AutocompleterBootstrap Diagnostics"
        window.styleMask = [.titled, .closable, .miniaturizable]
        // Keep the window alive on close so its @StateObject-backed state
        // survives between openings.
        window.isReleasedWhenClosed = false
        window.center()
        window.makeKeyAndOrderFront(nil)

        NSApp.activate(ignoringOtherApps: true)
        diagnosticsWindow = window
    }
}
