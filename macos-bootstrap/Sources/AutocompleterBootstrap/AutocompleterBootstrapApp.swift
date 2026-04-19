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
        // Show diagnostics immediately on first launch so the user knows
        // where to go.
        showDiagnostics()
    }

    func applicationWillTerminate(_ notification: Notification) {
        workerCoordinator?.stop()
        chromiumWaker?.stop()
        trajectoryStack?.stop()
        intentController?.stop()
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

    private func startTrajectoryStack() {
        let stack = TrajectoryStack()
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
        workerCoordinator = coord
        coord.start()
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
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(
            title: "Quit",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        ))
        for mi in menu.items where mi.action == #selector(showDiagnostics)
            || mi.action == #selector(showTrajectoryInspector) {
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
