import ApplicationServices
import AppKit
import Foundation

/// Keeps Chromium-based apps' accessibility engines awake.
///
/// Context: on macOS 15, Chromium apps (Chrome, Edge, Electron, etc.) keep
/// their AX engine dormant until an AX client with a live observer attaches.
/// Passive AX reads don't wake them — we literally need to hold an
/// `AXObserver` subscribed to at least one notification on the app element.
/// The callback itself can be a no-op; it's the *presence* of the observer
/// that flips Chromium from "sleep" into "normal AX."
///
/// Empirically verified via v0.1 diagnostic: Cmd+F5 (VoiceOver on/off)
/// woke Chrome on a friend's macOS 15 machine when our v0.1 app alone could
/// not read `AXFocusedUIElement` on a focused tab. See project memory:
/// `project_chromium_lazy_ax.md`.
///
/// Behavior:
///   * On `start()`: walk running apps once, attach observers to every
///     Chromium-family app we find.
///   * Listen to `NSWorkspace.didLaunchApplicationNotification` so new
///     Chromium launches get an observer attached immediately.
///   * Listen to `didTerminateApplicationNotification` to release observers
///     cleanly and avoid leaking AX run-loop sources.
@MainActor
final class ChromiumWaker {
    /// Bundle IDs we consider Chromium-family. Incomplete on purpose — we
    /// extend it when a user reports a browser this list misses.
    static let knownBundles: Set<String> = [
        "com.google.Chrome",
        "com.google.Chrome.beta",
        "com.google.Chrome.dev",
        "com.google.Chrome.canary",
        "com.microsoft.edgemac",
        "com.microsoft.edgemac.beta",
        "com.microsoft.edgemac.dev",
        "com.microsoft.edgemac.canary",
        "com.brave.Browser",
        "com.brave.Browser.beta",
        "com.brave.Browser.dev",
        "com.brave.Browser.nightly",
        "com.vivaldi.Vivaldi",
        "com.operasoftware.Opera",
        "company.thebrowser.Browser",  // Arc
        "com.google.Chrome.for.Testing",
    ]

    /// Per-PID registry of retained observers so we can release them when
    /// the app quits (otherwise `AXObserverGetRunLoopSource` keeps feeding
    /// a zombie runloop).
    private var observers: [pid_t: AXObserver] = [:]

    private var launchToken: NSObjectProtocol?
    private var terminateToken: NSObjectProtocol?

    /// Optional hook for the status model. Fires on every wake/attach event.
    var onEvent: ((String) -> Void)?

    // MARK: - Lifecycle

    func start() {
        attachAll()
        launchToken = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didLaunchApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            else { return }
            Task { @MainActor in
                self?.attachIfChromium(app)
            }
        }
        terminateToken = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didTerminateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] note in
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            else { return }
            Task { @MainActor in
                self?.release(pid: app.processIdentifier)
            }
        }
    }

    func stop() {
        if let token = launchToken {
            NSWorkspace.shared.notificationCenter.removeObserver(token)
        }
        if let token = terminateToken {
            NSWorkspace.shared.notificationCenter.removeObserver(token)
        }
        for pid in observers.keys {
            release(pid: pid)
        }
    }

    // MARK: - Attach / release

    private func attachAll() {
        for app in NSWorkspace.shared.runningApplications {
            attachIfChromium(app)
        }
    }

    private func attachIfChromium(_ app: NSRunningApplication) {
        guard let bid = app.bundleIdentifier, Self.knownBundles.contains(bid) else { return }
        attach(pid: app.processIdentifier, label: bid)
    }

    private func attach(pid: pid_t, label: String) {
        guard observers[pid] == nil else { return }

        var observer: AXObserver?
        let createErr = AXObserverCreate(pid, chromiumWakerCallback, &observer)
        guard createErr == .success, let created = observer else {
            onEvent?("ChromiumWaker: observer create failed pid=\(pid) bundle=\(label) err=\(createErr.rawValue)")
            return
        }

        let appElement = AXUIElementCreateApplication(pid)
        // Subscribing to focused-window-changed is cheap, high-signal, and
        // keeps the AX engine warm across tab / window changes.
        let notification = kAXFocusedWindowChangedNotification as CFString
        let addErr = AXObserverAddNotification(
            created,
            appElement,
            notification,
            nil
        )
        guard addErr == .success else {
            onEvent?("ChromiumWaker: addNotification failed pid=\(pid) err=\(addErr.rawValue)")
            return
        }

        CFRunLoopAddSource(
            CFRunLoopGetMain(),
            AXObserverGetRunLoopSource(created),
            .defaultMode
        )

        observers[pid] = created
        onEvent?("ChromiumWaker: attached pid=\(pid) bundle=\(label)")
    }

    private func release(pid: pid_t) {
        guard let observer = observers.removeValue(forKey: pid) else { return }
        CFRunLoopRemoveSource(
            CFRunLoopGetMain(),
            AXObserverGetRunLoopSource(observer),
            .defaultMode
        )
        onEvent?("ChromiumWaker: released pid=\(pid)")
    }
}

/// C-compatible callback. We don't need to do anything — Chromium waking up
/// is a side effect of having an active observer, not of the callback
/// firing. We keep the signature correct so the AX machinery is happy.
private func chromiumWakerCallback(
    observer: AXObserver,
    element: AXUIElement,
    notification: CFString,
    userInfo: UnsafeMutableRawPointer?
) {
    // Intentionally empty.
}
