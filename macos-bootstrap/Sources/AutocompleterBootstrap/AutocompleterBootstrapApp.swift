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

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var diagnosticsWindow: NSWindow?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Behave as a menu-bar accessory even if Info.plist's LSUIElement
        // was missed for some reason (e.g. running the raw binary during dev).
        NSApp.setActivationPolicy(.accessory)

        installStatusItem()
        // Show diagnostics immediately on first launch so the user knows
        // where to go.
        showDiagnostics()
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
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(
            title: "Quit",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q"
        ))
        for mi in menu.items where mi.action == #selector(showDiagnostics) {
            mi.target = self
        }
        item.menu = menu
        statusItem = item
    }

    @objc func showDiagnostics() {
        if let window = diagnosticsWindow {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let hosting = NSHostingController(rootView: DiagnosticsView())
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
