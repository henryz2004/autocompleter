import ApplicationServices
import AppKit
import Foundation

/// Converts a live AX snapshot plus frontmost-app metadata into an
/// `IPCMessage.Trigger` payload to hand to the Python worker.
///
/// Phase A only needs enough of the payload for Python to compute the
/// reverse-echo response: `focused_element.before_cursor`. The rest of
/// the shape (subtree, conversation_turns, cross_app_snapshots) is
/// declared so Phase B can fill it in without a protocol change, but
/// stays empty here.
enum TriggerBuilder {
    @MainActor
    static func build(seq: Int, triggerType: String = "manual") -> IPCMessage.Trigger? {
        let snap: FocusedElementSnapshot
        do {
            snap = try AXProbe.snapshotFocusedElement()
        } catch {
            return nil
        }

        let insertion = snap.insertionPoint ?? snap.value.count
        let safeInsertion = max(0, min(insertion, snap.value.count))
        let beforeCursor = String(snap.value.prefix(safeInsertion))
        let afterCursor = String(snap.value.dropFirst(safeInsertion))

        let frontApp = NSWorkspace.shared.frontmostApplication
        let windowTitle = frontmostWindowTitle(pid: snap.appPID)
        let sourceURL = frontmostBrowserURL(appBundleID: frontApp?.bundleIdentifier)

        let focused = IPCMessage.FocusedElement(
            role: snap.role,
            value: snap.value,
            insertionPoint: snap.insertionPoint,
            selectionLength: snap.selectionLength,
            placeholderDetected: snap.hasPlaceholder,
            beforeCursor: beforeCursor,
            afterCursor: afterCursor,
            position: nil,
            size: nil
        )

        return IPCMessage.Trigger(
            seq: seq,
            type: "trigger",
            triggerType: triggerType,
            timestamp: Date().timeIntervalSince1970,
            appName: snap.appName,
            appPID: snap.appPID,
            windowTitle: windowTitle,
            sourceURL: sourceURL,
            focusedElement: focused,
            caretPosition: nil,
            axSubtree: IPCMessage.AXSubtree(bottomUp: "", topDown: ""),
            conversationTurns: [],
            crossAppSnapshots: []
        )
    }

    // MARK: - Helpers

    /// Best-effort window title lookup. Not all apps expose a meaningful
    /// title via AX; we degrade to nil rather than a synthetic string.
    @MainActor
    private static func frontmostWindowTitle(pid: pid_t) -> String? {
        let app = AXUIElementCreateApplication(pid)
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            app,
            kAXFocusedWindowAttribute as CFString,
            &ref
        )
        guard err == .success,
              let window = ref,
              CFGetTypeID(window) == AXUIElementGetTypeID()
        else { return nil }

        let win = window as! AXUIElement
        var titleRef: CFTypeRef?
        let titleErr = AXUIElementCopyAttributeValue(
            win,
            kAXTitleAttribute as CFString,
            &titleRef
        )
        guard titleErr == .success, let title = titleRef as? String, !title.isEmpty else {
            return nil
        }
        return title
    }

    /// Browser-URL pull for Chromium/Safari. Phase A only reads the AX
    /// `AXURL` field or the `AXDocument` attribute when present. If the
    /// lazy-AX fix lands in a later iteration, this keeps working. For
    /// non-browser apps we return nil.
    @MainActor
    private static func frontmostBrowserURL(appBundleID: String?) -> String? {
        guard let bid = appBundleID else { return nil }
        let browserBundles: Set<String> = [
            "com.apple.Safari",
            "com.google.Chrome",
            "com.google.Chrome.beta",
            "com.google.Chrome.dev",
            "com.google.Chrome.canary",
            "com.microsoft.edgemac",
            "com.brave.Browser",
            "company.thebrowser.Browser",
            "com.vivaldi.Vivaldi",
            "com.operasoftware.Opera",
            "org.mozilla.firefox",
        ]
        guard browserBundles.contains(bid) else { return nil }

        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        var winRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            axApp,
            kAXFocusedWindowAttribute as CFString,
            &winRef
        ) == .success,
              let winUnwrapped = winRef,
              CFGetTypeID(winUnwrapped) == AXUIElementGetTypeID()
        else { return nil }
        let window = winUnwrapped as! AXUIElement

        // Try AXDocument — Safari / Chrome expose the current URL here.
        var docRef: CFTypeRef?
        if AXUIElementCopyAttributeValue(
            window,
            "AXDocument" as CFString,
            &docRef
        ) == .success, let s = docRef as? String, !s.isEmpty {
            return s
        }
        return nil
    }
}
