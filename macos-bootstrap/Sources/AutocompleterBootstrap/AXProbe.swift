import ApplicationServices
import AppKit
import Foundation

/// Minimal read of the currently focused text element.
///
/// Mirrors the shape of `FocusedElement` in `autocompleter/input_observer.py`
/// — only the fields needed by the v0.1 diagnostic checks. When the Python
/// pipeline later moves behind this Swift binary, more attributes can be
/// added here.
struct FocusedElementSnapshot {
    var appName: String
    var appPID: pid_t
    var role: String
    var value: String
    var valueLength: Int
    var insertionPoint: Int?
    var selectionLength: Int
    var hasPlaceholder: Bool
    /// True when `value` was recovered by walking child AXStaticText nodes
    /// rather than read directly from AXValue. Useful for diagnosing
    /// Electron/Chromium apps where top-level AXValue is blank.
    var valueFromChildText: Bool
}

/// JSON-encodable, **redacted** view of a snapshot. The raw `value` is
/// never copied to the pasteboard because the diagnostic report is meant
/// to be sent back to Henry by friends, and that field could contain a
/// password manager entry or a private message.
struct RedactedFocusedElementSnapshot: Codable {
    var appName: String
    var appPID: pid_t
    var role: String
    var valueLength: Int
    var valuePreviewFirstChar: String   // first char of value, or "" — just enough to spot "typed vs empty"
    var insertionPoint: Int?
    var selectionLength: Int
    var hasPlaceholder: Bool
    var valueFromChildText: Bool

    init(from snap: FocusedElementSnapshot) {
        self.appName = snap.appName
        self.appPID = snap.appPID
        self.role = snap.role
        self.valueLength = snap.valueLength
        self.valuePreviewFirstChar = snap.value.first.map { String($0) } ?? ""
        self.insertionPoint = snap.insertionPoint
        self.selectionLength = snap.selectionLength
        self.hasPlaceholder = snap.hasPlaceholder
        self.valueFromChildText = snap.valueFromChildText
    }
}

enum AXProbeError: Error, CustomStringConvertible {
    case notTrusted
    case noFocusedElement
    case unexpectedType(String)

    var description: String {
        switch self {
        case .notTrusted:
            return "Process is not trusted for Accessibility. Grant it in System Settings → Privacy & Security → Accessibility."
        case .noFocusedElement:
            return "AXFocusedUIElement returned nil. Is a text field focused in the frontmost app?"
        case let .unexpectedType(what):
            return "AX API returned an unexpected CoreFoundation type for \(what)."
        }
    }
}

enum AXProbe {
    /// Returns whether the current process is trusted by the Accessibility
    /// system. When `prompt` is true and the process is not yet trusted,
    /// macOS displays the standard permission prompt.
    static func isProcessTrusted(prompt: Bool) -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let options = [key: prompt] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    /// Read the currently focused UI element from the system-wide AXUIElement
    /// and extract the fields needed for diagnostics.
    static func snapshotFocusedElement() throws -> FocusedElementSnapshot {
        guard isProcessTrusted(prompt: false) else {
            throw AXProbeError.notTrusted
        }

        let systemWide = AXUIElementCreateSystemWide()

        var focusedRef: CFTypeRef?
        let focusedErr = AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedRef
        )
        guard focusedErr == .success, let focusedUnwrapped = focusedRef else {
            throw AXProbeError.noFocusedElement
        }
        guard CFGetTypeID(focusedUnwrapped) == AXUIElementGetTypeID() else {
            throw AXProbeError.unexpectedType("AXFocusedUIElement")
        }
        let focused = focusedUnwrapped as! AXUIElement

        let role = copyStringAttribute(focused, kAXRoleAttribute as CFString) ?? ""
        var value = copyStringAttribute(focused, kAXValueAttribute as CFString) ?? ""
        var valueFromChildText = false

        // Chromium/Electron often return blank AXValue for contenteditable
        // divs that show up as AXTextArea, AXWebArea, or AXGroup. Mirror
        // input_observer.py:147-152 and fall back to collecting child
        // AXStaticText so the e2e diagnostic doesn't false-negative
        // on Slack/Chrome/Gemini/etc.
        let childTextRoles: Set<String> = ["AXTextArea", "AXWebArea", "AXGroup"]
        if value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           childTextRoles.contains(role) {
            let collected = collectChildText(focused)
            if !collected.isEmpty {
                value = collected
                valueFromChildText = true
            }
        }

        let placeholder = copyStringAttribute(focused, kAXPlaceholderValueAttribute as CFString)

        let (insertionPoint, selectionLength) = readSelectedTextRange(focused)

        var pid: pid_t = 0
        AXUIElementGetPid(focused, &pid)
        let appName = NSRunningApplication(processIdentifier: pid)?.localizedName ?? "Unknown"

        return FocusedElementSnapshot(
            appName: appName,
            appPID: pid,
            role: role,
            value: value,
            valueLength: value.count,
            insertionPoint: insertionPoint,
            selectionLength: selectionLength,
            hasPlaceholder: placeholder != nil,
            valueFromChildText: valueFromChildText
        )
    }

    // MARK: - helpers

    private static func copyStringAttribute(_ element: AXUIElement, _ attribute: CFString) -> String? {
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attribute, &ref)
        guard err == .success else { return nil }
        return ref as? String
    }

    /// Extract (location, length) from AXSelectedTextRange. Returns
    /// (nil, 0) when the attribute is missing or unreadable.
    private static func readSelectedTextRange(_ element: AXUIElement) -> (Int?, Int) {
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &ref
        )
        guard err == .success, let ref else { return (nil, 0) }
        guard CFGetTypeID(ref) == AXValueGetTypeID() else { return (nil, 0) }

        let axValue = ref as! AXValue
        guard AXValueGetType(axValue) == .cfRange else { return (nil, 0) }

        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(axValue, .cfRange, &range) else { return (nil, 0) }
        return (range.location, range.length)
    }

    /// Walk AX children up to `maxDepth` levels and concatenate AXStaticText
    /// values. Mirrors the intent of `_collect_child_text` in
    /// `autocompleter/conversation_extractors.py:78-137` — a minimal port
    /// that handles the Electron "empty AXValue" case without dragging in
    /// the full speaker-separation heuristics.
    private static func collectChildText(
        _ element: AXUIElement,
        maxDepth: Int = 5,
        maxChars: Int = 2000
    ) -> String {
        var buffer = ""
        walkForText(element, depth: 0, maxDepth: maxDepth, maxChars: maxChars, buffer: &buffer)
        return buffer
    }

    private static func walkForText(
        _ element: AXUIElement,
        depth: Int,
        maxDepth: Int,
        maxChars: Int,
        buffer: inout String
    ) {
        if depth > maxDepth || buffer.count >= maxChars { return }

        let role = copyStringAttribute(element, kAXRoleAttribute as CFString) ?? ""

        // Skip obvious UI chrome so we don't pull in button labels and
        // toolbar strings that aren't part of the user's text.
        let skipRoles: Set<String> = [
            "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton",
            "AXMenuButton", "AXImage", "AXToolbar", "AXScrollBar",
        ]
        if skipRoles.contains(role) { return }

        if role == "AXStaticText" {
            if let text = copyStringAttribute(element, kAXValueAttribute as CFString),
               !text.isEmpty {
                if !buffer.isEmpty { buffer.append("\n") }
                buffer.append(text)
                if buffer.count >= maxChars { return }
            }
            return
        }

        var childrenRef: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            element,
            kAXChildrenAttribute as CFString,
            &childrenRef
        )
        guard err == .success, let arr = childrenRef as? [AXUIElement] else { return }

        for child in arr {
            walkForText(child, depth: depth + 1, maxDepth: maxDepth, maxChars: maxChars, buffer: &buffer)
            if buffer.count >= maxChars { return }
        }
    }
}
