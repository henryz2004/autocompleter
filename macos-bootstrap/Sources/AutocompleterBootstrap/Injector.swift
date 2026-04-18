import ApplicationServices
import AppKit
import Foundation

/// Applies an `inject` command from the Python worker to the currently
/// focused UI element. Phase A supports three strategies in preference
/// order: AX value set → clipboard paste → synthetic keystrokes. `cdp`
/// is a no-op that tells Python to keep owning that path for now.
///
/// Mirrors the strategy-ordering of `autocompleter/text_injector.py`.
/// Deliberately *not* generalized — Phase A hasn't proven what injection
/// looks like across different apps, so we keep the code path linear
/// and easy to trace rather than abstracting prematurely.
enum Injector {
    struct Outcome {
        let ok: Bool
        let reason: String?
        let strategyUsed: String
    }

    /// Inject `text` at the insertion point of the focused element.
    /// - Parameters:
    ///   - text: the text to insert
    ///   - strategy: caller-requested strategy (`auto` for best-effort)
    ///   - requireMatchingPID: if non-nil, abort when the currently focused
    ///     app's PID differs. Lets Swift refuse to inject into an app the
    ///     user switched away from between trigger and accept.
    @MainActor
    static func inject(
        text: String,
        strategy: InjectStrategy,
        requireMatchingPID: pid_t? = nil
    ) -> Outcome {
        guard !text.isEmpty else {
            return Outcome(ok: false, reason: "empty text", strategyUsed: strategy.rawValue)
        }

        // Re-read the focused element every time. This is intentionally
        // fresh; Python's `requires_focused_element` flag is the contract
        // for "only inject if focus is still reasonable."
        let focused: FocusedElementSnapshot
        do {
            focused = try AXProbe.snapshotFocusedElement()
        } catch {
            return Outcome(
                ok: false,
                reason: "no focused element: \(error)",
                strategyUsed: strategy.rawValue
            )
        }

        if let expected = requireMatchingPID, focused.appPID != expected {
            return Outcome(
                ok: false,
                reason: "focus changed (pid \(focused.appPID) != \(expected))",
                strategyUsed: strategy.rawValue
            )
        }

        switch strategy {
        case .cdp:
            // Phase A doesn't route CDP through Swift; Python owns it.
            return Outcome(
                ok: false,
                reason: "cdp strategy is handled in Python — Swift cannot satisfy it in Phase A",
                strategyUsed: "cdp"
            )
        case .axValue:
            return injectViaAXValue(text: text, focused: focused)
        case .clipboard:
            return injectViaClipboard(text: text)
        case .keystrokes:
            return injectViaKeystrokes(text: text)
        case .auto:
            // Try AX first (fast, targeted) then fall back to clipboard
            // (reliable, works in almost any editable field). Keystrokes
            // is a last resort — noisy and slow.
            let ax = injectViaAXValue(text: text, focused: focused)
            if ax.ok { return ax }
            let cb = injectViaClipboard(text: text)
            if cb.ok { return cb }
            return injectViaKeystrokes(text: text)
        }
    }

    // MARK: - Strategies

    /// Rebuild the AXValue string by splicing `text` at the insertion point
    /// and set it back. Works cleanly for NSTextField / NSTextView style
    /// controls; many web apps will accept the mutation but their JS
    /// handlers won't fire — the clipboard path handles those.
    @MainActor
    private static func injectViaAXValue(
        text: String,
        focused: FocusedElementSnapshot
    ) -> Outcome {
        // Fields with baked-in placeholders (common in web composers)
        // mis-handle a direct AXValue set: the placeholder stays visible
        // and the value never "sticks" because the JS layer didn't see a
        // synthetic input event. Skip AX and go to the next strategy.
        if focused.hasPlaceholder {
            return Outcome(
                ok: false,
                reason: "placeholder detected — AX value set is unsafe here",
                strategyUsed: "ax_value"
            )
        }

        let systemWide = AXUIElementCreateSystemWide()
        var focusedRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedRef
        ) == .success,
              let focusedUnwrapped = focusedRef,
              CFGetTypeID(focusedUnwrapped) == AXUIElementGetTypeID()
        else {
            return Outcome(
                ok: false,
                reason: "could not re-grab AXFocusedUIElement",
                strategyUsed: "ax_value"
            )
        }
        let element = focusedUnwrapped as! AXUIElement

        let insertion = focused.insertionPoint ?? focused.value.count
        // Clamp to bounds; AX can hand us stale values during fast focus changes.
        let safeInsertion = max(0, min(insertion, focused.value.count))
        let before = String(focused.value.prefix(safeInsertion))
        let after = String(focused.value.dropFirst(safeInsertion))
        let newValue = before + text + after
        let newInsertion = (before + text).count

        let setErr = AXUIElementSetAttributeValue(
            element,
            kAXValueAttribute as CFString,
            newValue as CFString
        )
        guard setErr == .success else {
            return Outcome(
                ok: false,
                reason: "AXUIElementSetAttributeValue failed: \(setErr.rawValue)",
                strategyUsed: "ax_value"
            )
        }

        // Best-effort: move the cursor to just after the injected text.
        // Some apps reject this silently; that's OK, the value is already set.
        var range = CFRange(location: newInsertion, length: 0)
        if let axRange = AXValueCreate(.cfRange, &range) {
            _ = AXUIElementSetAttributeValue(
                element,
                kAXSelectedTextRangeAttribute as CFString,
                axRange
            )
        }

        return Outcome(ok: true, reason: nil, strategyUsed: "ax_value")
    }

    /// Save clipboard → write `text` → post Cmd+V → restore clipboard.
    /// This path handles web apps that listen for paste events (Slack,
    /// Gmail, Facebook composers, etc.) where AX value set silently fails.
    @MainActor
    private static func injectViaClipboard(text: String) -> Outcome {
        let pb = NSPasteboard.general
        let saved = pb.string(forType: .string)

        pb.clearContents()
        pb.setString(text, forType: .string)

        // Synthesize Cmd+V. Post to the HID tap so the frontmost app
        // receives it as if the user typed it.
        guard let src = CGEventSource(stateID: .hidSystemState) else {
            return Outcome(
                ok: false,
                reason: "CGEventSource creation failed",
                strategyUsed: "clipboard"
            )
        }
        let vKey: CGKeyCode = 9  // 'v'
        let down = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: true)
        let up = CGEvent(keyboardEventSource: src, virtualKey: vKey, keyDown: false)
        down?.flags = .maskCommand
        up?.flags = .maskCommand
        down?.post(tap: .cghidEventTap)
        up?.post(tap: .cghidEventTap)

        // Restore clipboard on a short delay so the target app has time to
        // actually consume the paste. 120ms is a common choice; we'll tune
        // if dogfooding shows lost content.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            pb.clearContents()
            if let saved { pb.setString(saved, forType: .string) }
        }

        return Outcome(ok: true, reason: nil, strategyUsed: "clipboard")
    }

    /// Type each character as a synthetic keystroke. Slowest and least
    /// robust (unicode mapping is approximate) — reserved for cases where
    /// both AX and clipboard have failed.
    @MainActor
    private static func injectViaKeystrokes(text: String) -> Outcome {
        guard let src = CGEventSource(stateID: .hidSystemState) else {
            return Outcome(
                ok: false,
                reason: "CGEventSource creation failed",
                strategyUsed: "keystrokes"
            )
        }

        for scalar in text.unicodeScalars {
            // CGEventKeyboardSetUnicodeString lets us emit arbitrary text
            // without a keycode-to-char map — noisy but simple.
            let down = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: true)
            let up = CGEvent(keyboardEventSource: src, virtualKey: 0, keyDown: false)
            let value = UInt16(scalar.value & 0xFFFF)
            [down, up].forEach { event in
                guard let event else { return }
                var chars = [value]
                event.keyboardSetUnicodeString(stringLength: 1, unicodeString: &chars)
                event.post(tap: .cghidEventTap)
            }
        }

        return Outcome(ok: true, reason: nil, strategyUsed: "keystrokes")
    }
}
