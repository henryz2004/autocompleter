import ApplicationServices
import CoreGraphics
import Foundation

/// Thin wrapper around `CGEventTapCreate` for global-hotkey detection.
///
/// Ports the behavior of `autocompleter/hotkey.py:HotkeyListener`:
/// session-level tap, head-inserted, default options, listens for keyDown,
/// auto-re-enables on `tapDisabledByTimeout`.
///
/// Used in v0.1 for:
///   - Diagnostic check #3: verify the tap can be created at all.
///   - Diagnostic check #4: capture a hotkey press and snapshot AX state.
final class HotkeyTap {
    struct Hotkey: Equatable {
        var keycode: Int64
        var modifierMask: CGEventFlags
    }

    /// Callback invoked on main queue when the registered hotkey fires.
    var onFire: (() -> Void)?

    /// Callback invoked on main queue when `CGEvent.tapCreate` fails inside
    /// `start()`'s background thread (typically because accessibility
    /// permission has been revoked or was never granted).
    var onTapCreateFailed: (() -> Void)?

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var runLoopRef: CFRunLoop?
    private var thread: Thread?
    private let hotkey: Hotkey
    private var stopped = false
    /// Retained pointer to `self` handed to `CGEvent.tapCreate` so it stays
    /// alive for the lifetime of the tap. Released in `runLoop`'s teardown.
    private var retainedSelfPtr: UnsafeMutableRawPointer?

    init(hotkey: Hotkey) {
        self.hotkey = hotkey
    }

    /// Try to create a tap and immediately tear it down. Returns true if
    /// `CGEventTapCreate` succeeded — the test used by diagnostic check #3.
    static func canCreateTap() -> Bool {
        let mask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: CGEventMask(mask),
            callback: { _, _, event, _ in Unmanaged.passUnretained(event) },
            userInfo: nil
        ) else {
            return false
        }
        CGEvent.tapEnable(tap: tap, enable: false)
        return true
    }

    /// Start the tap on a dedicated background thread. Matches the threading
    /// model of the Python listener: the event tap callback must return
    /// quickly (<1s) or macOS disables the tap.
    func start() {
        guard thread == nil else { return }
        stopped = false

        let t = Thread { [weak self] in
            self?.runLoop()
        }
        t.name = "HotkeyTap"
        t.start()
        thread = t
    }

    func stop() {
        stopped = true
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let rl = runLoopRef {
            CFRunLoopStop(rl)
        }
    }

    // MARK: - private

    private func runLoop() {
        runLoopRef = CFRunLoopGetCurrent()

        let mask = (1 << CGEventType.keyDown.rawValue)
                 | (1 << CGEventType.keyUp.rawValue)
                 | (1 << CGEventType.tapDisabledByTimeout.rawValue)
                 | (1 << CGEventType.tapDisabledByUserInput.rawValue)

        // passRetained hands a +1 reference to the tap. That reference is
        // released in the teardown below once the run loop exits. This
        // guarantees that the C callback can safely dereference userInfo
        // even if the owning model has already dropped its reference.
        let retained = Unmanaged.passRetained(self).toOpaque()
        retainedSelfPtr = retained

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: CGEventMask(mask),
            callback: hotkeyTapCallback,
            userInfo: retained
        ) else {
            // tapCreate failed — balance the retain and surface the failure.
            Unmanaged<HotkeyTap>.fromOpaque(retained).release()
            retainedSelfPtr = nil
            DispatchQueue.main.async { [weak self] in
                self?.onTapCreateFailed?()
            }
            return
        }

        eventTap = tap

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), source, .commonModes)
        runLoopSource = source
        CGEvent.tapEnable(tap: tap, enable: true)

        while !stopped {
            CFRunLoopRunInMode(.defaultMode, 1.0, false)
        }

        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil

        if let retained = retainedSelfPtr {
            Unmanaged<HotkeyTap>.fromOpaque(retained).release()
            retainedSelfPtr = nil
        }
    }

    /// Match logic mirrors hotkey.py: required modifier bits must be present,
    /// and no *extra* modifier bits may be held beyond what was registered.
    fileprivate func handle(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let tap = eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        guard type == .keyDown else {
            return Unmanaged.passUnretained(event)
        }

        let keycode = event.getIntegerValueField(.keyboardEventKeycode)
        let flags = event.flags

        let allMods: CGEventFlags = [
            .maskControl, .maskCommand, .maskAlternate, .maskShift,
        ]

        guard keycode == hotkey.keycode else {
            return Unmanaged.passUnretained(event)
        }

        // Required modifier bits must be exactly the set that's held,
        // restricted to standard modifier keys (ignore caps-lock / device
        // bits that the OS sometimes sets). Matches the Python check:
        //   (flags & reg_flags) == reg_flags && (flags & modmask & ~reg_flags) == 0
        let required = hotkey.modifierMask.intersection(allMods)
        let held = flags.intersection(allMods)
        guard held == required else {
            return Unmanaged.passUnretained(event)
        }

        DispatchQueue.main.async { [weak self] in
            self?.onFire?()
        }
        // Suppress the event so it doesn't reach the focused app.
        return nil
    }
}

/// C-compatible callback — CGEventTapCreate requires a @convention(c)
/// function pointer. We recover `self` from the opaque userInfo pointer.
private func hotkeyTapCallback(
    proxy: CGEventTapProxy,
    type: CGEventType,
    event: CGEvent,
    userInfo: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    guard let userInfo else { return Unmanaged.passUnretained(event) }
    // takeUnretainedValue — the +1 retain handed to tapCreate keeps `self`
    // alive until the run loop exits and we explicitly release it.
    let tap = Unmanaged<HotkeyTap>.fromOpaque(userInfo).takeUnretainedValue()
    return tap.handle(type: type, event: event)
}

// MARK: - Hotkey parsing

enum HotkeyParseError: Error, CustomStringConvertible {
    case unknownComponent(String)
    var description: String {
        switch self {
        case .unknownComponent(let c): return "Unknown hotkey component: \(c)"
        }
    }
}

extension HotkeyTap.Hotkey {
    /// Parse strings like "ctrl+space" into a Hotkey. Mirrors
    /// `parse_hotkey` in autocompleter/hotkey.py.
    static func parse(_ s: String) throws -> HotkeyTap.Hotkey {
        var keycode: Int64 = 0
        var flags: CGEventFlags = []

        for raw in s.split(separator: "+") {
            let part = raw.trimmingCharacters(in: .whitespaces).lowercased()
            if let mod = modifierMap[part] {
                flags.insert(mod)
            } else if let code = keyMap[part] {
                keycode = code
            } else {
                throw HotkeyParseError.unknownComponent(part)
            }
        }

        return HotkeyTap.Hotkey(keycode: keycode, modifierMask: flags)
    }

    private static let modifierMap: [String: CGEventFlags] = [
        "ctrl": .maskControl,
        "control": .maskControl,
        "cmd": .maskCommand,
        "command": .maskCommand,
        "alt": .maskAlternate,
        "opt": .maskAlternate,
        "option": .maskAlternate,
        "shift": .maskShift,
    ]

    private static let keyMap: [String: Int64] = [
        "space": 49, "tab": 48, "return": 36, "enter": 36, "escape": 53, "esc": 53,
        "up": 126, "down": 125, "left": 123, "right": 124,
        "0": 29, "1": 18, "2": 19, "3": 20, "4": 21, "5": 23,
        "6": 22, "7": 26, "8": 28, "9": 25,
        "/": 44, "-": 27, "=": 24, ",": 43, ".": 47, ";": 41, "'": 39,
        "[": 33, "]": 30, "`": 50,
        "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
        "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
        "a": 0, "b": 11, "c": 8, "d": 2, "e": 14, "f": 3,
        "g": 5, "h": 4, "i": 34, "j": 38, "k": 40, "l": 37,
        "m": 46, "n": 45, "o": 31, "p": 35, "q": 12, "r": 15,
        "s": 1, "t": 17, "u": 32, "v": 9, "w": 13, "x": 7,
        "y": 16, "z": 6,
    ]
}
