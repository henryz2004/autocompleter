import ApplicationServices
import AppKit
import Foundation

/// Event-driven capture of the user's recent "trajectory" — a chronological
/// ring buffer of (app, window_title, url, focused_role, focused_value_preview)
/// snapshots. Feeds the trajectory-intent-probe experiment: the question is
/// whether a fast LLM, given the last N snapshots, can infer what the user is
/// currently trying to do.
///
/// Not wired into the core v0.2 suggestion pipeline. This is an experimental
/// side-channel on the `trajectory-intent-probe` branch.
///
/// Capture sources (all event-driven, no 2s polling):
///   1. `NSWorkspace.didActivateApplicationNotification` — app switches.
///   2. `AXObserver` on the frontmost app for
///        - `kAXFocusedWindowChangedNotification`
///        - `kAXTitleChangedNotification`
///        - `kAXFocusedUIElementChangedNotification`
///      Re-attached to the new frontmost app whenever app focus changes.
///   3. 30s heartbeat timer — emits a snapshot even when nothing fired, so the
///      LLM sees "user has been staring at this for 30s" as a signal rather
///      than silence.
///
/// A snapshot is taken after an 800ms debounce: rapid-fire events (e.g. a
/// window title streaming in as a page loads) collapse to one entry. If the
/// newest snapshot is identical to the previous one (same app/title/url/role
/// and same value preview) we drop it to keep the buffer signal-dense.
@MainActor
final class TrajectoryStack {
    /// One point on the trajectory. Deliberately small — a value preview,
    /// not the raw value — to keep the buffer cheap to JSON-serialize and
    /// to avoid shipping entire text fields to the LLM on every tick.
    /// Whether the ring buffer appended a new row or coalesced into the
    /// existing top entry (heartbeat dwell).
    enum Change {
        case appended(Snapshot)
        case updated(Snapshot)
    }

    struct Snapshot: Codable {
        let timestamp: Double           // seconds since epoch (first seen)
        var lastSeen: Double            // bumped on heartbeat coalesce; equals timestamp initially
        let trigger: String             // "app_switch" | "title_change" | "focus_change" | "heartbeat" | "initial"
        let appName: String
        let appPID: pid_t
        let bundleID: String?
        let windowTitle: String?
        let url: String?
        let focusedRole: String?
        var focusedValuePreview: String?  // up to ~400 chars (tail) of the focused element's value
        var focusedValueLength: Int?
        /// Jaccard similarity to the previous snapshot's content at capture time.
        /// Logged to eval JSONL for offline threshold tuning.
        var contentSimilarity: Double?
    }

    /// Ring buffer of recent snapshots. User explicitly asked not to limit
    /// the stack too much for this experiment, so we keep a generous window —
    /// ~200 snapshots is <100 KB JSON even in the worst case and covers
    /// roughly 1-2 hours of typical activity at the debounced event rate.
    private var capacity: Int
    private var buffer: [Snapshot] = []

    /// Currently-attached AX observer for the frontmost app. Released and
    /// reattached on every app switch — we only observe *one* app at a time
    /// to keep AX overhead bounded.
    private var focusedObserver: AXObserver?
    private var focusedPID: pid_t = 0

    /// NSWorkspace notification tokens.
    private var activateToken: NSObjectProtocol?

    /// 800ms debounce + heartbeat.
    private var debounceWork: DispatchWorkItem?
    private var heartbeatTimer: Timer?
    private let debounceInterval: TimeInterval = 0.8
    private var heartbeatInterval: TimeInterval

    /// Maximum chars to capture in the value preview. Tunable via the
    /// Trajectory Inspector so Henry can dial it live without rebuilding.
    private(set) var previewCharLimit: Int = 2000

    /// Fires whenever a snapshot is appended. Multicast — every registered
    /// observer gets called. Started life as a single-slot closure, but the
    /// diagnostic-log subscriber and the intent controller both want to
    /// observe, so we switched to a list after one of them silently
    /// overwrote the other.
    private var observers: [(Change) -> Void] = []

    /// Register a new observer. Returns a token the caller can use to
    /// deregister later; today nothing deregisters so we don't actually
    /// surface the token.
    func addObserver(_ handler: @escaping (Change) -> Void) {
        observers.append(handler)
    }

    init(capacity: Int = 200, heartbeatInterval: TimeInterval = 30.0) {
        self.capacity = capacity
        self.heartbeatInterval = heartbeatInterval
    }

    // MARK: - Lifecycle

    func start() {
        // Prime the buffer with an initial snapshot so the LLM has something
        // to reason about even before the user touches anything.
        capture(trigger: "initial", debounced: false)

        attachToFrontmost()

        activateToken = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            Task { @MainActor in
                self?.onAppActivated()
            }
        }

        heartbeatTimer = Timer.scheduledTimer(
            withTimeInterval: heartbeatInterval,
            repeats: true
        ) { [weak self] _ in
            Task { @MainActor in
                self?.capture(trigger: "heartbeat", debounced: false)
            }
        }
    }

    func stop() {
        if let t = activateToken {
            NSWorkspace.shared.notificationCenter.removeObserver(t)
            activateToken = nil
        }
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
        debounceWork?.cancel()
        debounceWork = nil
        releaseFocusedObserver()
    }

    /// Snapshot the current buffer. Returned in chronological order (oldest
    /// first). Safe to call from any actor context via `MainActor.run` — the
    /// caller should already be on MainActor.
    func currentSnapshots() -> [Snapshot] {
        return buffer
    }

    // MARK: - Tunable parameters (for Trajectory Inspector)

    var currentCapacity: Int { capacity }
    var currentHeartbeatInterval: TimeInterval { heartbeatInterval }

    func setCapacity(_ newCapacity: Int) {
        capacity = max(10, min(newCapacity, 1000))
        if buffer.count > capacity {
            buffer.removeFirst(buffer.count - capacity)
        }
    }

    func setHeartbeatInterval(_ interval: TimeInterval) {
        heartbeatInterval = max(5, min(interval, 120))
        heartbeatTimer?.invalidate()
        heartbeatTimer = Timer.scheduledTimer(
            withTimeInterval: heartbeatInterval,
            repeats: true
        ) { [weak self] _ in
            Task { @MainActor in
                self?.capture(trigger: "heartbeat", debounced: false)
            }
        }
    }

    func setPreviewCharLimit(_ limit: Int) {
        previewCharLimit = max(100, min(limit, 5000))
    }

    // MARK: - Event plumbing

    private func onAppActivated() {
        scheduleCapture(trigger: "app_switch")
        attachToFrontmost()
    }

    fileprivate func onFocusedAXNotification(_ name: CFString) {
        let kind: String
        if CFStringCompare(name, kAXTitleChangedNotification as CFString, []) == .compareEqualTo {
            kind = "title_change"
        } else if CFStringCompare(name, kAXFocusedUIElementChangedNotification as CFString, []) == .compareEqualTo {
            kind = "focus_change"
        } else if CFStringCompare(name, kAXFocusedWindowChangedNotification as CFString, []) == .compareEqualTo {
            kind = "focus_change"
        } else if CFStringCompare(name, kAXValueChangedNotification as CFString, []) == .compareEqualTo {
            kind = "value_change"
        } else {
            kind = "ax_event"
        }
        scheduleCapture(trigger: kind)
    }

    private func scheduleCapture(trigger: String) {
        debounceWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            Task { @MainActor in
                self?.capture(trigger: trigger, debounced: true)
            }
        }
        debounceWork = work
        DispatchQueue.main.asyncAfter(
            deadline: .now() + debounceInterval,
            execute: work
        )
    }

    // MARK: - Capture

    /// Jaccard similarity threshold below which we treat the capture as a
    /// context break (new snapshot) rather than content evolution (coalesce).
    private let similarityThreshold: Double = 0.4

    /// If content similarity is high (same context) but the number of new
    /// words exceeds this count, treat it as "content evolved enough to
    /// re-query" — emit .appended instead of .updated.
    private let newWordThreshold: Int = 20

    private func capture(trigger: String, debounced: Bool) {
        let snap = buildSnapshot(trigger: trigger)

        guard let lastIdx = buffer.indices.last else {
            // Empty buffer — always append.
            appendAndNotify(snap)
            return
        }

        let prev = buffer[lastIdx]

        // Hard breaks: different app or different focused element type
        // are always a context switch regardless of content.
        if snap.appPID != prev.appPID || snap.focusedRole != prev.focusedRole {
            var tagged = snap
            tagged.contentSimilarity = 0.0  // hard break
            appendAndNotify(tagged)
            return
        }

        // Content similarity determines break vs evolution.
        let (jaccard, newWords) = Self.contentSimilarity(
            old: prev.focusedValuePreview,
            new: snap.focusedValuePreview
        )

        if jaccard < similarityThreshold {
            // Low similarity = context break (tmux switch, new tab, different file).
            var tagged = snap
            tagged.contentSimilarity = jaccard
            appendAndNotify(tagged)
        } else if newWords >= newWordThreshold {
            // High similarity but enough new content appeared (terminal output
            // streaming, new chat messages) — worth re-querying.
            var tagged = snap
            tagged.contentSimilarity = jaccard
            appendAndNotify(tagged)
        } else {
            // Same context, minor evolution — coalesce into top entry.
            buffer[lastIdx].lastSeen = snap.timestamp
            buffer[lastIdx].focusedValuePreview = snap.focusedValuePreview
            buffer[lastIdx].focusedValueLength = snap.focusedValueLength
            let change = Change.updated(buffer[lastIdx])
            for observer in observers { observer(change) }
        }
    }

    private func appendAndNotify(_ snap: Snapshot) {
        buffer.append(snap)
        if buffer.count > capacity {
            buffer.removeFirst(buffer.count - capacity)
        }
        let change = Change.appended(snap)
        for observer in observers { observer(change) }
    }

    /// Jaccard similarity on whitespace-split word sets + count of new words.
    /// Returns (jaccard: 0..1, newWordCount: Int).
    private static func contentSimilarity(
        old: String?,
        new: String?
    ) -> (jaccard: Double, newWords: Int) {
        let oldWords = Set((old ?? "").split(separator: " "))
        let newWords = Set((new ?? "").split(separator: " "))
        let union = oldWords.union(newWords)
        guard !union.isEmpty else { return (1.0, 0) }
        let intersection = oldWords.intersection(newWords)
        let jaccard = Double(intersection.count) / Double(union.count)
        let delta = newWords.subtracting(oldWords).count
        return (jaccard, delta)
    }

    /// Diagnostic log hook for AX profiling. Wired by AppDelegate.
    var onProfile: ((String) -> Void)?

    private func buildSnapshot(trigger: String) -> Snapshot {
        let now = Date().timeIntervalSince1970
        let t0 = CFAbsoluteTimeGetCurrent()

        let frontmost = NSWorkspace.shared.frontmostApplication
        let pid = frontmost?.processIdentifier ?? 0
        let appName = frontmost?.localizedName ?? "Unknown"
        let bundleID = frontmost?.bundleIdentifier

        var windowTitle: String? = nil
        var url: String? = nil
        var focusedRole: String? = nil
        var focusedValuePreview: String? = nil
        var focusedValueLength: Int? = nil

        var profileParts: [String] = []

        if pid > 0 {
            let appElement = AXUIElementCreateApplication(pid)

            var t = CFAbsoluteTimeGetCurrent()
            windowTitle = Self.readFocusedWindowTitle(appElement)
            profileParts.append(String(format: "title=%.1fms", (CFAbsoluteTimeGetCurrent() - t) * 1000))

            t = CFAbsoluteTimeGetCurrent()
            if let focused = Self.readFocusedElement(appElement) {
                profileParts.append(String(format: "focus=%.1fms", (CFAbsoluteTimeGetCurrent() - t) * 1000))

                focusedRole = Self.readStringAttribute(focused, kAXRoleAttribute as CFString)

                t = CFAbsoluteTimeGetCurrent()
                let value = Self.readStringAttribute(focused, kAXValueAttribute as CFString) ?? ""
                profileParts.append(String(format: "value=%.1fms(%d)", (CFAbsoluteTimeGetCurrent() - t) * 1000, value.count))

                t = CFAbsoluteTimeGetCurrent()
                let visibleText = Self.readVisibleText(focused, fullValue: value) ?? value
                profileParts.append(String(format: "visible=%.1fms", (CFAbsoluteTimeGetCurrent() - t) * 1000))

                focusedValueLength = visibleText.count
                focusedValuePreview = Self.preview(of: visibleText, limit: previewCharLimit)

                t = CFAbsoluteTimeGetCurrent()
                url = Self.readURLAttribute(focused)
                profileParts.append(String(format: "url=%.1fms", (CFAbsoluteTimeGetCurrent() - t) * 1000))
            } else {
                profileParts.append(String(format: "focus=%.1fms(nil)", (CFAbsoluteTimeGetCurrent() - t) * 1000))
            }

            if url == nil {
                t = CFAbsoluteTimeGetCurrent()
                url = Self.readURLFromFocusedWindow(appElement)
                profileParts.append(String(format: "urlBFS=%.1fms", (CFAbsoluteTimeGetCurrent() - t) * 1000))
            }
        }

        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        if !profileParts.isEmpty {
            onProfile?(String(format: "ax[%@] %@ total=%.1fms | %@",
                              trigger, appName, totalMs, profileParts.joined(separator: " ")))
        }

        return Snapshot(
            timestamp: now,
            lastSeen: now,
            trigger: trigger,
            appName: appName,
            appPID: pid,
            bundleID: bundleID,
            windowTitle: windowTitle,
            url: url,
            focusedRole: focusedRole,
            focusedValuePreview: focusedValuePreview,
            focusedValueLength: focusedValueLength
        )
    }

    // MARK: - AX observer plumbing

    private func attachToFrontmost() {
        guard let app = NSWorkspace.shared.frontmostApplication else { return }
        let pid = app.processIdentifier
        guard pid > 0 else { return }
        if pid == focusedPID, focusedObserver != nil { return }
        releaseFocusedObserver()

        var observer: AXObserver?
        let createErr = AXObserverCreate(pid, trajectoryObserverCallback, &observer)
        guard createErr == .success, let created = observer else { return }

        let appElement = AXUIElementCreateApplication(pid)
        let selfPtr = Unmanaged.passUnretained(self).toOpaque()

        let notifications: [CFString] = [
            kAXFocusedWindowChangedNotification as CFString,
            kAXTitleChangedNotification as CFString,
            kAXFocusedUIElementChangedNotification as CFString,
            kAXValueChangedNotification as CFString,
        ]
        for note in notifications {
            // Errors are non-fatal — some apps don't emit every notification
            // we ask for and we still want the ones they do emit.
            _ = AXObserverAddNotification(created, appElement, note, selfPtr)
        }

        CFRunLoopAddSource(
            CFRunLoopGetMain(),
            AXObserverGetRunLoopSource(created),
            .defaultMode
        )

        focusedObserver = created
        focusedPID = pid
    }

    private func releaseFocusedObserver() {
        if let observer = focusedObserver {
            CFRunLoopRemoveSource(
                CFRunLoopGetMain(),
                AXObserverGetRunLoopSource(observer),
                .defaultMode
            )
        }
        focusedObserver = nil
        focusedPID = 0
    }

    // MARK: - AX helpers

    private static func readStringAttribute(_ element: AXUIElement, _ attr: CFString) -> String? {
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, attr, &ref)
        guard err == .success else { return nil }
        return ref as? String
    }

    private static func readFocusedElement(_ appElement: AXUIElement) -> AXUIElement? {
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            appElement,
            kAXFocusedUIElementAttribute as CFString,
            &ref
        )
        guard err == .success, let ref else { return nil }
        guard CFGetTypeID(ref) == AXUIElementGetTypeID() else { return nil }
        return (ref as! AXUIElement)
    }

    private static func readFocusedWindowTitle(_ appElement: AXUIElement) -> String? {
        var windowRef: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            appElement,
            kAXFocusedWindowAttribute as CFString,
            &windowRef
        )
        guard err == .success, let windowRef else { return nil }
        guard CFGetTypeID(windowRef) == AXUIElementGetTypeID() else { return nil }
        let window = windowRef as! AXUIElement
        return readStringAttribute(window, kAXTitleAttribute as CFString)
    }

    private static func readURLAttribute(_ element: AXUIElement) -> String? {
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(element, kAXURLAttribute as CFString, &ref)
        guard err == .success, let ref else { return nil }
        if let nsurl = ref as? NSURL {
            return nsurl.absoluteString
        }
        if let str = ref as? String {
            return str
        }
        return nil
    }

    /// Best-effort URL read from the focused window. Browsers often expose
    /// AXURL on the AXWebArea child of the focused window rather than on the
    /// window itself. We do one shallow walk; if we can't find it cheaply we
    /// leave it nil — the experiment can learn to cope.
    private static func readURLFromFocusedWindow(_ appElement: AXUIElement) -> String? {
        var windowRef: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            appElement,
            kAXFocusedWindowAttribute as CFString,
            &windowRef
        )
        guard err == .success, let windowRef else { return nil }
        guard CFGetTypeID(windowRef) == AXUIElementGetTypeID() else { return nil }
        let window = windowRef as! AXUIElement
        if let direct = readURLAttribute(window) { return direct }

        // Shallow BFS for AXWebArea. Capped at ~30 nodes to keep cost bounded.
        var queue: [AXUIElement] = [window]
        var visited = 0
        while !queue.isEmpty && visited < 30 {
            let node = queue.removeFirst()
            visited += 1
            let role = readStringAttribute(node, kAXRoleAttribute as CFString)
            if role == "AXWebArea" {
                if let u = readURLAttribute(node) { return u }
            }
            var childrenRef: CFTypeRef?
            let cerr = AXUIElementCopyAttributeValue(node, kAXChildrenAttribute as CFString, &childrenRef)
            if cerr == .success, let children = childrenRef as? [AXUIElement] {
                queue.append(contentsOf: children.prefix(10))
            }
        }
        return nil
    }

    /// Read the visible portion of a text element using AXVisibleCharacterRange.
    /// Returns nil if the attribute isn't supported (non-text elements, some apps).
    private static func readVisibleText(_ element: AXUIElement, fullValue: String) -> String? {
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            element,
            kAXVisibleCharacterRangeAttribute as CFString,
            &ref
        )
        guard err == .success, let ref else { return nil }
        var cfRange = CFRange(location: 0, length: 0)
        guard AXValueGetValue(ref as! AXValue, .cfRange, &cfRange) else { return nil }
        let nsStr = fullValue as NSString
        let loc = max(0, cfRange.location)
        let len = min(cfRange.length, nsStr.length - loc)
        guard loc < nsStr.length, len > 0 else { return nil }
        return nsStr.substring(with: NSRange(location: loc, length: len))
    }

    private static func preview(of value: String, limit: Int) -> String? {
        if value.isEmpty { return nil }
        // Collapse whitespace so multi-line previews stay legible in a single
        // LLM line.
        let collapsed = value.replacingOccurrences(
            of: "\\s+",
            with: " ",
            options: .regularExpression
        )
        if collapsed.count <= limit { return collapsed }
        // Take the TAIL — for terminals the last N chars are what's on screen
        // near the prompt, not the login banner at the top of scrollback.
        let idx = collapsed.index(collapsed.endIndex, offsetBy: -limit)
        return "…" + String(collapsed[idx...])
    }
}

/// C-compatible AX observer callback. We recover the `TrajectoryStack` via
/// the userInfo pointer we stored when subscribing. The callback must be
/// cheap — it hops onto the MainActor and schedules a debounced capture.
private func trajectoryObserverCallback(
    observer: AXObserver,
    element: AXUIElement,
    notification: CFString,
    userInfo: UnsafeMutableRawPointer?
) {
    guard let userInfo else { return }
    let stack = Unmanaged<TrajectoryStack>.fromOpaque(userInfo).takeUnretainedValue()
    // The AX callback fires on the main run loop — we're already on the main
    // thread, but MainActor isolation is compile-time-checked, so hop via
    // Task to make the compiler happy.
    Task { @MainActor in
        stack.onFocusedAXNotification(notification)
    }
}
