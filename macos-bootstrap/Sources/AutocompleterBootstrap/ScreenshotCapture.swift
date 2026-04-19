import AppKit
import CoreGraphics
import Foundation
import ScreenCaptureKit
import Vision

/// Event-triggered screenshot + OCR capture. Takes a screenshot of the
/// frontmost window when the user switches apps, tabs, or windows, then
/// runs local OCR via Apple Vision to extract readable text.
///
/// Triggers piggyback on TrajectoryStack's existing AXObserver (title
/// change, window change) plus NSWorkspace app-deactivation. A 1.5s
/// debounce collapses rapid-fire events (page loads rewriting the title
/// multiple times). Dedup skips captures when app + title haven't
/// changed since the last capture.
///
/// All heavy work (screenshot + OCR) runs on a background DispatchQueue
/// so the main thread is never blocked.
@MainActor
final class ScreenshotCapture {
    private let store: CaptureStore

    /// Debounce timer — 1.5s because screenshot + OCR is more expensive
    /// than a metadata snapshot.
    private var debounceWork: DispatchWorkItem?
    private let debounceInterval: TimeInterval = 1.5

    /// Dedup: skip capture if app + title haven't changed.
    private var lastCapturedApp: String?
    private var lastCapturedTitle: String?

    /// NSWorkspace observer token.
    private var deactivateToken: NSObjectProtocol?

    /// Diagnostic log hook (same pattern as TrajectoryStack).
    var onLog: ((String) -> Void)?

    init(store: CaptureStore) {
        self.store = store
    }

    // MARK: - Lifecycle

    func start() {
        // App deactivation — fires when user leaves any app.
        deactivateToken = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didDeactivateApplicationNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey]
                    as? NSRunningApplication else { return }
            Task { @MainActor in
                self?.scheduleCapture(
                    trigger: "app_switch",
                    pid: app.processIdentifier,
                    appName: app.localizedName ?? "Unknown",
                    bundleID: app.bundleIdentifier
                )
            }
        }

        onLog?("capture: started")
    }

    func stop() {
        if let t = deactivateToken {
            NSWorkspace.shared.notificationCenter.removeObserver(t)
            deactivateToken = nil
        }
        debounceWork?.cancel()
        debounceWork = nil
    }

    /// Called by TrajectoryStack (or AppDelegate) when an AX title/window
    /// change fires. Piggybacks on the existing observer rather than
    /// running a second AXObserver.
    func onAXEvent(trigger: String) {
        guard let app = NSWorkspace.shared.frontmostApplication else { return }
        scheduleCapture(
            trigger: trigger,
            pid: app.processIdentifier,
            appName: app.localizedName ?? "Unknown",
            bundleID: app.bundleIdentifier
        )
    }

    /// Explicit capture triggered by hotkey (ctrl+shift+c). Captures the
    /// current frontmost window immediately — no debounce, no dedup.
    func captureNow() {
        guard let app = NSWorkspace.shared.frontmostApplication else { return }
        let pid = app.processIdentifier
        let appName = app.localizedName ?? "Unknown"
        let bundleID = app.bundleIdentifier
        onLog?("capture: manual → \(appName)")
        performCapture(
            trigger: "manual",
            pid: pid,
            appName: appName,
            bundleID: bundleID,
            skipDedup: true
        )
    }

    // MARK: - Debounce + dedup

    private func scheduleCapture(
        trigger: String,
        pid: pid_t,
        appName: String,
        bundleID: String?
    ) {
        // Skip system apps that aren't useful to capture.
        if let bid = bundleID, Self.ignoredBundleIDs.contains(bid) { return }

        debounceWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            Task { @MainActor in
                self?.performCapture(
                    trigger: trigger,
                    pid: pid,
                    appName: appName,
                    bundleID: bundleID
                )
            }
        }
        debounceWork = work
        DispatchQueue.main.asyncAfter(
            deadline: .now() + debounceInterval,
            execute: work
        )
    }

    private func performCapture(
        trigger: String,
        pid: pid_t,
        appName: String,
        bundleID: String?,
        skipDedup: Bool = false
    ) {
        // Read window title for dedup.
        let title = Self.readWindowTitle(pid: pid)

        // Dedup: skip if app + title haven't changed (unless manual capture).
        if !skipDedup && appName == lastCapturedApp && title == lastCapturedTitle {
            onLog?("capture: dedup skip — \(appName) \(title ?? "")")
            return
        }
        lastCapturedApp = appName
        lastCapturedTitle = title

        // Read URL if available (for browsers).
        let url = Self.readURL(pid: pid)

        onLog?("capture: → \(trigger) \(appName) — \(title ?? "(no title)")")

        // Heavy work off the main thread via async.
        Task.detached(priority: .utility) { [weak self] in
            let t0 = CFAbsoluteTimeGetCurrent()

            guard let image = await Self.captureWindowSCK(pid: pid) else {
                await MainActor.run {
                    self?.onLog?("capture: screenshot failed for pid=\(pid)")
                }
                return
            }
            let screenshotMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0

            let t1 = CFAbsoluteTimeGetCurrent()
            let ocrText = Self.runOCR(on: image)
            let ocrMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000.0

            let t2 = CFAbsoluteTimeGetCurrent()
            // Save screenshot to disk.
            let screenshotDir = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".autocompleter/screenshots")
            try? FileManager.default.createDirectory(
                at: screenshotDir,
                withIntermediateDirectories: true
            )
            let filename = "\(Int(Date().timeIntervalSince1970 * 1000)).png"
            let screenshotPath = screenshotDir.appendingPathComponent(filename)
            Self.savePNG(image: image, to: screenshotPath)
            let saveMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000.0

            let totalMs = screenshotMs + ocrMs + saveMs

            let entry = CaptureEntry(
                id: UUID(),
                timestamp: Date(),
                trigger: trigger,
                appName: appName,
                bundleID: bundleID,
                windowTitle: title,
                url: url,
                screenshotPath: screenshotPath,
                ocrText: ocrText,
                ocrLatencyMs: totalMs
            )

            await MainActor.run { [weak self] in
                self?.store.append(entry)
                self?.onLog?(String(
                    format: "capture: ← %@ %@ — %d chars | screenshot=%.0fms ocr=%.0fms save=%.0fms total=%.0fms",
                    trigger, appName, ocrText.count, screenshotMs, ocrMs, saveMs, totalMs
                ))
            }
        }
    }

    // MARK: - Caret-localized capture

    /// Capture a region around the focused element and OCR it.
    /// Returns the OCR text with bounding boxes, or nil on failure.
    /// Used by paste trigger to give the LLM context about what field
    /// the user is in (the label is usually above/beside the field).
    nonisolated static func captureAroundFocusedElement(
        pid: pid_t,
        padding: CGFloat = 200
    ) async -> String? {
        // Get focused element position + size via AX.
        let app = AXUIElementCreateApplication(pid)
        var focRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(
            app, kAXFocusedUIElementAttribute as CFString, &focRef
        ) == .success, let focRef,
              CFGetTypeID(focRef) == AXUIElementGetTypeID() else { return nil }
        let el = focRef as! AXUIElement

        var posRef: CFTypeRef?
        var sizeRef: CFTypeRef?
        guard AXUIElementCopyAttributeValue(el, kAXPositionAttribute as CFString, &posRef) == .success,
              AXUIElementCopyAttributeValue(el, kAXSizeAttribute as CFString, &sizeRef) == .success,
              let posRef, let sizeRef else { return nil }

        var pos = CGPoint.zero
        var size = CGSize.zero
        AXValueGetValue(posRef as! AXValue, .cgPoint, &pos)
        AXValueGetValue(sizeRef as! AXValue, .cgSize, &size)

        // Expand the rect with padding to catch labels above/beside.
        let captureRect = CGRect(
            x: pos.x - padding,
            y: pos.y - padding,
            width: size.width + padding * 2,
            height: size.height + padding * 2
        )

        // Use ScreenCaptureKit to capture the region.
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true
            )
            guard let display = content.displays.first else { return nil }

            let filter = SCContentFilter(display: display, excludingWindows: [])
            let config = SCStreamConfiguration()
            config.sourceRect = captureRect
            config.width = Int(captureRect.width * 2)
            config.height = Int(captureRect.height * 2)
            config.showsCursor = false
            config.captureResolution = .best

            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter,
                configuration: config
            )
            return runOCR(on: image)
        } catch {
            return nil
        }
    }

    // MARK: - Screenshot (ScreenCaptureKit)

    /// Capture the frontmost window of a given PID using ScreenCaptureKit.
    private nonisolated static func captureWindowSCK(pid: pid_t) async -> CGImage? {
        do {
            let content = try await SCShareableContent.excludingDesktopWindows(
                false, onScreenWindowsOnly: true
            )
            // Find the first on-screen window owned by this PID.
            guard let window = content.windows.first(where: {
                $0.owningApplication?.processID == pid
                && $0.isOnScreen
                && $0.frame.width > 100  // skip tiny utility windows
            }) else { return nil }

            let filter = SCContentFilter(desktopIndependentWindow: window)
            let config = SCStreamConfiguration()
            config.width = Int(window.frame.width * 2)   // Retina
            config.height = Int(window.frame.height * 2)
            config.showsCursor = false
            config.captureResolution = .best

            let image = try await SCScreenshotManager.captureImage(
                contentFilter: filter,
                configuration: config
            )
            return image
        } catch {
            return nil
        }
    }

    /// Save a CGImage as PNG.
    private nonisolated static func savePNG(image: CGImage, to url: URL) {
        guard let dest = CGImageDestinationCreateWithURL(
            url as CFURL,
            "public.png" as CFString,
            1,
            nil
        ) else { return }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
    }

    // MARK: - OCR

    /// Run Apple Vision OCR on a CGImage. Returns text with bounding box
    /// metadata per line so the LLM (or downstream processing) can
    /// understand spatial layout — distinguishing main content from
    /// sidebar UI chrome, detecting columns, etc.
    ///
    /// Output format per line (coordinates converted to top-left origin,
    /// 0-100 scale for readability):
    ///   `[x:12 y:15 w:30 h:2] The actual recognized text`
    ///
    /// Blank lines inserted where vertical gaps suggest paragraph breaks.
    private nonisolated static func runOCR(on image: CGImage) -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true

        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try? handler.perform([request])

        guard let observations = request.results, !observations.isEmpty else { return "" }

        // Sort top-to-bottom (Vision coords are bottom-left origin, so
        // higher Y = higher on screen = earlier in reading order).
        let sorted = observations.sorted {
            $0.boundingBox.origin.y > $1.boundingBox.origin.y
        }

        // Compute median line height for gap detection.
        let heights = sorted.map { $0.boundingBox.height }
        let medianHeight = heights.sorted()[heights.count / 2]

        var lines: [String] = []
        var prevBottomY: CGFloat?

        for obs in sorted {
            guard let text = obs.topCandidates(1).first?.string else { continue }
            let box = obs.boundingBox

            if let prevBottom = prevBottomY {
                let gap = prevBottom - (box.origin.y + box.height)
                if gap > medianHeight * 0.7 {
                    lines.append("")
                }
            }

            // Normalized 0-1, converted to top-left origin.
            let x = String(format: "%.2f", box.origin.x)
            let y = String(format: "%.2f", 1.0 - box.origin.y - box.height)
            let w = String(format: "%.2f", box.width)
            let h = String(format: "%.2f", box.height)
            lines.append("[x:\(x) y:\(y) w:\(w) h:\(h)] \(text)")

            prevBottomY = box.origin.y
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - AX helpers (lightweight, for metadata only)

    private static func readWindowTitle(pid: pid_t) -> String? {
        let app = AXUIElementCreateApplication(pid)
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            app, kAXFocusedWindowAttribute as CFString, &ref
        )
        guard err == .success, let ref,
              CFGetTypeID(ref) == AXUIElementGetTypeID() else { return nil }
        let window = ref as! AXUIElement
        var titleRef: CFTypeRef?
        let terr = AXUIElementCopyAttributeValue(
            window, kAXTitleAttribute as CFString, &titleRef
        )
        guard terr == .success else { return nil }
        return titleRef as? String
    }

    private static func readURL(pid: pid_t) -> String? {
        // Best-effort URL from the focused window's AXWebArea child.
        let app = AXUIElementCreateApplication(pid)
        var ref: CFTypeRef?
        let err = AXUIElementCopyAttributeValue(
            app, kAXFocusedWindowAttribute as CFString, &ref
        )
        guard err == .success, let ref,
              CFGetTypeID(ref) == AXUIElementGetTypeID() else { return nil }
        let window = ref as! AXUIElement

        // Shallow BFS for AXWebArea, capped at 20 nodes.
        var queue: [AXUIElement] = [window]
        var visited = 0
        while !queue.isEmpty && visited < 20 {
            let node = queue.removeFirst()
            visited += 1
            var urlRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(
                node, kAXURLAttribute as CFString, &urlRef
            ) == .success, let urlRef {
                if let nsurl = urlRef as? NSURL { return nsurl.absoluteString }
                if let str = urlRef as? String { return str }
            }
            var childrenRef: CFTypeRef?
            if AXUIElementCopyAttributeValue(
                node, kAXChildrenAttribute as CFString, &childrenRef
            ) == .success, let children = childrenRef as? [AXUIElement] {
                queue.append(contentsOf: children.prefix(5))
            }
        }
        return nil
    }

    // MARK: - Ignored apps

    private static let ignoredBundleIDs: Set<String> = [
        "com.apple.loginwindow",
        "com.apple.notificationcenterui",
        "com.apple.controlcenter",
        "com.apple.Spotlight",
        "com.apple.dock",
        "com.apple.SystemUIServer",
        "com.henryzhang.autocompleter.bootstrap",
    ]
}
