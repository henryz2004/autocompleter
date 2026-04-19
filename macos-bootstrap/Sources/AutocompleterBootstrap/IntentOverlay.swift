import AppKit
import Foundation

/// Small always-on-top floating panel that shows the latest trajectory-intent
/// guess. Positioned near the top-right of the main screen. Experimental —
/// lives on the `trajectory-intent-probe` branch only.
///
/// The panel uses `NSPanel` with `.nonactivatingPanel` so clicking it doesn't
/// steal focus from whatever the user is doing. It never grabs keyboard input.
/// Visibility is tied to the most recent guess: we show the panel when the
/// first guess arrives and leave it there, updating in place, until the
/// user quits the app.
@MainActor
final class IntentOverlay {
    private var panel: NSPanel?
    private var label: NSTextField?
    private var meta: NSTextField?
    /// Timestamp of the most recent guess, for age display.
    private var lastGuessAt: Date?
    /// Refresh the "age" label once a second so "5s ago" stays accurate
    /// without forcing a query.
    private var ageTimer: Timer?

    // Panel sizing constants. The panel pins to the top-right of the
    // visibleFrame and grows downward as the sentence gets longer, capped at
    // maxBodyHeight so a verbose model doesn't eat the screen.
    private let panelWidth: CGFloat = 420
    private let horizPad: CGFloat = 12
    private let titleAreaHeight: CGFloat = 18  // top "intent" header + gap
    private let metaAreaHeight: CGFloat = 18   // bottom meta row + gap
    private let bodyTopPadding: CGFloat = 4
    private let bodyBottomPadding: CGFloat = 4
    private let minBodyHeight: CGFloat = 32
    private let maxBodyHeight: CGFloat = 500

    func show() {
        if panel == nil {
            panel = makePanel()
        }
        panel?.orderFrontRegardless()
        startAgeTimer()
    }

    func hide() {
        panel?.orderOut(nil)
        ageTimer?.invalidate()
        ageTimer = nil
    }

    /// Push a new guess into the overlay. Shows the panel on first call.
    func update(sentence: String, latencyMs: Double?, model: String?) {
        if panel == nil {
            panel = makePanel()
        }
        label?.stringValue = sentence
        lastGuessAt = Date()
        resizePanel(forBodyText: sentence)
        renderMeta(latencyMs: latencyMs, model: model)
        panel?.orderFrontRegardless()
        startAgeTimer()
    }

    /// Display a transient status line without a proper guess (e.g. "no LLM
    /// key configured"). Replaces whatever was showing.
    func updateStatus(_ text: String) {
        if panel == nil {
            panel = makePanel()
        }
        label?.stringValue = text
        lastGuessAt = nil
        meta?.stringValue = ""
        resizePanel(forBodyText: text)
        panel?.orderFrontRegardless()
    }

    // MARK: - Panel construction

    private func makePanel() -> NSPanel {
        let initialHeight = titleAreaHeight + bodyTopPadding + minBodyHeight
            + bodyBottomPadding + metaAreaHeight
        let origin = topRightOrigin(for: initialHeight)

        let panel = NSPanel(
            contentRect: NSRect(origin: origin, size: NSSize(width: panelWidth, height: initialHeight)),
            styleMask: [.borderless, .nonactivatingPanel, .hudWindow],
            backing: .buffered,
            defer: false
        )
        panel.level = .statusBar
        panel.isFloatingPanel = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.backgroundColor = NSColor.black.withAlphaComponent(0.72)
        panel.hasShadow = true
        panel.isOpaque = false
        panel.ignoresMouseEvents = true
        panel.hidesOnDeactivate = false

        let container = NSView(frame: NSRect(x: 0, y: 0, width: panelWidth, height: initialHeight))
        container.autoresizingMask = [.width, .height]

        let titleView = NSTextField(labelWithString: "intent")
        titleView.font = NSFont.systemFont(ofSize: 9, weight: .semibold)
        titleView.textColor = NSColor.white.withAlphaComponent(0.55)
        titleView.autoresizingMask = [.minYMargin, .width]  // pin to top
        titleView.frame = NSRect(
            x: horizPad,
            y: initialHeight - titleAreaHeight + 4,
            width: panelWidth - horizPad * 2,
            height: 12
        )
        container.addSubview(titleView)

        let label = NSTextField(wrappingLabelWithString: "(waiting for activity…)")
        label.font = NSFont.systemFont(ofSize: 12, weight: .regular)
        label.textColor = .white
        label.isEditable = false
        label.isBordered = false
        label.drawsBackground = false
        label.lineBreakMode = .byWordWrapping
        // 0 = as many lines as fit. We control bounding via the label frame.
        label.maximumNumberOfLines = 0
        label.autoresizingMask = [.width, .height]
        label.frame = bodyFrame(forPanelHeight: initialHeight)
        container.addSubview(label)
        self.label = label

        let meta = NSTextField(labelWithString: "")
        meta.font = NSFont.monospacedSystemFont(ofSize: 9, weight: .regular)
        meta.textColor = NSColor.white.withAlphaComponent(0.55)
        meta.autoresizingMask = [.maxYMargin, .width]  // pin to bottom
        meta.frame = NSRect(x: horizPad, y: 6, width: panelWidth - horizPad * 2, height: 12)
        container.addSubview(meta)
        self.meta = meta

        panel.contentView = container
        return panel
    }

    // MARK: - Sizing helpers

    /// Compute the content-rect origin (bottom-left, AppKit convention) so
    /// the panel's *top-right corner* pins to the same place on the screen
    /// regardless of height.
    private func topRightOrigin(for height: CGFloat) -> NSPoint {
        let screen = NSScreen.main ?? NSScreen.screens.first
        let screenFrame = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let x = screenFrame.maxX - panelWidth - 16
        let y = screenFrame.maxY - height - 16
        return NSPoint(x: x, y: y)
    }

    private func bodyFrame(forPanelHeight h: CGFloat) -> NSRect {
        let bodyHeight = h - titleAreaHeight - bodyTopPadding - bodyBottomPadding - metaAreaHeight
        return NSRect(
            x: horizPad,
            y: metaAreaHeight + bodyBottomPadding,
            width: panelWidth - horizPad * 2,
            height: max(bodyHeight, minBodyHeight)
        )
    }

    /// Measure the sentence at the label's font and return how tall the
    /// body text needs to be. Clamped to [minBodyHeight, maxBodyHeight].
    private func measureBodyHeight(for text: String) -> CGFloat {
        guard let label else { return minBodyHeight }
        let bodyWidth = panelWidth - horizPad * 2
        let ns = text as NSString
        let attrs: [NSAttributedString.Key: Any] = [.font: label.font ?? NSFont.systemFont(ofSize: 12)]
        let rect = ns.boundingRect(
            with: NSSize(width: bodyWidth, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: attrs
        )
        let measured = ceil(rect.height)
        return min(max(measured, minBodyHeight), maxBodyHeight)
    }

    /// Resize the panel so the body exactly fits `sentence`, keeping the
    /// top-right corner anchored. Called from `update(...)` and
    /// `updateStatus(...)`.
    private func resizePanel(forBodyText sentence: String) {
        guard let panel else { return }
        let bodyHeight = measureBodyHeight(for: sentence)
        let newHeight = titleAreaHeight + bodyTopPadding + bodyHeight
            + bodyBottomPadding + metaAreaHeight
        let origin = topRightOrigin(for: newHeight)
        panel.setFrame(
            NSRect(origin: origin, size: NSSize(width: panelWidth, height: newHeight)),
            display: true,
            animate: false
        )
        // Autoresizing keeps title pinned-top and meta pinned-bottom; the
        // label's frame needs to be set explicitly because it fills the gap.
        label?.frame = bodyFrame(forPanelHeight: newHeight)
    }

    private func startAgeTimer() {
        ageTimer?.invalidate()
        ageTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.refreshAge()
            }
        }
    }

    private func refreshAge() {
        renderMeta(latencyMs: lastLatency, model: lastModel)
    }

    // Retain the last guess's meta so the 1s age tick can re-render without
    // the caller passing them in again.
    private var lastLatency: Double?
    private var lastModel: String?

    private func renderMeta(latencyMs: Double?, model: String?) {
        lastLatency = latencyMs
        lastModel = model

        var pieces: [String] = []
        if let ts = lastGuessAt {
            let age = Int(Date().timeIntervalSince(ts))
            pieces.append("\(age)s ago")
        }
        if let ms = latencyMs {
            pieces.append("\(Int(ms))ms")
        }
        if let model {
            pieces.append(model)
        }
        meta?.stringValue = pieces.joined(separator: "  ·  ")
    }
}
