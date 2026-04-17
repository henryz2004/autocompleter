import SwiftUI
import AppKit

/// Observable state for the four diagnostic checks.
@MainActor
final class DiagnosticsModel: ObservableObject {
    enum CheckStatus: String, Codable {
        case pending
        case running
        case pass
        case fail
    }

    struct CheckResult: Identifiable, Codable {
        let id: String
        var title: String
        var status: CheckStatus
        var detail: String
    }

    @Published var checks: [CheckResult] = [
        .init(id: "tcc", title: "Accessibility permission granted", status: .pending, detail: ""),
        .init(id: "focus", title: "AXFocusedUIElement returns a non-nil element", status: .pending, detail: ""),
        .init(id: "tap", title: "CGEventTapCreate succeeds", status: .pending, detail: ""),
        .init(id: "e2e", title: "Hotkey fires → captures focused element value", status: .pending, detail: "Waiting for you to press the hotkey in a text field..."),
    ]

    @Published var hotkeyString: String = "ctrl+space"
    /// Redacted JSON of the last hotkey-fired snapshot. Safe to show in the
    /// UI and to copy to the pasteboard — the raw AXValue is NOT included.
    @Published var lastCaptureRedactedJSON: String = ""

    private var hotkeyTap: HotkeyTap?

    // MARK: - Check #1: accessibility permission

    func runTCCCheck(prompt: Bool) {
        update("tcc", .running, "Calling AXIsProcessTrustedWithOptions...")
        let trusted = AXProbe.isProcessTrusted(prompt: prompt)
        update(
            "tcc",
            trusted ? .pass : .fail,
            trusted
                ? "AXIsProcessTrustedWithOptions returned true."
                : "Not trusted. Open System Settings → Privacy & Security → Accessibility, enable AutocompleterBootstrap, then re-run."
        )
    }

    // MARK: - Check #2: focused element readable

    func runFocusCheck() {
        update("focus", .running, "Reading AXFocusedUIElement from frontmost app...")
        do {
            let snap = try AXProbe.snapshotFocusedElement()
            let insertion = snap.insertionPoint.map(String.init) ?? "nil"
            let fromChildren = snap.valueFromChildText ? " (via child-text fallback)" : ""
            update("focus", .pass, "app=\(snap.appName), role=\(snap.role), valueLen=\(snap.valueLength), insertion=\(insertion)\(fromChildren)")
        } catch {
            update("focus", .fail, String(describing: error))
        }
    }

    // MARK: - Check #3: tap creation

    func runTapCreationCheck() {
        update("tap", .running, "Creating and immediately tearing down a CGEvent tap...")
        let ok = HotkeyTap.canCreateTap()
        update("tap", ok ? .pass : .fail, ok
            ? "Tap creation succeeded."
            : "CGEvent.tapCreate returned nil. Usually indicates missing accessibility permission.")
    }

    // MARK: - Check #4: end-to-end hotkey capture

    func armHotkeyCapture() {
        // Tear down any previous tap first.
        hotkeyTap?.stop()
        hotkeyTap = nil

        let hk: HotkeyTap.Hotkey
        do {
            hk = try HotkeyTap.Hotkey.parse(hotkeyString)
        } catch {
            update("e2e", .fail, "Could not parse hotkey '\(hotkeyString)': \(error)")
            return
        }

        let tap = HotkeyTap(hotkey: hk)
        tap.onFire = { [weak self] in
            self?.handleHotkeyFire()
        }
        tap.onTapCreateFailed = { [weak self] in
            self?.update("e2e", .fail, "CGEvent.tapCreate failed inside the background thread. Usually means accessibility permission is missing or was just revoked — re-run the permission check.")
        }
        tap.start()
        hotkeyTap = tap

        update("e2e", .pending, "Armed. Focus a text field in any app and press \(hotkeyString).")
    }

    private func handleHotkeyFire() {
        do {
            let snap = try AXProbe.snapshotFocusedElement()
            let redacted = RedactedFocusedElementSnapshot(from: snap)
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? encoder.encode(redacted),
               let s = String(data: data, encoding: .utf8) {
                lastCaptureRedactedJSON = s
            }

            // v0.1 success criterion: we actually got something back. A
            // non-nil insertion_point OR non-empty value (including via
            // child-text fallback) is enough to prove AX is working.
            let gotSomething = snap.insertionPoint != nil || snap.valueLength > 0
            if !gotSomething {
                update("e2e", .fail, "Hotkey fired, but AXValue was empty and no insertion point could be read. This is the macOS 15 failure we're trying to reproduce.")
            } else {
                var detail = "Captured \(snap.valueLength) chars from \(snap.appName) (\(snap.role))."
                if snap.valueFromChildText {
                    detail += " Recovered via child-text fallback (Electron/Chromium)."
                }
                detail += " See 'Last capture' for the redacted JSON."
                update("e2e", .pass, detail)
            }
        } catch {
            update("e2e", .fail, "Hotkey fired but capture failed: \(error)")
        }
    }

    // MARK: - Report

    func diagnosticReportJSON() -> String {
        struct Report: Codable {
            var checks: [CheckResult]
            var osVersion: String
            var hotkey: String
            var lastCaptureRedacted: String
            var note: String
        }
        let report = Report(
            checks: checks,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            hotkey: hotkeyString,
            lastCaptureRedacted: lastCaptureRedactedJSON,
            note: "Raw AXValue is redacted. Only length + first-character preview is included."
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = (try? encoder.encode(report)) ?? Data()
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    func copyReportToPasteboard() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(diagnosticReportJSON(), forType: .string)
    }

    // MARK: - helpers

    private func update(_ id: String, _ status: CheckStatus, _ detail: String) {
        guard let idx = checks.firstIndex(where: { $0.id == id }) else { return }
        checks[idx].status = status
        checks[idx].detail = detail
    }
}

// MARK: - View

struct DiagnosticsView: View {
    @StateObject var model = DiagnosticsModel()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("AutocompleterBootstrap Diagnostics")
                .font(.title2).bold()

            Text("Runs the four checks needed to confirm the Accessibility + CGEvent infrastructure works on this Mac. If any check fails, click 'Copy diagnostic report' and paste it back to Henry.")
                .font(.callout)
                .foregroundStyle(.secondary)

            Text("The captured text value is redacted from the report — only its length and first character are included, so you won't accidentally share the contents of a password field or private chat.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Divider()

            ForEach(model.checks) { check in
                CheckRow(check: check)
            }

            Divider()

            HStack {
                Text("Hotkey:")
                TextField("ctrl+space", text: $model.hotkeyString)
                    .frame(width: 160)
            }

            HStack(spacing: 8) {
                Button("Run permission check") {
                    model.runTCCCheck(prompt: true)
                }
                Button("Run focused-element check") {
                    model.runFocusCheck()
                }
                Button("Run tap-creation check") {
                    model.runTapCreationCheck()
                }
                Button("Arm hotkey capture") {
                    model.armHotkeyCapture()
                }
            }

            HStack {
                Button("Run all") {
                    model.runTCCCheck(prompt: true)
                    model.runFocusCheck()
                    model.runTapCreationCheck()
                    model.armHotkeyCapture()
                }
                .keyboardShortcut(.defaultAction)

                Spacer()

                Button("Copy diagnostic report") {
                    model.copyReportToPasteboard()
                }
            }

            if !model.lastCaptureRedactedJSON.isEmpty {
                GroupBox(label: Text("Last capture (redacted)")) {
                    ScrollView {
                        Text(model.lastCaptureRedactedJSON)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 180)
                }
            }
        }
        .padding(20)
        .frame(width: 620)
    }
}

private struct CheckRow: View {
    let check: DiagnosticsModel.CheckResult

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text(statusSymbol)
                .font(.system(size: 16))
                .frame(width: 20)
            VStack(alignment: .leading, spacing: 2) {
                Text(check.title).bold()
                if !check.detail.isEmpty {
                    Text(check.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer()
        }
    }

    private var statusSymbol: String {
        switch check.status {
        case .pending: return "○"
        case .running: return "…"
        case .pass:    return "✓"
        case .fail:    return "✗"
        }
    }
}
