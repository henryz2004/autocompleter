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
        // The end-to-end hotkey test is superseded by the Worker (Phase A)
        // pane, which already observes the worker's real hotkey tap. Installing
        // a *second* tap here would head-insert ahead of the worker's tap and
        // consume ctrl+space before the worker saw it — so that check is gone.
    ]

    @Published var hotkeyString: String = "ctrl+space"
    /// Redacted JSON of the last hotkey-fired snapshot. Safe to show in the
    /// UI and to copy to the pasteboard — the raw AXValue is NOT included.
    @Published var lastCaptureRedactedJSON: String = ""

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

    // MARK: - Report

    struct WorkerReport: Codable {
        var alive: Bool
        var exitedAbnormally: Bool
        var lastTriggerSeq: Int
        var lastInjectSeq: Int
        var logTail: [String]
    }

    func diagnosticReportJSON(worker: WorkerReport? = nil) -> String {
        struct Report: Codable {
            var checks: [CheckResult]
            var osVersion: String
            var hotkey: String
            var lastCaptureRedacted: String
            var worker: WorkerReport?
            var note: String
        }
        let report = Report(
            checks: checks,
            osVersion: ProcessInfo.processInfo.operatingSystemVersionString,
            hotkey: hotkeyString,
            lastCaptureRedacted: lastCaptureRedactedJSON,
            worker: worker,
            note: "Raw AXValue is redacted. Only length + first-character preview is included."
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = (try? encoder.encode(report)) ?? Data()
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    func copyReportToPasteboard(worker: WorkerReport? = nil) {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(diagnosticReportJSON(worker: worker), forType: .string)
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
    /// Shared worker-status model owned by `AppDelegate`. Must be injected at
    /// the call site — a default value closure here caused SwiftUI to silently
    /// bind to a detached fresh model on re-init, leaving the pane stuck on
    /// "stopped" even though the worker was alive.
    @ObservedObject var workerStatus: WorkerStatusModel

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
            }

            HStack {
                Button("Run all") {
                    model.runTCCCheck(prompt: true)
                    model.runFocusCheck()
                    model.runTapCreationCheck()
                }
                .keyboardShortcut(.defaultAction)

                Spacer()

                Button("Copy diagnostic report") {
                    let report = DiagnosticsModel.WorkerReport(
                        alive: workerStatus.workerAlive,
                        exitedAbnormally: workerStatus.workerExitedAbnormally,
                        lastTriggerSeq: workerStatus.lastTriggerSeq,
                        lastInjectSeq: workerStatus.lastInjectSeq,
                        logTail: workerStatus.lines
                    )
                    model.copyReportToPasteboard(worker: report)
                }
            }

            Divider()

            HStack(spacing: 8) {
                Button("Trajectory Inspector…") {
                    NSApp.sendAction(#selector(AppDelegate.showTrajectoryInspector), to: nil, from: nil)
                }
                Button("Capture Inspector…") {
                    NSApp.sendAction(#selector(AppDelegate.showCaptureInspector), to: nil, from: nil)
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

            WorkerPane(status: workerStatus)
        }
        .padding(20)
        .frame(width: 620)
    }
}

/// Phase A worker status pane. Tails every IPC event so dogfood debugging
/// doesn't require opening a log file.
private struct WorkerPane: View {
    @ObservedObject var status: WorkerStatusModel

    var body: some View {
        GroupBox(label: Text("Worker (Phase A)")) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 12) {
                    Label(
                        status.workerAlive ? "running" : "stopped",
                        systemImage: status.workerAlive ? "circle.fill" : "circle"
                    )
                    .foregroundStyle(status.workerAlive ? .green : .secondary)
                    Text("triggers: \(status.lastTriggerSeq)")
                    Text("injects: \(status.lastInjectSeq)")
                    if status.workerExitedAbnormally {
                        Text("(last exit was abnormal)")
                            .foregroundStyle(.red)
                    }
                    Spacer()
                }
                .font(.caption)

                ScrollView {
                    Text(status.lines.joined(separator: "\n"))
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 180)
            }
            .padding(6)
        }
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
