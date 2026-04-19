import SwiftUI
import AppKit

/// Developer-facing debug/tuning window for the trajectory-intent-probe.
/// Shows the snapshot ring buffer, the assembled LLM prompt, the latest
/// guess, and sliders for tuning capture parameters live.
struct TrajectoryInspectorView: View {
    @ObservedObject var model: TrajectoryInspectorModel

    var body: some View {
        NavigationSplitView {
            snapshotList
                .navigationSplitViewColumnWidth(min: 220, ideal: 260, max: 320)
        } detail: {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if let idx = model.selectedSnapshotIndex,
                       idx < model.snapshots.count {
                        SnapshotDetailSection(
                            snapshot: model.snapshots[idx],
                            baseTimestamp: model.snapshots.first?.timestamp ?? 0
                        )
                    }
                    PromptPreviewSection(promptText: model.promptPreview)
                    LatestGuessSection(guess: model.latestGuess)
                    TuningSection(model: model)
                }
                .padding()
            }
        }
        .onChange(of: model.cooldownInterval) { _, val in model.applyCooldownInterval(val) }
        .onChange(of: model.previewCharBudget) { _, val in model.applyPreviewCharBudget(val) }
        .onChange(of: model.heartbeatInterval) { _, val in model.applyHeartbeatInterval(val) }
        .onChange(of: model.bufferCapacity) { _, val in model.applyBufferCapacity(val) }
    }

    // MARK: - Snapshot list (sidebar)

    private var snapshotList: some View {
        List(selection: $model.selectedSnapshotIndex) {
            ForEach(Array(model.snapshots.enumerated()), id: \.offset) { idx, snap in
                SnapshotRow(
                    snapshot: snap,
                    baseTimestamp: model.snapshots.first?.timestamp ?? 0
                )
                .tag(idx)
            }
        }
    }
}

// MARK: - Snapshot row

private struct SnapshotRow: View {
    let snapshot: TrajectoryStack.Snapshot
    let baseTimestamp: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text(triggerBadge)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(triggerColor)
                    .frame(width: 55, alignment: .leading)
                Text(snapshot.appName)
                    .font(.caption)
                    .bold()
                    .lineLimit(1)
                Spacer()
                Text(deltaString)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            if let title = snapshot.windowTitle, !title.isEmpty {
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 2)
    }

    private var triggerBadge: String {
        switch snapshot.trigger {
        case "app_switch": return "app"
        case "title_change": return "title"
        case "focus_change": return "focus"
        case "heartbeat": return "beat"
        case "initial": return "init"
        default: return snapshot.trigger
        }
    }

    private var triggerColor: Color {
        switch snapshot.trigger {
        case "app_switch": return .blue
        case "title_change": return .orange
        case "focus_change": return .green
        case "heartbeat": return .gray
        case "initial": return .purple
        default: return .primary
        }
    }

    private var deltaString: String {
        let dt = Int(max(0, snapshot.timestamp - baseTimestamp))
        let dwell = Int(max(0, snapshot.lastSeen - snapshot.timestamp))
        if dwell > 0 {
            return "+\(dt)s (\(dwell)s)"
        }
        return "+\(dt)s"
    }
}

// MARK: - Snapshot detail

private struct SnapshotDetailSection: View {
    let snapshot: TrajectoryStack.Snapshot
    let baseTimestamp: Double

    var body: some View {
        GroupBox("Snapshot Detail") {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 4) {
                GridRow { fieldLabel("App"); Text(snapshot.appName) }
                GridRow { fieldLabel("PID"); Text("\(snapshot.appPID)") }
                GridRow { fieldLabel("Trigger"); Text(snapshot.trigger) }
                GridRow { fieldLabel("Time"); Text("+\(Int(snapshot.timestamp - baseTimestamp))s") }
                GridRow { fieldLabel("Dwell"); Text("\(Int(snapshot.lastSeen - snapshot.timestamp))s") }
                if let bid = snapshot.bundleID {
                    GridRow { fieldLabel("Bundle"); Text(bid) }
                }
                if let title = snapshot.windowTitle {
                    GridRow { fieldLabel("Window"); Text(title).textSelection(.enabled) }
                }
                if let url = snapshot.url {
                    GridRow { fieldLabel("URL"); Text(url).textSelection(.enabled) }
                }
                if let role = snapshot.focusedRole {
                    GridRow { fieldLabel("Role"); Text(role) }
                }
                if let len = snapshot.focusedValueLength {
                    GridRow { fieldLabel("Value Len"); Text("\(len) chars") }
                }
            }
            .font(.system(.caption, design: .monospaced))

            if let preview = snapshot.focusedValuePreview, !preview.isEmpty {
                Divider()
                Text("Value Preview")
                    .font(.caption).bold()
                ScrollView {
                    Text(preview)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 120)
            }
        }
    }

    private func fieldLabel(_ text: String) -> some View {
        Text(text)
            .foregroundStyle(.secondary)
            .frame(width: 70, alignment: .trailing)
    }
}

// MARK: - Prompt preview

private struct PromptPreviewSection: View {
    let promptText: String

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Assembled Prompt")
                        .font(.caption).bold()
                    Spacer()
                    Text("\(promptText.count) chars")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(promptText, forType: .string)
                    }
                    .buttonStyle(.borderless)
                    .font(.caption2)
                }
                ScrollView([.horizontal, .vertical]) {
                    Text(promptText)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 200)
            }
        }
    }
}

// MARK: - Latest guess

private struct LatestGuessSection: View {
    let guess: IPCMessage.TrajectoryGuess?

    var body: some View {
        GroupBox("Latest Guess") {
            if let g = guess {
                VStack(alignment: .leading, spacing: 4) {
                    Text(g.intentSentence)
                        .font(.callout)
                        .textSelection(.enabled)
                    HStack(spacing: 12) {
                        if let model = g.modelUsed {
                            Text(model).font(.caption2).foregroundStyle(.secondary)
                        }
                        if let ms = g.latencyMs {
                            Text("\(Int(ms))ms").font(.caption2).foregroundStyle(.secondary)
                        }
                        if let conf = g.confidence {
                            Text(String(format: "conf=%.2f", conf))
                                .font(.caption2).foregroundStyle(.secondary)
                        }
                    }
                }
            } else {
                Text("No guess yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - Tuning sliders

private struct TuningSection: View {
    @ObservedObject var model: TrajectoryInspectorModel

    var body: some View {
        GroupBox("Tuning") {
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 8) {
                GridRow {
                    Text("Preview chars").font(.caption).frame(width: 100, alignment: .trailing)
                    Slider(
                        value: Binding(
                            get: { Double(model.previewCharBudget) },
                            set: { model.previewCharBudget = Int($0) }
                        ),
                        in: 100...5000,
                        step: 100
                    )
                    Text("\(model.previewCharBudget)")
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 50, alignment: .trailing)
                }
                GridRow {
                    Text("Cooldown (s)").font(.caption).frame(width: 100, alignment: .trailing)
                    Slider(value: $model.cooldownInterval, in: 0.5...30.0, step: 0.5)
                    Text(String(format: "%.1f", model.cooldownInterval))
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 50, alignment: .trailing)
                }
                GridRow {
                    Text("Heartbeat (s)").font(.caption).frame(width: 100, alignment: .trailing)
                    Slider(value: $model.heartbeatInterval, in: 5...120, step: 5)
                    Text("\(Int(model.heartbeatInterval))")
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 50, alignment: .trailing)
                }
                GridRow {
                    Text("Buffer cap").font(.caption).frame(width: 100, alignment: .trailing)
                    Slider(
                        value: Binding(
                            get: { Double(model.bufferCapacity) },
                            set: { model.bufferCapacity = Int($0) }
                        ),
                        in: 10...1000,
                        step: 10
                    )
                    Text("\(model.bufferCapacity)")
                        .font(.system(.caption, design: .monospaced))
                        .frame(width: 50, alignment: .trailing)
                }
            }
        }
    }
}
