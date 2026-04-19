import SwiftUI
import AppKit

/// Developer debug window showing captured screenshots + OCR output.
/// Lets us visually verify: is the right window being captured? Is OCR
/// producing useful text? What's noise vs signal?
struct CaptureInspectorView: View {
    @ObservedObject var store: CaptureStore
    @State private var selectedID: UUID?

    var body: some View {
        NavigationSplitView {
            captureList
                .navigationSplitViewColumnWidth(min: 200, ideal: 240, max: 300)
        } detail: {
            if let entry = selectedEntry {
                CaptureDetailView(entry: entry)
            } else {
                Text("Select a capture to inspect")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
    }

    private var selectedEntry: CaptureEntry? {
        guard let id = selectedID else { return nil }
        return store.entries.first { $0.id == id }
    }

    private var captureList: some View {
        List(selection: $selectedID) {
            ForEach(store.entries.reversed()) { entry in
                CaptureRow(entry: entry)
                    .tag(entry.id)
            }
        }
        .onChange(of: store.entries.count) { _, _ in
            // Auto-select the newest capture if nothing is selected.
            if selectedID == nil, let newest = store.entries.last {
                selectedID = newest.id
            }
        }
    }
}

// MARK: - Row

private struct CaptureRow: View {
    let entry: CaptureEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(entry.appName)
                    .font(.caption)
                    .bold()
                    .lineLimit(1)
                Spacer()
                Text(timeString)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            if let title = entry.windowTitle, !title.isEmpty {
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            HStack(spacing: 8) {
                Text(entry.trigger)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(triggerColor)
                Text("\(entry.ocrText.count) chars")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(String(format: "%.0fms", entry.ocrLatencyMs))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
    }

    private var timeString: String {
        let fmt = DateFormatter()
        fmt.dateFormat = "HH:mm:ss"
        return fmt.string(from: entry.timestamp)
    }

    private var triggerColor: Color {
        switch entry.trigger {
        case "app_switch": return .blue
        case "title_change": return .orange
        case "window_switch": return .green
        default: return .primary
        }
    }
}

// MARK: - Detail

private struct CaptureDetailView: View {
    let entry: CaptureEntry
    @State private var image: NSImage?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                // Screenshot preview
                GroupBox("Screenshot") {
                    if let img = image {
                        Image(nsImage: img)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .frame(maxHeight: 400)
                    } else {
                        Text("Loading...")
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, minHeight: 200)
                    }
                }

                // OCR output
                GroupBox {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("OCR Output")
                                .font(.caption).bold()
                            Spacer()
                            Text("\(entry.ocrText.count) chars")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            Button("Copy") {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(
                                    entry.ocrText, forType: .string
                                )
                            }
                            .buttonStyle(.borderless)
                            .font(.caption2)
                        }
                        ScrollView {
                            Text(entry.ocrText.isEmpty ? "(no text recognized)" : entry.ocrText)
                                .font(.system(.caption, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 300)
                    }
                }

                // Metadata
                GroupBox("Metadata") {
                    Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 12, verticalSpacing: 4) {
                        GridRow {
                            label("App"); Text(entry.appName)
                        }
                        GridRow {
                            label("Trigger"); Text(entry.trigger)
                        }
                        GridRow {
                            label("Time")
                            Text(entry.timestamp.formatted(
                                .dateTime.hour().minute().second()
                            ))
                        }
                        GridRow {
                            label("OCR Latency")
                            Text(String(format: "%.0f ms", entry.ocrLatencyMs))
                        }
                        if let title = entry.windowTitle {
                            GridRow { label("Window"); Text(title) }
                        }
                        if let url = entry.url {
                            GridRow {
                                label("URL")
                                Text(url)
                                    .textSelection(.enabled)
                                    .lineLimit(2)
                            }
                        }
                        if let bid = entry.bundleID {
                            GridRow { label("Bundle"); Text(bid) }
                        }
                        GridRow {
                            label("File")
                            Text(entry.screenshotPath.lastPathComponent)
                                .textSelection(.enabled)
                        }
                    }
                    .font(.system(.caption, design: .monospaced))
                }
            }
            .padding()
        }
        .onAppear { image = entry.loadImage() }
        .onChange(of: entry.id) { _, _ in image = entry.loadImage() }
    }

    private func label(_ text: String) -> some View {
        Text(text)
            .foregroundStyle(.secondary)
            .frame(width: 80, alignment: .trailing)
    }
}
