import AppKit
import Foundation

/// One screenshot + OCR capture entry.
struct CaptureEntry: Identifiable {
    let id: UUID
    let timestamp: Date
    let trigger: String          // "app_switch" | "title_change" | "window_switch" | "manual"
    let appName: String
    let bundleID: String?
    let windowTitle: String?
    let url: String?
    let screenshotPath: URL
    let ocrText: String
    let ocrLatencyMs: Double
    /// How many times this capture's content has been pasted. Used for
    /// soft-consume: pasted entries are deprioritized but not removed.
    var timesUsed: Int = 0

    var isManual: Bool { trigger == "manual" }

    /// Load the screenshot image from disk. Returns nil if the file was
    /// deleted or is unreadable.
    func loadImage() -> NSImage? {
        return NSImage(contentsOf: screenshotPath)
    }
}

/// Rolling buffer of recent screenshot captures, observable by SwiftUI.
@MainActor
final class CaptureStore: ObservableObject {
    @Published private(set) var entries: [CaptureEntry] = []

    private let capacity: Int

    init(capacity: Int = 50) {
        self.capacity = capacity
    }

    func append(_ entry: CaptureEntry) {
        entries.append(entry)
        if entries.count > capacity {
            // Remove oldest entries. Optionally delete their screenshot files.
            let overflow = entries.count - capacity
            let removed = Array(entries.prefix(overflow))
            entries.removeFirst(overflow)
            // Clean up disk — best effort, don't block on failure.
            for old in removed {
                try? FileManager.default.removeItem(at: old.screenshotPath)
            }
        }
    }

    /// Manual captures only, sorted for paste: least-used first, then
    /// most recent. Entries marked as used are deprioritized but not removed.
    var manualEntriesForPaste: [CaptureEntry] {
        entries
            .filter { $0.isManual }
            .sorted {
                if $0.timesUsed != $1.timesUsed {
                    return $0.timesUsed < $1.timesUsed  // least-used first
                }
                return $0.timestamp > $1.timestamp      // then most recent
            }
    }

    /// Soft-consume: increment timesUsed for the entry whose OCR text
    /// best matches the pasted text (substring match).
    func markUsed(pastedText: String) {
        guard !pastedText.isEmpty else { return }
        // Find the manual entry whose OCR contains the pasted text (or
        // the closest match by shared prefix).
        if let idx = entries.firstIndex(where: {
            $0.isManual && $0.ocrText.contains(pastedText.prefix(50))
        }) {
            entries[idx].timesUsed += 1
        }
    }

    /// Total OCR chars captured across all entries.
    var totalOCRChars: Int {
        entries.reduce(0) { $0 + $1.ocrText.count }
    }
}
