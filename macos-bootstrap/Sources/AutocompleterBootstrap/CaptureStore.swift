import AppKit
import Foundation

/// One screenshot + OCR capture entry.
struct CaptureEntry: Identifiable {
    let id: UUID
    let timestamp: Date
    let trigger: String          // "app_switch" | "title_change" | "window_switch"
    let appName: String
    let bundleID: String?
    let windowTitle: String?
    let url: String?
    let screenshotPath: URL
    let ocrText: String
    let ocrLatencyMs: Double

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

    /// Total OCR chars captured across all entries.
    var totalOCRChars: Int {
        entries.reduce(0) { $0 + $1.ocrText.count }
    }
}
