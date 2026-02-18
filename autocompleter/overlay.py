"""Overlay Renderer - system-level floating overlay for suggestions.

Displays suggestions in a minimal dropdown anchored near the cursor position.
Uses PyObjC to create a borderless, floating NSWindow that stays above all
other windows. Supports keyboard navigation: arrow keys to select,
Tab/Enter to accept, Esc to dismiss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .suggestion_engine import Suggestion

logger = logging.getLogger(__name__)

try:
    import AppKit
    import objc
    from Foundation import NSMakeRect, NSObject

    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False


@dataclass
class OverlayConfig:
    width: int = 400
    max_height: int = 200
    font_size: int = 13
    opacity: float = 0.95
    bg_color: tuple[float, float, float] = (0.15, 0.15, 0.17)
    text_color: tuple[float, float, float] = (0.92, 0.92, 0.94)
    highlight_color: tuple[float, float, float] = (0.25, 0.45, 0.75)
    border_radius: float = 8.0
    padding: float = 8.0
    item_height: float = 32.0


if HAS_APPKIT:

    class SuggestionOverlayView(AppKit.NSView):
        """Custom view that renders the suggestion list."""

        def initWithFrame_config_(self, frame, config):
            self = objc.super(SuggestionOverlayView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._config = config
            self._suggestions: list[Suggestion] = []
            self._selected_index: int = 0
            return self

        def setSuggestions_(self, suggestions):
            self._suggestions = suggestions
            self._selected_index = 0
            self.setNeedsDisplay_(True)

        def setSelectedIndex_(self, index):
            if 0 <= index < len(self._suggestions):
                self._selected_index = index
                self.setNeedsDisplay_(True)

        def drawRect_(self, rect):
            cfg = self._config

            # Background
            bg = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.bg_color[0], cfg.bg_color[1], cfg.bg_color[2], cfg.opacity
            )
            path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                self.bounds(), cfg.border_radius, cfg.border_radius
            )
            bg.set()
            path.fill()

            # Draw each suggestion
            font = AppKit.NSFont.systemFontOfSize_(cfg.font_size)
            text_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.text_color[0], cfg.text_color[1], cfg.text_color[2], 1.0
            )
            highlight = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.highlight_color[0],
                cfg.highlight_color[1],
                cfg.highlight_color[2],
                1.0,
            )

            y = self.bounds().size.height - cfg.padding

            for i, suggestion in enumerate(self._suggestions):
                y -= cfg.item_height
                item_rect = NSMakeRect(
                    cfg.padding,
                    y,
                    self.bounds().size.width - 2 * cfg.padding,
                    cfg.item_height,
                )

                # Highlight selected item
                if i == self._selected_index:
                    highlight_path = (
                        AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                            item_rect, 4.0, 4.0
                        )
                    )
                    highlight.set()
                    highlight_path.fill()

                # Draw text
                attrs = {
                    AppKit.NSFontAttributeName: font,
                    AppKit.NSForegroundColorAttributeName: text_color,
                }
                text = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                    suggestion.text, attrs
                )
                text_rect = NSMakeRect(
                    item_rect.origin.x + 8,
                    item_rect.origin.y + 6,
                    item_rect.size.width - 16,
                    cfg.item_height - 12,
                )
                text.drawInRect_(text_rect)


class SuggestionOverlay:
    """Manages the floating overlay window for displaying suggestions."""

    def __init__(self, config: OverlayConfig | None = None):
        self._config = config or OverlayConfig()
        self._window = None
        self._view = None
        self._suggestions: list[Suggestion] = []
        self._selected_index: int = 0
        self._on_accept = None

    @property
    def is_visible(self) -> bool:
        if not HAS_APPKIT or self._window is None:
            return False
        return bool(self._window.isVisible())

    def show(
        self,
        suggestions: list[Suggestion],
        x: float,
        y: float,
        on_accept=None,
    ) -> None:
        """Show the overlay with suggestions at the given screen coordinates."""
        if not HAS_APPKIT:
            logger.warning("AppKit not available; overlay cannot be shown.")
            return

        self._suggestions = suggestions
        self._selected_index = 0
        self._on_accept = on_accept

        if not suggestions:
            self.hide()
            return

        height = min(
            self._config.padding * 2
            + self._config.item_height * len(suggestions),
            self._config.max_height,
        )

        # Convert coordinates: macOS screen origin is bottom-left
        screen = AppKit.NSScreen.mainScreen()
        if screen:
            screen_height = screen.frame().size.height
            # Position overlay below the cursor
            y_flipped = screen_height - y - height
        else:
            y_flipped = y

        frame = NSMakeRect(x, y_flipped, self._config.width, height)

        if self._window is None:
            self._create_window(frame)
        else:
            self._window.setFrame_display_(frame, True)

        if self._view is not None:
            self._view.setSuggestions_(suggestions)
            self._view.setSelectedIndex_(0)

        self._window.orderFront_(None)

    def hide(self) -> None:
        """Hide the overlay."""
        if self._window is not None:
            self._window.orderOut_(None)

    def move_selection(self, delta: int) -> None:
        """Move the selection up or down."""
        if not self._suggestions:
            return
        self._selected_index = (
            self._selected_index + delta
        ) % len(self._suggestions)
        if self._view is not None:
            self._view.setSelectedIndex_(self._selected_index)

    def accept_selection(self) -> Suggestion | None:
        """Accept the currently selected suggestion and hide the overlay."""
        if not self._suggestions:
            return None
        suggestion = self._suggestions[self._selected_index]
        self.hide()
        if self._on_accept:
            self._on_accept(suggestion)
        return suggestion

    def _create_window(self, frame) -> None:
        """Create the borderless floating window."""
        if not HAS_APPKIT:
            return

        style = AppKit.NSWindowStyleMaskBorderless
        window = AppKit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setLevel_(AppKit.NSFloatingWindowLevel + 1)
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setHasShadow_(True)
        window.setIgnoresMouseEvents_(False)
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )

        view = (
            SuggestionOverlayView.alloc().initWithFrame_config_(
                frame, self._config
            )
        )
        window.setContentView_(view)

        self._window = window
        self._view = view
