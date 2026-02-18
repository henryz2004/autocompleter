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


def _measure_text_height(text: str, font, available_width: float) -> float:
    """Measure the height needed to render wrapped text."""
    if not HAS_APPKIT:
        return 20.0
    paragraph_style = AppKit.NSMutableParagraphStyle.alloc().init()
    paragraph_style.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)
    attrs = {
        AppKit.NSFontAttributeName: font,
        AppKit.NSParagraphStyleAttributeName: paragraph_style,
    }
    attr_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
        text, attrs
    )
    bounding = attr_str.boundingRectWithSize_options_(
        AppKit.NSMakeSize(available_width, 1e6),
        AppKit.NSStringDrawingUsesLineFragmentOrigin,
    )
    return bounding.size.height


def _compute_item_heights(
    suggestions: list[Suggestion], config: OverlayConfig,
) -> list[float]:
    """Compute per-item heights accounting for text wrapping."""
    if not HAS_APPKIT:
        return [config.item_height] * len(suggestions)
    font = AppKit.NSFont.systemFontOfSize_(config.font_size)
    # Available width for text inside each item (item padding: 8 each side + 8 inner each side)
    available_width = config.width - 2 * config.padding - 16
    heights = []
    for s in suggestions:
        text_h = _measure_text_height(s.text, font, available_width)
        item_h = max(config.item_height, text_h + 12)
        heights.append(item_h)
    return heights


def _compute_overlay_height(
    suggestions: list[Suggestion], config: OverlayConfig,
) -> float:
    """Compute total overlay height with dynamic per-item sizing."""
    item_heights = _compute_item_heights(suggestions, config)
    total = config.padding * 2 + sum(item_heights)
    return min(total, config.max_height)


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
            self._item_heights: list[float] = []
            return self

        def setSuggestions_(self, suggestions):
            self._suggestions = suggestions
            self._selected_index = 0
            self._item_heights = _compute_item_heights(suggestions, self._config)
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

            paragraph_style = AppKit.NSMutableParagraphStyle.alloc().init()
            paragraph_style.setLineBreakMode_(AppKit.NSLineBreakByWordWrapping)

            y = self.bounds().size.height - cfg.padding

            for i, suggestion in enumerate(self._suggestions):
                item_h = (
                    self._item_heights[i]
                    if i < len(self._item_heights)
                    else cfg.item_height
                )
                y -= item_h
                item_rect = NSMakeRect(
                    cfg.padding,
                    y,
                    self.bounds().size.width - 2 * cfg.padding,
                    item_h,
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

                # Draw text with word wrapping
                attrs = {
                    AppKit.NSFontAttributeName: font,
                    AppKit.NSForegroundColorAttributeName: text_color,
                    AppKit.NSParagraphStyleAttributeName: paragraph_style,
                }
                text = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                    suggestion.text, attrs
                )
                text_rect = NSMakeRect(
                    item_rect.origin.x + 8,
                    item_rect.origin.y + 6,
                    item_rect.size.width - 16,
                    item_h - 12,
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
        self._visible = False

    @property
    def is_visible(self) -> bool:
        return self._visible

    def show(
        self,
        suggestions: list[Suggestion],
        x: float,
        y: float,
        on_accept=None,
        caret_height: float = 20.0,
    ) -> None:
        """Show the overlay with suggestions at the given screen coordinates.

        x, y are in macOS screen coordinates (origin = top-left of primary
        screen, y increases downward — matching the Accessibility API).
        caret_height is the height of the caret in AX coordinates, used
        to position the overlay above the caret when flipping.
        """
        if not HAS_APPKIT:
            logger.warning("AppKit not available; overlay cannot be shown.")
            return

        self._suggestions = suggestions
        self._selected_index = 0
        self._on_accept = on_accept

        if not suggestions:
            self.hide()
            return

        height = _compute_overlay_height(suggestions, self._config)

        # Convert from top-left origin (AX coords) to bottom-left origin (AppKit coords).
        # AX coords: origin at top-left of primary display, Y increases downward.
        # AppKit coords: origin at bottom-left of primary display, Y increases upward.
        # The primary screen (screens()[0]) defines both coordinate systems.
        # Its height is the pivot for the Y flip regardless of which monitor
        # the target is on.
        screens = AppKit.NSScreen.screens()
        primary_height = 0.0
        if screens and len(screens) > 0:
            # screens()[0] is always the primary display
            primary_height = screens[0].frame().size.height
            # Flip: ns_y = primary_height - ax_y - overlay_height
            # This places the overlay's top edge at the caret bottom (below caret)
            ns_y = primary_height - y - height
        else:
            ns_y = y

        # caret_ns_y is the caret's bottom edge in AppKit coords (used for flip)
        caret_ns_y = primary_height - y if primary_height else y

        logger.debug(
            f"Overlay show: AX pos=({x:.0f}, {y:.0f}) -> "
            f"NSWindow pos=({x:.0f}, {ns_y:.0f}), "
            f"size=({self._config.width}, {height:.0f}), "
            f"suggestions={len(suggestions)}, "
            f"primary_h={primary_height if screens else '?'}"
        )
        # Log all screens for multi-monitor debugging
        for i, scr in enumerate(AppKit.NSScreen.screens()):
            f = scr.frame()
            logger.debug(
                f"  Screen {i}: origin=({f.origin.x:.0f}, {f.origin.y:.0f}) "
                f"size=({f.size.width:.0f}, {f.size.height:.0f})"
            )

        x, ns_y = self._clamp_to_screen(
            x, ns_y, self._config.width, height,
            caret_ns_y=caret_ns_y, caret_height=caret_height,
        )
        frame = NSMakeRect(x, ns_y, self._config.width, height)

        if self._window is None:
            self._create_window(frame)
        else:
            self._window.setFrame_display_(frame, True)
            # Resize the view to match
            if self._view is not None:
                self._view.setFrame_(
                    NSMakeRect(0, 0, self._config.width, height)
                )

        if self._view is not None:
            self._view.setSuggestions_(suggestions)
            self._view.setSelectedIndex_(0)

        self._window.orderFront_(None)
        self._visible = True
        logger.debug("Overlay is now visible")

    def hide(self) -> None:
        """Hide the overlay."""
        if self._window is not None:
            self._window.orderOut_(None)
        self._visible = False
        logger.debug("Overlay hidden")

    def move_selection(self, delta: int) -> None:
        """Move the selection up or down."""
        if not self._suggestions:
            return
        old = self._selected_index
        self._selected_index = (
            self._selected_index + delta
        ) % len(self._suggestions)
        logger.debug(f"Overlay selection: {old} -> {self._selected_index}")
        if self._view is not None:
            self._view.setSelectedIndex_(self._selected_index)

    def accept_selection(self) -> Suggestion | None:
        """Accept the currently selected suggestion and hide the overlay."""
        if not self._suggestions:
            return None
        suggestion = self._suggestions[self._selected_index]
        logger.debug(f"Overlay accepted [{self._selected_index}]: {suggestion.text[:60]}")
        self.hide()
        if self._on_accept:
            self._on_accept(suggestion)
        return suggestion

    def _clamp_to_screen(
        self, x: float, ns_y: float, width: float, height: float,
        caret_ns_y: float = 0.0, caret_height: float = 20.0,
    ) -> tuple:
        """Adjust overlay position so it stays within screen bounds.

        Args:
            x, ns_y: Position in AppKit coordinates (origin bottom-left).
            width, height: Overlay dimensions.
            caret_ns_y: Caret bottom edge in AppKit coordinates.
            caret_height: Height of the caret in points.

        Returns:
            Adjusted (x, ns_y).
        """
        if not HAS_APPKIT:
            return (x, ns_y)

        # Find the screen containing the overlay origin
        target_screen = None
        for scr in AppKit.NSScreen.screens():
            sf = scr.frame()
            if (
                sf.origin.x <= x < sf.origin.x + sf.size.width
                and sf.origin.y <= ns_y < sf.origin.y + sf.size.height
            ):
                target_screen = scr
                break

        if target_screen is None:
            screens = AppKit.NSScreen.screens()
            if screens:
                target_screen = screens[0]
            else:
                return (x, ns_y)

        sf = target_screen.frame()
        screen_left = sf.origin.x
        screen_right = sf.origin.x + sf.size.width
        screen_bottom = sf.origin.y
        screen_top = sf.origin.y + sf.size.height

        # Shift left if overlay extends past the right edge
        if x + width > screen_right:
            x = screen_right - width

        # Shift right if overlay extends past the left edge
        if x < screen_left:
            x = screen_left

        # If overlay extends below the bottom edge, flip above the caret
        if ns_y < screen_bottom:
            # Place overlay bottom at the caret top (above the caret)
            ns_y = caret_ns_y + caret_height
            if ns_y + height > screen_top:
                ns_y = screen_top - height

        # Clamp to top if overlay extends above the top edge
        if ns_y + height > screen_top:
            ns_y = screen_top - height

        return (x, ns_y)

    def _create_window(self, frame) -> None:
        """Create the borderless floating window."""
        if not HAS_APPKIT:
            return

        logger.debug(f"Creating overlay window at frame={frame}")

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

        content_rect = NSMakeRect(0, 0, frame.size.width, frame.size.height)
        view = (
            SuggestionOverlayView.alloc().initWithFrame_config_(
                content_rect, self._config
            )
        )
        window.setContentView_(view)

        self._window = window
        self._view = view
