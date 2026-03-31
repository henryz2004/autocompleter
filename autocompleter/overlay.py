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
    width: int = 500
    max_height: int = 200
    font_size: int = 13
    opacity: float = 0.95
    text_color: tuple[float, float, float] = (0.92, 0.92, 0.94)
    highlight_color: tuple[float, float, float] = (0.25, 0.45, 0.75)
    border_radius: float = 8.0
    padding: float = 4.0
    item_height: float = 32.0
    max_suggestion_height: float = 150.0
    hint_bar_height: float = 22.0


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
    expanded_index: int = -1,
) -> list[float]:
    """Compute per-item heights accounting for text wrapping.

    Args:
        suggestions: List of suggestions to measure.
        config: Overlay configuration.
        expanded_index: Index of the currently selected/expanded suggestion.
            When a suggestion is expanded (preview mode), it is allowed up to
            ``config.max_height`` instead of ``config.max_suggestion_height``.

    Returns:
        List of heights, one per suggestion.
    """
    if not HAS_APPKIT:
        return [config.item_height] * len(suggestions)
    font = AppKit.NSFont.systemFontOfSize_(config.font_size)
    # Available width for text inside each item (item padding: 8 each side + 8 inner each side)
    available_width = config.width - 2 * config.padding - 16
    heights: list[float] = []
    for i, s in enumerate(suggestions):
        text_h = _measure_text_height(s.text, font, available_width)
        item_h = max(config.item_height, text_h + 12)
        # Clamp to max_suggestion_height (unless this item is expanded)
        if i == expanded_index:
            item_h = min(item_h, config.max_height)
        else:
            item_h = min(item_h, config.max_suggestion_height)
        heights.append(item_h)
    return heights


def _compute_overlay_height(
    suggestions: list[Suggestion], config: OverlayConfig,
    expanded_index: int = -1,
) -> float:
    """Compute total overlay height with dynamic per-item sizing."""
    item_heights = _compute_item_heights(suggestions, config, expanded_index)
    total = config.padding * 2 + sum(item_heights) + config.hint_bar_height
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
            self._preview_mode: bool = False
            self._item_heights: list[float] = []
            self._full_text_heights: list[float] = []  # unclamped heights
            self._auto_trigger_active: bool = False
            return self

        def setSuggestions_(self, suggestions):
            self._suggestions = suggestions
            self._selected_index = 0
            self._preview_mode = False
            self._recompute_heights()
            self.setNeedsDisplay_(True)

        def setSelectedIndex_(self, index):
            if 0 <= index < len(self._suggestions):
                old = self._selected_index
                self._selected_index = index
                # Enable preview mode when selection changes to a truncated item
                if old != index and self._is_truncated(index):
                    self._preview_mode = True
                elif old != index:
                    self._preview_mode = False
                self._recompute_heights()
                self.setNeedsDisplay_(True)

        def _recompute_heights(self):
            expanded = self._selected_index if self._preview_mode else -1
            self._item_heights = _compute_item_heights(
                self._suggestions, self._config, expanded_index=expanded,
            )
            # Also compute unclamped heights for truncation detection
            if HAS_APPKIT:
                font = AppKit.NSFont.systemFontOfSize_(self._config.font_size)
                avail_w = self._config.width - 2 * self._config.padding - 16
                self._full_text_heights = []
                for s in self._suggestions:
                    text_h = _measure_text_height(s.text, font, avail_w)
                    self._full_text_heights.append(max(self._config.item_height, text_h + 12))
            else:
                self._full_text_heights = list(self._item_heights)

        def _is_truncated(self, index: int) -> bool:
            """Check if a suggestion at the given index is being truncated."""
            if index < 0 or index >= len(self._suggestions):
                return False
            if index < len(self._full_text_heights) and index < len(self._item_heights):
                return self._full_text_heights[index] > self._item_heights[index]
            return False

        def setAutoTriggerActive_(self, active):
            self._auto_trigger_active = active
            self.setNeedsDisplay_(True)

        def setOnClickCallback_(self, callback):
            """Set callback for click-to-select: callback(index)."""
            self._on_click = callback

        def mouseDown_(self, event):
            """Handle mouse clicks to select a suggestion."""
            if not self._suggestions:
                return
            # Convert click point to view coordinates
            pt = self.convertPoint_fromView_(event.locationInWindow(), None)
            # Determine which suggestion was clicked by walking item rects
            cfg = self._config
            y = self.bounds().size.height - cfg.padding
            for i in range(len(self._suggestions)):
                item_h = (
                    self._item_heights[i]
                    if i < len(self._item_heights)
                    else cfg.item_height
                )
                y -= item_h
                if pt.y >= y and pt.y < y + item_h:
                    # Clicked on suggestion i
                    logger.debug("Click on suggestion %d", i)
                    callback = getattr(self, "_on_click", None)
                    if callback:
                        callback(i)
                    return
            # Click was in the hint bar or padding — ignore
            return

        def drawRect_(self, rect):
            cfg = self._config

            # No solid background fill — NSVisualEffectView provides the blur

            # Draw each suggestion
            font = AppKit.NSFont.systemFontOfSize_(cfg.font_size)
            ellipsis_font = AppKit.NSFont.systemFontOfSize_(cfg.font_size - 1)
            text_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.text_color[0], cfg.text_color[1], cfg.text_color[2], 1.0
            )
            dim_text_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.text_color[0] * 0.6,
                cfg.text_color[1] * 0.6,
                cfg.text_color[2] * 0.6,
                1.0,
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

                # Draw text with word wrapping — clip to item rect
                AppKit.NSGraphicsContext.currentContext().saveGraphicsState()
                clip_path = AppKit.NSBezierPath.bezierPathWithRect_(item_rect)
                clip_path.addClip()

                attrs = {
                    AppKit.NSFontAttributeName: font,
                    AppKit.NSForegroundColorAttributeName: text_color,
                    AppKit.NSParagraphStyleAttributeName: paragraph_style,
                }
                text = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                    suggestion.text, attrs
                )
                text_rect = NSMakeRect(
                    item_rect.origin.x + 6,
                    item_rect.origin.y + 4,
                    item_rect.size.width - 12,
                    item_h - 8,
                )
                text.drawInRect_(text_rect)

                AppKit.NSGraphicsContext.currentContext().restoreGraphicsState()

                # Draw "..." truncation indicator if text is clipped
                is_truncated = self._is_truncated(i)
                if is_truncated:
                    ellipsis_attrs = {
                        AppKit.NSFontAttributeName: ellipsis_font,
                        AppKit.NSForegroundColorAttributeName: dim_text_color,
                    }
                    ellipsis = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                        "...", ellipsis_attrs,
                    )
                    ellipsis_rect = NSMakeRect(
                        item_rect.origin.x + item_rect.size.width - 30,
                        item_rect.origin.y + 2,
                        24,
                        14,
                    )
                    ellipsis.drawInRect_(ellipsis_rect)

            # ---- Hint bar at the bottom ----
            hint_bar_y = 0.0
            hint_bar_h = cfg.hint_bar_height
            bounds_w = self.bounds().size.width

            # Separator line
            separator_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                1.0, 1.0, 1.0, 0.3
            )
            separator_color.set()
            separator_rect = NSMakeRect(cfg.padding, hint_bar_y + hint_bar_h - 1, bounds_w - 2 * cfg.padding, 1.0)
            AppKit.NSBezierPath.fillRect_(separator_rect)

            # Hint text
            hint_font = AppKit.NSFont.systemFontOfSize_(10.0)
            hint_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                1.0, 1.0, 1.0, 0.5
            )
            hint_attrs = {
                AppKit.NSFontAttributeName: hint_font,
                AppKit.NSForegroundColorAttributeName: hint_color,
            }
            hint_text = "Tab/Click accept  \u2191\u2193 navigate  Esc dismiss"
            hint_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                hint_text, hint_attrs
            )
            hint_text_rect = NSMakeRect(cfg.padding + 4, hint_bar_y + 2, bounds_w * 0.7, hint_bar_h - 4)
            hint_str.drawInRect_(hint_text_rect)

            # AUTO badge (right side of hint bar)
            if self._auto_trigger_active:
                badge_font = AppKit.NSFont.boldSystemFontOfSize_(9.0)
                badge_text_attrs = {
                    AppKit.NSFontAttributeName: badge_font,
                    AppKit.NSForegroundColorAttributeName: AppKit.NSColor.whiteColor(),
                }
                badge_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                    "AUTO", badge_text_attrs
                )
                badge_size = badge_str.size()
                badge_w = badge_size.width + 10
                badge_h = badge_size.height + 4
                badge_x = bounds_w - cfg.padding - badge_w - 4
                badge_y = hint_bar_y + (hint_bar_h - badge_h) / 2
                badge_rect = NSMakeRect(badge_x, badge_y, badge_w, badge_h)
                accent_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    0.2, 0.5, 0.9, 1.0
                )
                badge_path = AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    badge_rect, 4.0, 4.0
                )
                accent_color.set()
                badge_path.fill()
                badge_text_rect = NSMakeRect(badge_x + 5, badge_y + 2, badge_size.width, badge_size.height)
                badge_str.drawInRect_(badge_text_rect)


    class NonActivatingWindow(AppKit.NSWindow):
        """NSWindow subclass that never steals focus from the active app.

        Prevents the overlay from taking keyboard focus when the user
        clicks on it to select a suggestion.
        """

        def canBecomeKeyWindow(self):
            return False

        def canBecomeMainWindow(self):
            return False


class SuggestionOverlay:
    """Manages the floating overlay window for displaying suggestions."""

    def __init__(self, config: OverlayConfig | None = None):
        self._config = config or OverlayConfig()
        self._window = None
        self._view = None
        self._effect_view = None
        self._suggestions: list[Suggestion] = []
        self._selected_index: int = 0
        self._on_accept = None
        self._on_dismiss = None
        self._global_monitor = None
        self._local_monitor = None
        self._visible = False
        self._auto_trigger_active: bool = False
        self._hide_generation: int = 0  # guards fade-out completion handler

        # Stored position for resizing during preview mode
        self._last_x: float = 0.0
        self._last_ns_y: float = 0.0
        self._last_caret_ns_y: float = 0.0
        self._last_caret_height: float = 20.0
        # Cached AX position + primary_height for stable updates
        self._last_ax_x: float = 0.0
        self._last_ax_y: float = 0.0
        self._last_primary_height: float = 0.0

    @property
    def is_visible(self) -> bool:
        return self._visible

    def set_auto_trigger_active(self, active: bool) -> None:
        """Update the auto-trigger indicator state."""
        self._auto_trigger_active = active
        if self._view is not None:
            self._view.setAutoTriggerActive_(active)

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

        width = self._config.width
        height = _compute_overlay_height(suggestions, self._config)

        # When the overlay is already visible at the same AX position, anchor
        # the top edge and only resize downward.  This avoids recomputing the
        # coordinate flip (which can produce different results when the overlay
        # height changes between "Generating..." and the final suggestions).
        same_position = (
            self._visible
            and self._window is not None
            and abs(x - self._last_ax_x) < 1.0
            and abs(y - self._last_ax_y) < 1.0
        )

        if same_position:
            # Reuse the cached primary_height and caret_ns_y from the first show
            primary_height = self._last_primary_height
            caret_ns_y = self._last_caret_ns_y

            # Anchor the top edge of the existing window and resize downward
            current_frame = self._window.frame()
            top_edge = current_frame.origin.y + current_frame.size.height
            ns_y = top_edge - height
            x_pos = current_frame.origin.x

            logger.debug(
                f"Overlay update (same pos): top_edge={top_edge:.0f}, "
                f"new_height={height:.0f}, ns_y={ns_y:.0f}, "
                f"suggestions={len(suggestions)}"
            )
        else:
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
                # Add a small gap (4px) so the overlay doesn't touch the caret edge
                gap = 4.0
                ns_y = primary_height - y - height - gap
            else:
                ns_y = y

            # caret_ns_y is the caret's bottom edge in AppKit coords (used for flip)
            caret_ns_y = primary_height - y if primary_height else y
            x_pos = x

            logger.debug(
                f"Overlay show: AX pos=({x:.0f}, {y:.0f}) -> "
                f"NSWindow pos=({x_pos:.0f}, {ns_y:.0f}), "
                f"size=({width}, {height:.0f}), "
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

        x_pos, ns_y = self._clamp_to_screen(
            x_pos, ns_y, width, height,
            caret_ns_y=caret_ns_y, caret_height=caret_height,
        )

        # Store position for later resizing (preview mode) and stable updates
        self._last_x = x_pos
        self._last_ns_y = ns_y
        self._last_caret_ns_y = caret_ns_y
        self._last_caret_height = caret_height
        self._last_ax_x = x
        self._last_ax_y = y
        self._last_primary_height = primary_height

        frame = NSMakeRect(x_pos, ns_y, width, height)

        if self._window is None:
            self._create_window(frame)
        else:
            self._window.setFrame_display_(frame, True)
            # Resize the effect view and suggestion view to match
            content_rect = NSMakeRect(0, 0, width, height)
            if self._effect_view is not None:
                self._effect_view.setFrame_(content_rect)
            if self._view is not None:
                self._view.setFrame_(content_rect)

        if self._view is not None:
            self._view.setAutoTriggerActive_(self._auto_trigger_active)
            self._view.setOnClickCallback_(self._on_click_suggestion)
            self._view.setSuggestions_(suggestions)
            self._view.setSelectedIndex_(0)

        # Cancel any in-flight fade-out so its completion handler won't hide us
        self._hide_generation += 1

        # Fade-in animation
        was_visible = self._visible
        self._window.setAlphaValue_(1.0)
        self._window.orderFront_(None)
        if not was_visible:
            self._window.setAlphaValue_(0.0)
            AppKit.NSAnimationContext.beginGrouping()
            AppKit.NSAnimationContext.currentContext().setDuration_(0.15)
            self._window.animator().setAlphaValue_(1.0)
            AppKit.NSAnimationContext.endGrouping()
        self._visible = True
        self._install_click_monitor()
        logger.debug("Overlay is now visible")

    def hide(self) -> None:
        """Hide the overlay with a fade-out animation."""
        self._remove_click_monitor()
        self._visible = False
        # Reset cached AX position so next show() does a full position compute
        self._last_ax_x = 0.0
        self._last_ax_y = 0.0
        if self._window is not None:
            window = self._window
            self._hide_generation += 1
            gen = self._hide_generation

            def _on_complete():
                # Only hide if no show() has been called since this fade started
                if self._hide_generation == gen:
                    window.orderOut_(None)
                    window.setAlphaValue_(1.0)

            AppKit.NSAnimationContext.beginGrouping()
            ctx = AppKit.NSAnimationContext.currentContext()
            ctx.setDuration_(0.1)
            ctx.setCompletionHandler_(_on_complete)
            window.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()
        logger.debug("Overlay hidden")

    def move_selection(self, delta: int) -> None:
        """Move the selection up or down.

        When the newly selected item is a multi-line suggestion, preview
        mode is activated which may resize the overlay to accommodate the
        expanded content.
        """
        if not self._suggestions:
            return
        old = self._selected_index
        self._selected_index = (
            self._selected_index + delta
        ) % len(self._suggestions)
        logger.debug(f"Overlay selection: {old} -> {self._selected_index}")
        if self._view is not None:
            self._view.setSelectedIndex_(self._selected_index)
            # Resize overlay if preview mode changed the item heights
            self._resize_to_fit()

    def _resize_to_fit(self) -> None:
        """Resize the overlay window to match updated item heights.

        Called after selection changes to accommodate preview mode expansion.
        """
        if not HAS_APPKIT or self._window is None or self._view is None:
            return
        if not self._suggestions:
            return

        # Recompute height using the view's current item heights
        new_height = (
            self._config.padding * 2
            + sum(self._view._item_heights)
            + self._config.hint_bar_height
        )
        new_height = min(new_height, self._config.max_height)

        current_frame = self._window.frame()
        if abs(current_frame.size.height - new_height) < 1.0:
            return  # No meaningful change

        # Keep the top edge in the same place by adjusting ns_y
        ns_y = current_frame.origin.y + current_frame.size.height - new_height
        x = current_frame.origin.x

        x, ns_y = self._clamp_to_screen(
            x, ns_y, self._config.width, new_height,
            caret_ns_y=self._last_caret_ns_y,
            caret_height=self._last_caret_height,
        )

        frame = NSMakeRect(x, ns_y, self._config.width, new_height)
        self._window.setFrame_display_(frame, True)
        content_rect = NSMakeRect(0, 0, self._config.width, new_height)
        if self._effect_view is not None:
            self._effect_view.setFrame_(content_rect)
        self._view.setFrame_(content_rect)

    def set_dismiss_callback(self, callback):
        """Set callback invoked when user clicks outside the overlay."""
        self._on_dismiss = callback

    def _install_click_monitor(self):
        if not HAS_APPKIT or self._global_monitor is not None:
            return

        def _on_mouse_down(event):
            if not self._visible or self._window is None:
                return
            # For global monitors, locationInWindow() returns screen coords
            click_pt = event.locationInWindow()
            frame = self._window.frame()
            if not AppKit.NSPointInRect(click_pt, frame):
                logger.debug("Click outside overlay detected, dismissing")
                if self._on_dismiss:
                    self._on_dismiss()
            # Clicks inside the overlay are handled by SuggestionOverlayView.mouseDown_

        # NSEventMaskLeftMouseDown = 1 << 1; use numeric value for
        # compatibility with older PyObjC that lacks the named constant.
        mask = getattr(AppKit, "NSEventMaskLeftMouseDown", 1 << 1)
        try:
            self._global_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                mask, _on_mouse_down
            )
            logger.debug("Installed global click monitor for overlay dismiss")
        except Exception:
            logger.warning("Failed to install global click monitor", exc_info=True)

        # Local monitor catches clicks on our own window (global monitors
        # only see events destined for other processes).
        def _on_local_mouse_down(event):
            if not self._visible or self._window is None:
                return event
            # Let the view's mouseDown_ handle clicks inside the overlay
            return event

        try:
            self._local_monitor = AppKit.NSEvent.addLocalMonitorForEventsMatchingMask_handler_(
                mask, _on_local_mouse_down
            )
        except Exception:
            logger.warning("Failed to install local click monitor", exc_info=True)

    def _remove_click_monitor(self):
        if self._global_monitor is not None:
            AppKit.NSEvent.removeMonitor_(self._global_monitor)
            self._global_monitor = None
        if self._local_monitor is not None:
            AppKit.NSEvent.removeMonitor_(self._local_monitor)
            self._local_monitor = None

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

    def _on_click_suggestion(self, index: int) -> None:
        """Handle a mouse click on a suggestion at *index*."""
        if index < 0 or index >= len(self._suggestions):
            return
        self._selected_index = index
        if self._view is not None:
            self._view.setSelectedIndex_(index)
        self.accept_selection()

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
            # Place overlay bottom at the caret top + gap (above the caret)
            ns_y = caret_ns_y + caret_height + 4.0
            if ns_y + height > screen_top:
                ns_y = screen_top - height

        # Clamp to top if overlay extends above the top edge
        if ns_y + height > screen_top:
            ns_y = screen_top - height

        return (x, ns_y)

    def _create_window(self, frame) -> None:
        """Create the borderless floating window with NSVisualEffectView backdrop."""
        if not HAS_APPKIT:
            return

        logger.debug(f"Creating overlay window at frame={frame}")

        style = AppKit.NSWindowStyleMaskBorderless
        window = NonActivatingWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame,
            style,
            AppKit.NSBackingStoreBuffered,
            False,
        )
        window.setLevel_(AppKit.NSPopUpMenuWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setHasShadow_(True)
        # Do NOT setIgnoresMouseEvents_ — we want click-to-select
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )

        content_rect = NSMakeRect(0, 0, frame.size.width, frame.size.height)

        # NSVisualEffectView provides the frosted-glass blur backdrop
        effect_view = AppKit.NSVisualEffectView.alloc().initWithFrame_(content_rect)
        # NSVisualEffectMaterialHUDWindow = 13 (dark translucent, like Spotlight)
        material = getattr(AppKit, "NSVisualEffectMaterialHUDWindow", 13)
        effect_view.setMaterial_(material)
        # NSVisualEffectBlendingModeBehindWindow = 0
        effect_view.setBlendingMode_(0)
        # NSVisualEffectStateActive = 1 (always active, even when app not focused)
        effect_view.setState_(1)
        effect_view.setWantsLayer_(True)
        effect_view.layer().setCornerRadius_(self._config.border_radius)
        effect_view.layer().setMasksToBounds_(True)

        # Force dark appearance so backdrop stays dark regardless of system theme
        dark_appearance = AppKit.NSAppearance.appearanceNamed_("NSAppearanceNameVibrantDark")
        effect_view.setAppearance_(dark_appearance)

        window.setContentView_(effect_view)

        view = SuggestionOverlayView.alloc().initWithFrame_config_(
            content_rect, self._config
        )
        effect_view.addSubview_(view)

        self._window = window
        self._effect_view = effect_view
        self._view = view
