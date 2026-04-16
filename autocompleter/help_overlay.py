"""Help overlay — a floating panel listing every hotkey and what it does.

Shown when the user presses the help hotkey (default ``ctrl+/``). Mirrors the
visual style of the suggestion overlay (HUD-style blur, rounded corners) but
is centered on the primary screen rather than anchored to the caret.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import AppKit
    import objc
    from Foundation import NSMakeRect

    HAS_APPKIT = True
except ImportError:
    HAS_APPKIT = False


@dataclass
class HelpEntry:
    """A single row in the help panel."""

    keys: str  # e.g. "⌃Space" or "Shift+Tab"
    description: str  # e.g. "Trigger suggestions"


@dataclass
class HelpOverlayConfig:
    width: int = 460
    max_height: int = 560
    title: str = "Autocompleter shortcuts"
    footer: str = "Esc or ⌃/ to close"
    font_size: int = 13
    title_font_size: int = 15
    footer_font_size: int = 11
    padding: float = 16.0
    row_height: float = 26.0
    keys_column_width: float = 130.0
    section_gap: float = 8.0
    title_gap: float = 12.0
    opacity: float = 0.98
    text_color: tuple[float, float, float] = (0.94, 0.94, 0.96)
    dim_text_color: tuple[float, float, float] = (0.65, 0.65, 0.70)
    border_radius: float = 10.0
    sections: list[tuple[str, list[HelpEntry]]] = field(default_factory=list)


def _default_entries(
    *,
    trigger_hotkey: str,
    regenerate_hotkey: str,
    help_hotkey: str,
    report_hotkey: str,
) -> list[tuple[str, list[HelpEntry]]]:
    """Build the default list of help sections."""
    return [
        (
            "Generate",
            [
                HelpEntry(
                    _format_hotkey(trigger_hotkey),
                    "Trigger suggestions (or dismiss)",
                ),
                HelpEntry(
                    _format_hotkey(f"shift+{trigger_hotkey}"),
                    "Toggle auto-trigger on/off",
                ),
                HelpEntry(
                    _format_hotkey(regenerate_hotkey),
                    "Regenerate with fresh sampling",
                ),
            ],
        ),
        (
            "Navigate & accept",
            [
                HelpEntry("↑ / ↓", "Move selection up/down"),
                HelpEntry("1 / 2 / 3", "Accept suggestion by number"),
                HelpEntry("Tab or Return", "Accept highlighted suggestion"),
                HelpEntry("Shift+Tab", "Accept only the first sentence/line"),
                HelpEntry("Esc", "Dismiss the overlay"),
                HelpEntry("Click outside", "Dismiss the overlay"),
            ],
        ),
        (
            "Help & feedback",
            [
                HelpEntry(
                    _format_hotkey(help_hotkey),
                    "Show/hide this help panel",
                ),
                HelpEntry(
                    _format_hotkey(report_hotkey),
                    "Report a bug for the current app (no content sent)",
                ),
            ],
        ),
    ]


_MODIFIER_GLYPHS = {
    "ctrl": "⌃",
    "control": "⌃",
    "cmd": "⌘",
    "command": "⌘",
    "alt": "⌥",
    "opt": "⌥",
    "option": "⌥",
    "shift": "⇧",
}

_KEY_DISPLAY = {
    "space": "Space",
    "tab": "Tab",
    "return": "Return",
    "escape": "Esc",
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
    "/": "/",
}


def _format_hotkey(hotkey: str) -> str:
    """Render a hotkey string like 'ctrl+space' as '⌃Space'."""
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    mods = []
    key = ""
    for p in parts:
        if p in _MODIFIER_GLYPHS:
            mods.append(_MODIFIER_GLYPHS[p])
        elif p in _KEY_DISPLAY:
            key = _KEY_DISPLAY[p]
        else:
            key = p.upper() if len(p) == 1 else p.capitalize()
    return "".join(mods) + key


if HAS_APPKIT:

    class HelpOverlayView(AppKit.NSView):
        """Renders the help panel content."""

        def initWithFrame_config_(self, frame, config):
            self = objc.super(HelpOverlayView, self).initWithFrame_(frame)
            if self is None:
                return None
            self._config = config
            return self

        def drawRect_(self, rect):
            cfg = self._config
            bounds = self.bounds()
            width = bounds.size.width
            height = bounds.size.height

            text_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.text_color[0], cfg.text_color[1], cfg.text_color[2], 1.0
            )
            dim_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                cfg.dim_text_color[0], cfg.dim_text_color[1], cfg.dim_text_color[2], 1.0
            )
            section_color = AppKit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.55, 0.7, 1.0, 1.0
            )

            title_font = AppKit.NSFont.boldSystemFontOfSize_(cfg.title_font_size)
            section_font = AppKit.NSFont.boldSystemFontOfSize_(cfg.font_size - 1)
            keys_font = AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(
                cfg.font_size, 0.0
            )
            desc_font = AppKit.NSFont.systemFontOfSize_(cfg.font_size)
            footer_font = AppKit.NSFont.systemFontOfSize_(cfg.footer_font_size)

            # Title (top)
            y_top = height - cfg.padding
            title_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                cfg.title,
                {
                    AppKit.NSFontAttributeName: title_font,
                    AppKit.NSForegroundColorAttributeName: text_color,
                },
            )
            title_size = title_str.size()
            title_str.drawAtPoint_(
                AppKit.NSMakePoint(cfg.padding, y_top - title_size.height)
            )
            y_cursor = y_top - title_size.height - cfg.title_gap

            # Sections
            for section_title, entries in cfg.sections:
                # Section heading (uppercase, dim, small)
                section_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                    section_title.upper(),
                    {
                        AppKit.NSFontAttributeName: section_font,
                        AppKit.NSForegroundColorAttributeName: section_color,
                    },
                )
                section_size = section_str.size()
                y_cursor -= section_size.height
                section_str.drawAtPoint_(
                    AppKit.NSMakePoint(cfg.padding, y_cursor)
                )
                y_cursor -= 4.0  # small gap after section heading

                # Rows
                for entry in entries:
                    y_cursor -= cfg.row_height
                    # Keys column (right-aligned)
                    keys_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                        entry.keys,
                        {
                            AppKit.NSFontAttributeName: keys_font,
                            AppKit.NSForegroundColorAttributeName: text_color,
                        },
                    )
                    keys_size = keys_str.size()
                    keys_x = (
                        cfg.padding + cfg.keys_column_width - keys_size.width
                    )
                    keys_y = y_cursor + (cfg.row_height - keys_size.height) / 2
                    keys_str.drawAtPoint_(AppKit.NSMakePoint(keys_x, keys_y))

                    # Description column
                    desc_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                        entry.description,
                        {
                            AppKit.NSFontAttributeName: desc_font,
                            AppKit.NSForegroundColorAttributeName: text_color,
                        },
                    )
                    desc_x = cfg.padding + cfg.keys_column_width + 14.0
                    desc_size = desc_str.size()
                    desc_y = y_cursor + (cfg.row_height - desc_size.height) / 2
                    desc_width = width - desc_x - cfg.padding
                    desc_rect = AppKit.NSMakeRect(
                        desc_x, desc_y, desc_width, desc_size.height + 2
                    )
                    desc_str.drawInRect_(desc_rect)

                y_cursor -= cfg.section_gap

            # Footer (bottom)
            footer_str = AppKit.NSAttributedString.alloc().initWithString_attributes_(
                cfg.footer,
                {
                    AppKit.NSFontAttributeName: footer_font,
                    AppKit.NSForegroundColorAttributeName: dim_color,
                },
            )
            footer_size = footer_str.size()
            footer_str.drawAtPoint_(
                AppKit.NSMakePoint(
                    cfg.padding,
                    max(6.0, cfg.padding - footer_size.height / 2),
                )
            )


    class HelpNonActivatingWindow(AppKit.NSWindow):
        """Window subclass that never becomes key (no focus steal)."""

        def canBecomeKeyWindow(self):
            return False

        def canBecomeMainWindow(self):
            return False


class HelpOverlay:
    """Floating help panel showing all shortcuts."""

    def __init__(
        self,
        *,
        trigger_hotkey: str = "ctrl+space",
        regenerate_hotkey: str = "ctrl+r",
        help_hotkey: str = "ctrl+/",
        report_hotkey: str = "ctrl+shift+b",
    ):
        self._config = HelpOverlayConfig()
        self._config.sections = _default_entries(
            trigger_hotkey=trigger_hotkey,
            regenerate_hotkey=regenerate_hotkey,
            help_hotkey=help_hotkey,
            report_hotkey=report_hotkey,
        )
        self._window = None
        self._effect_view = None
        self._view = None
        self._visible = False
        self._on_dismiss = None
        self._global_monitor = None

    @property
    def is_visible(self) -> bool:
        return self._visible

    def set_dismiss_callback(self, cb) -> None:
        """Callback invoked when the panel is dismissed via click-outside."""
        self._on_dismiss = cb

    def update_hotkeys(
        self,
        *,
        trigger_hotkey: str,
        regenerate_hotkey: str,
        help_hotkey: str,
        report_hotkey: str,
    ) -> None:
        self._config.sections = _default_entries(
            trigger_hotkey=trigger_hotkey,
            regenerate_hotkey=regenerate_hotkey,
            help_hotkey=help_hotkey,
            report_hotkey=report_hotkey,
        )
        if self._view is not None:
            self._view.setNeedsDisplay_(True)

    def toggle(self) -> None:
        if self._visible:
            self.hide()
        else:
            self.show()

    def _compute_height(self) -> float:
        cfg = self._config
        total = cfg.padding * 2 + cfg.title_font_size + cfg.title_gap
        for _, entries in cfg.sections:
            total += cfg.font_size + 4.0  # section heading + gap
            total += cfg.row_height * len(entries)
            total += cfg.section_gap
        total += cfg.footer_font_size + 6.0  # footer
        return min(total, cfg.max_height)

    def show(self) -> None:
        if not HAS_APPKIT:
            logger.warning("AppKit not available; help overlay cannot be shown.")
            return

        width = self._config.width
        height = self._compute_height()

        # Center on the primary screen
        screens = AppKit.NSScreen.screens()
        if screens:
            screen_frame = screens[0].frame()
            x = screen_frame.origin.x + (screen_frame.size.width - width) / 2
            y = screen_frame.origin.y + (screen_frame.size.height - height) / 2
        else:
            x, y = 200.0, 200.0

        frame = NSMakeRect(x, y, width, height)

        if self._window is None:
            self._create_window(frame)
        else:
            self._window.setFrame_display_(frame, True)
            content_rect = NSMakeRect(0, 0, width, height)
            if self._effect_view is not None:
                self._effect_view.setFrame_(content_rect)
            if self._view is not None:
                self._view.setFrame_(content_rect)
                self._view.setNeedsDisplay_(True)

        self._window.setAlphaValue_(0.0)
        self._window.orderFront_(None)
        AppKit.NSAnimationContext.beginGrouping()
        AppKit.NSAnimationContext.currentContext().setDuration_(0.12)
        self._window.animator().setAlphaValue_(1.0)
        AppKit.NSAnimationContext.endGrouping()

        self._visible = True
        self._install_click_monitor()
        logger.debug("Help overlay shown")

    def hide(self) -> None:
        self._remove_click_monitor()
        self._visible = False
        if self._window is not None:
            window = self._window

            def _complete():
                if not self._visible:
                    window.orderOut_(None)
                    window.setAlphaValue_(1.0)

            AppKit.NSAnimationContext.beginGrouping()
            ctx = AppKit.NSAnimationContext.currentContext()
            ctx.setDuration_(0.08)
            ctx.setCompletionHandler_(_complete)
            window.animator().setAlphaValue_(0.0)
            AppKit.NSAnimationContext.endGrouping()
        logger.debug("Help overlay hidden")

    def _install_click_monitor(self):
        if not HAS_APPKIT or self._global_monitor is not None:
            return

        def _on_mouse_down(event):
            if not self._visible or self._window is None:
                return
            pt = event.locationInWindow()
            frame = self._window.frame()
            if not AppKit.NSPointInRect(pt, frame):
                logger.debug("Click outside help overlay, dismissing")
                if self._on_dismiss:
                    self._on_dismiss()
                else:
                    self.hide()

        mask = getattr(AppKit, "NSEventMaskLeftMouseDown", 1 << 1)
        try:
            self._global_monitor = AppKit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
                mask, _on_mouse_down
            )
        except Exception:
            logger.warning("Failed to install help-overlay click monitor", exc_info=True)

    def _remove_click_monitor(self):
        if self._global_monitor is not None:
            AppKit.NSEvent.removeMonitor_(self._global_monitor)
            self._global_monitor = None

    def _create_window(self, frame) -> None:
        if not HAS_APPKIT:
            return

        style = AppKit.NSWindowStyleMaskBorderless
        window = HelpNonActivatingWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, style, AppKit.NSBackingStoreBuffered, False,
        )
        window.setLevel_(AppKit.NSPopUpMenuWindowLevel)
        window.setOpaque_(False)
        window.setBackgroundColor_(AppKit.NSColor.clearColor())
        window.setHasShadow_(True)
        window.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorStationary
        )

        content_rect = NSMakeRect(0, 0, frame.size.width, frame.size.height)
        effect_view = AppKit.NSVisualEffectView.alloc().initWithFrame_(content_rect)
        material = getattr(AppKit, "NSVisualEffectMaterialHUDWindow", 13)
        effect_view.setMaterial_(material)
        effect_view.setBlendingMode_(0)
        effect_view.setState_(1)
        effect_view.setWantsLayer_(True)
        effect_view.layer().setCornerRadius_(self._config.border_radius)
        effect_view.layer().setMasksToBounds_(True)
        dark = AppKit.NSAppearance.appearanceNamed_("NSAppearanceNameVibrantDark")
        effect_view.setAppearance_(dark)
        window.setContentView_(effect_view)

        view = HelpOverlayView.alloc().initWithFrame_config_(
            content_rect, self._config,
        )
        effect_view.addSubview_(view)

        self._window = window
        self._effect_view = effect_view
        self._view = view
