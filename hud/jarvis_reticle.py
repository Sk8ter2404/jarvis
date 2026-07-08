#!/usr/bin/env python3
"""
JARVIS Reticle Overlay — full-virtual-screen translucent target reticle.

Spawned as a subprocess by bobert_companion.py at startup. Spans the entire
multi-monitor virtual desktop and draws a 2-second translucent target reticle
at the click/type coordinates published by JARVIS whenever it executes a
UI-automation action (ui_click, ui_type, ui_press, ui_hotkey, ui_scroll,
_act_focus_window).

The host publishes reticle events to ``hud_reticles.json`` (sibling to
bobert_companion.py). Each event has ``x``, ``y`` (virtual-desktop pixel
coordinates — may be negative on the left monitor), an optional ``label``,
and a ``created_at`` epoch timestamp. The overlay reads the file at every
animation tick and draws every reticle whose age is < ``RETICLE_TTL``.

Closes cleanly when its parent process exits — the ``--parent-pid`` argument
lets us detect that without IPC plumbing.

CLI:
  python hud/jarvis_reticle.py --x -2560 --y -1440 --width 7680 --height 2880 \
                               --parent-pid 12345
"""
import argparse
import json
import os
import sys
import time
import tkinter as tk

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ──────────────────────────────────────────────────────────────────────────
#  Appearance
#  Palette matches hud/jarvis_hud.py so the visual language reads as a
#  single coherent overlay system.
# ──────────────────────────────────────────────────────────────────────────
BG_KEY        = "#010101"   # near-black, keyed transparent on Win32
RING_COLOR    = "#4cc9ff"   # cyan (matches HUD CYAN)
RING_GLOW     = "#9ee7ff"   # bright inner (matches HUD CYAN_BRIGHT)
RING_FADE     = "#1b4a66"   # dim cyan during the fade-out tail
RING_FADE_2   = "#5d8aa3"   # dim mid (matches HUD DIM_TEXT)
LABEL_COLOR   = "#cfeefb"   # matches HUD TEXT_COLOR
LABEL_FADE    = "#5d8aa3"

# Priority-1 palette — reticles with color=="red" use this so a boss-mode
# alert reads as visually distinct from a normal UI-automation click.
RED_RING_COLOR  = "#ff3344"
RED_RING_GLOW   = "#ffb0b0"
RED_RING_FADE   = "#5a1620"
RED_RING_FADE_2 = "#8a3a44"
RED_LABEL_COLOR = "#ffd2d6"
RED_LABEL_FADE  = "#8a3a44"

RETICLE_TTL       = 2.0     # spec: "2-second translucent target reticle"
GROW_DURATION     = 0.35    # outer ring grows for the first 0.35s, then holds
FADE_TAIL_SECS    = 0.5     # last 0.5s fade to the dim palette
START_RADIUS      = 14
MAX_RADIUS        = 46
CROSSHAIR_LEN     = 14
TICK_MS           = 33      # ~30 fps animation
LABEL_FONT        = ("Consolas", 9, "bold")

# STALE-STATE EXIT (P0-2): mirrors jarvis_air_cursor's STATE_STALE_EXIT_S. If
# hud_reticles.json hasn't been updated (newest mtime) for this long, the host
# that publishes reticles is gone (crashed / killed without running the shutdown
# handler) — exit so a full-virtual-desktop click-through layer can't strand over
# all four monitors. The reticle file is touched on EVERY UI-automation action,
# and the launcher rewrites it at boot, so a live JARVIS keeps it fresh well
# inside this window even when no reticle is currently on screen.
STATE_STALE_EXIT_S = 600.0   # 10 min of no state-file update → exit
# Orphan guard (mirrors jarvis_air_cursor / holo HUD): with --parent-pid 0/absent
# there's no parent to track, so self-exit after this long rather than linger
# forever as a parentless click-through layer.
ORPHAN_MAX_LIFETIME_S = 1800.0

# Win32 extended-window-style bits for the click-through colour-key backstop
# (P2-7) — same contract jarvis_air_cursor uses. Module-level so the transparency
# / click-through math is unit-testable without constructing a Tk root.
GWL_EXSTYLE       = -20
WS_EX_LAYERED     = 0x00080000
WS_EX_TRANSPARENT = 0x00000020   # makes the window click-through (hit-test pass)
WS_EX_TOOLWINDOW  = 0x00000080   # keep it out of the alt-tab / taskbar list
WS_EX_NOACTIVATE  = 0x08000000   # never steal focus from the app underneath
LWA_COLORKEY      = 0x00000001   # SetLayeredWindowAttributes: key a colour out

STATE_FILE_NAME = "hud_reticles.json"
PROJECT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_FILE      = os.path.join(PROJECT_DIR, STATE_FILE_NAME)


def _click_through_exstyle(cur: int) -> int:
    """OR the click-through / layered / no-activate / tool-window bits onto an
    existing GWL_EXSTYLE value. Pure (int in, int out) so the transparency +
    click-through contract is unit-testable with no Tk root / no Win32. Mirrors
    jarvis_air_cursor._click_through_exstyle exactly."""
    return (cur
            | WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOOLWINDOW
            | WS_EX_NOACTIVATE)


def _colorref(hex_color: str) -> int:
    """``#rrggbb`` → Win32 COLORREF (0x00bbggrr), the colour-key passed to
    SetLayeredWindowAttributes so the keyed background composites fully
    transparent instead of as an opaque block. Pure; testable."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b << 16) | (g << 8) | r


def _is_parent_alive(pid: int, start_time: "float | None" = None) -> bool:
    """True while the spawning JARVIS is still alive.

    PID-RECYCLE GUARD (P0-2): a bare pid_exists() can read TRUE for a STRANGER
    that the OS handed our dead parent's recycled PID — leaving this full-virtual-
    desktop click-through overlay stranded on top of an unrelated process. When
    ``start_time`` (the parent's psutil create_time() captured at spawn, passed
    via --parent-start) is given, we additionally require the live process at
    ``pid`` to report the SAME create_time (within a small epsilon): a recycled
    PID has a DIFFERENT start time and so reads as DEAD. Without a start_time we
    fall back to the historical PID-exists-only check. A transient/unknowable
    lookup is treated as ALIVE so a hiccup can't strand the overlay."""
    if pid <= 0:
        return True
    if _HAS_PSUTIL:
        # pid_exists can raise on Windows for a transient handle/permission
        # error. The reticle spans the full virtual desktop, so a frozen
        # frame here is the worst case — treat an unknowable parent as alive
        # (matches the PyQt HUDs' guard) so a hiccup can't strand it.
        try:
            if not psutil.pid_exists(pid):
                return False
        except Exception:
            return True
        if start_time is not None:
            try:
                # A recycled PID reads as DEAD: same number, different birth time.
                if abs(psutil.Process(pid).create_time() - start_time) > 1.0:
                    return False
            except Exception:
                # Can't read the live process's start time — treat as alive so a
                # transient race can't strand the overlay.
                return True
        return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _read_reticles() -> list:
    if not os.path.exists(STATE_FILE):
        return []
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = data.get("reticles", [])
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _state_file_mtime() -> float:
    """Last-modified epoch of hud_reticles.json, or 0.0 when it's missing /
    unreadable. The host rewrites this file at launch and re-touches it on EVERY
    UI-automation action, so its mtime is the freshness signal the stale-state
    exit timer (P0-2) watches. NEVER raises."""
    try:
        return os.path.getmtime(STATE_FILE)
    except OSError:
        return 0.0


def _primary_work_area_bottom():
    """Bottom edge (virtual-desktop pixel y) of the primary monitor's work
    area — i.e. the top of the Windows taskbar. Returns None when it can't be
    determined (non-Windows, or the Win32 call fails) so the caller leaves the
    full-screen geometry untouched. Best-effort and side-effect-free."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        SPI_GETWORKAREA = 0x0030
        rect = wintypes.RECT()
        ok = ctypes.windll.user32.SystemParametersInfoW(
            SPI_GETWORKAREA, 0, ctypes.byref(rect), 0
        )
        if not ok:
            return None
        return int(rect.bottom)
    except Exception:
        return None


def _clamp_to_work_area(x: int, y: int, w: int, h: int, work_bottom):
    """Trim the overlay so its bottom edge stops at the taskbar instead of
    extending behind it.

    The overlay spans the full virtual desktop ``y .. y+h``. The Windows
    taskbar is topmost too, so the band of our window that overlaps it is
    occluded — a reticle for a click near the bottom edge would be sliced off.
    When ``work_bottom`` (top of the taskbar) falls inside our window we shrink
    the height to end there. We only ever shrink, never move the top or grow,
    so reticles anywhere in the usable work area still render unchanged.

    ``work_bottom`` None (couldn't be queried / non-Windows) returns the
    geometry verbatim. Returns ``(x, y, w, h)``."""
    if work_bottom is None:
        return x, y, w, h
    # Only act when the taskbar top sits within our vertical span; clamp so the
    # window ends at the taskbar and keep at least a 1px-tall canvas.
    if y < work_bottom < y + h:
        h = max(1, work_bottom - y)
    return x, y, w, h


class ReticleOverlay:
    def __init__(self, x: int, y: int, w: int, h: int, parent_pid: int,
                 parent_start: "float | None" = None):
        # Keep the overlay above the Windows taskbar so reticles near the
        # bottom edge aren't sliced off behind it (the taskbar is topmost too).
        x, y, w, h = _clamp_to_work_area(x, y, w, h, _primary_work_area_bottom())

        self.parent_pid = parent_pid
        # Parent's create_time() at spawn (--parent-start) for the PID-recycle
        # guard; None → PID-exists-only fallback.
        self.parent_start = parent_start
        self.origin_x   = x
        self.origin_y   = y
        self.width      = w
        self.height     = h
        self._started_at = time.time()
        # Freshest hud_reticles.json mtime we've seen — drives the stale-state
        # exit timer (the host re-touches the file on every UI-automation action
        # and at boot, so a live JARVIS keeps this advancing).
        self._last_state_mtime = _state_file_mtime()

        self.root = tk.Tk()
        self.root.title("JARVIS Reticle")
        self.root.configure(bg=BG_KEY)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        # On Windows, -transparentcolor makes the keyed background fully
        # transparent AND click-through. Drawn ring pixels (cyan) remain
        # opaque, which is desired so the reticle is visible.
        self._has_colorkey = False
        try:
            self.root.attributes("-transparentcolor", BG_KEY)
            self._has_colorkey = True
        except tk.TclError:
            # Fallback: plain alpha. BLACKOUT BACKSTOP (P2-7): keep this LOW
            # (~0.25) so a degraded reticle whose colour-key was lost reads as a
            # FAINT wash, not a dim sheet over four monitors. The old 0.85 made a
            # colour-key failure paint a near-opaque full-desktop block.
            try:
                self.root.attributes("-alpha", 0.25)
            except Exception:
                pass

        self.root.geometry(f"{w}x{h}+{x}+{y}")

        self.canvas = tk.Canvas(
            self.root, bg=BG_KEY, width=w, height=h,
            highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        # BLACKOUT BACKSTOP (P2-7): actively re-assert the colour-key via Win32
        # SetLayeredWindowAttributes(LWA_COLORKEY) — the SAME mechanism
        # jarvis_air_cursor uses — instead of merely trusting Tk's
        # -transparentcolor to hold. A full-virtual-desktop layered window whose
        # colour-key is lost composites FULLY OPAQUE (the four-monitor blackout),
        # so we (re)key it ourselves after touching the ex-style.
        self._make_click_through_win32()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.tick()

    def _make_click_through_win32(self):
        """On Windows, set WS_EX_LAYERED | WS_EX_TRANSPARENT (+ NOACTIVATE +
        TOOLWINDOW) on the real top-level HWND and RE-KEY the background
        transparent via SetLayeredWindowAttributes(LWA_COLORKEY). Re-asserting
        WS_EX_LAYERED without re-keying is exactly what turned the old full-
        desktop overlay opaque, so the re-key is mandatory whenever we touch the
        ex-style. NEVER raises — the colour-key already gives transparency +
        click-through; this is a backstop."""
        try:
            if os.name != "nt":
                return
            import ctypes
            self.root.update_idletasks()
            user32 = ctypes.windll.user32
            # Tk wraps the toplevel in a frame on Win32; walk up to the real HWND
            # Tk colour-keyed, or styling a child leaves the toplevel opaque.
            hwnd = user32.GetParent(self.root.winfo_id())
            if not hwnd:
                hwnd = self.root.winfo_id()
            cur = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, _click_through_exstyle(cur))
            # Re-assert the colour-key — a layered window with no
            # SetLayeredWindowAttributes/UpdateLayeredWindow composites OPAQUE.
            # Skip when -transparentcolor was unavailable (we rely on -alpha then).
            if self._has_colorkey:
                user32.SetLayeredWindowAttributes(
                    hwnd, _colorref(BG_KEY), 0, LWA_COLORKEY)
        except Exception:
            # Worst case the keyed background is trusted to -transparentcolor (or
            # the low -alpha fallback) — never a hard failure.
            pass

    def _on_close(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def _draw_reticle(self, x: int, y: int, age: float, label: str,
                      color: str = ""):
        # Outer ring grows over the first GROW_DURATION seconds, then holds.
        grow = min(1.0, age / GROW_DURATION)
        radius = START_RADIUS + (MAX_RADIUS - START_RADIUS) * grow

        # Last FADE_TAIL_SECS dims toward the fade palette
        fading = age > (RETICLE_TTL - FADE_TAIL_SECS)

        # Palette selection. The default (cyan) is unchanged so every existing
        # UI-automation reticle still renders identically. Boss-mode entries
        # come through with color=="red" and pick up the priority-1 palette.
        if color == "red":
            outer_color = RED_RING_FADE if fading else RED_RING_COLOR
            inner_color = RED_RING_FADE_2 if fading else RED_RING_GLOW
            text_color  = RED_LABEL_FADE if fading else RED_LABEL_COLOR
        else:
            outer_color = RING_FADE if fading else RING_COLOR
            inner_color = RING_FADE_2 if fading else RING_GLOW
            text_color  = LABEL_FADE if fading else LABEL_COLOR

        # Crosshair (always at fixed length so the center is precise)
        self.canvas.create_line(
            x - CROSSHAIR_LEN, y, x + CROSSHAIR_LEN, y,
            fill=outer_color, width=1,
        )
        self.canvas.create_line(
            x, y - CROSSHAIR_LEN, x, y + CROSSHAIR_LEN,
            fill=outer_color, width=1,
        )

        # Two concentric rings (thick outer + thin inner for the gauge look)
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            outline=outer_color, width=2,
        )
        inner_r = max(4, radius - 5)
        self.canvas.create_oval(
            x - inner_r, y - inner_r, x + inner_r, y + inner_r,
            outline=inner_color, width=1,
        )

        # Center dot
        self.canvas.create_oval(
            x - 2, y - 2, x + 2, y + 2,
            fill=inner_color, outline="",
        )

        if label:
            self.canvas.create_text(
                x, y + radius + 8,
                text=label[:24],
                fill=text_color, font=LABEL_FONT, anchor="n",
            )

    def _should_exit(self, now: float) -> bool:
        """True when this overlay should CLOSE: parent dead (PID-recycle-aware),
        an orphan (no real parent) that has outlived the cap, or the host has
        stopped updating hud_reticles.json for STATE_STALE_EXIT_S (crashed
        without a clean shutdown). Keeps a full-virtual-desktop click-through
        layer from stranding over every monitor."""
        if not _is_parent_alive(self.parent_pid, self.parent_start):
            return True
        # Orphan cap: no real parent to track and we've lived too long → exit.
        if self.parent_pid <= 0 and (now - self._started_at) > ORPHAN_MAX_LIFETIME_S:
            return True
        # Stale-state exit: advance the freshest-seen mtime for bookkeeping, but
        # only ACT on staleness for an ORPHAN (no real parent to trust). When we
        # DO have a live real parent (checked above via _is_parent_alive, which is
        # PID-recycle-aware), a stale hud_reticles.json is NOT a crash signal: a
        # healthy JARVIS only rewrites that file on a UI-automation action, so any
        # voice-only / idle stretch longer than STATE_STALE_EXIT_S (10 min) would
        # otherwise self-exit this overlay for good — nothing re-spawns it, so
        # click/type reticles silently stopped for the rest of the session. A
        # crashed real parent is already caught above; trust that and keep the
        # stale-exit only where there's no parent liveness to rely on. 2026-07-08.
        mtime = _state_file_mtime()
        if mtime > self._last_state_mtime:
            self._last_state_mtime = mtime
        if (self.parent_pid <= 0 and self._last_state_mtime > 0
                and (now - self._last_state_mtime) > STATE_STALE_EXIT_S):
            return True
        return False

    def tick(self):
        now = time.time()
        if self._should_exit(now):
            self._on_close()
            return

        entries = _read_reticles()

        # Filter to live entries up-front so the idle path can short-circuit.
        live: list = []
        for r in entries:
            try:
                t0 = float(r.get("created_at", 0) or 0)
            except (TypeError, ValueError):
                continue
            age = now - t0
            if age < 0 or age > RETICLE_TTL:
                continue
            live.append((r, age))

        # task-13:22 CPU runaway fix: when there's nothing on-screen and
        # there was nothing last frame either, skip the 7680×2880 canvas
        # delete+repaint and back off to a slow idle poll. A 30fps full-
        # screen redraw of an empty canvas was burning a full CPU core
        # (observed PID 15964 at 99% on 2026-05-29).
        if not hasattr(self, "_prev_live_count"):
            self._prev_live_count = 0
        if not live and self._prev_live_count == 0:
            # Both this frame and last had no reticles — idle poll at 4 Hz
            # instead of redrawing 30 times a second.
            self.root.after(250, self.tick)
            return
        self._prev_live_count = len(live)

        self.canvas.delete("all")
        # Repaint the keyed background so old reticles don't smear
        self.canvas.create_rectangle(
            0, 0, self.width, self.height, fill=BG_KEY, outline="",
        )

        for r, age in live:
            try:
                rx_abs = int(r.get("x", 0))
                ry_abs = int(r.get("y", 0))
            except (TypeError, ValueError):
                continue
            # Translate virtual-desktop coords into our canvas-local coords.
            cx = rx_abs - self.origin_x
            cy = ry_abs - self.origin_y
            # Skip reticles that fall outside our canvas (shouldn't happen
            # for valid clicks, but defensive against bogus data).
            if cx < -64 or cy < -64 or cx > self.width + 64 or cy > self.height + 64:
                continue
            color = str(r.get("color", "") or "").lower()
            self._draw_reticle(
                cx, cy, age, str(r.get("label", "") or ""), color=color,
            )

        self.root.after(TICK_MS, self.tick)

    def run(self):
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, default=0)
    parser.add_argument("--y", type=int, default=0)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--parent-pid", type=int, default=0)
    # Parent's psutil create_time() at spawn — enables the PID-recycle guard in
    # _is_parent_alive. Optional (<=0 / absent → PID-exists-only fallback).
    parser.add_argument("--parent-start", type=float, default=0.0)
    args = parser.parse_args()

    overlay = ReticleOverlay(
        args.x, args.y, args.width, args.height, args.parent_pid,
        parent_start=(args.parent_start if args.parent_start > 0 else None),
    )
    overlay.run()


if __name__ == "__main__":
    main()
