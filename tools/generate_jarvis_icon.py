#!/usr/bin/env python3
"""Render the JARVIS arc-reactor icon to assets/jarvis_icon.{png,ico}.

This is a one-shot build script. The output PNG/ICO are checked in next to
this file under assets/ so the tray and Windows shortcut can load them
directly without needing Pillow at runtime. tray.py still falls back to a
procedural draw if the asset is missing, so re-running this script is only
necessary if the design itself changes.

The look matches the HUD's arc-reactor (cyan core ring, inner spoked ring,
glowing hub) so that the tray icon, HUD, and desktop shortcut all read as
the same visual identity.
"""
from __future__ import annotations

import math
import os
import sys

try:
    from PIL import Image, ImageDraw, ImageFilter
except Exception as e:
    print(f"[generate_icon] Pillow required: {e}")
    sys.exit(1)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR  = os.path.join(PROJECT_DIR, "assets")
PNG_PATH    = os.path.join(ASSETS_DIR, "jarvis_icon.png")
ICO_PATH    = os.path.join(ASSETS_DIR, "jarvis_icon.ico")

# Palette mirrors hud/jarvis_hud.py — keep these in sync if the HUD palette
# shifts so the tray, HUD, and shortcut continue to read as one identity.
CYAN        = (76, 201, 255)
CYAN_BRIGHT = (158, 231, 255)
CYAN_DIM    = (27, 74, 102)
CYAN_GLOW   = (110, 220, 255)
BG_DARK     = (4, 8, 13)
HUB_WHITE   = (235, 250, 255)


def render_arc_reactor(size: int = 256) -> Image.Image:
    """Draw the JARVIS arc-reactor on a transparent RGBA canvas.

    Layered (back to front):
      1. soft cyan radial glow
      2. dark inner disc (panel)
      3. outer cyan ring
      4. eight inner-segment spokes (arc-reactor "petals")
      5. inner cyan ring
      6. central white-hot hub
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d   = ImageDraw.Draw(img)

    cx = cy = size / 2

    # Soft outer glow — paint a fat dim ring, then blur, so the icon reads
    # as "lit from within" at the small tray rasterisations.
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd   = ImageDraw.Draw(glow)
    r_glow = size * 0.46
    for i in range(8):
        t = i / 7
        rr = r_glow - i * (size * 0.02)
        alpha = int(50 * (1 - t))
        gd.ellipse(
            [cx - rr, cy - rr, cx + rr, cy + rr],
            outline=CYAN_GLOW + (alpha,), width=3,
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=size * 0.025))
    img.alpha_composite(glow)

    # Dark panel disc — gives the spokes and hub somewhere to land.
    r_panel = size * 0.40
    d.ellipse(
        [cx - r_panel, cy - r_panel, cx + r_panel, cy + r_panel],
        fill=BG_DARK + (255,),
        outline=CYAN_DIM + (255,), width=max(1, size // 128),
    )

    # Outer cyan ring — the strongest visual signature of the arc reactor.
    r_outer = size * 0.42
    ring_w  = max(2, size // 32)
    d.ellipse(
        [cx - r_outer, cy - r_outer, cx + r_outer, cy + r_outer],
        outline=CYAN + (255,), width=ring_w,
    )
    # Slim brighter accent just inside it for the "two-tone" reactor look.
    r_accent = r_outer - ring_w - 1
    d.ellipse(
        [cx - r_accent, cy - r_accent, cx + r_accent, cy + r_accent],
        outline=CYAN_BRIGHT + (200,), width=max(1, size // 80),
    )

    # Eight spokes / petals between the outer ring and the inner hub.
    r_spoke_out = r_panel - max(2, size // 64)
    r_spoke_in  = size * 0.16
    spoke_w     = max(2, size // 28)
    for i in range(8):
        ang = i * (math.pi / 4) + math.pi / 8
        x1 = cx + math.cos(ang) * r_spoke_in
        y1 = cy + math.sin(ang) * r_spoke_in
        x2 = cx + math.cos(ang) * r_spoke_out
        y2 = cy + math.sin(ang) * r_spoke_out
        d.line([(x1, y1), (x2, y2)], fill=CYAN + (220,), width=spoke_w)

    # Inner cyan ring just inside the spokes — encloses the hub.
    r_inner = size * 0.18
    d.ellipse(
        [cx - r_inner, cy - r_inner, cx + r_inner, cy + r_inner],
        outline=CYAN_BRIGHT + (255,), width=max(2, size // 48),
    )

    # White-hot hub at the dead center — small radial gradient via stacked
    # ellipses so it reads as a glow at any DPI rasterisation.
    hub_layers = [
        (size * 0.16, CYAN_GLOW + (110,)),
        (size * 0.13, CYAN_BRIGHT + (200,)),
        (size * 0.10, HUB_WHITE + (255,)),
        (size * 0.06, (255, 255, 255, 255)),
    ]
    for r, color in hub_layers:
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)

    return img


def main() -> int:
    os.makedirs(ASSETS_DIR, exist_ok=True)

    # Master is rendered at 512px so downsampled tray copies stay crisp at
    # 16/24/32/40/48px. Pillow's LANCZOS resampling preserves the ring
    # antialiasing for the small tray sizes.
    master = render_arc_reactor(512)
    png    = master.resize((256, 256), Image.LANCZOS)
    png.save(PNG_PATH, format="PNG")
    print(f"[generate_icon] wrote {PNG_PATH}")

    # Multi-resolution ICO so Windows picks the sharpest variant for the
    # tray (16/24/32/48), taskbar (32/48), and Start menu (256). Pillow
    # builds the ICO from a list of (size, size) tuples; pass the master
    # PNG and let Pillow downsample each frame.
    sizes = [(16, 16), (24, 24), (32, 32), (40, 40), (48, 48),
             (64, 64), (128, 128), (256, 256)]
    master.save(ICO_PATH, format="ICO", sizes=sizes)
    print(f"[generate_icon] wrote {ICO_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
