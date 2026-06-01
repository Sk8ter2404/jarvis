"""Real-display render of the unified HUD to a PNG so we can verify fonts +
layout. Uses the normal Qt platform (offscreen can't load glyph fonts → tofu).
The widget flashes briefly bottom-right, grabs after 400 ms, then quits.
Injects realistic preview data and stops the refresh timer so the injected
state persists for the grab."""
import sys
sys.path.insert(0, r"C:\JARVIS")
sys.path.insert(0, r"C:\JARVIS\hud")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
import jarvis_unified_hud as U

app = QApplication(sys.argv[:1])
slow = U._SlowData()
slow.weather = {"emoji": "⛅", "temp_c": 21, "desc": "partly cloudy"}
slow.forecast = [
    {"emoji": "☀", "label": "Today", "high_c": 24, "low_c": 13},
    {"emoji": "🌧", "label": "Tomorrow", "high_c": 19, "low_c": 11},
    {"emoji": "⛅", "label": "Mon", "high_c": 22, "low_c": 12},
]
slow.calendar = [{"time": "2:30 PM", "subject": "Bambu H2D maintenance call"}]
slow.unread_mail = 3

hud = U.UnifiedHud(parent_pid=0, slow=slow)
hud.timer.stop()  # keep injected preview data from being overwritten
hud.setGeometry(60, 60, 420, 560)
hud.state = "speaking"
hud.now_doing = "EXECUTING: play_music"
hud.now_playing = "Michael Jackson — Billie Jean"
hud.transcript = ["Jarvis, play some Michael Jackson", "Turn it up a bit, would you"]
hud.tts_amp = 0.7
hud.cpu = 34.0
hud.ram = 61.0
hud.gpu_temp = 54.0
hud.net_mbps = 2.4
hud.bambu_active = True
hud.bambu_pct = 47
hud.bambu_gcode = "RUNNING"
hud.bambu_eta_min = 92
hud.frame = 12
hud.show()

out = r"C:\JARVIS\tools\unified_hud_preview.png"


def _grab_and_quit():
    hud.update()
    pix = hud.grab()
    pix.save(out)
    print("saved", out, pix.width(), "x", pix.height())
    app.quit()


QTimer.singleShot(450, _grab_and_quit)
sys.exit(app.exec())
