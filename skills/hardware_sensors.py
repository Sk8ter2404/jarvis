"""Voice/HUD access to the full HWiNFO sensor set -- CPU/GPU temperatures, fan
speeds, package power, clocks and voltages -- via the shared-memory reader.

Action:
  hardware_sensors                 -> spoken systems snapshot (temps + fans + power)
  hardware_sensors, <fragment>     -> the single sensor whose label contains
                                      <fragment> (e.g. "VRM", "GPU Hot Spot")

Requires HWiNFO running with Settings -> Shared Memory Support enabled; degrades
to a clear, actionable message otherwise and never raises. Unlike system_pulse
(which only has the nvidia-smi GPU temp), this surfaces EVERY sensor HWiNFO polls.
"""
from __future__ import annotations

_SM_OFF = ("HWiNFO shared memory is off, sir -- enable HWiNFO Settings, "
           "Shared Memory Support and it activates live, no restart.")


def register(actions: dict) -> None:
    def hardware_sensors(arg: str = "") -> str:
        try:
            from audio import hwinfo
        except Exception:
            return "Hardware sensor access isn't wired up on this machine, sir."

        q = (arg or "").strip()
        if q:
            try:
                hit = hwinfo.find(q)
            except Exception:
                hit = None
            if hit:
                label, value, unit = hit
                return f"{label}: {value:g} {unit}, sir."
            try:
                up = hwinfo.available()
            except Exception:
                up = False
            return f"I couldn't find a sensor matching '{q}', sir." if up else _SM_OFF

        try:
            s = hwinfo.summary()
        except Exception:
            s = {"available": False}
        if not s.get("available"):
            return _SM_OFF

        parts: list[str] = []
        if s.get("cpu_temp_c") is not None:
            parts.append(f"CPU {s['cpu_temp_c']:.0f} degrees")
        if s.get("gpu_temp_c") is not None:
            parts.append(f"GPU {s['gpu_temp_c']:.0f} degrees")
        if s.get("cpu_load_pct") is not None:
            parts.append(f"CPU load {s['cpu_load_pct']:.0f} percent")
        fans = s.get("fans_rpm") or []
        if fans:
            avg = sum(v for _, v in fans) / len(fans)
            parts.append(f"{len(fans)} fans averaging {avg:.0f} RPM")
        power = s.get("power_w") or []
        if power:
            _, top_val = max(power, key=lambda lv: lv[1])
            parts.append(f"{top_val:.0f} watts package power")
        if not parts:
            return (f"HWiNFO is up with {s.get('count', 0)} sensors, sir, but none "
                    "matched the usual temperature or fan labels.")
        return "Systems check, sir -- " + ", ".join(parts) + "."

    actions["hardware_sensors"] = hardware_sensors
