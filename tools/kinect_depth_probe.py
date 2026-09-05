"""kinect_depth_probe — measure what the Kinect v2 ACTUALLY delivers on this box.

    python tools\\kinect_depth_probe.py [seconds] [--json out.jsonl]

WHY THIS EXISTS
===============
The 2026-09-04 capability audit could only measure the Kinect INDIRECTLY, by
parsing 33 archived session logs (219,000+ air-mouse telemetry lines), because
the live JARVIS holds the sensor as a single consumer. Three numbers that
decide real tuning could not be obtained that way:

  1. When the HAND joint drops below TrackingState 2 (measured at 12-58% of
     tracked frames), is the same-side WRIST still fully tracked? That ratio is
     exactly how much the wrist fallback in audio/kinect_bridge.arm_extension
     actually buys.
  2. How often does the SDK flag a grip LOW-confidence? That decides whether
     kinect_bridge.HAND_CONFIDENCE_GATE can be turned on without killing clicks.
  3. What do depth_corroborate()'s agree_frac / surround_nearer_frac look like
     for a REAL person at this desk? The mirror/TV thresholds are currently
     seeded from geometry, not calibrated, and cannot be set honestly until the
     real-person distribution is known.

It also reports clipped-edge rates (the owner sits ~0.6-0.9 m out, at/below the
v2 body-tracking envelope, so clipping is expected and explains a whole class of
"tracking is unreliable" that no software change can fix).

>>> READ THIS BEFORE RUNNING <<<
The Kinect v2 is SINGLE-CONSUMER. This script opens its own PyKinectRuntime, so
it CANNOT run while JARVIS holds the sensor — one of the two will get a handle
that opens but never streams. Run it only with JARVIS stopped, or with
KINECT_ENABLED off in the running instance. It writes nothing under C:\\JARVIS\\data
unless you pass --json with an explicit path.

Read-only otherwise: no config is changed, no state is persisted, and the
sensor is closed on exit.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio import kinect_bridge as kb   # noqa: E402


def _pct(n, d):
    return "n/a" if not d else f"{100.0 * n / d:5.1f}%"


def main(argv):
    seconds = 30.0
    out_path = None
    args = list(argv[1:])
    if args and not args[0].startswith("--"):
        seconds = float(args.pop(0))
    if "--json" in args:
        out_path = args[args.index("--json") + 1]

    kb.set_enabled(True)
    ok, reason = kb.available()
    if not ok:
        print(f"Kinect unavailable: {reason}")
        print("If JARVIS is running it OWNS the sensor — stop it and retry.")
        return 2

    print(f"probing for {seconds:.0f}s ... (Ctrl-C to stop early)")
    stats = {
        "frames": 0, "frames_with_body": 0, "bodies": 0,
        "hand_tracked": {"left": 0, "right": 0},
        "hand_unreliable": {"left": 0, "right": 0},
        # THE headline number: hand unreliable AND wrist reliable = frames the
        # wrist fallback rescues that used to yield lift_m=None.
        "wrist_rescues": {"left": 0, "right": 0},
        "both_unreliable": {"left": 0, "right": 0},
        "conf": {"low": 0, "high": 0, "unknown": 0},
        "clipped": {"left": 0, "right": 0, "top": 0, "bottom": 0, "any": 0},
        "verdicts": {},
        "agree_fracs": [], "surround_fracs": [], "distances": [],
        "joint_states": {"tracked": 0, "inferred": 0, "not_tracked": 0},
    }
    rows = []
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            stats["frames"] += 1
            bodies = kb.get_bodies()
            if not bodies:
                time.sleep(1.0 / 30.0)
                continue
            stats["frames_with_body"] += 1
            reports = {r.get("id"): r for r in kb.depth_check_bodies(bodies)}
            for b in bodies:
                stats["bodies"] += 1
                joints = b.get("joints") or {}
                for side in ("left", "right"):
                    hand = joints.get(f"hand_{side}")
                    wrist = joints.get(f"wrist_{side}")
                    if kb._joint_reliable(hand):
                        stats["hand_tracked"][side] += 1
                    else:
                        stats["hand_unreliable"][side] += 1
                        if kb._joint_reliable(wrist):
                            stats["wrist_rescues"][side] += 1
                        else:
                            stats["both_unreliable"][side] += 1
                    stats["conf"][b.get(f"hand_{side}_conf", "unknown")] += 1
                clip = b.get("clipped") or {}
                for edge in ("left", "right", "top", "bottom", "any"):
                    if clip.get(edge):
                        stats["clipped"][edge] += 1
                js = b.get("joint_states") or {}
                for k in ("tracked", "inferred", "not_tracked"):
                    stats["joint_states"][k] += int(js.get(k, 0))
                if isinstance(b.get("distance_m"), (int, float)):
                    stats["distances"].append(float(b["distance_m"]))
                rep = reports.get(b.get("id"))
                if rep:
                    v = rep.get("verdict", "?")
                    stats["verdicts"][v] = stats["verdicts"].get(v, 0) + 1
                    if rep.get("agree_frac") is not None:
                        stats["agree_fracs"].append(rep["agree_frac"])
                    if rep.get("surround_nearer_frac") is not None:
                        stats["surround_fracs"].append(rep["surround_nearer_frac"])
                    if out_path:
                        rows.append({"t": round(time.time(), 3),
                                     "id": b.get("id"),
                                     "distance_m": b.get("distance_m"),
                                     "clipped": clip,
                                     "joint_states": js,
                                     "hand_left_conf": b.get("hand_left_conf"),
                                     "hand_right_conf": b.get("hand_right_conf"),
                                     "depth": rep})
            time.sleep(1.0 / 30.0)
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        kb.close(final=True)

    n = stats["bodies"]
    print("\n─── stream health ───")
    print(kb.get_stream_health())
    print(f"\n─── {n} body-samples over {stats['frames']} polls "
          f"({stats['frames_with_body']} with a body) ───")
    if not n:
        print("no body was ever tracked — stand in front of the sensor and retry")
        return 1

    d = stats["distances"]
    if d:
        d_sorted = sorted(d)
        print(f"distance_m: min={d_sorted[0]:.2f} median="
              f"{d_sorted[len(d_sorted) // 2]:.2f} max={d_sorted[-1]:.2f} "
              f"(v2 body tracking is rated ~0.8-4.0 m)")

    print("\n─── FINDING 3: does the wrist fallback actually rescue frames? ───")
    for side in ("left", "right"):
        unrel = stats["hand_unreliable"][side]
        print(f"  {side:5s}: hand reliable {_pct(stats['hand_tracked'][side], n)}"
              f" | hand UNRELIABLE {_pct(unrel, n)}"
              f" -> wrist rescues {_pct(stats['wrist_rescues'][side], unrel)}"
              f" of those, both dead {_pct(stats['both_unreliable'][side], unrel)}")

    js = stats["joint_states"]
    tot = sum(js.values())
    print(f"\njoint tracking-state census over all bodies: "
          f"tracked {_pct(js['tracked'], tot)} "
          f"inferred {_pct(js['inferred'], tot)} "
          f"not_tracked {_pct(js['not_tracked'], tot)}")

    print("\n─── clipped edges (is the owner simply too close?) ───")
    for edge in ("any", "left", "right", "top", "bottom"):
        print(f"  {edge:6s}: {_pct(stats['clipped'][edge], n)}")

    print("\n─── grip confidence (decides HAND_CONFIDENCE_GATE) ───")
    ctot = sum(stats["conf"].values())
    for k in ("high", "low", "unknown"):
        print(f"  {k:8s}: {_pct(stats['conf'][k], ctot)}")

    print("\n─── depth corroboration verdicts (mirror/TV heuristic) ───")
    for v, c in sorted(stats["verdicts"].items(), key=lambda kv: -kv[1]):
        print(f"  {v:16s}: {_pct(c, n)}  ({c})")
    for label, vals in (("agree_frac", stats["agree_fracs"]),
                        ("surround_nearer_frac", stats["surround_fracs"])):
        if vals:
            v = sorted(vals)
            print(f"  {label}: min={v[0]:.2f} p10={v[len(v) // 10]:.2f} "
                  f"median={v[len(v) // 2]:.2f} max={v[-1]:.2f}")
    print("\nTUNING: for a REAL person these should be agree_frac HIGH and "
          "surround_nearer_frac LOW. Set DEPTH_SURROUND_SUSPECT_FRAC above the "
          "p90 you see here, then repeat the run standing so a mirror/TV "
          "reflection is also tracked and check the two distributions separate.")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nwrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
