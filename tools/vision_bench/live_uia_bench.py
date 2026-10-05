"""tools/vision_bench/live_uia_bench.py - the screen-vision code against a
REAL browser's UI Automation tree, without touching the owner's screen.

  1. writes the synthetic pages (tools/vision_bench/pages.py) into
     <work_dir>/pages;
  2. creates a separate, hidden Windows desktop (it never becomes the input
     desktop: nothing on it is visible, takes focus or receives the
     owner's input);
  3. starts a throw-away Chrome there (its own profile under <work_dir>,
     DevTools on 127.0.0.1 only, GPU off);
  4. runs tools/vision_bench/live_probe.py on that desktop at idle priority
     with JARVIS_DATA_DIR=<work_dir>/data, which drives the shipped
     core.screen_text / core.grounded_click code against that Chrome only;
  5. kills only the Chrome processes whose command line names our profile,
     closes the desktop, and prints the result.

Nothing is captured as pixels; no model, OCR or cloud call is made; no
mouse / keyboard input is sent anywhere ("clicks" are UI Automation Invoke
on our own hidden Chrome). Needs pywin32, psutil, websocket-client (all
already in the JARVIS environment).

    python tools/vision_bench/live_uia_bench.py <work_dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
DESK = "jarvis_vision_bench"
PORT = 9339
IDLE = 0x00000040
BELOW_NORMAL = 0x00004000
NO_WINDOW = 0x08000000
UNICODE_ENV = 0x00000400


def _our_chrome(profile):
    import psutil
    out = []
    for p in psutil.process_iter(["name", "cmdline"]):
        try:
            if (p.info["name"] or "").lower() == "chrome.exe" and any(
                    profile in (a or "") for a in (p.info["cmdline"] or [])):
                out.append(p)
        except Exception:
            pass
    return out


def _cpu_s(procs):
    s = 0.0
    for p in procs:
        try:
            t = p.cpu_times()
            s += t.user + t.system
        except Exception:
            pass
    return s


def _launch(cmd, flags, env=None, cwd=None):
    import win32process
    si = win32process.STARTUPINFO()
    si.lpDesktop = DESK
    hp, _ht, pid, _tid = win32process.CreateProcess(
        None, cmd, None, None, False, flags, env, cwd, si)
    return hp, pid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("work_dir")
    ap.add_argument("--json", help="write the probe result here too")
    ap.add_argument("--probe", default=os.path.join(_HERE, "live_probe.py"),
                    help="the script to run on the hidden desktop")
    a = ap.parse_args(argv)
    work = os.path.abspath(a.work_dir)
    if os.path.normcase(work).startswith(os.path.normcase(_PROJECT)) or \
            os.path.normcase(work).startswith(os.path.normcase(r"C:\JARVIS")):
        print("work_dir must be outside the repository and C:\\JARVIS")
        return 2
    import psutil
    import win32con
    import win32event
    import win32service
    pages = os.path.join(work, "pages")
    profile = os.path.join(work, "chrome_profile")
    data = os.path.join(work, "data")
    for d in (pages, profile, data):
        os.makedirs(d, exist_ok=True)
    subprocess.run([sys.executable, "-B", os.path.join(_HERE, "pages.py"),
                    pages], check=True, cwd=_PROJECT)
    first = "file:///" + os.path.join(pages, "fresh2_light.html").replace(
        "\\", "/")
    flags = ["--no-first-run", "--no-default-browser-check", "--disable-sync",
             "--disable-extensions", "--disable-gpu",
             "--disable-background-networking", "--window-position=0,0",
             "--window-size=2560,1440", f"--remote-debugging-port={PORT}",
             "--remote-debugging-address=127.0.0.1",
             f"--user-data-dir={profile}"]
    hdesk = win32service.CreateDesktop(DESK, 0, win32con.GENERIC_ALL, None)
    out = os.path.join(work, "live_result.json")
    if os.path.exists(out):
        os.remove(out)
    report = {}
    try:
        cmd = f'"{CHROME}" ' + " ".join(f'"{f}"' for f in flags) + \
            f' "{first}"'
        _launch(cmd, BELOW_NORMAL)
        ok = False
        for _ in range(80):
            try:
                tg = json.load(urllib.request.urlopen(
                    f"http://127.0.0.1:{PORT}/json", timeout=1))
                if any(t.get("type") == "page" for t in tg):
                    ok = True
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if not ok:
            print("Chrome DevTools never came up")
            return 1
        time.sleep(2.0)
        procs = _our_chrome(profile)
        report["chrome_processes"] = len(procs)
        c0 = _cpu_s(procs)
        time.sleep(5.0)
        report["chrome_idle_cpu_pct_before"] = round(
            (_cpu_s(procs) - c0) / 5 * 100 / psutil.cpu_count(), 3)
        env = {k: v for k, v in os.environ.items()
               if k not in ("JARVIS_NO_SCREEN_READ", "JARVIS_TEST_MODE")}
        env.update({"JARVIS_DATA_DIR": data, "CUDA_VISIBLE_DEVICES": "-1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8"})
        pids = ",".join(str(p.pid) for p in procs)
        cmd = (f'"{sys.executable}" -B "{os.path.abspath(a.probe)}"'
               f' "{out}" "{pages}" {PORT} {pids}')
        c1 = _cpu_s(procs)
        t0 = time.perf_counter()
        hp, _pid = _launch(cmd, IDLE | NO_WINDOW | UNICODE_ENV, env=env,
                           cwd=_PROJECT)
        win32event.WaitForSingleObject(hp, 300000)
        report["probe_wall_s"] = round(time.perf_counter() - t0, 1)
        procs = _our_chrome(profile)
        report["chrome_cpu_s_during_probe"] = round(_cpu_s(procs) - c1, 2)
        c2 = _cpu_s(procs)
        time.sleep(5.0)
        report["chrome_idle_cpu_pct_after"] = round(
            (_cpu_s(procs) - c2) / 5 * 100 / psutil.cpu_count(), 3)
        mem = 0
        for p in procs:
            try:
                mem += p.memory_info().private
            except Exception:
                pass
        report["chrome_private_mb_after"] = round(mem / 2 ** 20)
    finally:
        for p in _our_chrome(profile):
            try:
                p.kill()
            except Exception:
                pass
        time.sleep(1.0)
        try:
            hdesk.CloseDesktop()
        except Exception:
            pass
    try:
        with open(out, encoding="utf-8") as f:
            probe = json.load(f)
    except Exception as e:
        print("no probe result:", e)
        return 1
    probe["orchestrator"] = report
    with open(out, "w", encoding="utf-8") as f:
        json.dump(probe, f, indent=1, default=str)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(probe, f, indent=1, default=str)
    _summary(probe)
    return 0


def _summary(p):
    print("orchestrator:", p.get("orchestrator"))
    print("uia host ready:", p.get("uia_host_ready"),
          p.get("uia_host_start_ms"), "ms; windows:", p.get("windows"))
    for name, pr in p.get("pages", {}).items():
        print(f"  {name:<14}", {k: v for k, v in pr.items() if k != "title"})
    sets = {}
    for q in p.get("queries", ()):
        s = sets.setdefault(q["set"], {"n": 0, "ok": 0, "wrong": 0,
                                       "declined": 0, "ms": []})
        s["n"] += 1
        s[q["verdict"]] += 1
        s["ms"].append(q["ms"])
    for name, s in sets.items():
        ms = sorted(s["ms"])
        print(f"  find {name:<9} {s['ok']}/{s['n']} ok, {s['wrong']} wrong, "
              f"{s['declined']} declined; median {ms[len(ms) // 2]:.0f} ms, "
              f"max {ms[-1]:.0f} ms")
    for q in p.get("queries", ()):
        if q["verdict"] != "ok":
            print("     ", q)
    for c in p.get("clicks", ()):
        print("  click", c)
    print("  not that one:", p.get("not_that_one"))
    for e in p.get("errors", ()):
        print("ERROR:", e)


if __name__ == "__main__":
    sys.exit(main())
