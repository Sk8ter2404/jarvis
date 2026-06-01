"""Full LAN inventory: ping-sweep every /24 the host is on, read the ARP table,
identify each device by MAC vendor (OUI), and re-run a longer Kasa discovery.
Goal: surface EVERY smart-home device, not just the ones that answered the
first quick broadcast. No cloud/Amazon needed."""
import asyncio
import concurrent.futures as cf
import json
import re
import socket
import subprocess
import sys
import time

CF = 0x08000000  # CREATE_NO_WINDOW

# Common smart-home / IoT MAC OUI prefixes (first 3 octets, upper, no sep).
OUI = {
    "TP-Link/Kasa/Tapo": ["003192","1C3BF3","50C7BF","6032B1","AC84C6","B0BE76",
                           "1027F5","5091E3","98DAC4","30DE4B","7CC294","E848B8",
                           "788CB5","C006C3","9C5322","54AF97","F0A731","005F67"],
    "Amazon Echo/Fire":  ["00FC8B","34D270","40B4CD","440049","6854FD","747548",
                           "F0272D","FC65DE","08A6BC","68F73B","ACCF85","B47C9C",
                           "50DCE7","A002DC","F0F0A4","4CEFC0","CC9EA2","380A94"],
    "Espressif (ESP/IoT)":["240AC4","30AEA4","3C71BF","84F3EB","A020A6","807D3A",
                           "246F28","B4E62D","CC50E3","D8A01D","E09806","EC94CB",
                           "5443B2","8CAAB5","A4CF12","C82B96","DC4F22"],
    "LIFX":              ["D073D5"],
    "Philips Hue":       ["001788"],
    "Google/Nest":       ["F4F5D8","F4F5E8","6466B3","D8EB46","1CF29A","30FD38",
                           "548998","A47733","E4F042","CC3ADF"],
    "Roku":              ["B0A737","CC6DA0","D0004B","DC3A5E","8C49B6","AC3A7A"],
    "Wyze":              ["2CAA8E","7C78B2","D03F27"],
    "Sonos":             ["000E58","347E5C","5CAAFD","78282D","B8E937","949F3E"],
    "Ring/Amazon":       ["0C4710","34EA34","B47443"],
    "Govee":             ["D43D39","C4D0AE","E0B6F5"],
}
_PREFIX2VENDOR = {pfx: v for v, lst in OUI.items() for pfx in lst}


def _host_subnets():
    nets = []
    try:
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            for a in addrs:
                if getattr(a, "family", None) == socket.AF_INET:
                    ip = a.address
                    if ip.startswith("169.254.") or ip == "127.0.0.1":
                        continue
                    nets.append(ip)
    except Exception:
        pass
    return nets


def _ping(ip):
    try:
        r = subprocess.run(["ping", "-n", "1", "-w", "250", ip],
                           capture_output=True, text=True, timeout=2,
                           creationflags=CF)
        return ip if ("TTL=" in r.stdout or "ttl=" in r.stdout) else None
    except Exception:
        return None


def _sweep(prefix):
    hosts = [f"{prefix}.{i}" for i in range(1, 255)]
    alive = []
    with cf.ThreadPoolExecutor(max_workers=64) as ex:
        for r in ex.map(_ping, hosts):
            if r:
                alive.append(r)
    return alive


def _arp_table():
    out = {}
    try:
        raw = subprocess.run(["arp", "-a"], capture_output=True, text=True,
                             timeout=15, creationflags=CF).stdout
    except Exception:
        return out
    for line in raw.splitlines():
        m = re.search(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9A-Fa-f]{2}(?:[-:][0-9A-Fa-f]{2}){5})", line)
        if m:
            ip = m.group(1)
            mac = m.group(2).upper().replace("-", ":")
            if mac.startswith("FF:FF") or ip.endswith(".255"):
                continue
            out[ip] = mac
    return out


def _vendor(mac):
    pfx = mac.replace(":", "")[:6].upper()
    return _PREFIX2VENDOR.get(pfx)


def _kasa_rescan():
    devs = {}
    try:
        import kasa

        async def go():
            try:
                return await kasa.Discover.discover(discovery_timeout=10)
            except TypeError:
                return await kasa.Discover.discover()
        found = asyncio.run(go()) or {}
        for ip, d in found.items():
            try:
                asyncio.run(d.update())
            except Exception:
                pass
            devs[ip] = getattr(d, "alias", None) or getattr(d, "model", "?")
    except Exception as e:
        devs["_error"] = str(e)[:120]
    return devs


def main():
    subnets = _host_subnets()
    print("host IPv4s:", subnets)
    prefixes = sorted({".".join(ip.split(".")[:3]) for ip in subnets})
    print("sweeping subnets:", prefixes)
    for pfx in prefixes:
        alive = _sweep(pfx)
        print(f"  {pfx}.0/24 -> {len(alive)} hosts responded to ping")

    arp = _arp_table()
    print(f"\nARP table: {len(arp)} entries")
    identified = {}
    for ip, mac in sorted(arp.items(), key=lambda x: tuple(int(o) for o in x[0].split('.'))):
        v = _vendor(mac)
        tag = v or ""
        if v:
            identified.setdefault(v, []).append(ip)
        print(f"  {ip:16s} {mac}  {tag}")

    print("\n=== SMART-HOME / IoT DEVICES IDENTIFIED BY VENDOR ===")
    if identified:
        for v, ips in sorted(identified.items()):
            print(f"  {v}: {len(ips)}  -> {', '.join(ips)}")
    else:
        print("  (none matched the built-in OUI list)")

    print("\n=== KASA RE-SCAN (10s) ===")
    kd = _kasa_rescan()
    for ip, alias in kd.items():
        print(f"  {ip}: {alias}")
    print(f"  kasa total: {len([k for k in kd if k != '_error'])}")


if __name__ == "__main__":
    main()
