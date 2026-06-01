"""Read the ARP table, look up each unique MAC OUI via macvendors.com
(throttled), and group LAN devices by manufacturer."""
import re
import subprocess
import time
import urllib.request

CF = 0x08000000


def arp_table():
    out = {}
    raw = subprocess.run(["arp", "-a"], capture_output=True, text=True,
                         timeout=15, creationflags=CF).stdout
    for line in raw.splitlines():
        m = re.search(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9A-Fa-f]{2}(?:[-:][0-9A-Fa-f]{2}){5})", line)
        if m:
            ip, mac = m.group(1), m.group(2).upper().replace("-", ":")
            if mac.startswith(("FF:FF", "01:00:5E")) or ip.endswith(".255") or ip.startswith("224.") or ip.startswith("239."):
                continue
            out[ip] = mac
    return out


def lookup(mac):
    try:
        req = urllib.request.Request("https://api.macvendors.com/" + mac,
                                     headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.read().decode("utf-8", "replace").strip()
    except Exception:
        return "(unknown)"


arp = arp_table()
# unique OUI -> vendor
ouis = {}
for ip, mac in arp.items():
    ouis.setdefault(mac[:8], None)
print(f"{len(arp)} devices, {len(ouis)} unique vendors. Looking up...\n")
for i, oui in enumerate(list(ouis)):
    ouis[oui] = lookup(oui + ":00:00:00")
    time.sleep(1.2)  # macvendors free tier ~1 req/sec

# group
groups = {}
for ip, mac in arp.items():
    v = ouis.get(mac[:8]) or "(unknown)"
    groups.setdefault(v, []).append((ip, mac))

# Smart-home-relevant keywords to flag
SH = ("tp-link", "tplink", "amazon", "espressif", "lifx", "philips", "signify",
      "google", "nest", "roku", "wyze", "sonos", "ring", "govee", "tuya",
      "shenzhen", "ecobee", "belkin", "wemo", "smartthings", "samsung", "lg ")

print("=== DEVICES BY MANUFACTURER ===")
for v in sorted(groups, key=lambda k: -len(groups[k])):
    devs = groups[v]
    flag = "  <-- smart-home" if any(s in v.lower() for s in SH) else ""
    ips = ", ".join(ip for ip, _ in sorted(devs, key=lambda x: tuple(int(o) for o in x[0].split('.'))))
    print(f"  [{len(devs)}] {v}{flag}")
    print(f"        {ips}")
