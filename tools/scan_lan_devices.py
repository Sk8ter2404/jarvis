"""One-shot LAN smart-home discovery — no Amazon/cloud login required.
Finds TP-Link Kasa (UDP 9999), LIFX (UDP 56700), and Philips Hue bridges."""
import asyncio
import json

results = {"kasa": [], "lifx": [], "hue_bridges": [], "errors": {}}


# ── TP-Link Kasa (older/local devices; no login) ──
async def _scan_kasa():
    import kasa
    try:
        try:
            devs = await kasa.Discover.discover(discovery_timeout=6)
        except TypeError:
            devs = await kasa.Discover.discover()
    except Exception as e:
        results["errors"]["kasa"] = str(e)[:150]
        return
    for ip, dev in (devs or {}).items():
        try:
            await dev.update()
        except Exception:
            pass
        results["kasa"].append({
            "ip": ip,
            "alias": getattr(dev, "alias", None),
            "model": getattr(dev, "model", None),
            "is_on": getattr(dev, "is_on", None),
        })


try:
    asyncio.run(_scan_kasa())
except Exception as e:
    results["errors"]["kasa"] = str(e)[:150]

# ── LIFX (UDP broadcast) ──
try:
    from lifxlan import LifxLAN
    lan = LifxLAN()
    for lt in (lan.get_lights() or []):
        try:
            results["lifx"].append({
                "label": lt.get_label(),
                "ip": lt.get_ip_addr(),
                "group": lt.get_group_label(),
            })
        except Exception:
            results["lifx"].append({"ip": getattr(lt, "ip_addr", "?")})
except Exception as e:
    results["errors"]["lifx"] = str(e)[:150]

# ── Philips Hue bridges (cloud discovery endpoint returns LOCAL IPs) ──
try:
    import requests
    r = requests.get("https://discovery.meethue.com/", timeout=6)
    for b in (r.json() or []):
        results["hue_bridges"].append(b)
except Exception as e:
    results["errors"]["hue"] = str(e)[:150]

print(json.dumps(results, indent=2, default=str))
print("\nSUMMARY: kasa=%d  lifx=%d  hue_bridges=%d" % (
    len(results["kasa"]), len(results["lifx"]), len(results["hue_bridges"])))
