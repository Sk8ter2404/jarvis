"""tools/vision_bench/pages.py - SYNTHETIC test pages for the screen-vision
bench (ported from the 2026-10-05 research; no real screen content
anywhere).

Pages mimic the layout metrics of a video-site home grid, a watch page, a
cloud-console page, a sign-in page and an account chooser, plus FRESH2 - a
second held-out home page whose titles and queries were written for this
build before anything was run on it. Every element of interest carries
data-gt="<kind>" and data-id so ground-truth rects can be read back from the
DOM. Video links point at per-video pages (watch_<id>.html), so a click that
works changes the address - what core.grounded_click verifies.

    python tools/vision_bench/pages.py <out_dir>
"""
import html, json, os, sys

# Where main() writes: the first argument (a scratch folder - never the repo).
OUT = sys.argv[1] if __name__ == "__main__" and len(sys.argv) > 1 else None

VIDEOS = [
    ("v01", "I Survived 7 Days In An Abandoned City", "MrBeast", "182M views", "2 weeks ago", "18:42", "$500,000"),
    ("v02", "Reacting To The Worst Pizza In New York", "Kai Cenat", "4.1M views", "3 days ago", "1:12:05", "NO WAY"),
    ("v03", "Why Every Bridge Has These Weird Gaps", "Veritasium", "9.3M views", "1 month ago", "21:17", "WHY?"),
    ("v04", "The Phone That Fixes Everything (Almost)", "Marques Brownlee", "3.8M views", "5 days ago", "14:03", "FINALLY"),
    ("v05", "We Built The Fastest Water Cooled PC", "Linus Tech Tips", "2.2M views", "1 week ago", "25:48", "200 FPS"),
    ("v06", "Glitter Bomb 7.0 vs Porch Pirates", "Mark Rober", "41M views", "6 months ago", "26:31", "GOTCHA"),
    ("v07", "Beast Mode Leg Day - No Equipment", "FitnessPro Daily", "880K views", "2 days ago", "32:10", "BEAST"),
    ("v08", "Trick Shots At The World's Tallest Tower", "Dude Perfect", "12M views", "3 weeks ago", "11:56", "1,800 FT"),
    ("v09", "How Do Helicopters Actually Fly?", "Smarter Every Day", "6.7M views", "2 years ago", "19:22", "LIFT"),
    ("v10", "Lofi Beats To Study And Relax To", "Chillhop Music", "57K watching", "LIVE", "LIVE", ""),
    ("v11", "Building A Cabin In The Woods - Part 3", "Off Grid Builds", "1.4M views", "4 days ago", "40:08", "PART 3"),
    ("v12", "Cooking Every Dish From A 1950s Cookbook", "Kitchen Time Machine", "2.9M views", "1 week ago", "16:44", "1950"),
    ("v13", "The Hardest Puzzle Game Ever Made", "GameTheory Clips", "5.5M views", "8 months ago", "22:30", "IMPOSSIBLE"),
    ("v14", "Formula 1 Onboard: Monaco Pole Lap", "Motorsport Daily", "3.1M views", "1 year ago", "2:14", "P1"),
    ("v15", "Ranking Every Fast Food Burger", "Food Ranker", "7.7M views", "2 months ago", "28:59", "TIER LIST"),
]

CHIPS = ["All", "Music", "Gaming", "Live", "Podcasts", "Comedy", "Science", "Cooking",
         "Computers", "Recently uploaded", "Watched", "New to you"]
SIDE = ["Home", "Shorts", "Subscriptions", "You", "History", "Playlists",
        "Watch later", "Liked videos", "Trending", "Shopping", "Music", "Movies",
        "Live", "Gaming", "News", "Sports", "Learning", "Settings", "Help", "Send feedback"]

PALETTES = [("#ff512f", "#f09819"), ("#1d976c", "#93f9b9"), ("#4776e6", "#8e54e9"),
            ("#c31432", "#240b36"), ("#f7971e", "#ffd200"), ("#00c6ff", "#0072ff"),
            ("#ee0979", "#ff6a00"), ("#11998e", "#38ef7d"), ("#fc4a1a", "#f7b733"),
            ("#3a1c71", "#d76d77"), ("#5614b0", "#dbd65c"), ("#e65c00", "#f9d423"),
            ("#134e5e", "#71b280"), ("#283c86", "#45a247"), ("#b24592", "#f15f79")]


def css(theme):
    if theme == "dark":
        bg, fg, sub, chip, chipfg, line = "#0f0f0f", "#f1f1f1", "#aaaaaa", "#272727", "#f1f1f1", "#303030"
    else:
        bg, fg, sub, chip, chipfg, line = "#ffffff", "#0f0f0f", "#606060", "#f2f2f2", "#0f0f0f", "#e5e5e5"
    return f"""
    *{{box-sizing:border-box}} body{{margin:0;background:{bg};color:{fg};
      font-family:Roboto,Arial,sans-serif;}}
    a{{color:inherit;text-decoration:none}}
    .top{{height:56px;display:flex;align-items:center;padding:0 16px;gap:24px}}
    .logo{{font-size:20px;font-weight:700;letter-spacing:-0.5px;width:200px}}
    .search{{flex:0 0 640px;height:40px;border:1px solid {line};border-radius:40px;
      display:flex;align-items:center;padding:0 16px;font-size:16px;color:{sub};background:transparent}}
    .search input{{border:0;outline:0;background:transparent;color:{fg};font-size:16px;width:100%}}
    .side{{position:fixed;top:56px;left:0;width:240px;padding:12px}}
    .side a{{display:block;height:40px;line-height:40px;padding-left:12px;font-size:14px;border-radius:10px}}
    .main{{margin-left:240px;padding:0 24px}}
    .chips{{display:flex;gap:12px;height:56px;align-items:center}}
    .chip{{background:{chip};color:{chipfg};font-size:14px;font-weight:500;padding:6px 12px;border-radius:8px;white-space:nowrap}}
    .grid{{display:grid;grid-template-columns:repeat(5, 1fr);gap:40px 16px;padding-top:12px}}
    .card .th{{position:relative;aspect-ratio:16/9;border-radius:12px;overflow:hidden}}
    .card .th .cap{{position:absolute;left:18px;bottom:26px;font:900 46px Impact,Arial Black,sans-serif;
      color:#fff;text-shadow:3px 3px 0 #000}}
    .dur{{position:absolute;right:8px;bottom:8px;background:rgba(0,0,0,.8);color:#fff;font-size:12px;
      font-weight:500;padding:2px 4px;border-radius:4px}}
    .meta{{display:flex;gap:12px;padding-top:12px}}
    .av{{flex:0 0 36px;height:36px;border-radius:50%}}
    .title{{font-size:16px;line-height:22px;font-weight:500;display:block;max-height:44px;overflow:hidden}}
    .chan,.views{{font-size:14px;line-height:20px;color:{sub};display:block}}
    .player{{background:#000;border-radius:12px;position:relative;overflow:hidden}}
    .player .frame{{position:absolute;inset:0;}}
    .wtitle{{font-size:20px;line-height:28px;font-weight:700;margin:12px 0 8px}}
    .btn{{background:{chip};color:{chipfg};font-size:14px;font-weight:500;padding:9px 16px;border-radius:18px;border:0}}
    .rec{{display:flex;gap:8px;margin-bottom:8px}}
    .rec .th{{flex:0 0 168px;height:94px;border-radius:8px;position:relative;overflow:hidden}}
    .rec .title{{font-size:14px;line-height:20px;font-weight:500;max-height:40px}}
    .rec .chan,.rec .views{{font-size:12px;line-height:18px}}
    table{{border-collapse:collapse;font-size:14px}} td,th{{border-bottom:1px solid {line};padding:10px 16px;text-align:left}}
    """


def thumb(i, cap, dur, vid=None):
    a, b = PALETTES[i % len(PALETTES)]
    capd = f'<div class="cap">{html.escape(cap)}</div>' if cap else ""
    vid = vid or f"v{i+1:02d}"
    return (f'<div class="th" data-gt="thumb" data-id="{vid}" style="background:'
            f'linear-gradient(135deg,{a},{b})">{capd}<div class="dur">{html.escape(dur)}</div></div>')


def page(title, body, theme):
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
            f"<style>{css(theme)}</style></head><body>{body}</body></html>")


def topbar():
    return ('<div class="top"><div class="logo" data-gt="logo" data-id="logo">VideoSite</div>'
            '<div class="search" data-gt="search" data-id="search"><input aria-label="Search" '
            'placeholder="Search"></div><button class="btn" data-gt="button" data-id="signin">Sign in</button></div>')


def sidebar():
    return '<div class="side">' + "".join(
        f'<a href="#" data-gt="side" data-id="side-{html.escape(s)}">{html.escape(s)}</a>' for s in SIDE) + "</div>"


def home(theme, vids=VIDEOS):
    chips = '<div class="chips">' + "".join(
        f'<a href="#" class="chip" data-gt="chip" data-id="chip-{html.escape(c)}">{html.escape(c)}</a>' for c in CHIPS) + "</div>"
    cards = []
    for i, (vid, t, ch, views, age, dur, cap) in enumerate(vids):
        a, b = PALETTES[(i + 3) % len(PALETTES)]
        cards.append(
            f'<div class="card"><a href="watch_{vid}.html">{thumb(i, cap, dur, vid)}</a><div class="meta">'
            f'<div class="av" style="background:{b}"></div><div>'
            f'<a href="watch_{vid}.html" class="title" data-gt="title" data-id="{vid}" title="{html.escape(t)}">{html.escape(t)}</a>'
            f'<a href="#ch-{vid}" class="chan" data-gt="channel" data-id="{vid}">{html.escape(ch)}</a>'
            f'<span class="views" data-gt="views" data-id="{vid}">{html.escape(views)} &middot; {html.escape(age)}</span>'
            f'</div></div></div>')
    body = topbar() + sidebar() + f'<div class="main">{chips}<div class="grid">{"".join(cards)}</div></div>'
    return page("Home - VideoSite", body, theme)


def watch(theme):
    playing = ("w00", "Eat Everything In This Store, Win $10,000", "MrBeast", "96M views", "1 week ago")
    recs = []
    for i, (vid, t, ch, views, age, dur, cap) in enumerate(VIDEOS[1:13]):
        recs.append(
            f'<div class="rec"><a href="watch_{vid}.html">{thumb(i+1, "", dur, vid)}</a><div>'
            f'<a href="watch_{vid}.html" class="title" data-gt="title" data-id="{vid}">{html.escape(t)}</a>'
            f'<a href="#ch-{vid}" class="chan" data-gt="channel" data-id="{vid}">{html.escape(ch)}</a>'
            f'<span class="views" data-gt="views" data-id="{vid}">{html.escape(views)} &middot; {html.escape(age)}</span></div></div>')
    body = (topbar() +
            '<div style="display:flex;gap:24px;padding:24px 24px 0 24px">'
            '<div style="flex:0 0 1754px">'
            '<div class="player" data-gt="player" data-id="player" style="width:1754px;height:987px">'
            '<div class="frame" style="background:radial-gradient(circle at 30% 40%,#e8b04a,#7a2e12 60%,#160805)"></div>'
            '<div style="position:absolute;left:0;right:0;bottom:0;height:48px;background:linear-gradient(transparent,rgba(0,0,0,.7))">'
            '<span style="position:absolute;left:16px;bottom:14px;color:#fff;font-size:14px">4:21 / 19:58</span></div></div>'
            f'<div class="wtitle" data-gt="title" data-id="{playing[0]}">{html.escape(playing[1])}</div>'
            '<div style="display:flex;gap:12px;align-items:center">'
            f'<a href="#ch-w00" class="chan" data-gt="channel" data-id="w00" style="font-size:16px;font-weight:500">{playing[2]}</a>'
            '<button class="btn" data-gt="button" data-id="subscribe">Subscribe</button>'
            '<button class="btn" data-gt="button" data-id="like">Like</button>'
            '<button class="btn" data-gt="button" data-id="share">Share</button></div></div>'
            f'<div style="flex:1">{"".join(recs)}</div></div>')
    return page(f"{playing[1]} - VideoSite", body, theme)


def console(theme):
    rows = "".join(
        f'<tr><td data-gt="cell" data-id="key-{i}">key-{n}</td><td>sk-...{1000+i*37}</td><td>{d}</td>'
        f'<td><a href="#" data-gt="link" data-id="revoke-{i}">Revoke</a></td></tr>'
        for i, (n, d) in enumerate([("jarvis-local", "Oct 1, 2026"), ("ci-runner", "Sep 12, 2026"),
                                    ("prototype", "Aug 2, 2026"), ("old-laptop", "Jun 30, 2026")]))
    nav = "".join(f'<a href="#" data-gt="side" data-id="nav-{n}">{n}</a>'
                  for n in ["Dashboard", "Workbench", "API keys", "Usage", "Limits", "Billing", "Members", "Settings"])
    body = (topbar().replace("VideoSite", "Acme Cloud Console") + f'<div class="side">{nav}</div>'
            '<div class="main"><h1 data-gt="heading" data-id="h1" style="font-size:28px">API keys</h1>'
            '<button class="btn" data-gt="button" data-id="create-key">Create key</button>'
            f'<table style="margin-top:24px"><tr><th>Name</th><th>Key</th><th>Created</th><th></th></tr>{rows}</table>'
            '<h2 data-gt="heading" data-id="h2" style="font-size:20px;margin-top:40px">Usage this month</h2>'
            '<p data-gt="para" data-id="p1" style="font-size:14px">$3.42 of $5.00 daily budget used today.</p></div>')
    return page("API keys - Acme Cloud Console", body, theme)


def signin(theme):
    body = ('<div style="width:448px;margin:160px auto;padding:48px;border:1px solid #dadce0;border-radius:8px">'
            '<h1 data-gt="heading" data-id="h1" style="font-size:24px;font-weight:400">Sign in</h1>'
            '<p style="font-size:16px">to continue to Example Mail</p>'
            '<input data-gt="input" data-id="email" aria-label="Email" placeholder="Email" style="width:100%;height:56px;font-size:16px;margin:12px 0">'
            '<input data-gt="input" data-id="password" type="password" aria-label="Password" placeholder="Enter your password" style="width:100%;height:56px;font-size:16px;margin:12px 0">'
            '<button class="btn" data-gt="button" data-id="next">Next</button></div>')
    return page("Sign in - Example Accounts", body, theme)


def chooser(theme):
    accts = "".join(
        f'<a href="#" data-gt="account" data-id="acct-{i}" style="display:flex;gap:16px;padding:16px;border-bottom:1px solid #dadce0">'
        f'<span style="width:40px;height:40px;border-radius:50%;background:{c}"></span><span><b>{n}</b><br>'
        f'<span style="font-size:14px">{e}</span></span></a>'
        for i, (n, e, c) in enumerate([("Test User", "test.user@example.com", "#4776e6"),
                                       ("Work Account", "someone@example.org", "#11998e")]))
    body = ('<div style="width:448px;margin:160px auto;padding:48px;border:1px solid #dadce0;border-radius:8px">'
            '<h1 data-gt="heading" data-id="h1" style="font-size:24px;font-weight:400">Choose an account</h1>'
            f'<p style="font-size:16px">to continue to Example Console</p>{accts}'
            '<a href="#" data-gt="link" data-id="other" style="display:block;padding:16px">Use another account</a></div>')
    return page("Sign in - Example Accounts", body, theme)


# HOLDOUT-1 (research, 2026-10-05): 6 columns, wrapped 2-line titles and
# real-site-style long aria-labels on the title links. It was written before
# the research resolver ran on it, but the shipped resolver was TUNED after
# seeing its misses - treat it as a dev set now; FRESH2 is the clean one.
HOLDOUT = [
    ("h01", "Surviving 100 Days In Hardcore Mode", "Dream Clips", "3.3M views", "1 week ago", "31:02", "DAY 100"),
    ("h02", "Minecraft But Every Block Is Random", "Blocky Builds", "1.1M views", "2 days ago", "18:20", "RANDOM"),
    ("h03", "Visiting Japan For 24 Hours Straight", "IShowSpeed", "9.8M views", "5 days ago", "1:02:44", "JAPAN"),
    ("h04", "The Truth About Budget Gaming Laptops", "Tech Bench", "640K views", "3 weeks ago", "13:13", "$499"),
    ("h05", "Sidemen Among Us In Real Life 3", "Sidemen", "14M views", "2 months ago", "58:07", "IRL"),
    ("h06", "Every Country Explained In One Video", "Map Nerd", "2.0M views", "1 year ago", "42:42", ""),
    ("h07", "We Dug The Deepest Hole In Our Backyard", "Backyard Science", "5.1M views", "4 months ago", "22:05", "40 FT"),
    ("h08", "Ranking Every Pokemon Game Ever", "Retro Ranks", "870K views", "6 days ago", "35:55", "S TIER"),
    ("h09", "Ludwig Reacts To His Old Streams", "Ludwig", "2.4M views", "3 days ago", "27:31", "2019"),
    ("h10", "How Rockets Land Themselves", "Space Explained", "4.4M views", "9 months ago", "15:48", "LANDING"),
    ("h11", "What The James Webb Telescope Just Found", "Cosmos Today", "6.2M views", "2 weeks ago", "19:09", "NEW"),
    ("h12", "Cooking A Steak With Lava", "Fire Kitchen", "12M views", "1 year ago", "10:58", "2000 C"),
    ("h13", "Learn Guitar In 30 Days Challenge", "Strum Daily", "330K views", "1 month ago", "24:17", "DAY 1"),
    ("h14", "Pranking My Neighbors For A Week", "JiDion", "7.0M views", "5 months ago", "20:46", "PRANK"),
    ("h15", "I Bought The Cheapest Tesla Cybertruck", "Car Flip Co", "3.7M views", "3 weeks ago", "17:12", "$40K?"),
    ("h16", "Try Not To Laugh Challenge 99", "Smosh Pit", "2.8M views", "1 month ago", "16:30", "99"),
    ("h17", "Landing A Plane With No Experience", "Sky Rookies", "8.1M views", "7 months ago", "21:21", "MAYDAY"),
    ("h18", "Danny Goes Back To High School", "Danny Duncan", "4.6M views", "2 months ago", "14:44", "SCHOOL"),
]


def home_holdout():
    chips = '<div class="chips">' + "".join(
        f'<a href="#" class="chip" data-gt="chip" data-id="chip-{html.escape(c)}">{html.escape(c)}</a>'
        for c in CHIPS) + "</div>"
    cards = []
    for i, (vid, t, ch, views, age, dur, cap) in enumerate(HOLDOUT):
        label = f"{t} by {ch} {views} {age} {dur.replace(':', ' minutes ', 1)} seconds"
        cards.append(
            f'<div class="card"><a href="watch_{vid}.html" aria-hidden="true" tabindex="-1">'
            f'{thumb(i + 4, cap, dur, vid)}</a><div class="meta">'
            f'<div class="av" style="background:#888"></div><div>'
            f'<a href="watch_{vid}.html" class="title" data-gt="title" data-id="{vid}" '
            f'aria-label="{html.escape(label)}">{html.escape(t)}</a>'
            f'<a href="#ch-{vid}" class="chan" data-gt="channel" data-id="{vid}">{html.escape(ch)}</a>'
            f'<span class="views" data-gt="views" data-id="{vid}">{views} &middot; {age}</span>'
            f'</div></div></div>')
    body = (topbar() + sidebar() + f'<div class="main">{chips}<div class="grid" '
            f'style="grid-template-columns:repeat(6,1fr)">{"".join(cards)}</div></div>')
    return page("Home - VideoSite", body, "light")


# FRESH2 (2026-10-05, written for this build BEFORE any resolver run on it):
# a 5-column home page with different creators and topics.
FRESH2 = [
    ("f01", "Racing A Drone Through An Abandoned Mall", "Drone Lab", "2.6M views", "6 days ago", "14:20", "MALL"),
    ("f02", "We Tried Every Gas Station Sushi", "Snack Squad", "1.9M views", "2 weeks ago", "22:41", "SUSHI?"),
    ("f03", "Explaining Black Holes With Bowling Balls", "Physics Girl Too", "3.3M views", "1 month ago", "17:05", "GRAVITY"),
    ("f04", "MrBeast Gave Away A Private Island", "MrBeast", "210M views", "3 weeks ago", "20:12", "ISLAND"),
    ("f05", "Restoring A Rusty 1970s Pocket Knife", "Hand Tool Rescue", "8.8M views", "4 months ago", "31:09", "RUST"),
    ("f06", "The Cheapest Flight Around The World", "Budget Travel Bros", "740K views", "5 days ago", "26:33", "$1,900"),
    ("f07", "Speedrunning Minecraft Blindfolded", "Blind Runs", "1.2M views", "1 week ago", "48:17", "BLIND"),
    ("f08", "Making Pizza In A Volcano Oven", "Fire Kitchen", "5.0M views", "9 months ago", "12:58", "LAVA"),
    ("f09", "Why Trains Can't Climb Hills", "Practical Engineering", "2.1M views", "2 months ago", "15:40", "GRADE"),
    ("f10", "Chess Grandmaster Plays Five Kids At Once", "Chess Daily", "980K views", "4 days ago", "33:02", "5 V 1"),
    ("f11", "Surviving A Night In The Desert With No Water", "Wild Survival", "6.4M views", "1 year ago", "27:45", "NIGHT 1"),
    ("f12", "Turning My Garage Into A Recording Studio", "Studio Builds", "450K views", "3 days ago", "19:11", "STUDIO"),
    ("f13", "Kai Cenat Learns To Ice Skate", "Kai Cenat", "7.1M views", "2 weeks ago", "38:30", "SKATE"),
    ("f14", "Testing Viral Kitchen Gadgets", "Gadget Test Kitchen", "3.6M views", "6 months ago", "16:22", "WORTH IT?"),
    ("f15", "The Loudest Engine Ever Built", "Garage Heroes", "2.2M views", "5 months ago", "21:03", "180 DB"),
]
# Queries for FRESH2: (query, expected) - expected a video id, ("kind",
# data-id), ("amb", {ids}) or None (nothing on the page: must decline).
FRESH2_QUERIES = [
    ("click that MrBeast video", "f04"),
    ("play the mister beast island one", "f04"),
    ("click the drone video", "f01"),
    ("open the one about black holes", "f03"),
    ("play the sushi video", "f02"),
    ("click the knife restoration video", "f05"),
    ("click the fifth video", "f05"),
    ("click the last one", "f15"),
    ("play the chess one", "f10"),
    ("click the kai senat video", "f13"),
    ("open the video about trains", "f09"),
    ("click the garage studio video", "f12"),
    ("play the volcano pizza video", "f08"),
    ("click the minecraft blindfold one", "f07"),
    ("open the desert survival video", "f11"),
    ("click the kitchen gadgets video", "f14"),
    ("click the loudest engine video", "f15"),
    ("click history", ("kind", "side-History")),
    ("click the gaming chip", ("kind", "chip-Gaming")),
    ("click the flight video", "f06"),
    # negatives: nothing like these is on the page
    ("click the dragon video", None),
    ("play the Veritasium video", None),
    ("click the formula 1 video", None),
    ("open the lofi one", None),
]


def two_beasts_videos():
    two = list(VIDEOS)
    two[11] = ("v12", "Last To Leave The Circle Wins $500,000", "MrBeast", "140M views", "5 months ago", "24:10", "LAST ONE")
    return two


def meta_dict():
    return {"videos": VIDEOS, "two_beasts": two_beasts_videos(),
            "holdout": HOLDOUT, "fresh2": FRESH2,
            "fresh2_queries": FRESH2_QUERIES}


def main():
    if not OUT:
        print("usage: python tools/vision_bench/pages.py <out_dir>")
        return 2
    os.makedirs(OUT, exist_ok=True)
    two_beasts = two_beasts_videos()
    specs = {
        "home_light": home("light"), "home_dark": home("dark"),
        "home2beast_dark": home("dark", two_beasts),
        "watch_light": watch("light"), "watch_dark": watch("dark"),
        "console_light": console("light"), "signin_light": signin("light"),
        "chooser_light": chooser("light"),
        "fresh2_light": home("light", FRESH2),
    }
    specs["holdout_light"] = home_holdout()
    seen = set()
    for vid, t, ch, views, age, _dur, _cap in VIDEOS + HOLDOUT + FRESH2:
        if vid in seen:
            continue
        seen.add(vid)
        specs[f"watch_{vid}"] = page(f"{t} - VideoSite",
                                     topbar() + f'<h1 data-gt="title" data-id="{vid}">'
                                     f'{html.escape(t)}</h1><p>{html.escape(ch)}</p>',
                                     "dark")
    for name, src in specs.items():
        with open(os.path.join(OUT, name + ".html"), "w", encoding="utf-8") as f:
            f.write(src)
    meta = meta_dict()
    with open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=list)
    print("wrote", len(specs), "pages to", OUT)


if __name__ == "__main__":
    sys.exit(main())
