"""
First-time setup wizard for the Bambu H2D printer.

Triggered by voice with phrases like 'JARVIS, set up the printer' /
'configure the printer' / 'first time printer setup'. The wizard:

  1. Listens for Bambu's SSDP-style broadcasts on multicast 239.255.255.250:2021
     for ~8 seconds. Bambu printers announce themselves on the LAN with a
     NOTIFY-shaped UDP packet containing Location (the printer's IP) and USN
     (the serial number). One sniffed packet = IP + serial captured.
  2. Confirms the discovered printer with the user verbally. With multiple
     printers on the LAN it lists them and asks which is the H2D.
  3. If nothing is broadcast it falls back to reading the IP off the printer
     screen by voice (digit-extraction tolerant of "one ninety two ... ").
  4. Asks the user to read out the 8-digit LAN Access Code (Settings →
     General → LAN). Repeats back what it heard and confirms.
  5. Writes BAMBU_PRINTER_IP / BAMBU_ACCESS_CODE / BAMBU_SERIAL into both
     the live `bobert_companion` module attributes AND the source file
     (regex-anchored on the variable name) so the config persists.
  6. Hot-restarts the bambu_monitor poller via its start_monitor() helper —
     no JARVIS restart required to start receiving print status.

Action names registered:
  setup_printer / configure_printer / bambu_setup / setup_bambu /
  first_time_printer_setup

Optional one-shot non-voice form: pass the three values as the action arg
(IP first, then access code, then serial), space-separated. Useful for
re-runs after a credentials change without sitting through the voice flow.

If paho-mqtt isn't installed the wizard still writes the config — the
monitor just can't poll until the dep is added. If multicast discovery
isn't available on the host (some VPN setups eat 2021) the wizard
gracefully falls back to voice-prompting for the IP.
"""
import importlib
import os
import re
import socket
import struct
import threading
import time

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_PROJECT_DIR, "bobert_companion.py")

# Bambu's discovery multicast — matches the SSDP convention but on the
# non-standard port 2021 to avoid colliding with Windows' UPnP service.
BAMBU_MCAST_GROUP = "239.255.255.250"
BAMBU_MCAST_PORT  = 2021
DISCOVERY_SECONDS = 8.0

# How long to wait for the user's spoken reply at each prompt (seconds).
# Long enough to walk over to the printer and read the screen.
VOICE_PROMPT_TIMEOUT = 30

# A wizard run is single-threaded: the main loop is blocked on this action
# anyway, but the lock guards against a second skill calling start_monitor
# concurrently from a different thread (e.g. a stray timer).
_wizard_lock = threading.Lock()

# Number words → digits for parsing spoken codes like "one two three four".
_NUM_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
    "four": "4", "for": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "ate": "8", "nine": "9",
    # Common Whisper compound-number outputs for short reads.
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
}


# ── speech bridge ────────────────────────────────────────────────────────
def _bc():
    """Lazy import of bobert_companion. Imported in a function so this
    module loads cleanly under py_compile / pytest without spinning up
    Whisper + audio backends just to register actions."""
    return importlib.import_module("bobert_companion")


def _say(text: str) -> None:
    """Speak through JARVIS's TTS pipeline."""
    try:
        _bc()._speak(text)
    except Exception as e:
        print(f"  [bambu-setup] _speak failed ({e}); falling back to console")
        print(f"  [bambu-setup] {text}")


def _listen(timeout: float = VOICE_PROMPT_TIMEOUT) -> str:
    """Capture one utterance from the user and return its transcript.
    Returns '' on timeout / failure."""
    try:
        bc = _bc()
        audio = bc.record_speech(timeout=timeout)
        if audio is None or len(audio) == 0:
            return ""
        text, _ = bc.transcribe(audio)
        return (text or "").strip()
    except Exception as e:
        print(f"  [bambu-setup] listen failed: {e}")
        return ""


def _affirmative(text: str) -> bool:
    """True when the user clearly said yes / correct / confirmed."""
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(t.startswith(w) for w in (
        "yes", "yeah", "yep", "yup", "correct", "confirm", "right",
        "that's right", "that's correct", "affirmative", "sure", "go ahead",
    ))


def _negative(text: str) -> bool:
    """True when the user clearly said no / wrong."""
    t = (text or "").strip().lower()
    if not t:
        return False
    return any(t.startswith(w) for w in (
        "no", "nope", "wrong", "incorrect", "negative", "cancel", "stop",
    ))


# ── discovery ────────────────────────────────────────────────────────────
def _parse_bambu_packet(payload: bytes) -> dict:
    """Parse one Bambu NOTIFY-style UDP packet. Returns {'ip', 'serial',
    'model', 'name'} when the packet looks like a Bambu broadcast,
    otherwise an empty dict."""
    try:
        text = payload.decode("utf-8", errors="ignore")
    except Exception:
        return {}
    if "bambulab" not in text.lower() and "3dprinter" not in text.lower():
        return {}
    out: dict = {}
    for raw in text.splitlines():
        if ":" not in raw:
            continue
        key, _, val = raw.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if key == "location":
            out["ip"] = val
        elif key == "usn":
            out["serial"] = val
        elif key == "devmodel.bambu.com":
            out["model"] = val
        elif key == "devname.bambu.com":
            out["name"] = val
    # Only return something when at minimum we got an IP — without it the
    # entry is useless downstream.
    return out if out.get("ip") else {}


def discover_printers(duration: float = DISCOVERY_SECONDS) -> list[dict]:
    """Listen on the Bambu multicast group for printer broadcasts.
    Returns a deduplicated list of {ip, serial, model, name} dicts.
    Empty list on failure / no broadcasts heard."""
    found: dict[str, dict] = {}   # keyed by IP so the same printer doesn't appear twice
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass   # SO_REUSEPORT not on Windows
        # Bind to all interfaces on the Bambu port.
        sock.bind(("", BAMBU_MCAST_PORT))
        # Join the Bambu multicast group on every interface (INADDR_ANY).
        mreq = struct.pack("4sl", socket.inet_aton(BAMBU_MCAST_GROUP),
                           socket.INADDR_ANY)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(1.0)
        deadline = time.time() + duration
        while time.time() < deadline:
            try:
                payload, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            parsed = _parse_bambu_packet(payload)
            if not parsed:
                continue
            # Prefer the IP advertised in the packet, but fall back to the
            # sender's address if Location is somehow missing/malformed.
            ip = parsed.get("ip") or addr[0]
            parsed["ip"] = ip
            # Latest broadcast for an IP wins — that way if the model name
            # appears in a later packet we still pick it up.
            existing = found.get(ip, {})
            existing.update({k: v for k, v in parsed.items() if v})
            found[ip] = existing
    except OSError as e:
        # Port-already-in-use, no multicast support, etc. Wizard will fall
        # back to voice-prompting for the IP.
        print(f"  [bambu-setup] discovery socket failed: {e}")
    finally:
        if sock is not None:
            try: sock.close()
            except Exception: pass
    return list(found.values())


# ── digit / voice helpers ────────────────────────────────────────────────
def _voice_to_digits(text: str) -> str:
    """Extract a digit string from a transcript, handling spoken digits
    ('one two three') AND spoken numbers ('one twenty three' = 123). The
    Bambu access code is always 8 digits, so callers can re-prompt if the
    result isn't the expected length."""
    if not text:
        return ""
    t = text.lower()
    # Strip everything that isn't a letter, digit, or whitespace so
    # punctuation like 'one-two-three' splits the same way as 'one two three'.
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    out: list[str] = []
    for token in t.split():
        if token.isdigit():
            out.append(token)
            continue
        mapped = _NUM_WORDS.get(token)
        if mapped is not None:
            out.append(mapped)
    return "".join(out)


def _voice_to_ip(text: str) -> str:
    """Extract a dotted IPv4 address from a transcript. Handles three
    cases the user is likely to produce when reading off the printer
    screen: dotted ('192.168.1.42'), 'one ninety two dot one sixty
    eight ...', or '192 168 1 42' with 'dot' as the separator word."""
    if not text:
        return ""
    # Direct hit first — full dotted IPv4 anywhere in the utterance.
    m = re.search(r"\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b", text)
    if m:
        return m.group(1)
    # Voice form: digit/word tokens separated by literal "dot".
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    parts = re.split(r"\bdot\b|\bpoint\b", t)
    octets: list[str] = []
    for chunk in parts:
        digits = _voice_to_digits(chunk)
        if not digits:
            continue
        # An octet that came out as "192168" because the user didn't say
        # 'dot' is no good — we only accept one octet per dot-delimited piece.
        try:
            n = int(digits)
        except ValueError:
            continue
        if 0 <= n <= 255:
            octets.append(str(n))
    if len(octets) == 4:
        return ".".join(octets)
    return ""


def _format_digits_for_speech(digits: str) -> str:
    """Render a digit string like '12345678' as '1-2-3-4-5-6-7-8' so the
    TTS reads it as discrete digits rather than 'twelve million...'."""
    return "-".join(digits)


def _humanise_printer(p: dict) -> str:
    """JARVIS-friendly description of a discovered printer."""
    name = p.get("name") or p.get("model") or "printer"
    ip = p.get("ip", "?")
    return f"{name} at {ip}"


# ── credential persistence ──────────────────────────────────────────────
def _persist_credentials(ip: str, access: str, serial: str) -> bool:
    """Patch bobert_companion's runtime attributes AND rewrite the three
    `BAMBU_*` lines in the source file so the config survives restart.
    Returns True on success.

    Match is regex-anchored on the variable name + '=' so cosmetic
    formatting (spacing, trailing comments) doesn't break the rewrite.
    A trailing inline comment is preserved if present."""
    # Live module attrs — affects this process immediately so the monitor
    # restart below picks up the new credentials via _read_config().
    try:
        bc = _bc()
        bc.BAMBU_PRINTER_IP  = ip
        bc.BAMBU_ACCESS_CODE = access
        bc.BAMBU_SERIAL      = serial
    except Exception as e:
        print(f"  [bambu-setup] could not patch live module attrs: {e}")
        return False

    # Source rewrite for persistence.
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            src = f.read()
    except Exception as e:
        print(f"  [bambu-setup] could not read config source: {e}")
        return False

    def _replace(src_text: str, var: str, value: str) -> str:
        # Capture the leading var-name+= and any trailing inline comment so
        # we keep the existing # comments. Match a double- or single-quoted
        # literal; we always re-emit double-quoted for file-style consistency.
        pattern = re.compile(
            r"^(?P<lead>" + re.escape(var) + r"\s*=\s*)"
            r"(?P<quote>[\"'])(?P<old>[^\"']*)(?P=quote)"
            r"(?P<tail>.*)$",
            re.MULTILINE,
        )
        # Use a callable replacer so re doesn't try to interpret backslashes
        # in the value (a string replacer eats `\\` and turns `\"` into a
        # literal backslash-quote, which corrupted the rewritten lines).
        # Escape any embedded backslash or double-quote so the rewritten
        # literal stays syntactically valid even on unusual access codes.
        safe = value.replace("\\", "\\\\").replace("\"", "\\\"")

        def _do(m: re.Match) -> str:
            return f'{m.group("lead")}"{safe}"{m.group("tail")}'

        new_text, n = pattern.subn(_do, src_text, count=1)
        if n == 0:
            print(f"  [bambu-setup] could not locate '{var}' line — skipping persist")
            return src_text
        return new_text

    new_src = src
    new_src = _replace(new_src, "BAMBU_PRINTER_IP",  ip)
    new_src = _replace(new_src, "BAMBU_ACCESS_CODE", access)
    new_src = _replace(new_src, "BAMBU_SERIAL",      serial)

    if new_src == src:
        # No lines matched — the file structure has changed since this
        # skill was written. Live attrs are still patched so the user
        # gets a working session, but warn them to re-run after restart.
        print("  [bambu-setup] source file unchanged — credentials live "
              "for this session only")
        return True

    try:
        # Atomic-ish write: same-dir tempfile + replace.
        dir_ = os.path.dirname(_CONFIG_PATH)
        tmp = _CONFIG_PATH + ".bambusetup.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(new_src)
        os.replace(tmp, _CONFIG_PATH)
    except Exception as e:
        print(f"  [bambu-setup] could not write config source: {e}")
        return False
    return True


def _restart_monitor() -> bool:
    """Hot-restart the bambu_monitor poller so the new credentials take
    effect without a JARVIS restart. Returns True if polling is active
    afterwards."""
    try:
        mod = importlib.import_module("skill_bambu_monitor")
    except Exception as e:
        print(f"  [bambu-setup] bambu_monitor not loaded yet ({e}); "
              "credentials are saved — restart JARVIS to begin polling")
        return False
    try:
        return bool(mod.start_monitor())
    except Exception as e:
        print(f"  [bambu-setup] monitor restart failed: {e}")
        return False


# ── wizard ──────────────────────────────────────────────────────────────
def _wizard_pick_printer(found: list[dict]) -> dict | None:
    """Confirm which discovered printer to set up. Returns the chosen
    dict, or None if the user cancels."""
    if len(found) == 1:
        p = found[0]
        _say(f"I have one candidate, sir — {_humanise_printer(p)}. "
             "Shall I set that one up?")
        ans = _listen()
        return p if _affirmative(ans) else None

    # Multiple — read them out with indices.
    desc = "; ".join(f"{i + 1}: {_humanise_printer(p)}"
                     for i, p in enumerate(found))
    _say(f"I'm seeing {len(found)} printers on the network, sir. "
         f"{desc}. Which one is the H2D?")
    ans = _listen()
    digits = _voice_to_digits(ans)
    if digits:
        try:
            idx = int(digits) - 1
            if 0 <= idx < len(found):
                return found[idx]
        except ValueError:
            pass
    # Last-ditch: look for the model name in the reply.
    lower = (ans or "").lower()
    for p in found:
        for key in ("name", "model"):
            v = (p.get(key) or "").lower()
            if v and v in lower:
                return p
    _say("I'm afraid I didn't catch which one, sir. We can try again "
         "with 'set up the printer'.")
    return None


def _wizard_prompt_ip() -> str:
    """No printer broadcast was heard. Ask the user to read off the IP.

    The instruction explicitly asks for digit-by-digit ('one nine two')
    because that's what _voice_to_ip + _voice_to_digits can reliably parse;
    compact spoken cardinals like 'one ninety two' decode to '1-90-2' which
    fails the 0-255 octet check.
    """
    _say("I'm afraid I couldn't see the H2D on the network, sir. "
         "Could you read off the IP address from the printer screen, "
         "digit by digit, with 'dot' between each section? "
         "Settings, then WLAN.")
    for attempt in range(3):
        text = _listen()
        if not text:
            if attempt < 2:
                _say("Once more, sir, when you're ready.")
            continue
        ip = _voice_to_ip(text)
        if ip:
            _say(f"I have {ip}. Is that correct?")
            if _affirmative(_listen(timeout=15)):
                return ip
            if attempt < 2:
                _say("My apologies, sir. Once more.")
        else:
            if attempt < 2:
                _say("I couldn't pick out four octets, sir. Try again, "
                     "with 'dot' between each number.")
    return ""


def _wizard_prompt_access_code() -> str:
    """Voice-capture the 8-digit LAN access code."""
    _say("Now I'll need the LAN Access Code, sir — eight digits, found "
         "under Settings, General, LAN on the printer screen.")
    for attempt in range(3):
        text = _listen()
        if not text:
            if attempt < 2:
                _say("Once more, sir.")
            continue
        digits = _voice_to_digits(text)
        if len(digits) == 8:
            spoken = _format_digits_for_speech(digits)
            _say(f"I have {spoken}. Is that correct?")
            if _affirmative(_listen(timeout=15)):
                return digits
            if attempt < 2:
                _say("Right — once more.")
        else:
            if attempt < 2:
                _say(f"That came through as {len(digits)} digits, sir. "
                     "It should be eight — try again.")
    return ""


def _wizard_prompt_serial() -> str:
    """Voice-capture the printer serial. Serials are long and mixed
    alpha+digit, so the wizard tells the user it's optional and accepts
    'skip' to defer (the serial isn't strictly required for the wizard
    to write the config — bambu_monitor will idle until it's filled in,
    but pinging the printer for it is a future enhancement)."""
    _say("And the serial number, sir — it's on a sticker under the "
         "printer, or on the same Settings page. You can say 'skip' "
         "if you'd rather copy it in manually later.")
    text = _listen()
    if not text:
        return ""
    if any(w in text.lower() for w in ("skip", "later", "don't know", "manually")):
        _say("Very good, sir. The serial can go in by hand at your leisure.")
        return ""
    # Pull alphanumerics out of the transcript. Bambu serials are typically
    # 15-16 chars, alphanumeric, often starting '03'. We accept anything
    # 8+ chars long here since this is a free-form best-effort capture;
    # the user confirms in the next step.
    candidate = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    if len(candidate) >= 6:
        _say(f"I have {' '.join(candidate)}. Is that correct?")
        if _affirmative(_listen(timeout=15)):
            return candidate
    _say("I couldn't quite parse that, sir. I'll leave the serial blank "
         "for now — you can edit it in directly.")
    return ""


def _parse_inline_args(arg: str) -> tuple[str, str, str] | None:
    """Allow the action to be invoked non-interactively with the three
    credentials passed as the action argument. Returns (ip, access,
    serial) on a successful parse, or None to fall through to voice."""
    if not arg or not arg.strip():
        return None
    tokens = arg.strip().split()
    if len(tokens) not in (2, 3):
        return None
    ip = tokens[0]
    access = tokens[1]
    serial = tokens[2] if len(tokens) == 3 else ""
    if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip):
        return None
    if not re.match(r"^\d{6,12}$", access):
        return None
    return (ip, access, serial)


def setup_printer(arg: str = "") -> str:
    """The wizard entry point. Registered under several names so the LLM
    can match the user's natural phrasing."""
    if not _wizard_lock.acquire(blocking=False):
        return "The setup wizard is already running, sir."
    try:
        # Inline-args shortcut for re-runs and scripted setups.
        inline = _parse_inline_args(arg)
        if inline is not None:
            ip, access, serial = inline
            if _persist_credentials(ip, access, serial):
                ok = _restart_monitor()
                return ("Credentials saved and the monitor is online, sir."
                        if ok else
                        "Credentials saved, sir. The monitor will start on "
                        "next launch.")
            return "I'm afraid I couldn't write the credentials, sir."

        _say("Right away, sir. Looking for your H2D on the network...")
        found = discover_printers()
        chosen: dict | None = None
        ip = ""
        serial = ""
        if found:
            chosen = _wizard_pick_printer(found)
            if chosen is None:
                return "Setup cancelled, sir."
            ip = chosen.get("ip", "")
            serial = chosen.get("serial", "")
        else:
            ip = _wizard_prompt_ip()
            if not ip:
                _say("I couldn't capture the IP, sir. We can try again "
                     "any time.")
                return "Setup aborted — no IP captured."

        access = _wizard_prompt_access_code()
        if not access:
            _say("I couldn't capture the access code, sir. The IP is on "
                 "file but the monitor will stay idle until I have the code.")
            return "Setup aborted — no access code captured."

        # Serial — auto-detected if we got it from discovery, otherwise
        # ask the user.
        if not serial:
            serial = _wizard_prompt_serial()

        if not _persist_credentials(ip, access, serial):
            _say("I'm afraid something went wrong saving the credentials, "
                 "sir. Have a look at the console output.")
            return "Setup failed — persist error."

        polling = _restart_monitor()
        if polling:
            _say("All set, sir. The H2D is now in our care — I'll let "
                 "you know when the first status comes through.")
            return f"Bambu monitor configured for {ip} and polling."
        else:
            _say("Credentials saved, sir. The poller will pick them up on "
                 "next launch — paho-mqtt may need installing.")
            return f"Bambu credentials saved for {ip} (poller idle)."
    finally:
        _wizard_lock.release()


def register(actions):
    actions["setup_printer"]            = setup_printer
    actions["setup_bambu"]              = setup_printer
    actions["configure_printer"]        = setup_printer
    actions["bambu_setup"]              = setup_printer
    actions["first_time_printer_setup"] = setup_printer
