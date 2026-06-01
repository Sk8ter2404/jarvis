"""JARVIS audio package — narrow helpers for audio-adjacent integrations
that should NOT be hung off bobert_companion. Kept import-cheap: nothing
here pulls in win32com, sounddevice, or any other heavy dependency at
package-import time.

Currently exposes:

    itunes_bridge — lazy iTunes COM client. `from audio import itunes_bridge`
        then `itunes_bridge.get_client(force=False)` returns (app, None) or
        (None, error). The bridge NEVER initialises COM at import — it does
        so only inside get_client(), and even then only when iTunes.exe is
        already running OR the caller explicitly requested it via force=True.
"""
