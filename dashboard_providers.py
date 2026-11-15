"""Edit the local dashboard's API credentials without echoing secrets."""
from getpass import getpass
from pathlib import Path
import json
import os
import subprocess
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

settings = Path.home() / "Library/Application Support/DysphaGrip"


def claim():
    identity = json.loads((settings / "neon/identity.json").read_text())
    root = "https://claimable.neon.tech/v1"
    body = urlencode({"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                      "assertion": identity["identity_assertion"], "resource": "https://claimable.neon.tech/"}).encode()
    with urlopen(Request(root + "/oauth2/token", data=body), timeout=20) as response:
        token = json.load(response)["access_token"]
    with urlopen(Request(root + "/projects/" + identity["project"]["id"] + "/claim", data=b"",
                         headers={"Authorization": "Bearer " + token}), timeout=20) as response:
        url = json.load(response)["verification_uri_complete"]
    subprocess.run(["open", url], check=True)


def configure():
    os.umask(0o077)
    settings.mkdir(parents=True, exist_ok=True)
    file = settings / "providers.env"
    values = {}
    if file.exists():
        for line in file.read_text().splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
    print("DysphaGrip providers · press Enter to keep the current setting.")
    for key, prompt in [
        ("BRIGHTDATA_API_TOKEN", "Bright Data API token"), ("BRIGHTDATA_ZONE", "Bright Data Web Unlocker zone"),
        ("ELEVENLABS_API_KEY", "ElevenLabs API key"), ("ELEVENLABS_VOICE_ID", "ElevenLabs voice ID (default George)"),
        ("DATABASE_URL", "Existing Neon connection string (optional)")]:
        value = getpass(prompt + ": ").strip()
        if value:
            values[key] = value
    if "ELEVENLABS_API_KEY" in values:
        values.setdefault("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    with tempfile.NamedTemporaryFile(mode="w", dir=settings, delete=False) as out:
        for key, value in values.items():
            print(key + "=" + value, file=out)
    Path(out.name).replace(file)
    print("Saved privately on this Mac. Restart the dashboard launcher to load these settings.")


if __name__ == "__main__":
    import sys
    try:
        claim() if "--claim" in sys.argv else configure()
    except (OSError, KeyError, ValueError) as exc:
        sys.exit("Could not complete setup: " + str(exc))
