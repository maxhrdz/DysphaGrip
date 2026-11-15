"""Run the DysphaGrip browser dashboard and the sensor service together on this Mac."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent
SIBLING_WEB = BASE.parent / "dysphagrip"
DEFAULT_WEB = SIBLING_WEB if (SIBLING_WEB / "server/src/app.ts").is_file() else Path.home() / "Documents/ChatGPT/HackFinal/dysphagrip"
WEB = Path(os.environ.get("DYSPHAGRIP_WEB_ROOT", DEFAULT_WEB)).expanduser().resolve()
APP_URL = "http://127.0.0.1:5173"


def status(url):
    try:
        with urlopen(url, timeout=1) as response:
            return json.load(response)
    except (OSError, ValueError):
        return {}


def ensure_build(folder, output):
    npm = shutil.which("npm")
    if not npm:
        raise RuntimeError("Install Node.js and npm to start the dashboard.")
    if not (folder / "node_modules").is_dir():
        subprocess.run([npm, "ci", "--prefix", str(folder)], check=True)
    source = [*folder.joinpath("src").rglob("*"), folder / "package.json", folder / "tsconfig.json"]
    if folder.name == "web":
        source += [folder / "index.html", folder / "vite.config.ts"]
    if not output.is_file() or any(p.stat().st_mtime > output.stat().st_mtime for p in source if p.is_file()):
        subprocess.run([npm, "run", "build", "--prefix", str(folder)], check=True)


def run(open_page=True):
    if status(APP_URL + "/api/health").get("local_dashboard"):
        if open_page:
            subprocess.run(["open", APP_URL], check=True)
        print("Dashboard already running:", APP_URL)
        return
    if not (WEB / "server/src/app.ts").is_file():
        raise RuntimeError("Web project missing. Set DYSPHAGRIP_WEB_ROOT to the dysphagrip web project folder.")
    ensure_build(WEB / "apps/web", WEB / "apps/web/dist/index.html")
    ensure_build(WEB / "server", WEB / "server/dist/index.js")
    env = {**os.environ, "PORT": "5173", "NODE_ENV": "production", "DYSPHAGRIP_LOCAL_DASHBOARD": "1",
           "DYSPHAGRIP_EXPERIMENTS_DIR": str(BASE / "experiments")}
    processes = []
    try:
        if not status("http://127.0.0.1:8770/api/monitor/status").get("state"):
            processes.append(subprocess.Popen([sys.executable, str(BASE / "web_monitor.py")], cwd=BASE, env=env))
        processes.append(subprocess.Popen(["node", "dist/index.js"], cwd=WEB / "server", env=env))
        for _ in range(80):
            if any(proc.poll() is not None for proc in processes):
                raise RuntimeError("A dashboard service exited. Check the messages above and ports 5173 / 8770.")
            if status(APP_URL + "/api/health").get("local_dashboard") and status(APP_URL + "/api/monitor/status").get("state"):
                break
            time.sleep(.2)
        else:
            raise RuntimeError("The dashboard services did not become ready.")
        print("DysphaGrip is ready at", APP_URL, flush=True)
        if open_page:
            subprocess.run(["open", APP_URL], check=True)
        while all(proc.poll() is None for proc in processes):
            time.sleep(1)
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        run(not args.no_browser)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
