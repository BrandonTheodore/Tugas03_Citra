"""Jalankan perintah CamVid di latar belakang untuk GUI MATLAB."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path


def start(folder):
    launcher = Path(__file__).resolve()
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    with (folder / "process.log").open("wb") as log:
        subprocess.Popen([sys.executable, "-u", str(launcher), "run", str(folder)],
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **options)


def run(folder):
    status = 1
    try:
        arguments = json.loads((folder / "arguments.json").read_text(encoding="utf-8"))
        script = Path(__file__).with_name("camvid.py").resolve()
        process = subprocess.Popen([sys.executable, "-u", str(script), *arguments], stdin=subprocess.DEVNULL)
        while process.poll() is None:
            if (folder / "stop").exists():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(0.2)
        status = process.returncode
    except (OSError, ValueError) as error:
        print(f"Proses CamVid gagal: {error}", file=sys.stderr)
    finally:
        temporary = folder / "status.tmp"
        temporary.write_text(str(status), encoding="ascii")
        temporary.replace(folder / "status.txt")


if __name__ == "__main__":
    action, directory = sys.argv[1:]
    folder = Path(directory).resolve()
    if action == "start":
        start(folder)
    elif action == "run":
        run(folder)
    else:
        sys.exit(f"Perintah proses CamVid tidak dikenal: {action}")
