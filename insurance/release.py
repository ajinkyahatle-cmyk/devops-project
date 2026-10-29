"""Continuous deployment to Streamlit Community Cloud.

Community Cloud redeploys an app whenever its watched Git branch changes. So a
"release" here is: assemble a small, runtime-only tree (app + model + version
stamp), push it to the `streamlit-release` branch, then poll the live app until
it reports the commit we just shipped.

  python -m insurance.release build  --out release --sha <commit>
  python -m insurance.release verify --url https://<app>.streamlit.app --sha <commit>
"""
import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Everything the running app needs, and nothing else (no training code, tests or data).
RUNTIME_FILES = [
    "app.py",
    "requirements.txt",
    "assets/hospital.jpg",
    "insurance/__init__.py",
    "insurance/schema.py",
    "insurance/metrics.py",
    "insurance/predict.py",
]
MODEL_FILES = ["metrics.json", "leaderboard.csv", "residuals.png", "feature.png"]
# Streamlit serves ./static at /app/static when enableStaticServing is on. We use it
# to publish version.json so CI can prove which commit is live.
STREAMLIT_CONFIG = "[server]\nenableStaticServing = true\n"
APT_PACKAGES = "libgomp1\n"  # needed by LightGBM, a PyCaret dependency
VERSION_PATH = "app/static/version.json"


def build_release(dest, model_dir="models", model_name="insurance_model", sha="local", root="."):
    """Assemble the deployable tree in `dest` and return the version stamp."""
    root, dest, model_dir = Path(root), Path(dest), Path(model_dir)
    if dest.exists():
        shutil.rmtree(dest)
    for rel in RUNTIME_FILES:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)

    (dest / "models").mkdir(parents=True)
    shutil.copy2(model_dir / f"{model_name}.pkl", dest / "models" / f"{model_name}.pkl")
    for name in MODEL_FILES:
        if (model_dir / name).exists():
            shutil.copy2(model_dir / name, dest / "models" / name)

    try:
        metrics = json.loads((model_dir / "metrics.json").read_text())
    except (OSError, ValueError):
        metrics = {}
    version = {
        "git_sha": sha,
        "model": metrics.get("best_model"),
        "holdout_r2": metrics.get("holdout_r2"),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (dest / "static").mkdir()
    (dest / "static" / "version.json").write_text(json.dumps(version, indent=2))
    (dest / ".streamlit").mkdir()
    (dest / ".streamlit" / "config.toml").write_text(STREAMLIT_CONFIG)
    (dest / "packages.txt").write_text(APT_PACKAGES)
    (dest / "README.md").write_text(
        "# streamlit-release\n\nGenerated deployment branch. Do not edit by hand; "
        "it is rebuilt by CI from `main`.\n"
    )
    return version


def fetch_version(base_url, timeout=10):
    """Return the live app's version.json as a dict, or None if it cannot be read."""
    url = base_url.rstrip("/") + "/" + VERSION_PATH
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_for_release(base_url, sha, timeout=900, interval=15, fetch=fetch_version):
    """Block until the live app reports `sha`; raise SystemExit on timeout."""
    deadline = time.time() + timeout
    last = None
    while True:
        live = fetch(base_url)
        if live is not None:
            last = live.get("git_sha")
            if last == sha:
                print(f"Verified: {base_url} is serving {sha[:7]} ({live.get('model')}).")
                return "verified"
        if time.time() >= deadline:
            break
        time.sleep(interval)
    if last is None:
        raise SystemExit(
            f"Could not read {VERSION_PATH} from {base_url} within {timeout}s. The app may be "
            "asleep or crashed: open it in a browser (click 'wake up' if shown) and check its logs."
        )
    raise SystemExit(
        f"Stale release: {base_url} is still serving {str(last)[:7]}, expected {sha[:7]}. "
        "Community Cloud sometimes needs a manual 'Reboot app'."
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="assemble the release tree")
    b.add_argument("--out", default="release")
    b.add_argument("--model-dir", default="models")
    b.add_argument("--sha", default="local")
    v = sub.add_parser("verify", help="poll the live app until it serves --sha")
    v.add_argument("--url", required=True)
    v.add_argument("--sha", required=True)
    v.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        print(json.dumps(build_release(args.out, args.model_dir, sha=args.sha), indent=2))
    else:
        wait_for_release(args.url, args.sha, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
