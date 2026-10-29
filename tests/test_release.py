import functools
import http.server
import json
import subprocess
import sys
import threading
import time

import pytest

from insurance.release import build_release, wait_for_release


def test_release_tree_contents(trained, tmp_path):
    path, _, _ = trained
    dest = tmp_path / "rel"
    version = build_release(dest, model_dir=path.parent, sha="abc1234")
    assert version["git_sha"] == "abc1234"
    assert (dest / "models" / "insurance_model.pkl").exists()
    assert json.loads((dest / "static" / "version.json").read_text())["git_sha"] == "abc1234"
    assert "enableStaticServing = true" in (dest / ".streamlit" / "config.toml").read_text()
    assert (dest / "packages.txt").exists()
    # training/deploy code must not ship to production
    assert not (dest / "insurance" / "train.py").exists()


def test_release_tree_runs_standalone(trained, tmp_path):
    """The stripped tree must serve predictions on its own, as on Community Cloud."""
    path, _, _ = trained
    dest = tmp_path / "rel"
    build_release(dest, model_dir=path.parent, sha="abc1234")
    script = (
        "from streamlit.testing.v1 import AppTest\n"
        "at = AppTest.from_file('app.py', default_timeout=120).run()\n"
        "assert not at.exception, at.exception\n"
        "at.button[0].click().run()\n"
        "assert at.success, [e.value for e in at.error]\n"
        "print(at.success[0].value)\n"
        "print(at.sidebar.caption[0].value)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], cwd=dest, capture_output=True, text=True, timeout=240
    )
    assert out.returncode == 0, out.stderr[-1500:]
    assert "Estimated insurance charge" in out.stdout
    assert "Build abc1234" in out.stdout


@pytest.fixture
def live_app(tmp_path):
    """A tiny local web server standing in for the deployed Streamlit app."""
    static = tmp_path / "app" / "static"
    static.mkdir(parents=True)
    version_file = static / "version.json"
    version_file.write_text(json.dumps({"git_sha": "old"}))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", version_file
    server.shutdown()


def test_verify_waits_for_new_release(live_app):
    url, version_file = live_app

    def deploy_lands_later():
        time.sleep(0.4)
        version_file.write_text(json.dumps({"git_sha": "new", "model": "gbr"}))

    threading.Thread(target=deploy_lands_later).start()
    assert wait_for_release(url, "new", timeout=10, interval=0.1) == "verified"


def test_verify_fails_on_stale_release(live_app):
    url, _ = live_app
    with pytest.raises(SystemExit, match="Stale release"):
        wait_for_release(url, "new", timeout=0.5, interval=0.1)


def test_verify_fails_when_unreachable():
    with pytest.raises(SystemExit, match="Could not read"):
        wait_for_release("http://127.0.0.1:1", "new", timeout=0.3, interval=0.1)
