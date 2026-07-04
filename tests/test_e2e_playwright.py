"""One end-to-end test: upload -> generate -> download, run across
Chromium, Firefox, and WebKit.

Marked `@pytest.mark.e2e` — slow (launches a real Streamlit server + real
browsers) and excluded from the default `pytest` run. Run explicitly with:

    pytest tests/test_e2e_playwright.py -m e2e -v

Uses EXAMCAST_MOCK_LLM=1 (see pipeline/llm_client.py) so this never touches
the real Gemini API — it verifies the mechanical upload/generate/download
flow and that the page doesn't throw, not generated-content quality.
"""
import os
import socket
import subprocess
import time

import pytest

pytestmark = pytest.mark.e2e

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(PROJECT_ROOT, "tests", "fixtures")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def streamlit_server(tmp_path_factory):
    port = _free_port()
    chroma_dir = tmp_path_factory.mktemp("e2e_chroma")
    env = {
        **os.environ,
        "EXAMCAST_MOCK_LLM": "1",
        "GEMINI_API_KEY": "sk-mock-key-for-e2e-only",
        # Never touch the real ./chroma_db from an E2E run.
        "EXAMCAST_CHROMA_PATH": str(chroma_dir),
    }
    proc = subprocess.Popen(
        [
            "venv/bin/streamlit" if os.path.exists(os.path.join(PROJECT_ROOT, "venv/bin/streamlit")) else "streamlit",
            "run",
            "app.py",
            "--server.headless",
            "true",
            "--server.port",
            str(port),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    url = f"http://localhost:{port}"
    deadline = time.time() + 30
    up = False
    while time.time() < deadline:
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                up = True
                break
        except OSError:
            time.sleep(0.5)
    if not up:
        proc.terminate()
        raise RuntimeError("Streamlit test server did not start within 30s")

    time.sleep(2)  # let the first script run complete (model loads etc.)
    yield url

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(params=["chromium", "firefox", "webkit"])
def browser_name(request):
    return request.param


def test_upload_generate_download_no_console_errors(streamlit_server, browser_name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    console_errors = []

    with sync_playwright() as p:
        launcher = getattr(p, browser_name)
        try:
            browser = launcher.launch()
        except Exception as e:
            pytest.skip(f"{browser_name} not installed: {e}")

        page = browser.new_page(viewport={"width": 1280, "height": 1400})
        page.on(
            "console",
            lambda msg: console_errors.append(msg.text) if msg.type == "error" else None,
        )
        page.on("pageerror", lambda exc: console_errors.append(str(exc)))

        page.goto(streamlit_server, timeout=30000)
        page.wait_for_selector('div[data-testid="stTextInput"] input', timeout=20000)
        page.wait_for_timeout(1000)

        page.fill('div[data-testid="stTextInput"] input', "E2E Test Subject")
        file_input = page.query_selector_all('input[type="file"]')[0]
        file_input.set_input_files(
            [
                os.path.join(FIXTURES, "paper_2022.pdf"),
                os.path.join(FIXTURES, "paper_2023.pdf"),
            ]
        )
        page.wait_for_timeout(1000)

        page.click('button:has-text("Generate practice paper")')
        page.wait_for_selector("text=Reading your papers", timeout=15000)

        deadline = time.time() + 60
        reached_results = False
        while time.time() < deadline:
            if page.query_selector("text=Practice paper ready"):
                reached_results = True
                break
            if page.query_selector('[data-testid="stAlert"]'):
                alert_text = page.query_selector('[data-testid="stAlert"]').inner_text()
                pytest.fail(f"pipeline reported an error instead of reaching results: {alert_text}")
            page.wait_for_timeout(1000)

        assert reached_results, "did not reach the results screen within 60s"

        pdf_button = page.query_selector('button:has-text("PDF")')
        assert pdf_button is not None, "PDF download button not found"
        assert pdf_button.get_attribute("disabled") is None, "PDF button is disabled — generation likely failed silently"

        with page.expect_download(timeout=10000) as download_info:
            pdf_button.click()
        download = download_info.value
        saved_path = download.path()
        assert saved_path is not None
        assert os.path.getsize(saved_path) > 0

        browser.close()

    real_errors = [
        e for e in console_errors
        if "favicon" not in e.lower() and "chrome-extension" not in e.lower()
    ]
    assert not real_errors, f"console errors during the flow: {real_errors}"
