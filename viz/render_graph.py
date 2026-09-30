#!/usr/bin/env python3
"""
Render a coalition-formation transition graph as an image from the command line.

This produces images of the transition graph that is shown in the interactive
visualizer (the right-hand panel of http://localhost:3000/#<profile>). It drives
the actual web app with a headless Chromium (via Playwright) and screenshots the
graph panel, so the output matches what you see in the browser.

Requirements:
  - Playwright (pip install playwright) with a Chromium browser.
  - The visualizer backend (python viz/service_viz.py) and frontend
    (cd viz && npm run dev) reachable on their default ports. Both are
    auto-started by this script if they are not already running (unless
    --no-start-servers is given).

Examples:
  # Render the profile referenced by the URL hash (key = filename stem)
  python render_graph.py eq_n3_power_threshold_RICE_by_GDP_fbbdac

  # By strategy-table filename or path
  python render_graph.py strategy_tables/weak_governance.xlsx --out weak.png

  # Customise the view (matches the "Visualisation Options" panel in the app)
  python render_graph.py --profile eq_n3_power_threshold_RICE_by_GDP_fbbdac \
      --coloring absorbing --threshold 0.05 --out absorbing.png
"""

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Make repo root importable so this script can be run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
VIZ_DIR = Path(__file__).resolve().parent
DEFAULT_STRATEGY_DIR = REPO_ROOT / "strategy_tables"

DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_FRONTEND_URL = "http://localhost:3000"

# Browser executables to try, in order. The first existing one wins.
BROWSER_CANDIDATES = [
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
]

# Names of the keyboard shortcuts used by the app to toggle display elements.
KEY_TOGGLES = {
    "self_loops": "s",
    "edge_labels": "e",
    "node_labels": "l",
    "geo_level": "g",
}


class _StartedServer:
    """A background server started by this script (stopped at exit)."""

    def __init__(self, proc, name):
        self.proc = proc
        self.name = name

    def stop(self):
        try:
            # Kill the whole process group so child processes (e.g. vite's
            # node child) are terminated too.
            os.killpg(os.getpgid(self.proc.pid), 15)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass


_started_servers = []


def _stop_started_servers():
    for server in _started_servers:
        server.stop()


atexit.register(_stop_started_servers)


def log(msg):
    print(f"[render_graph] {msg}", file=sys.stderr)


def is_up(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def _wait_until(url, timeout_seconds, interval=0.5):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if is_up(url):
            return True
        time.sleep(interval)
    return False


def _parse_base_url(url, default_port):
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname, parsed.port or default_port


def ensure_backend(api_base, no_start_servers):
    """Make sure the visualizer backend responds; start it if necessary."""
    host, port = _parse_base_url(api_base, default_port=8000)
    if is_up(f"{api_base}/profiles"):
        log(f"Backend already running at {api_base}")
        return
    if no_start_servers:
        raise SystemExit(
            f"Backend not reachable at {api_base} and --no-start-servers given. "
            "Start it with: cd viz && python service_viz.py"
        )
    log(f"Starting backend: python service_viz.py (host={host}, port={port})")
    proc = subprocess.Popen(
        [sys.executable, "service_viz.py", "--host", host, "--port", str(port)],
        cwd=str(VIZ_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _started_servers.append(_StartedServer(proc, "backend"))
    if not _wait_until(f"{api_base}/profiles", timeout_seconds=60):
        raise SystemExit(f"Backend failed to start at {api_base}")


def ensure_frontend(frontend_url, no_start_servers):
    """Make sure the frontend responds; start vite if necessary."""
    if is_up(frontend_url):
        log(f"Frontend already running at {frontend_url}")
        return
    if no_start_servers:
        raise SystemExit(
            f"Frontend not reachable at {frontend_url} and --no-start-servers given. "
            "Start it with: cd viz && npm run dev"
        )
    npm = shutil.which("npm")
    if not npm:
        raise SystemExit("npm not found on PATH; cannot start the frontend.")
    log(f"Starting frontend: npm run dev (cwd={VIZ_DIR})")
    proc = subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(VIZ_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    _started_servers.append(_StartedServer(proc, "frontend"))
    if not _wait_until(frontend_url, timeout_seconds=60):
        raise SystemExit(f"Frontend failed to start at {frontend_url}")


def normalize_key(s):
    """Match the frontend's normalizeProfileKey(): strip path + extension, lowercase."""
    s = s.strip().lstrip("#/")
    s = s.split("/")[-1]
    for ext in (".xlsx", ".json"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return s.lower()


def resolve_profile(profile_arg, api_base, strategy_dir):
    """Resolve a profile key/name/path to an absolute XLSX path.

    Prefers the backend's profile list, since only those files can be loaded by
    the web app (the dropdown/URL matching only covers files listed there).
    Falls back to the local filesystem only when the backend is unreachable.
    """
    arg = str(profile_arg)

    if is_up(f"{api_base}/profiles"):
        with urllib.request.urlopen(f"{api_base}/profiles", timeout=15) as resp:
            profiles = json.load(resp)["profiles"]
        key = normalize_key(arg)

        exact = []
        for p in profiles:
            candidates = [p["name"], p["filename"], p["path"], p["path"].split("/")[-1]]
            if any(normalize_key(c) == key for c in candidates):
                exact.append(p)

        matches = exact or [p for p in profiles if key in normalize_key(p["name"])]
        if len(matches) == 1:
            return matches[0]["path"]
        if len(matches) > 1:
            names = "\n  ".join(p["name"] for p in matches[:10])
            raise SystemExit(
                f"Ambiguous profile key {arg!r}: matches multiple files:\n  {names}\n"
                "Pass the full filename or path instead."
            )
        raise SystemExit(
            f"Profile not found: {arg!r}\n"
            f"No matching file in the backend's profile list ({strategy_dir}). "
            "Subdirectory profiles cannot be rendered by the visualizer."
        )

    # Backend unreachable: fall back to the local filesystem.
    for candidate in (
        Path(arg),
        Path(arg + ".xlsx"),
        REPO_ROOT / arg,
        REPO_ROOT / (arg + ".xlsx"),
        strategy_dir / arg,
        strategy_dir / (arg + ".xlsx"),
    ):
        try:
            if candidate.is_file():
                return str(candidate.resolve())
        except OSError:
            pass

    raise SystemExit(
        f"Profile not found: {arg!r} (backend at {api_base} is unreachable and no "
        "matching local file was found).\nStart the backend or pass a full path."
    )


def _pick_browser(executable):
    if executable:
        return executable
    for cand in BROWSER_CANDIDATES:
        if Path(cand).is_file():
            return cand
    return None  # Fall back to whatever Playwright has bundled / configured.


def _expand_visualisation_options(page):
    """Open the 'Visualisation Options' collapsible so radios/inputs are visible."""
    is_active = page.evaluate(
        "() => document.querySelector('button.collapsible')?.classList.contains('active') ?? false"
    )
    if not is_active:
        page.click("button.collapsible")
        page.wait_for_timeout(200)


def _apply_options(page, args):
    """Apply visualization options through the app's own UI and re-render."""
    before = page.evaluate("() => window.__graphRenderCount || 0")
    changed = False

    # Threshold input (listener is on 'change', so dispatch it after filling).
    if args.threshold is not None:
        page.locator("#prob-threshold").fill(str(args.threshold))
        page.locator("#prob-threshold").dispatch_event("change")
        changed = True

    if args.filter_mode is not None or args.coloring is not None or args.layout is not None:
        _expand_visualisation_options(page)

    if args.filter_mode is not None:
        page.check(f'input[name="filter-mode"][value="{args.filter_mode}"]')
        changed = True

    if args.coloring is not None:
        page.check(f'input[name="node-coloring"][value="{args.coloring}"]')
        changed = True

    if args.layout is not None:
        page.check(f'input[name="layout-mode"][value="{args.layout}"]')
        changed = True

    # Keyboard toggles: blur any focused input first so the keys aren't typed.
    if any(getattr(args, f"no_{key}") for key in KEY_TOGGLES):
        page.evaluate("() => { if (document.activeElement) document.activeElement.blur(); }")
        for key, hotkey in KEY_TOGGLES.items():
            if getattr(args, f"no_{key}"):
                page.keyboard.press(hotkey)
                changed = True

    if changed:
        # Wait until the app re-rendered the graph (counter incremented in
        # GraphRenderer.render), then give the browser a moment to paint.
        try:
            page.wait_for_function(
                f"(window.__graphRenderCount || 0) > {before}",
                timeout=args.timeout * 1000,
            )
        except Exception as e:
            log(f"Warning: did not observe a re-render event ({e}); continuing.")
        page.wait_for_timeout(300)


def _screenshot_graph(page, out_path, with_overlay, args):
    # Hide the floating result box (states/transitions/E_pi[G] summary in the
    # top-right corner) unless explicitly requested; the summary is still
    # printed to the terminal.
    if not args.with_result_indicator:
        page.evaluate(
            "document.getElementById('result-indicator').style.display = 'none'"
        )
    if with_overlay:
        # Capture the whole visible graph panel rather than just the graph
        # container element.
        box = page.locator("#graph-container").bounding_box()
        if box is None:
            raise SystemExit("Could not locate the graph panel.")
        clip = {
            "x": box["x"],
            "y": 0,
            "width": box["width"],
            "height": page.viewport_size["height"],
        }
        page.screenshot(path=str(out_path), clip=clip)
    else:
        page.locator("#graph-container").screenshot(path=str(out_path))


def _fetch_graph_summary(api_base, profile_path, args):
    """Fetch the same graph JSON the app uses, for a summary in the output.

    Mirrors the frontend's fallback: if the live API fails (e.g. legacy profiles
    that no longer compute), fall back to the precomputed JSON in viz/data/.
    """
    if args.no_summary:
        return None
    url = f"{api_base}/graph?profile={urllib.parse.quote(profile_path)}"
    try:
        with urllib.request.urlopen(url, timeout=args.timeout * 2) as resp:
            return json.load(resp)
    except Exception as e:
        log(f"Warning: could not fetch graph summary from API ({e}); trying static data.")
    static_path = VIZ_DIR / "data" / f"{Path(profile_path).stem}.json"
    if static_path.is_file():
        try:
            return json.loads(static_path.read_text())
        except Exception as e:
            log(f"Warning: could not read static summary {static_path}: {e}")
    return None


def _print_summary(out_path, profile_path, graph, show_json):
    name = Path(profile_path).stem
    print(f"Wrote {out_path}")
    print(f"  profile: {name}")
    if graph is None:
        return
    meta = graph.get("metadata", {})
    print(f"  states: {meta.get('num_states')}, transitions: {meta.get('num_transitions')}")
    print(f"  players: {meta.get('num_players')}")
    cfg = meta.get("config", {})
    print(
        "  config: "
        f"power_rule={cfg.get('power_rule')}, "
        f"unanimity={cfg.get('unanimity_required')}, "
        f"min_power={cfg.get('min_power')}"
    )
    g = meta.get("expected_geo_level")
    if g is not None:
        print(f"  E_pi[G]: {g:.2f} °C")
    mixing = meta.get("mixing_time")
    if mixing is not None:
        print(f"  mixing_time: {mixing}")
    abs_t = meta.get("absorption_time")
    if abs_t:
        print(f"  absorption_time (worst-case): {round(abs_t['max'])} steps")
    if show_json:
        print(json.dumps(graph, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Render a transition graph as an image (headless browser screenshot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "profile",
        nargs="?",
        help="Profile key, filename, or path. The key equals the URL hash, e.g. "
        "eq_n3_power_threshold_RICE_by_GDP_fbbdac, or a file like "
        "strategy_tables/weak_governance.xlsx.",
    )
    parser.add_argument(
        "-p", "--profile", dest="profile_opt",
        help="Alias for the positional profile argument.",
    )
    parser.add_argument(
        "-o", "--out",
        help="Output image path (default: <cwd>/<profile>.png).",
    )
    parser.add_argument("--width", type=int, default=1600, help="Browser viewport width in pixels.")
    parser.add_argument("--height", type=int, default=1000, help="Browser viewport height in pixels.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Edge probability threshold (0-1).")
    parser.add_argument(
        "--filter-mode", choices=["absolute", "cumulative"], default="absolute",
        help="How the probability threshold filters edges.",
    )
    parser.add_argument(
        "--coloring", choices=["none", "absorbing", "geoengineering", "deployer", "internal", "external", "internal-external", "gamma-core", "ricke"], default="none",
        help="Node coloring mode.",
    )
    parser.add_argument(
        "--layout", choices=["default", "connections", "deployer", "geo-level"], default="default",
        help="Graph layout mode.",
    )
    parser.add_argument("--no-self-loops", action="store_true", help="Hide self-loop edges.")
    parser.add_argument("--no-edge-labels", action="store_true", help="Hide edge probability labels.")
    parser.add_argument("--no-node-labels", action="store_true", help="Hide node labels.")
    parser.add_argument("--no-geo-level", action="store_true", help="Hide geoengineering levels on nodes.")
    parser.add_argument(
        "--with-overlay", action="store_true",
        help="Capture the full visible graph panel (not just the graph container element).",
    )
    parser.add_argument(
        "--with-result-indicator", action="store_true",
        help="Keep the small result box (states/transitions/E_pi[G]) that floats in "
             "the top-right corner of the graph; it is hidden by default.",
    )
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Backend API base URL.")
    parser.add_argument("--frontend-url", default=DEFAULT_FRONTEND_URL, help="Frontend (vite) URL.")
    parser.add_argument(
        "--browser", help="Path to a Chromium/Chrome executable (auto-detected by default).",
    )
    parser.add_argument(
        "--no-start-servers", action="store_true",
        help="Fail instead of auto-starting backend/frontend.",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Maximum seconds to wait for rendering.")
    parser.add_argument("--no-summary", action="store_true", help="Skip printing the graph summary.")
    parser.add_argument("--json", action="store_true", help="Also print the full graph JSON to stdout.")
    args = parser.parse_args()

    profile_arg = args.profile_opt or args.profile
    if not profile_arg:
        parser.error("profile is required")

    profile_path = resolve_profile(profile_arg, args.api_base, DEFAULT_STRATEGY_DIR)
    log(f"Resolved profile: {profile_path}")

    ensure_backend(args.api_base, args.no_start_servers)
    ensure_frontend(args.frontend_url, args.no_start_servers)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path.cwd() / f"{Path(profile_path).stem}.png"
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.suffix:
        out_path = out_path.with_suffix(".png")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit(
            "Playwright is required to render images. Install it with:\n"
            "  pip install playwright\n"
            "and make sure a Chromium browser is available "
            "(playwright install chromium or a system Chrome/Chromium)."
        )

    browser_exec = _pick_browser(args.browser)
    with sync_playwright() as p:
        launch_kwargs = {"headless": True}
        if browser_exec:
            launch_kwargs["executable_path"] = browser_exec
        browser = p.chromium.launch(**launch_kwargs)
        try:
            page = browser.new_page(viewport={"width": args.width, "height": args.height})
            page.goto(f"{args.frontend_url}/#", wait_until="domcontentloaded")

            # Wait until the profile dropdown is populated, then select our
            # profile by its exact path (same code path as picking it in the
            # dropdown, which re-renders the graph).
            page.wait_for_selector(
                "#profile-select option:not([value=''])",
                state="attached",
                timeout=args.timeout * 1000,
            )

            # Selecting a profile triggers a fresh graph load (the page may have
            # already auto-loaded a different profile). Wait deterministically for
            # the re-render that our selection triggers by watching the render
            # counter instead of transient status text.
            render_count_before = page.evaluate("() => window.__graphRenderCount || 0")
            page.select_option("#profile-select", value=profile_path)
            try:
                page.wait_for_function(
                    f"(window.__graphRenderCount || 0) > {render_count_before}",
                    timeout=args.timeout * 1000,
                )
            except Exception:
                status_text = (page.text_content("#status") or "").strip()
                raise SystemExit(
                    f"Graph did not render for profile {Path(profile_path).name}. "
                    f"Page status: {status_text!r}. "
                    "Open the app manually and check the browser console for details."
                )
            page.wait_for_timeout(300)

            _apply_options(page, args)
            _screenshot_graph(page, out_path, args.with_overlay, args)
        finally:
            browser.close()

    graph = _fetch_graph_summary(args.api_base, profile_path, args)
    _print_summary(out_path, profile_path, graph, args.json)
    log("Done.")


if __name__ == "__main__":
    main()
