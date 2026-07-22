from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, sync_playwright

LOG = logging.getLogger("queue_watcher")
DEFAULT_URL = "https://www.pokemoncenter.com/"


class QueueSignal(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class QueueConfig:
    url: str
    interval_seconds: int
    jitter_seconds: int
    profile_dir: Path
    state_file: Path
    discord_webhook_url: str | None
    headless: bool

    @classmethod
    def from_env(cls) -> "QueueConfig":
        return cls(
            url=os.getenv("QUEUE_URL", DEFAULT_URL),
            interval_seconds=max(
                45, int(os.getenv("QUEUE_CHECK_INTERVAL_SECONDS", "90"))
            ),
            jitter_seconds=max(
                0, int(os.getenv("QUEUE_CHECK_JITTER_SECONDS", "15"))
            ),
            profile_dir=Path(
                os.getenv("PLAYWRIGHT_PROFILE_DIR", "data/browser-profile")
            ),
            state_file=Path(os.getenv("QUEUE_STATE_FILE", "data/queue-state.json")),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or None,
            headless=os.getenv("PLAYWRIGHT_HEADLESS", "false").casefold()
            in ("1", "true", "yes"),
        )


def detect_queue_signal(url: str, title: str, body: str) -> QueueSignal:
    combined = f"{url}\n{title}\n{body}".casefold()
    host = urlparse(url).netloc.casefold()
    queue_markers = (
        "virtual queue",
        "you are now in line",
        "you are in line",
        "estimated wait time",
        "your estimated wait",
        "please keep this page open",
        "waiting room",
        "queue-it",
    )
    if (
        "queue-it" in host
        or host.startswith("queue.")
        or any(marker in combined for marker in queue_markers)
    ):
        return QueueSignal.ACTIVE

    blocked_markers = (
        "pardon our interruption",
        "incapsula incident id",
        "error 17",
        "access denied",
    )
    if any(marker in combined for marker in blocked_markers):
        return QueueSignal.BLOCKED
    return QueueSignal.INACTIVE


class QueueWatcher:
    def __init__(self, config: QueueConfig) -> None:
        self.config = config
        self.previous_signal = self._load_signal()

    def _load_signal(self) -> QueueSignal:
        try:
            value = json.loads(self.config.state_file.read_text()).get("signal")
            return QueueSignal(value)
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError):
            return QueueSignal.INACTIVE

    def _save_signal(self, signal: QueueSignal) -> None:
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps({"signal": signal.value}, indent=2))
        temporary.replace(self.config.state_file)

    def _notify(self, page_url: str) -> None:
        if not self.config.discord_webhook_url:
            LOG.warning("Queue detected, but DISCORD_WEBHOOK_URL is not configured")
            return
        payload = {
            "content": "🚨🚨 **POKÉMON CENTER VIRTUAL QUEUE DETECTED**",
            "embeds": [
                {
                    "title": "Open Pokémon Center now",
                    "description": (
                        "The virtual waiting room is active. This often accompanies "
                        "heavy traffic or a product drop, but it does not guarantee "
                        "that a new product is available.\n\n"
                        f"[Open the queue]({page_url})"
                    ),
                    "url": page_url,
                    "color": 0xFFCB05,
                }
            ],
        }
        try:
            response = httpx.post(
                self.config.discord_webhook_url, json=payload, timeout=20
            )
            response.raise_for_status()
            LOG.info("Discord queue alert sent")
        except httpx.HTTPError as exc:
            LOG.error("Discord queue alert failed: %s", exc)

    def _inspect(self, page: Page) -> QueueSignal:
        try:
            title = page.title()
        except PlaywrightError:
            title = ""
        try:
            body = page.locator("body").inner_text(timeout=5_000)
        except PlaywrightError:
            body = ""
        signal = detect_queue_signal(page.url, title, body)
        LOG.info("Queue signal: %s (%s)", signal.value, page.url)
        return signal

    def _record(self, signal: QueueSignal, page: Page) -> None:
        if signal == QueueSignal.ACTIVE and self.previous_signal != signal:
            page.bring_to_front()
            self._notify(page.url)
        if signal != self.previous_signal:
            self.previous_signal = signal
            self._save_signal(signal)

    def run(self, once: bool = False) -> None:
        self.config.profile_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            try:
                context = playwright.chromium.launch_persistent_context(
                    user_data_dir=str(self.config.profile_dir),
                    channel="chrome",
                    headless=self.config.headless,
                    no_viewport=True,
                )
            except PlaywrightError as exc:
                raise RuntimeError(
                    "Google Chrome could not be started. Install Chrome and try again."
                ) from exc

            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(
                    self.config.url,
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
            except PlaywrightError as exc:
                LOG.warning("Initial page load did not finish: %s", exc)

            try:
                while True:
                    signal = self._inspect(page)
                    self._record(signal, page)
                    if once:
                        return

                    if signal == QueueSignal.ACTIVE:
                        # Preserve the live queue session; Queue-it redirects it
                        # automatically when the visitor is admitted.
                        time.sleep(15)
                        continue

                    delay = self.config.interval_seconds + random.uniform(
                        0, self.config.jitter_seconds
                    )
                    LOG.info("Next queue check in %.0f seconds", delay)
                    time.sleep(delay)
                    try:
                        page.reload(wait_until="domcontentloaded", timeout=30_000)
                    except PlaywrightError as exc:
                        LOG.warning("Page reload did not finish: %s", exc)
            finally:
                context.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Local Pokémon Center virtual queue detector"
    )
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    parser.add_argument(
        "--test-notification",
        action="store_true",
        help="Send a Discord queue-style test alert and exit",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    # httpx logs complete request URLs, which would expose a Discord webhook token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    watcher = QueueWatcher(QueueConfig.from_env())
    if args.test_notification:
        watcher._notify(DEFAULT_URL)
        return
    watcher.run(once=args.once)


if __name__ == "__main__":
    main()
