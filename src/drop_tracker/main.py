from __future__ import annotations

import argparse
import html
import json
import logging
import os
import random
import re
import smtplib
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
from enum import Enum
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

LOG = logging.getLogger("drop_tracker")
DEFAULT_DISCOVERY_URL = "https://www.pokemoncenter.com/category/trading-card-game"
DEFAULT_DISCOVERY_URLS = (
    "https://www.pokemoncenter.com/category/new-releases",
    DEFAULT_DISCOVERY_URL,
    "https://www.pokemoncenter.com/category/plush",
    "https://www.pokemoncenter.com/category/figures-and-pins",
)


class Availability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProductListing:
    url: str
    title: str
    availability: Availability
    is_tcg: bool


@dataclass(frozen=True)
class Config:
    target_urls: tuple[str, ...]
    discovery_urls: tuple[str, ...]
    match_terms: tuple[str, ...]
    product_terms: tuple[str, ...]
    track_all_products: bool
    interval_seconds: int
    jitter_seconds: int
    request_timeout: float
    state_file: Path
    discord_webhook_url: str | None
    smtp_host: str | None
    smtp_port: int
    smtp_username: str | None
    smtp_password: str | None
    email_from: str | None
    email_to: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Config":
        target_urls = _split_values(os.getenv("TARGET_URLS", ""))
        discovery_urls = _split_values(
            os.getenv("DISCOVERY_URLS", ",".join(DEFAULT_DISCOVERY_URLS))
        )
        match_terms = tuple(
            term.casefold()
            for term in _split_values(
                os.getenv("MATCH_TERMS", "30th celebration")
            )
        )
        # Migrate the original single-product default without breaking existing .env files.
        if match_terms == ("30th celebration", "elite trainer box"):
            match_terms = ("30th celebration",)
        product_terms = tuple(
            term.casefold()
            for term in _split_values(
                os.getenv(
                    "PRODUCT_TERMS",
                    "elite trainer box,booster bundle,3 pack,three pack,"
                    "sylveon ex,greninja ex,poster collection,binder collection,"
                    "mew figure collection,mewtwo figure collection,"
                    "ditto premium collection",
                )
            )
        )
        interval = max(60, int(os.getenv("CHECK_INTERVAL_SECONDS", "300")))
        jitter = max(0, int(os.getenv("CHECK_JITTER_SECONDS", "30")))
        email_to = _split_values(os.getenv("EMAIL_TO", ""))
        return cls(
            target_urls=target_urls,
            discovery_urls=discovery_urls,
            match_terms=match_terms,
            product_terms=product_terms,
            track_all_products=os.getenv("TRACK_ALL_PRODUCTS", "true").casefold()
            in ("1", "true", "yes"),
            interval_seconds=interval,
            jitter_seconds=jitter,
            request_timeout=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "20")),
            state_file=Path(os.getenv("STATE_FILE", "data/state.json")),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or None,
            smtp_host=os.getenv("SMTP_HOST") or None,
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME") or None,
            smtp_password=os.getenv("SMTP_PASSWORD") or None,
            email_from=os.getenv("EMAIL_FROM") or None,
            email_to=email_to,
        )


def _split_values(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.replace("\n", ",").split(",") if part.strip())


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def detect_availability(page: str) -> Availability:
    soup = BeautifulSoup(page, "html.parser")

    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for item in _walk_json(data):
            availability = str(item.get("availability", "")).casefold()
            if availability.endswith("instock") or availability.endswith("preorder"):
                return Availability.AVAILABLE
            if availability.endswith("outofstock") or availability.endswith("soldout"):
                return Availability.UNAVAILABLE

    for element in soup.find_all(["button", "input"]):
        label = " ".join(
            filter(
                None,
                [
                    element.get_text(" ", strip=True),
                    element.get("value"),
                    element.get("aria-label"),
                ],
            )
        ).casefold()
        disabled = element.has_attr("disabled") or element.get("aria-disabled") == "true"
        if ("add to cart" in label or "preorder" in label) and not disabled:
            return Availability.AVAILABLE

    text = " ".join(soup.stripped_strings).casefold()
    unavailable_markers = (
        "out of stock",
        "sold out",
        "currently unavailable",
        "not available for purchase",
    )
    if any(marker in text for marker in unavailable_markers):
        return Availability.UNAVAILABLE
    return Availability.UNKNOWN


def _looks_like_tcg(value: str) -> bool:
    normalized = _normalize(value)
    return any(
        term in normalized
        for term in (
            "pokemon tcg",
            "trading card game",
            "booster",
            "elite trainer box",
            "battle deck",
            "trainer toolkit",
        )
    )


def _clean_product_title(value: str, url: str) -> str:
    title = re.sub(
        r"\b(sold out|out of stock|preorder)\b", "", value, flags=re.IGNORECASE
    )
    title = re.sub(r"\$\d+(?:\.\d{2})?", "", title)
    title = " ".join(title.split()).strip(" -|")
    if title:
        return title
    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").title()


def discover_product_listings(page: str, base_url: str) -> dict[str, ProductListing]:
    soup = BeautifulSoup(page, "html.parser")
    base_host = urlparse(base_url).netloc
    collected: dict[str, dict[str, list[str]]] = {}

    for link in soup.find_all("a", href=True):
        href = urljoin(base_url, link["href"]).split("#", 1)[0]
        parsed = urlparse(href)
        if parsed.netloc != base_host or "/product/" not in parsed.path:
            continue
        entry = collected.setdefault(href, {"text": [], "titles": []})
        text = link.get_text(" ", strip=True)
        if text:
            entry["text"].append(text)
            entry["titles"].append(text)
        aria_label = link.get("aria-label")
        if aria_label:
            entry["titles"].append(aria_label)
        for image in link.find_all("img", alt=True):
            if image["alt"]:
                entry["titles"].append(image["alt"])

    listings: dict[str, ProductListing] = {}
    for url, values in collected.items():
        combined = " ".join(values["text"])
        normalized = _normalize(combined)
        availability = (
            Availability.UNAVAILABLE
            if "sold out" in normalized or "out of stock" in normalized
            else Availability.AVAILABLE
        )
        candidates = [
            _clean_product_title(candidate, url)
            for candidate in values["titles"]
            if candidate
        ]
        title = max(candidates, key=len, default=_clean_product_title("", url))
        listings[url] = ProductListing(
            url=url,
            title=title,
            availability=availability,
            is_tcg="/trading-card-game" in urlparse(base_url).path
            or _looks_like_tcg(f"{title} {url}"),
        )
    return listings


def discover_product_urls(
    page: str,
    base_url: str,
    match_terms: tuple[str, ...],
    product_terms: tuple[str, ...] = (),
) -> set[str]:
    matches: set[str] = set()
    for listing in discover_product_listings(page, base_url).values():
        searchable = _normalize(f"{listing.title} {urlparse(listing.url).path}")
        matches_collection = all(
            _normalize(term) in searchable for term in match_terms
        )
        matches_product = not product_terms or any(
            _normalize(term) in searchable for term in product_terms
        )
        if matches_collection and matches_product:
            matches.add(listing.url)
    return matches


class Tracker:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = httpx.Client(
            timeout=config.request_timeout,
            follow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; DropTracker/0.1; "
                    "+personal stock notifications)"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        self.state = self._load_state()

    def close(self) -> None:
        self.client.close()

    def _load_state(self) -> dict[str, str]:
        try:
            data = json.loads(self.config.state_file.read_text())
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_state(self) -> None:
        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, sort_keys=True))
        temporary.replace(self.config.state_file)

    def _fetch(self, url: str) -> str | None:
        try:
            response = self.client.get(url)
            if response.status_code in (403, 429):
                LOG.warning("Request blocked or rate-limited (%s): %s", response.status_code, url)
                return None
            response.raise_for_status()
            challenge_markers = (
                "incapsula incident id",
                "pardon our interruption",
                "distil_referrer",
            )
            if any(
                marker in response.text.casefold() for marker in challenge_markers
            ):
                LOG.warning("Pokémon Center returned an anti-bot challenge: %s", url)
                return None
            return response.text
        except httpx.HTTPError as exc:
            LOG.warning("Could not check %s: %s", url, exc)
            return None

    def _matches_focus(self, listing: ProductListing) -> bool:
        searchable = _normalize(f"{listing.title} {urlparse(listing.url).path}")
        return all(
            _normalize(term) in searchable for term in self.config.match_terms
        ) and (
            not self.config.product_terms
            or any(
                _normalize(term) in searchable for term in self.config.product_terms
            )
        )

    def _discover_listings(self) -> dict[str, ProductListing]:
        listings: dict[str, ProductListing] = {}
        for discovery_url in self.config.discovery_urls:
            page = self._fetch(discovery_url)
            if page is None:
                continue
            discovered = discover_product_listings(page, discovery_url)
            if not self.config.track_all_products:
                discovered = {
                    url: listing
                    for url, listing in discovered.items()
                    if self._matches_focus(listing)
                }
            if discovered:
                LOG.info(
                    "Found %d product listing(s) on %s",
                    len(discovered),
                    discovery_url,
                )
            for url, listing in discovered.items():
                existing = listings.get(url)
                if existing is None:
                    listings[url] = listing
                    continue
                listings[url] = ProductListing(
                    url=url,
                    title=max(existing.title, listing.title, key=len),
                    availability=(
                        Availability.AVAILABLE
                        if Availability.AVAILABLE
                        in (existing.availability, listing.availability)
                        else Availability.UNAVAILABLE
                    ),
                    is_tcg=existing.is_tcg or listing.is_tcg,
                )

        for url in self.config.target_urls:
            if url in listings:
                continue
            page = self._fetch(url)
            if page is None:
                continue
            soup = BeautifulSoup(page, "html.parser")
            heading = soup.find("h1")
            title = _clean_product_title(
                heading.get_text(" ", strip=True) if heading else "", url
            )
            listings[url] = ProductListing(
                url=url,
                title=title,
                availability=detect_availability(page),
                is_tcg=_looks_like_tcg(f"{title} {url}"),
            )
        return listings

    def check_once(self) -> None:
        listings = self._discover_listings()
        if not listings:
            LOG.info("No product listings discovered; leaving saved state unchanged")
            return

        initialized = self.state.get("__catalog_initialized__") == "true"
        changed = False
        for url, listing in sorted(listings.items()):
            status = listing.availability
            previous = self.state.get(url)
            LOG.info("%s: %s (%s)", status, listing.title, url)
            if status == Availability.UNKNOWN:
                continue
            self.state[url] = status.value
            changed = changed or previous != status.value
            if not initialized or status != Availability.AVAILABLE:
                continue
            event = None
            if previous is None:
                event = "NEW DROP"
            elif previous == Availability.UNAVAILABLE.value:
                event = "RESTOCK"
            if event:
                priority = "🔥 TCG PRIORITY" if listing.is_tcg else "NEW MERCH"
                self.notify(
                    f"{priority}: {event}",
                    f"**{listing.title}**\n\n"
                    f"Status: **{event}**\n\n"
                    f"[Open this product on Pokémon Center]({url})\n\n"
                    "Open the official page and complete checkout manually.",
                    url,
                    is_tcg=listing.is_tcg,
                )
        if not initialized:
            self.state["__catalog_initialized__"] = "true"
            changed = True
            LOG.info(
                "Saved initial catalog baseline; future new products and restocks "
                "will trigger alerts"
            )
        if changed:
            self._save_state()

    def notify(
        self,
        subject: str,
        message: str,
        url: str | None = None,
        is_tcg: bool = False,
    ) -> None:
        sent = False
        if self.config.discord_webhook_url:
            payload = {
                "content": (
                    "🔥🔥 **TCG DROP — PRIORITY ALERT**"
                    if is_tcg
                    else "🚨 **Pokémon Center merchandise alert**"
                ),
                "embeds": [
                    {
                        "title": subject,
                        "description": message,
                        "url": url,
                        "color": 0xFFCB05 if is_tcg else 0xE3350D,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ],
            }
            try:
                response = self.client.post(
                    self.config.discord_webhook_url, json=payload
                )
                response.raise_for_status()
                sent = True
                LOG.info("Discord notification sent")
            except httpx.HTTPError as exc:
                LOG.error("Discord notification failed: %s", exc)

        if self.config.smtp_host and self.config.email_from and self.config.email_to:
            email = EmailMessage()
            email["Subject"] = subject
            email["From"] = self.config.email_from
            email["To"] = ", ".join(self.config.email_to)
            email.set_content(message)
            email.add_alternative(
                f"<p>{html.escape(message).replace(chr(10), '<br>')}</p>",
                subtype="html",
            )
            try:
                context = ssl.create_default_context()
                with smtplib.SMTP(
                    self.config.smtp_host, self.config.smtp_port, timeout=20
                ) as smtp:
                    smtp.starttls(context=context)
                    if self.config.smtp_username and self.config.smtp_password:
                        smtp.login(
                            self.config.smtp_username, self.config.smtp_password
                        )
                    smtp.send_message(email)
                sent = True
                LOG.info("Email notification sent")
            except (OSError, smtplib.SMTPException) as exc:
                LOG.error("Email notification failed: %s", exc)

        if not sent:
            LOG.warning("No notification was sent; configure Discord and/or email")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pokémon Center stock notifier")
    parser.add_argument("--once", action="store_true", help="Run one check and exit")
    parser.add_argument(
        "--test-notification", action="store_true", help="Send a test alert and exit"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()
    config = Config.from_env()
    tracker = Tracker(config)
    try:
        if args.test_notification:
            tracker.notify(
                "Drop Tracker test",
                "Your Drop Tracker notification is configured.\n\n"
                f"[Open Pokémon Center]({DEFAULT_DISCOVERY_URL})",
                DEFAULT_DISCOVERY_URL,
            )
            return
        while True:
            tracker.check_once()
            if args.once:
                return
            delay = config.interval_seconds + random.uniform(
                0, config.jitter_seconds
            )
            LOG.info("Next check in %.0f seconds", delay)
            time.sleep(delay)
    except KeyboardInterrupt:
        LOG.info("Stopped")
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
