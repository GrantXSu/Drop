"""Daily scan quotas and Stripe-backed device entitlements."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple


FREE_DAILY_LIMIT = 3
PRO_MONTHLY_PRICE = "$9.99"
PRO_ANNUAL_PRICE = "$59.99"
COOKIE_NAME = "cardlens_device"
DEFAULT_BILLING_PATH = Path("data/grading/billing.sqlite")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            subscription_status TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS usage (
            device_id TEXT NOT NULL,
            usage_day TEXT NOT NULL,
            scan_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(device_id, usage_day)
        );
        CREATE TABLE IF NOT EXISTS scan_events (
            device_id TEXT NOT NULL,
            usage_day TEXT NOT NULL,
            scan_id TEXT NOT NULL,
            PRIMARY KEY(device_id, usage_day, scan_id)
        );
        CREATE TABLE IF NOT EXISTS grade_history (
            device_id TEXT NOT NULL,
            scan_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            card_id TEXT,
            card_name TEXT,
            set_name TEXT,
            card_number TEXT,
            image_url TEXT,
            grade REAL NOT NULL,
            grade_label TEXT NOT NULL,
            categories_json TEXT NOT NULL,
            PRIMARY KEY(device_id, scan_id)
        );
        CREATE INDEX IF NOT EXISTS devices_customer_idx
            ON devices(stripe_customer_id);
        """
    )
    return connection


def billing_path() -> Path:
    return Path(os.getenv("CARDLENS_BILLING_DB", str(DEFAULT_BILLING_PATH)))


def _secret() -> bytes:
    return os.getenv(
        "CARDLENS_COOKIE_SECRET", "local-cardlens-change-before-deployment"
    ).encode("utf-8")


def _signature(device_id: str) -> str:
    return hmac.new(_secret(), device_id.encode("utf-8"), hashlib.sha256).hexdigest()


def device_token(device_id: str) -> str:
    return f"{device_id}.{_signature(device_id)}"


def resolve_device(token: Optional[str]) -> Tuple[str, str]:
    if token and "." in token:
        device_id, signature = token.rsplit(".", 1)
        if hmac.compare_digest(signature, _signature(device_id)):
            return device_id, token
    device_id = secrets.token_urlsafe(24)
    return device_id, device_token(device_id)


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _is_pro(row: Optional[sqlite3.Row]) -> bool:
    if os.getenv("CARDLENS_DEV_PRO") == "1":
        return True
    return bool(row and row["subscription_status"] in {"active", "trialing"})


def usage_status(device_id: str, path: Optional[Path] = None) -> Dict[str, object]:
    connection = _connect(path or billing_path())
    device = connection.execute(
        "SELECT * FROM devices WHERE device_id = ?", (device_id,)
    ).fetchone()
    usage = connection.execute(
        "SELECT scan_count FROM usage WHERE device_id = ? AND usage_day = ?",
        (device_id, _today()),
    ).fetchone()
    connection.close()
    used = int(usage["scan_count"]) if usage else 0
    pro = _is_pro(device)
    return {
        "plan": "pro" if pro else "free",
        "is_pro": pro,
        "used_today": used,
        "daily_limit": None if pro else FREE_DAILY_LIMIT,
        "remaining_today": None if pro else max(0, FREE_DAILY_LIMIT - used),
        "can_scan": pro or used < FREE_DAILY_LIMIT,
        "monthly_price": PRO_MONTHLY_PRICE,
        "annual_price": PRO_ANNUAL_PRICE,
        "billing_configured": bool(
            os.getenv("STRIPE_SECRET_KEY")
            and os.getenv("STRIPE_PRO_MONTHLY_PRICE_ID")
            and os.getenv("STRIPE_PRO_ANNUAL_PRICE_ID")
            and os.getenv("CARDLENS_COOKIE_SECRET")
        ),
    }


def consume_scan(
    device_id: str,
    scan_id: Optional[str] = None,
    path: Optional[Path] = None,
) -> Dict[str, object]:
    current = usage_status(device_id, path)
    if current["is_pro"]:
        return current
    if not current["can_scan"]:
        return current
    connection = _connect(path or billing_path())
    if scan_id:
        inserted = connection.execute(
            """
            INSERT OR IGNORE INTO scan_events(device_id, usage_day, scan_id)
            VALUES (?, ?, ?)
            """,
            (device_id, _today(), scan_id),
        ).rowcount
        if not inserted:
            connection.close()
            return current
    connection.execute(
        """
        INSERT INTO usage(device_id, usage_day, scan_count)
        VALUES (?, ?, 1)
        ON CONFLICT(device_id, usage_day)
        DO UPDATE SET scan_count = scan_count + 1
        """,
        (device_id, _today()),
    )
    connection.commit()
    connection.close()
    return usage_status(device_id, path)


def scan_already_counted(
    device_id: str,
    scan_id: Optional[str],
    path: Optional[Path] = None,
) -> bool:
    if not scan_id:
        return False
    connection = _connect(path or billing_path())
    row = connection.execute(
        """
        SELECT 1 FROM scan_events
        WHERE device_id = ? AND usage_day = ? AND scan_id = ?
        """,
        (device_id, _today(), scan_id),
    ).fetchone()
    connection.close()
    return row is not None


def set_subscription(
    *,
    status: str,
    device_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    subscription_id: Optional[str] = None,
    path: Optional[Path] = None,
) -> None:
    connection = _connect(path or billing_path())
    now = datetime.now(timezone.utc).isoformat()
    if device_id:
        connection.execute(
            """
            INSERT INTO devices(
                device_id, stripe_customer_id, stripe_subscription_id,
                subscription_status, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                stripe_customer_id = COALESCE(excluded.stripe_customer_id, stripe_customer_id),
                stripe_subscription_id = COALESCE(excluded.stripe_subscription_id, stripe_subscription_id),
                subscription_status = excluded.subscription_status,
                updated_at = excluded.updated_at
            """,
            (device_id, customer_id, subscription_id, status, now),
        )
    elif customer_id:
        connection.execute(
            """
            UPDATE devices SET
                stripe_subscription_id = COALESCE(?, stripe_subscription_id),
                subscription_status = ?,
                updated_at = ?
            WHERE stripe_customer_id = ?
            """,
            (subscription_id, status, now, customer_id),
        )
    connection.commit()
    connection.close()


def customer_for_device(
    device_id: str, path: Optional[Path] = None
) -> Optional[str]:
    connection = _connect(path or billing_path())
    row = connection.execute(
        "SELECT stripe_customer_id FROM devices WHERE device_id = ?", (device_id,)
    ).fetchone()
    connection.close()
    return str(row["stripe_customer_id"]) if row and row["stripe_customer_id"] else None


def record_grade(
    *,
    device_id: str,
    scan_id: str,
    grade: float,
    grade_label: str,
    categories: object,
    card: Optional[Dict[str, object]] = None,
    path: Optional[Path] = None,
) -> None:
    connection = _connect(path or billing_path())
    card = card or {}
    connection.execute(
        """
        INSERT INTO grade_history(
            device_id, scan_id, created_at, card_id, card_name, set_name,
            card_number, image_url, grade, grade_label, categories_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(device_id, scan_id) DO UPDATE SET
            created_at = excluded.created_at,
            card_id = excluded.card_id,
            card_name = excluded.card_name,
            set_name = excluded.set_name,
            card_number = excluded.card_number,
            image_url = excluded.image_url,
            grade = excluded.grade,
            grade_label = excluded.grade_label,
            categories_json = excluded.categories_json
        """,
        (
            device_id,
            scan_id,
            datetime.now(timezone.utc).isoformat(),
            card.get("id"),
            card.get("name"),
            card.get("set_name"),
            card.get("number"),
            card.get("image_url"),
            float(grade),
            grade_label,
            json.dumps(categories, separators=(",", ":")),
        ),
    )
    connection.commit()
    connection.close()


def grade_history(
    device_id: str, path: Optional[Path] = None, limit: int = 100
) -> list:
    connection = _connect(path or billing_path())
    rows = connection.execute(
        """
        SELECT scan_id, created_at, card_id, card_name, set_name, card_number,
               image_url, grade, grade_label, categories_json
        FROM grade_history
        WHERE device_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (device_id, max(1, min(limit, 500))),
    ).fetchall()
    connection.close()
    return [
        {
            "scan_id": row["scan_id"],
            "created_at": row["created_at"],
            "card_id": row["card_id"],
            "name": row["card_name"] or "Unidentified card",
            "set_name": row["set_name"],
            "number": row["card_number"],
            "image_url": row["image_url"],
            "grade": row["grade"],
            "label": row["grade_label"],
            "categories": json.loads(row["categories_json"]),
        }
        for row in rows
    ]


def clear_grade_history(device_id: str, path: Optional[Path] = None) -> None:
    connection = _connect(path or billing_path())
    connection.execute("DELETE FROM grade_history WHERE device_id = ?", (device_id,))
    connection.commit()
    connection.close()
