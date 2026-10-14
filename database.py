"""PostgreSQL persistence for upload batches and their file-level audit trail."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


ROOT = Path(__file__).resolve().parent
LEGACY_HISTORY_FILE = ROOT / "ingestion_history.json"


@contextmanager
def _session() -> Iterator[psycopg.Connection]:
    """Open a transaction, create the schema if needed, then close the connection."""
    with psycopg.connect(
        host=os.getenv("PGHOST", "127.0.0.1"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "oil_forecast"),
        user=os.getenv("PGUSER", "oil_forecast"),
        password=os.getenv("PGPASSWORD", "change-me"),
        connect_timeout=15,
        row_factory=dict_row,
    ) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_batches (
                batch_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                total_files INTEGER NOT NULL DEFAULT 0,
                saved_count INTEGER NOT NULL DEFAULT 0,
                rejected_count INTEGER NOT NULL DEFAULT 0,
                no_new_data_count INTEGER NOT NULL DEFAULT 0,
                total_new_rows INTEGER NOT NULL DEFAULT 0,
                total_overwritten_rows INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_files (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES ingestion_batches(batch_id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK(status IN ('saved', 'rejected', 'no_new_data')),
                filename TEXT NOT NULL,
                saved_as TEXT,
                sha256 TEXT,
                new_rows INTEGER NOT NULL DEFAULT 0,
                overwritten_rows INTEGER NOT NULL DEFAULT 0,
                date_range TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT '',
                details_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_ingestion_files_batch ON ingestion_files(batch_id, id)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS app_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        _migrate_legacy_history(connection)
        yield connection


def _insert_batch(connection: psycopg.Connection, entry: dict[str, Any]) -> None:
    batch_id = str(entry.get("batch_id", "")).strip()
    if not batch_id:
        return
    inserted = connection.execute(
        """INSERT INTO ingestion_batches
        (batch_id, created_at, total_files, saved_count, rejected_count,
         no_new_data_count, total_new_rows, total_overwritten_rows)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (batch_id) DO NOTHING""",
        (
            batch_id,
            str(entry.get("created_at", "")),
            int(entry.get("total_files", 0)),
            int(entry.get("saved_count", 0)),
            int(entry.get("rejected_count", 0)),
            int(entry.get("no_new_data_count", 0)),
            int(entry.get("total_new_rows", 0)),
            int(entry.get("total_overwritten_rows", 0)),
        ),
    )
    if inserted.rowcount == 0:
        return
    for status, key in (
        ("saved", "saved_files"),
        ("rejected", "rejected_files"),
        ("no_new_data", "no_new_data_files"),
    ):
        for item in entry.get(key, []) or []:
            connection.execute(
                """INSERT INTO ingestion_files
                (batch_id, status, filename, saved_as, sha256, new_rows,
                 overwritten_rows, date_range, reason, details_json)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    batch_id,
                    status,
                    str(item.get("filename", "unknown")),
                    item.get("saved_as"),
                    item.get("sha256"),
                    int(item.get("new_rows", 0)),
                    int(item.get("overwritten_rows", 0)),
                    str(item.get("date_range", "")),
                    str(item.get("reason", "")),
                    Jsonb(item),
                ),
            )


def _migrate_legacy_history(connection: psycopg.Connection) -> None:
    if not LEGACY_HISTORY_FILE.exists():
        return
    connection.execute("SELECT pg_advisory_xact_lock(%s)", (4815162342,))
    migrated = connection.execute(
        "SELECT value FROM app_metadata WHERE key = 'ingestion_json_migrated'"
    ).fetchone()
    if migrated:
        return
    try:
        entries = json.loads(LEGACY_HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(entries, list):
        return
    for entry in reversed(entries):
        if isinstance(entry, dict):
            _insert_batch(connection, entry)
    connection.execute(
        """INSERT INTO app_metadata(key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
        ("ingestion_json_migrated", "1"),
    )


def record_ingestion(entry: dict[str, Any]) -> None:
    """Persist one upload batch and its file outcomes in a single transaction."""
    with _session() as connection:
        _insert_batch(connection, entry)


def list_ingestion_history(limit: int = 100) -> list[dict[str, Any]]:
    """Return upload history in the shape expected by the Streamlit UI."""
    limit = max(1, min(int(limit), 1000))
    with _session() as connection:
        batches = connection.execute(
            """SELECT * FROM ingestion_batches
            ORDER BY created_at DESC, batch_id DESC LIMIT %s""",
            (limit,),
        ).fetchall()
        result = []
        for batch in batches:
            files = connection.execute(
                "SELECT * FROM ingestion_files WHERE batch_id = %s ORDER BY id",
                (batch["batch_id"],),
            ).fetchall()
            entry = dict(batch)
            entry["saved_files"] = []
            entry["rejected_files"] = []
            entry["no_new_data_files"] = []
            for item in files:
                bucket = {
                    "saved": "saved_files",
                    "rejected": "rejected_files",
                    "no_new_data": "no_new_data_files",
                }[item["status"]]
                entry[bucket].append(item["details_json"])
            result.append(entry)
        return result
