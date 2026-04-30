"""Failure pattern aggregator for Sky-Lynx.

Consumes the shared event stream (Metroplex, CMD, etc. — JSON files in
``~/.local/share/skylynx-events/``) and upserts recurring failure modes into
a SQLite ``failure_patterns`` table. Future work can mine this table for
cross-repo pattern trends without having to re-read every event.

Schema (auto-created on first call)::

    CREATE TABLE IF NOT EXISTS failure_patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        source_repo TEXT NOT NULL,
        first_seen TEXT NOT NULL,
        last_seen TEXT NOT NULL,
        occurrences INTEGER NOT NULL DEFAULT 1,
        confidence REAL NOT NULL DEFAULT 0.0,
        correlation_ids TEXT NOT NULL DEFAULT '[]',
        UNIQUE(pattern_type, source_repo)
    );
    CREATE INDEX IF NOT EXISTS idx_fp_last_seen ON failure_patterns(last_seen);

No new dependencies — stdlib ``sqlite3`` and ``json`` only.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# Keyword markers that flag an event as a failure worth aggregating.
_FAILURE_MARKERS: tuple[str, ...] = ("fail", "failed", "rollback", "timeout")

# Cap on how many correlation_ids we persist per pattern — bounded storage.
_MAX_CORRELATION_IDS = 50

# Default DB location follows the Sky-Lynx convention of ``data/*.db`` under
# the project root (see proposals.db).
DEFAULT_DB_PATH = Path.home() / "projects" / "sky-lynx" / "data" / "patterns.db"


def _get_db_path() -> Path:
    return Path(os.environ.get("SKYLYNX_PATTERNS_DB", str(DEFAULT_DB_PATH)))


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS failure_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            source_repo TEXT NOT NULL,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            occurrences INTEGER NOT NULL DEFAULT 1,
            confidence REAL NOT NULL DEFAULT 0.0,
            correlation_ids TEXT NOT NULL DEFAULT '[]',
            UNIQUE(pattern_type, source_repo)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fp_last_seen ON failure_patterns(last_seen)"
    )
    conn.commit()


def _is_failure_event(event_type: str | None) -> bool:
    if not event_type:
        return False
    et = event_type.lower()
    return any(marker in et for marker in _FAILURE_MARKERS)


def _source_repo(event: dict) -> str:
    """Pick ``source_repo``, falling back to the legacy ``source`` alias."""
    return event.get("source_repo") or event.get("source") or "unknown"


def _confidence(occurrences: int) -> float:
    """Boring heuristic: scale linearly to 1.0 at 10 occurrences."""
    return min(1.0, occurrences / 10.0)


def _merge_correlation_ids(existing_json: str, new_id: str | None) -> str:
    """Append ``new_id`` to the JSON array if not already present.

    - ``None`` correlation_ids are dropped silently.
    - Dedupes by identity (order preserved).
    - Caps at ``_MAX_CORRELATION_IDS`` entries (FIFO — oldest dropped first).
    """
    try:
        ids = json.loads(existing_json) if existing_json else []
        if not isinstance(ids, list):
            ids = []
    except (ValueError, TypeError):
        ids = []

    if new_id is None:
        return json.dumps(ids)

    if new_id not in ids:
        ids.append(new_id)

    if len(ids) > _MAX_CORRELATION_IDS:
        ids = ids[-_MAX_CORRELATION_IDS:]

    return json.dumps(ids)


def aggregate_patterns(
    events: Iterable[dict],
    db_path: Path | None = None,
) -> int:
    """Upsert failure patterns from ``events`` into the SQLite store.

    Args:
        events: Iterable of event dicts following the shared schema:
            ``event_type``, ``source_repo`` (or legacy ``source``),
            ``correlation_id``, ``timestamp``, ``details``.
        db_path: Override DB path. Defaults to ``SKYLYNX_PATTERNS_DB`` env var
            or ``~/projects/sky-lynx/data/patterns.db``.

    Returns:
        Number of distinct ``(pattern_type, source_repo)`` rows touched.
    """
    db_path = db_path or _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Bucket failure events in-memory first so repeated events in the same
    # batch only produce one upsert per (pattern_type, source_repo).
    buckets: dict[tuple[str, str], dict] = {}
    for event in events:
        event_type = event.get("event_type")
        if not _is_failure_event(event_type):
            continue

        source_repo = _source_repo(event)
        # Sanity: event_type is non-None because _is_failure_event returned True.
        assert event_type is not None
        timestamp = event.get("timestamp") or ""
        correlation_id = event.get("correlation_id")

        key = (event_type, source_repo)
        bucket = buckets.setdefault(
            key,
            {
                "occurrences": 0,
                "timestamps": [],
                "correlation_ids": [],
            },
        )
        bucket["occurrences"] += 1
        if timestamp:
            bucket["timestamps"].append(timestamp)
        if correlation_id is not None:
            bucket["correlation_ids"].append(correlation_id)

    if not buckets:
        return 0

    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        _init_schema(conn)

        touched = 0
        for (pattern_type, source_repo), bucket in buckets.items():
            timestamps = sorted(bucket["timestamps"])
            batch_first = timestamps[0] if timestamps else ""
            batch_last = timestamps[-1] if timestamps else ""
            delta_occ = bucket["occurrences"]

            row = conn.execute(
                "SELECT id, first_seen, last_seen, occurrences, correlation_ids "
                "FROM failure_patterns WHERE pattern_type = ? AND source_repo = ?",
                (pattern_type, source_repo),
            ).fetchone()

            if row is None:
                merged_ids = "[]"
                for cid in bucket["correlation_ids"]:
                    merged_ids = _merge_correlation_ids(merged_ids, cid)

                conn.execute(
                    "INSERT INTO failure_patterns "
                    "(pattern_type, source_repo, first_seen, last_seen, "
                    " occurrences, confidence, correlation_ids) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        pattern_type,
                        source_repo,
                        batch_first,
                        batch_last,
                        delta_occ,
                        _confidence(delta_occ),
                        merged_ids,
                    ),
                )
            else:
                new_occ = row["occurrences"] + delta_occ
                new_first = (
                    min(row["first_seen"], batch_first)
                    if batch_first and row["first_seen"]
                    else (row["first_seen"] or batch_first)
                )
                new_last = (
                    max(row["last_seen"], batch_last)
                    if batch_last and row["last_seen"]
                    else (row["last_seen"] or batch_last)
                )

                merged_ids = row["correlation_ids"] or "[]"
                for cid in bucket["correlation_ids"]:
                    merged_ids = _merge_correlation_ids(merged_ids, cid)

                conn.execute(
                    "UPDATE failure_patterns SET "
                    "first_seen = ?, last_seen = ?, occurrences = ?, "
                    "confidence = ?, correlation_ids = ? "
                    "WHERE id = ?",
                    (
                        new_first,
                        new_last,
                        new_occ,
                        _confidence(new_occ),
                        merged_ids,
                        row["id"],
                    ),
                )

            touched += 1

        conn.commit()
        return touched
    finally:
        conn.close()
