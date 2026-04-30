"""Tests for the Sky-Lynx failure-pattern aggregator."""

from __future__ import annotations

import json
import sqlite3

import pytest

from sky_lynx.pattern_aggregator import aggregate_patterns


def _event(
    event_type: str,
    source_repo: str = "metroplex",
    correlation_id: str | None = None,
    timestamp: str = "2026-04-19T12:00:00+00:00",
    details: dict | None = None,
) -> dict:
    return {
        "event_type": event_type,
        "source_repo": source_repo,
        "correlation_id": correlation_id,
        "timestamp": timestamp,
        "source": source_repo,
        "details": details or {},
    }


def _read_rows(db_path) -> list[dict]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM failure_patterns ORDER BY pattern_type, source_repo"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def test_happy_path_three_failures_two_correlation_ids(tmp_path):
    db = tmp_path / "patterns.db"
    events = [
        _event("build_failed", correlation_id="corr-a", timestamp="2026-04-19T10:00:00+00:00"),
        _event("build_failed", correlation_id="corr-b", timestamp="2026-04-19T11:00:00+00:00"),
        _event("build_failed", correlation_id="corr-a", timestamp="2026-04-19T12:00:00+00:00"),
    ]

    touched = aggregate_patterns(events, db_path=db)
    assert touched == 1

    rows = _read_rows(db)
    assert len(rows) == 1
    row = rows[0]
    assert row["pattern_type"] == "build_failed"
    assert row["source_repo"] == "metroplex"
    assert row["occurrences"] == 3
    assert row["first_seen"] == "2026-04-19T10:00:00+00:00"
    assert row["last_seen"] == "2026-04-19T12:00:00+00:00"

    cids = json.loads(row["correlation_ids"])
    assert set(cids) == {"corr-a", "corr-b"}
    # Confidence heuristic: min(1.0, occurrences / 10.0)
    assert row["confidence"] == pytest.approx(0.3)


def test_non_failure_events_are_ignored(tmp_path):
    db = tmp_path / "patterns.db"
    events = [
        _event("build_completed", correlation_id="corr-ok-1"),
        _event("agent_completed", correlation_id="corr-ok-2"),
        _event("message_routed"),
    ]

    touched = aggregate_patterns(events, db_path=db)
    assert touched == 0

    # No failures means no DB write at all — schema never materializes.
    assert not db.exists()


def test_none_correlation_id_not_stored(tmp_path):
    db = tmp_path / "patterns.db"
    events = [
        _event("build_failed", correlation_id=None),
        _event("build_failed", correlation_id="corr-real"),
        _event("build_failed", correlation_id=None),
    ]

    aggregate_patterns(events, db_path=db)
    rows = _read_rows(db)
    assert len(rows) == 1
    cids = json.loads(rows[0]["correlation_ids"])
    assert cids == ["corr-real"]
    assert None not in cids
    assert "null" not in cids  # JSON null should never land here


def test_idempotent_second_run_bumps_occurrences(tmp_path):
    db = tmp_path / "patterns.db"
    events = [
        _event("build_failed", correlation_id="corr-a"),
        _event("build_failed", correlation_id="corr-b"),
    ]

    aggregate_patterns(events, db_path=db)
    first_rows = _read_rows(db)
    assert first_rows[0]["occurrences"] == 2

    # Running over the same batch again should bump occurrences by the
    # expected amount (2), not create a duplicate row.
    aggregate_patterns(events, db_path=db)
    second_rows = _read_rows(db)
    assert len(second_rows) == 1
    assert second_rows[0]["occurrences"] == 4

    # correlation_ids are deduped — still just two.
    cids = json.loads(second_rows[0]["correlation_ids"])
    assert set(cids) == {"corr-a", "corr-b"}


def test_source_repo_falls_back_to_legacy_source_field(tmp_path):
    db = tmp_path / "patterns.db"
    # No source_repo key, only legacy 'source'.
    event = {
        "event_type": "a2a_dispatch_failed",
        "correlation_id": "cid-1",
        "timestamp": "2026-04-19T09:00:00+00:00",
        "source": "command-center",
        "details": {},
    }
    aggregate_patterns([event], db_path=db)
    rows = _read_rows(db)
    assert len(rows) == 1
    assert rows[0]["source_repo"] == "command-center"


def test_different_source_repos_keep_separate_rows(tmp_path):
    db = tmp_path / "patterns.db"
    events = [
        _event("build_failed", source_repo="metroplex", correlation_id="m-1"),
        _event("build_failed", source_repo="command-center", correlation_id="c-1"),
        _event("build_failed", source_repo="command-center", correlation_id="c-2"),
    ]
    touched = aggregate_patterns(events, db_path=db)
    assert touched == 2

    rows = _read_rows(db)
    by_repo = {r["source_repo"]: r for r in rows}
    assert by_repo["metroplex"]["occurrences"] == 1
    assert by_repo["command-center"]["occurrences"] == 2


def test_confidence_caps_at_one(tmp_path):
    db = tmp_path / "patterns.db"
    events = [_event("build_failed", correlation_id=f"c-{i}") for i in range(15)]
    aggregate_patterns(events, db_path=db)
    rows = _read_rows(db)
    assert rows[0]["occurrences"] == 15
    assert rows[0]["confidence"] == 1.0


def test_rollback_and_timeout_markers_are_captured(tmp_path):
    db = tmp_path / "patterns.db"
    events = [
        _event("ego_rollback", correlation_id="r-1"),
        _event("agent_timeout", correlation_id="t-1"),
    ]
    touched = aggregate_patterns(events, db_path=db)
    assert touched == 2
    types = {r["pattern_type"] for r in _read_rows(db)}
    assert types == {"ego_rollback", "agent_timeout"}


def test_correlation_ids_cap_at_fifty(tmp_path):
    db = tmp_path / "patterns.db"
    events = [_event("build_failed", correlation_id=f"c-{i}") for i in range(60)]
    aggregate_patterns(events, db_path=db)
    rows = _read_rows(db)
    cids = json.loads(rows[0]["correlation_ids"])
    assert len(cids) == 50
    # FIFO eviction keeps the newest entries.
    assert "c-59" in cids
    assert "c-0" not in cids


def test_empty_events_no_touch(tmp_path):
    db = tmp_path / "patterns.db"
    touched = aggregate_patterns([], db_path=db)
    assert touched == 0
    # DB file should not be created when there's nothing to write.
    assert not db.exists()
