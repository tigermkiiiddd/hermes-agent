"""Tests for the todo tool module."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from tools.todo_tool import (
    TodoStore,
    _normalize_priority,
    _normalize_severity,
    todo_tool,
)
from hermes_state import SessionDB


# ── Normalize helpers ────────────────────────────────────────────────────


class TestNormalizePriority:
    def test_valid_labels(self):
        for label in ("P0", "P1", "P2", "P3"):
            assert _normalize_priority(label) == label

    def test_lowercase(self):
        assert _normalize_priority("p0") == "P0"
        assert _normalize_priority("p3") == "P3"

    def test_bare_number_string(self):
        assert _normalize_priority("0") == "P0"
        assert _normalize_priority("1") == "P1"
        assert _normalize_priority("2") == "P2"
        assert _normalize_priority("3") == "P3"

    def test_int_zero_is_falsy_default(self):
        """int 0 is falsy, so it defaults to P2 (realistic: agents send strings)."""
        assert _normalize_priority(0) == "P2"

    def test_none_defaults_to_p2(self):
        assert _normalize_priority(None) == "P2"

    def test_empty_defaults_to_p2(self):
        assert _normalize_priority("") == "P2"

    def test_invalid_defaults_to_p2(self):
        assert _normalize_priority("P5") == "P2"
        assert _normalize_priority("high") == "P2"


class TestNormalizeSeverity:
    def test_valid_labels(self):
        for label in ("S1", "S2", "S3"):
            assert _normalize_severity(label) == label

    def test_lowercase(self):
        assert _normalize_severity("s1") == "S1"

    def test_bare_number(self):
        assert _normalize_severity(1) == "S1"
        assert _normalize_severity("2") == "S2"
        assert _normalize_severity("3") == "S3"

    def test_none_defaults_to_s2(self):
        assert _normalize_severity(None) == "S2"

    def test_invalid_defaults_to_s2(self):
        assert _normalize_severity("S5") == "S2"


# ── Write / Read ─────────────────────────────────────────────────────────


class TestWriteAndRead:
    def test_write_replaces_list(self):
        store = TodoStore()
        items = [
            {"id": "1", "content": "First task", "status": "pending"},
            {"id": "2", "content": "Second task", "status": "in_progress"},
        ]
        result = store.write(items)
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["status"] == "in_progress"

    def test_read_returns_copy(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        items = store.read()
        items[0]["content"] = "MUTATED"
        assert store.read()[0]["content"] == "Task"

    def test_write_deduplicates_duplicate_ids(self):
        store = TodoStore()
        result = store.write([
            {"id": "1", "content": "First version", "status": "pending"},
            {"id": "2", "content": "Other task", "status": "pending"},
            {"id": "1", "content": "Latest version", "status": "in_progress"},
        ])
        assert len(result) == 2
        assert result[0]["id"] == "2"
        assert result[1]["id"] == "1"
        assert result[1]["content"] == "Latest version"
        assert result[1]["status"] == "in_progress"

    def test_write_adds_default_priority_severity(self):
        store = TodoStore()
        result = store.write([{"id": "1", "content": "Task", "status": "pending"}])
        assert result[0]["priority"] == "P2"
        assert result[0]["severity"] == "S2"

    def test_write_preserves_explicit_priority_severity(self):
        store = TodoStore()
        result = store.write([
            {"id": "1", "content": "Urgent", "status": "pending", "priority": "P0", "severity": "S1"},
        ])
        assert result[0]["priority"] == "P0"
        assert result[0]["severity"] == "S1"

    def test_write_normalizes_priority_severity(self):
        store = TodoStore()
        result = store.write([
            {"id": "1", "content": "Task", "status": "pending", "priority": "p1", "severity": "s3"},
        ])
        assert result[0]["priority"] == "P1"
        assert result[0]["severity"] == "S3"

    def test_write_invalid_priority_defaults(self):
        store = TodoStore()
        result = store.write([
            {"id": "1", "content": "Task", "status": "pending", "priority": "P99", "severity": "S99"},
        ])
        assert result[0]["priority"] == "P2"
        assert result[0]["severity"] == "S2"


# ── HasItems ─────────────────────────────────────────────────────────────


class TestHasItems:
    def test_empty_store(self):
        store = TodoStore()
        assert store.has_items() is False

    def test_non_empty_store(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "x", "status": "pending"}])
        assert store.has_items() is True


# ── FormatForInjection ───────────────────────────────────────────────────


class TestFormatForInjection:
    def test_empty_returns_none(self):
        store = TodoStore()
        assert store.format_for_injection() is None

    def test_non_empty_has_markers(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Do thing", "status": "completed"},
            {"id": "2", "content": "Next", "status": "pending"},
            {"id": "3", "content": "Working", "status": "in_progress"},
        ])
        text = store.format_for_injection()
        assert "[x]" not in text
        assert "Do thing" not in text
        assert "[ ]" in text
        assert "[>]" in text
        assert "Next" in text
        assert "Working" in text
        assert "context compression" in text.lower()

    def test_includes_priority_severity_label(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Urgent", "status": "pending", "priority": "P0", "severity": "S1"},
        ])
        text = store.format_for_injection()
        assert "P0S1" in text

    def test_default_label_is_p2s2(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Normal", "status": "pending"},
        ])
        text = store.format_for_injection()
        assert "P2S2" in text


# ── MergeMode ────────────────────────────────────────────────────────────


class TestMergeMode:
    def test_update_existing_by_id(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Original", "status": "pending"},
        ])
        store.write(
            [{"id": "1", "status": "completed"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 1
        assert items[0]["status"] == "completed"
        assert items[0]["content"] == "Original"

    def test_merge_appends_new(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "First", "status": "pending"}])
        store.write(
            [{"id": "2", "content": "Second", "status": "pending"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 2

    def test_merge_updates_priority_severity(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Task", "status": "pending", "priority": "P2", "severity": "S2"},
        ])
        store.write(
            [{"id": "1", "priority": "P0", "severity": "S1"}],
            merge=True,
        )
        items = store.read()
        assert len(items) == 1
        assert items[0]["priority"] == "P0"
        assert items[0]["severity"] == "S1"

    def test_merge_preserves_priority_when_not_provided(self):
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Task", "status": "pending", "priority": "P1"},
        ])
        store.write(
            [{"id": "1", "status": "completed"}],
            merge=True,
        )
        items = store.read()
        assert items[0]["priority"] == "P1"
        assert items[0]["status"] == "completed"


# ── TodoToolFunction ─────────────────────────────────────────────────────


class TestTodoToolFunction:
    def test_read_mode(self):
        store = TodoStore()
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        result = json.loads(todo_tool(store=store))
        assert result["summary"]["total"] == 1
        assert result["summary"]["pending"] == 1

    def test_write_mode(self):
        store = TodoStore()
        result = json.loads(todo_tool(
            todos=[{"id": "1", "content": "New", "status": "in_progress"}],
            store=store,
        ))
        assert result["summary"]["in_progress"] == 1

    def test_no_store_returns_error(self):
        result = json.loads(todo_tool())
        assert "error" in result

    def test_write_with_priority_severity(self):
        store = TodoStore()
        result = json.loads(todo_tool(
            todos=[{
                "id": "1", "content": "Critical", "status": "pending",
                "priority": "P0", "severity": "S1",
            }],
            store=store,
        ))
        assert result["summary"]["total"] == 1
        items = store.read()
        assert items[0]["priority"] == "P0"
        assert items[0]["severity"] == "S1"

    def test_result_includes_priority_severity(self):
        store = TodoStore()
        result = json.loads(todo_tool(
            todos=[{"id": "1", "content": "Task", "status": "pending", "priority": "P3"}],
            store=store,
        ))
        assert result["todos"][0]["priority"] == "P3"
        assert result["todos"][0]["severity"] == "S2"  # default


# ── DB Persistence ───────────────────────────────────────────────────────


class TestDBPersistence:
    """Tests for TodoStore persistence via SessionDB."""

    @staticmethod
    def _make_db(tmp_path) -> SessionDB:
        """Create a temp-file SessionDB with the schema applied."""
        db = SessionDB(tmp_path / "state.db")
        # Ensure a session row exists for FK constraint
        db.create_session("test-session", source="test")
        return db

    def test_persist_on_write(self, tmp_path):
        db = self._make_db(tmp_path)
        store = TodoStore(db=db, session_id="test-session")
        store.write([
            {"id": "1", "content": "Task A", "status": "pending", "priority": "P0", "severity": "S1"},
        ])
        # Load directly from DB
        loaded = db.load_todos("test-session")
        assert loaded is not None
        assert len(loaded) == 1
        assert loaded[0]["content"] == "Task A"
        assert loaded[0]["priority"] == "P0"

    def test_restore_from_db(self, tmp_path):
        db = self._make_db(tmp_path)
        # Write with one store
        store1 = TodoStore(db=db, session_id="test-session")
        store1.write([
            {"id": "1", "content": "Persisted", "status": "in_progress", "priority": "P1"},
        ])
        # Restore with a fresh store
        store2 = TodoStore(db=db, session_id="test-session")
        assert store2.has_items() is False
        assert store2.restore_from_db() is True
        items = store2.read()
        assert len(items) == 1
        assert items[0]["content"] == "Persisted"
        assert items[0]["priority"] == "P1"
        assert items[0]["severity"] == "S2"

    def test_restore_from_db_empty_returns_false(self, tmp_path):
        db = self._make_db(tmp_path)
        store = TodoStore(db=db, session_id="test-session")
        assert store.restore_from_db() is False

    def test_restore_from_db_no_db_returns_false(self):
        store = TodoStore()
        assert store.restore_from_db() is False

    def test_persist_on_merge(self, tmp_path):
        db = self._make_db(tmp_path)
        store = TodoStore(db=db, session_id="test-session")
        store.write([{"id": "1", "content": "First", "status": "pending"}])
        store.write(
            [{"id": "2", "content": "Second", "status": "pending", "priority": "P0"}],
            merge=True,
        )
        # Verify DB has both items
        loaded = db.load_todos("test-session")
        assert len(loaded) == 2
        assert loaded[1]["priority"] == "P0"

    def test_persist_without_db_is_noop(self):
        store = TodoStore()  # no db
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        assert store.has_items() is True
        # Should not raise

    def test_db_error_does_not_crash(self):
        mock_db = MagicMock()
        mock_db.save_todos.side_effect = sqlite3.OperationalError("locked")
        store = TodoStore(db=mock_db, session_id="test-session")
        # Should not raise
        store.write([{"id": "1", "content": "Task", "status": "pending"}])
        assert store.has_items() is True  # in-memory still works


# ── get_todo_display_items (AIAgent public interface) ────────────────────


class TestGetTodoDisplayItems:
    """Tests for the AIAgent.get_todo_display_items() interface."""

    def test_no_agent_returns_empty(self):
        """Simulate calling on an object without _todo_store."""
        from run_agent import AIAgent
        # We can't fully construct AIAgent without providers, so mock it
        agent = MagicMock(spec=AIAgent)
        # Use the real method
        agent.get_todo_display_items = AIAgent.get_todo_display_items.__get__(agent)
        # _todo_store not set -> returns []
        assert agent.get_todo_display_items() == []

    def test_returns_items_from_store(self):
        from run_agent import AIAgent
        agent = MagicMock(spec=AIAgent)
        agent.get_todo_display_items = AIAgent.get_todo_display_items.__get__(agent)
        store = TodoStore()
        store.write([
            {"id": "1", "content": "Task", "status": "pending", "priority": "P1"},
        ])
        agent._todo_store = store
        items = agent.get_todo_display_items()
        assert len(items) == 1
        assert items[0]["content"] == "Task"
        assert items[0]["priority"] == "P1"
