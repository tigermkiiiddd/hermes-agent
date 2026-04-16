#!/usr/bin/env python3
"""
Todo Tool Module - Planning & Task Management

Provides a task list the agent uses to decompose complex tasks,
track progress, and maintain focus across long conversations.

Persistence:
- When a session_db + session_id are provided, every write() call
  automatically persists the full list to the ``session_todos`` table
  in state.db.  On next turn (gateway creates fresh AIAgent), the store
  is rehydrated from DB before falling back to history scan.

Design:
- Single `todo` tool: provide `todos` param to write, omit to read
- Every call returns the full current list
- No system prompt mutation, no tool response modification
- Behavioral guidance lives entirely in the tool schema description
"""

import json
from typing import Dict, Any, List, Optional


# Valid status values for todo items
VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# Valid priority / severity levels
VALID_PRIORITIES = {"P0", "P1", "P2", "P3"}
VALID_SEVERITIES = {"S1", "S2", "S3"}


def _normalize_priority(value: Any) -> str:
    """Normalize a priority value to P0-P3."""
    if not value:
        return "P2"
    s = str(value).strip().upper()
    if s in VALID_PRIORITIES:
        return s
    # Accept bare number: 0->P0, 1->P1, etc.
    if s.isdigit() and 0 <= int(s) <= 3:
        return f"P{int(s)}"
    return "P2"


def _normalize_severity(value: Any) -> str:
    """Normalize a severity value to S1-S3."""
    if not value:
        return "S2"
    s = str(value).strip().upper()
    if s in VALID_SEVERITIES:
        return s
    if s.isdigit() and 1 <= int(s) <= 3:
        return f"S{int(s)}"
    return "S2"


class TodoStore:
    """
    Task list with optional DB persistence.

    One instance per AIAgent (one per session).  When ``db`` and
    ``session_id`` are provided, every write is persisted to state.db.

    Items are ordered -- list position is priority. Each item has:
      - id: unique string identifier (agent-chosen)
      - content: task description
      - status: pending | in_progress | completed | cancelled
      - priority: P0 (critical) | P1 (high) | P2 (normal) | P3 (low)
      - severity: S1 (blocker) | S2 (major) | S3 (minor)
    """

    def __init__(self, db=None, session_id: Optional[str] = None):
        self._items: List[Dict[str, str]] = []
        self._db = db
        self._session_id = session_id

    def _persist(self) -> None:
        """Write current items to DB if db + session_id are set."""
        if self._db and self._session_id:
            try:
                self._db.save_todos(self._session_id, self._items)
            except Exception:
                pass  # Don't crash agent on DB errors

    def restore_from_db(self) -> bool:
        """Try to load items from DB. Returns True if items were found."""
        if not self._db or not self._session_id:
            return False
        try:
            items = self._db.load_todos(self._session_id)
            if items:
                self._items = [self._validate(t) for t in items]
                return True
        except Exception:
            pass
        return False

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """
        Write todos. Returns the full current list after writing.

        Args:
            todos: list of {id, content, status, priority?, severity?} dicts
            merge: if False, replace the entire list. If True, update
                   existing items by id and append new ones.
        """
        if not merge:
            # Replace mode: new list entirely
            self._items = [self._validate(t) for t in self._dedupe_by_id(todos)]
        else:
            # Merge mode: update existing items by id, append new ones
            existing = {item["id"]: item for item in self._items}
            for t in self._dedupe_by_id(todos):
                item_id = str(t.get("id", "")).strip()
                if not item_id:
                    continue  # Can't merge without an id

                if item_id in existing:
                    # Update only the fields the LLM actually provided
                    if "content" in t and t["content"]:
                        existing[item_id]["content"] = str(t["content"]).strip()
                    if "status" in t and t["status"]:
                        status = str(t["status"]).strip().lower()
                        if status in VALID_STATUSES:
                            existing[item_id]["status"] = status
                    if "priority" in t:
                        existing[item_id]["priority"] = _normalize_priority(t["priority"])
                    if "severity" in t:
                        existing[item_id]["severity"] = _normalize_severity(t["severity"])
                else:
                    # New item -- validate fully and append to end
                    validated = self._validate(t)
                    existing[validated["id"]] = validated
                    self._items.append(validated)
            # Rebuild _items preserving order for existing items
            seen = set()
            rebuilt = []
            for item in self._items:
                current = existing.get(item["id"], item)
                if current["id"] not in seen:
                    rebuilt.append(current)
                    seen.add(current["id"])
            self._items = rebuilt
        self._persist()
        return self.read()

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        """Check if there are any items in the list."""
        return bool(self._items)

    def format_for_injection(self) -> Optional[str]:
        """
        Render the todo list for post-compression injection.

        Returns a human-readable string to append to the compressed
        message history, or None if the list is empty.
        """
        if not self._items:
            return None

        # Status markers for compact display
        markers = {
            "completed": "[x]",
            "in_progress": "[>]",
            "pending": "[ ]",
            "cancelled": "[~]",
        }

        # Only inject pending/in_progress items — completed/cancelled ones
        # cause the model to re-do finished work after compression.
        active_items = [
            item for item in self._items
            if item["status"] in ("pending", "in_progress")
        ]
        if not active_items:
            return None

        lines = ["[Your active task list was preserved across context compression]"]
        for item in active_items:
            marker = markers.get(item["status"], "[?]")
            p = item.get("priority", "P2")
            s = item.get("severity", "S2")
            label = f"{p}{s}"
            lines.append(f"- {marker} {item['id']}. {item['content']} ({item['status']}) [{label}]")

        return "\n".join(lines)

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """
        Validate and normalize a todo item.

        Ensures required fields exist and status/priority/severity are valid.
        Returns a clean dict with {id, content, status, priority, severity}.
        """
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            item_id = "?"

        content = str(item.get("content", "")).strip()
        if not content:
            content = "(no description)"

        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        return {
            "id": item_id,
            "content": content,
            "status": status,
            "priority": _normalize_priority(item.get("priority")),
            "severity": _normalize_severity(item.get("severity")),
        }

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
) -> str:
    """
    Single entry point for the todo tool. Reads or writes depending on params.

    Args:
        todos: if provided, write these items. If None, read current list.
        merge: if True, update by id. If False (default), replace entire list.
        store: the TodoStore instance from the AIAgent.

    Returns:
        JSON string with the full current list and summary metadata.
    """
    if store is None:
        return tool_error("TodoStore not initialized")

    if todos is not None:
        items = store.write(todos, merge)
    else:
        items = store.read()

    # Build summary counts
    pending = sum(1 for i in items if i["status"] == "pending")
    in_progress = sum(1 for i in items if i["status"] == "in_progress")
    completed = sum(1 for i in items if i["status"] == "completed")
    cancelled = sum(1 for i in items if i["status"] == "cancelled")

    return json.dumps({
        "todos": items,
        "summary": {
            "total": len(items),
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "cancelled": cancelled,
        },
    }, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# =============================================================================
# OpenAI Function-Calling Schema
# =============================================================================
# Behavioral guidance is baked into the description so it's part of the
# static tool schema (cached, never changes mid-conversation).

TODO_SCHEMA = {
    "name": "todo",
    "description": (
        "Manage your task list for the current session. Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "Call with no parameters to read the current list.\n\n"
        "Writing:\n"
        "- Provide 'todos' array to create/update items\n"
        "- merge=false (default): replace the entire list with a fresh plan\n"
        "- merge=true: update existing items by id, add any new ones\n\n"
        "Each item: {id: string, content: string, "
        "status: pending|in_progress|completed|cancelled, "
        "priority?: P0|P1|P2|P3, severity?: S1|S2|S3}\n"
        "Priority: P0=critical/blocking, P1=high, P2=normal(default), P3=low\n"
        "Severity: S1=blocker/unusable, S2=major impact(default), S3=minor/cosmetic\n"
        "List order is priority. Only ONE item in_progress at a time.\n"
        "Mark items completed immediately when done. If something fails, "
        "cancel it and add a revised item.\n\n"
        "Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write. Omit to read current list.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique item identifier"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Current status"
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["P0", "P1", "P2", "P3"],
                            "description": "Priority level (P0=critical, P1=high, P2=normal, P3=low). Default: P2"
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["S1", "S2", "S3"],
                            "description": "Severity/impact (S1=blocker, S2=major, S3=minor). Default: S2"
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list."
                ),
                "default": False
            }
        },
        "required": []
    }
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="todo",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store")),
    check_fn=check_todo_requirements,
    emoji="📋",
)
