"""
Project management tool for hermes-agent.

Allows the LLM to create, list, update, delete projects and associate
the current session with a project. Projects provide context (working
directory, description) to the agent without isolating conversations.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy DB access — avoids circular imports at module load time.
_db = None


def _get_db():
    global _db
    if _db is None:
        from hermes_state import SessionDB
        _home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        _db = SessionDB(_home / "state.db")
    return _db


def _get_session_id(override: str = None) -> Optional[str]:
    """Try to get the current session ID — from explicit arg or environment."""
    if override:
        return override
    return os.environ.get("HERMES_SESSION_ID")


def project_tool(
    action: str,
    name: str = None,
    path: str = None,
    description: str = None,
    project_id: str = None,
    session_id: str = None,
) -> dict:
    """
    Manage projects — named contexts that provide working directory and
    description to the current session.

    Actions:
      - list:   List all projects.
      - create: Create a project (requires name; path and description optional).
      - update: Update project fields (name required, other fields optional).
      - delete: Delete a project by name.
      - info:   Get detailed info about a project.
      - set:    Associate current session with a project (by name).
      - unset:  Disassociate current session from its project.
    """
    db = _get_db()

    if action == "list":
        projects = db.list_projects()
        if not projects:
            return {"result": "No projects exist yet.", "projects": []}
        return {
            "result": f"{len(projects)} project(s) found.",
            "projects": projects,
        }

    elif action == "create":
        if not name:
            return {"error": "Project name is required for create."}
        existing = db.get_project(name)
        if existing:
            return {"error": f"Project '{name}' already exists."}
        resolved_path = None
        if path:
            resolved_path = str(Path(path).expanduser().resolve())
        db.create_project(name, path=resolved_path, description=description)
        proj = db.get_project(name)
        return {"result": f"Project '{name}' created.", "project": proj}

    elif action == "update":
        if not name:
            return {"error": "Project name is required for update."}
        existing = db.get_project(name)
        if not existing:
            return {"error": f"Project '{name}' not found."}
        resolved_path = path  # keep Ellipsis sentinel if not provided
        if path and path is not ...:
            resolved_path = str(Path(path).expanduser().resolve())
        db.update_project(name, path=resolved_path, description=description)
        proj = db.get_project(name)
        return {"result": f"Project '{name}' updated.", "project": proj}

    elif action == "delete":
        if not name:
            return {"error": "Project name is required for delete."}
        ok = db.delete_project(name)
        if ok:
            return {"result": f"Project '{name}' deleted."}
        return {"error": f"Project '{name}' not found."}

    elif action == "info":
        target = name or project_id
        if not target:
            return {"error": "Project name required. Use action='info', name='project_name'."}
        proj = db.get_project(target)
        if not proj:
            return {"error": f"Project '{target}' not found."}
        sessions = db.get_sessions_by_project(target)
        proj["session_count"] = len(sessions)
        return {"result": f"Project info for '{target}'.", "project": proj}

    elif action == "set":
        if not name:
            return {"error": "Project name is required for set."}
        proj = db.get_project(name)
        if not proj:
            return {"error": f"Project '{name}' not found. Create it first with action='create'."}
        sid = _get_session_id(session_id)
        if not sid:
            return {"error": "No active session ID found."}
        db.set_session_project(sid, name)
        # Update CWD if project has a path
        if proj.get("path"):
            os.environ["TERMINAL_CWD"] = proj["path"]
        return {
            "result": f"Session associated with project '{name}'.",
            "project": proj,
            "cwd_updated": bool(proj.get("path")),
        }

    elif action == "unset":
        sid = _get_session_id(session_id)
        if not sid:
            return {"error": "No active session ID found."}
        db.set_session_project(sid, None)
        # Reset CWD to default
        messaging_cwd = os.getenv("MESSAGING_CWD", str(Path.home()))
        os.environ["TERMINAL_CWD"] = messaging_cwd
        return {"result": "Session disassociated from project. CWD reset to default."}

    else:
        return {"error": f"Unknown action '{action}'. Valid: list, create, update, delete, info, set, unset."}


# ── Schema & Registry ──────────────────────────────────────────────────────

PROJECT_SCHEMA = {
    "name": "project",
    "description": (
        "Manage projects — named contexts that provide working directory, description, "
        "and configuration to the current session. Projects are NOT isolated: global "
        "memory, tools, and skills remain fully accessible.\n\n"
        "ACTIONS:\n"
        "- list: List all projects.\n"
        "- create: Create a new project (name required, path and description optional).\n"
        "- update: Update project fields (name required, path/description optional).\n"
        "- delete: Delete a project.\n"
        "- info: Get detailed info about a project.\n"
        "- set: Associate the current session with a project. Updates CWD if project has a path.\n"
        "- unset: Remove project association from current session.\n\n"
        "WHEN TO USE:\n"
        "- User mentions working on a specific project/codebase → create or set project.\n"
        "- Starting work in a specific directory → set project to update CWD.\n"
        "- User asks 'what projects do we have' → list.\n"
        "- Switching between different work contexts → set/unset."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "create", "update", "delete", "info", "set", "unset"],
                "description": "The action to perform.",
            },
            "name": {
                "type": "string",
                "description": "Project name. Required for create, update, delete, info, set.",
            },
            "path": {
                "type": "string",
                "description": "Working directory path for the project. Optional. Expanduser is applied automatically.",
            },
            "description": {
                "type": "string",
                "description": "Human-readable description of the project. Optional.",
            },
        },
        "required": ["action"],
    },
}


from tools.registry import registry

registry.register(
    name="project",
    toolset="memory",
    schema=PROJECT_SCHEMA,
    handler=lambda args, **kw: project_tool(
        action=args.get("action", ""),
        name=args.get("name"),
        path=args.get("path"),
        description=args.get("description"),
    ),
    check_fn=lambda: True,
    emoji="📁",
    max_result_size_chars=50_000,
)
