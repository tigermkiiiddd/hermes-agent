#!/usr/bin/env python3
"""
Project Tool — manage named projects with directory, git URL, and description.

Projects associate sessions with a working context.  When a project is active:
  - The project name, path, and description are injected into the system prompt.
  - The terminal CWD is set to the project's path.
  - Branch sessions inherit the parent's project.

Actions: create, update, delete, list, set, unset, info, sessions.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

PROJECT_SCHEMA = {
    "name": "project",
    "description": (
        "Manage projects — named work contexts with directory, git URL, and description.\n\n"
        "A project binds a session to a working directory and metadata. When set:\n"
        "- System prompt includes project name/path/description.\n"
        "- Terminal CWD is set to the project's path.\n"
        "- Branch sessions inherit the project.\n\n"
        "ACTIONS:\n"
        "- create: Create a new project (name required, path/description/git_url optional).\n"
        "- update: Update fields of an existing project.\n"
        "- delete: Delete a project (sessions keep their project_id as NULL).\n"
        "- list: List all projects.\n"
        "- set: Associate the current session with a project (sets CWD).\n"
        "- unset: Remove project association from current session.\n"
        "- info: Get details of a specific project.\n"
        "- sessions: List sessions associated with a project."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "update", "delete", "list", "set", "unset", "info", "sessions"],
                "description": "The action to perform.",
            },
            "name": {
                "type": "string",
                "description": "Project name (used as identifier). Required for create, update, delete, set, info, sessions.",
            },
            "path": {
                "type": "string",
                "description": "Directory path for the project. Used in create and update.",
            },
            "description": {
                "type": "string",
                "description": "Project description. Used in create and update.",
            },
            "git_url": {
                "type": "string",
                "description": "Git repository URL. Used in create and update.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db():
    """Lazily open SessionDB for the current profile."""
    from hermes_state import SessionDB
    return SessionDB(get_hermes_home() / "state.db")


def _success(data: dict) -> str:
    data["success"] = True
    return json.dumps(data, ensure_ascii=False)


def _error(msg: str, **extra) -> str:
    result = {"success": False, "error": msg}
    result.update(extra)
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def project_tool(
    action: str = "",
    name: str = None,
    path: str = None,
    description: str = None,
    git_url: str = None,
    session_id: str = None,
) -> str:
    """Execute a project management action."""

    try:
        db = _get_db()

        # -- create ----------------------------------------------------------
        if action == "create":
            if not name:
                return _error("Project name is required for create.")
            existing = db.get_project(name)
            if existing:
                return _error(f"Project '{name}' already exists.", name=name)
            if path:
                path = str(Path(path).expanduser().resolve())
            db.create_project(name, path=path, description=description, git_url=git_url)
            proj = db.get_project(name)
            return _success({
                "message": f"Project '{name}' created.",
                "project": proj,
            })

        # -- update ----------------------------------------------------------
        elif action == "update":
            if not name:
                return _error("Project name is required for update.")
            existing = db.get_project(name)
            if not existing:
                return _error(f"Project '{name}' not found.", name=name)
            if path is not None:
                path = str(Path(path).expanduser().resolve())
            ok = db.update_project(
                name,
                path=path if path is not None else ...,
                description=description if description is not None else ...,
                git_url=git_url if git_url is not None else ...,
            )
            if ok:
                proj = db.get_project(name)
                return _success({
                    "message": f"Project '{name}' updated.",
                    "project": proj,
                })
            return _error(f"Project '{name}' not found.", name=name)

        # -- delete ----------------------------------------------------------
        elif action == "delete":
            if not name:
                return _error("Project name is required for delete.")
            ok = db.delete_project(name)
            if ok:
                return _success({"message": f"Project '{name}' deleted.", "name": name})
            return _error(f"Project '{name}' not found.", name=name)

        # -- list ------------------------------------------------------------
        elif action == "list":
            projects = db.list_projects()
            return _success({
                "projects": projects,
                "count": len(projects),
            })

        # -- set -------------------------------------------------------------
        elif action == "set":
            if not name:
                return _error("Project name is required for set.")
            if not session_id:
                return _error("No active session.")
            proj = db.get_project(name)
            if not proj:
                return _error(f"Project '{name}' not found.", name=name)
            db.set_session_project(session_id, name)
            # Set CWD so terminal commands go to the project directory
            if proj.get("path"):
                os.environ["TERMINAL_CWD"] = proj["path"]
            return _success({
                "message": f"Session bound to project '{name}'.",
                "project": proj,
            })

        # -- unset -----------------------------------------------------------
        elif action == "unset":
            if not session_id:
                return _error("No active session.")
            db.set_session_project(session_id, None)
            # Reset CWD to default
            messaging_cwd = os.getenv("MESSAGING_CWD", str(Path.home()))
            os.environ["TERMINAL_CWD"] = messaging_cwd
            return _success({"message": "Project unset from current session."})

        # -- info ------------------------------------------------------------
        elif action == "info":
            if not name:
                # Try to get from current session
                if session_id:
                    meta = db.get_session(session_id)
                    if meta:
                        name = meta.get("project_id")
                if not name:
                    return _error("No project specified or associated with current session.")
            proj = db.get_project(name)
            if not proj:
                return _error(f"Project '{name}' not found.", name=name)
            sessions = db.get_sessions_by_project(name)
            return _success({
                "project": proj,
                "session_count": len(sessions),
                "sessions": [
                    {
                        "id": s["id"],
                        "started_at": s.get("started_at"),
                        "title": s.get("title"),
                    }
                    for s in sessions
                ],
            })

        # -- sessions --------------------------------------------------------
        elif action == "sessions":
            if not name:
                return _error("Project name is required for sessions.")
            proj = db.get_project(name)
            if not proj:
                return _error(f"Project '{name}' not found.", name=name)
            sessions = db.get_sessions_by_project(name)
            return _success({
                "project": name,
                "sessions": [
                    {
                        "id": s["id"],
                        "started_at": s.get("started_at"),
                        "title": s.get("title"),
                        "model": s.get("model"),
                    }
                    for s in sessions
                ],
                "count": len(sessions),
            })

        else:
            return _error(f"Unknown action '{action}'. Valid: create, update, delete, list, set, unset, info, sessions.")

    except Exception as e:
        logger.exception("project_tool error: %s", e)
        return _error(str(e))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
        git_url=args.get("git_url"),
        session_id=kw.get("session_id"),
    ),
    emoji="📁",
)
