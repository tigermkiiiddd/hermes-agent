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
    # Issue-related params
    issue_id: int = None,
    title: str = None,
    priority: str = None,
    labels: list = None,
    status: str = None,
    assignee_session_id: str = None,
) -> dict:
    """
    Manage projects — named contexts that provide working directory and
    description to the current session.

    Actions:
      Project management:
      - list:   List all projects.
      - create: Create a project (requires name; path and description optional).
      - update: Update project fields (name required, other fields optional).
      - delete: Delete a project by name.
      - info:   Get detailed info about a project.
      - set:    Associate current session with a project (by name).
      - unset:  Disassociate current session from its project.

      Issue management:
      - issue-create:  Create an issue in a project (name + title required).
      - issue-list:    List issues for a project (name required, status optional).
      - issue-update:  Update an issue (issue_id required, other fields optional).
      - issue-close:   Close an issue (issue_id required).
      - issue-assign:  Assign an issue to a session (issue_id required).
      - issue-delete:  Delete an issue (issue_id required).
      - issues:        List open/in_progress issues for current project.
    """
    db = _get_db()

    # ── Project management ──────────────────────────────────────────────
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

    # ── Issue management ────────────────────────────────────────────────
    elif action == "issue-create":
        if not name:
            return {"error": "Project name is required (use 'name' param)."}
        if not title:
            return {"error": "Issue title is required (use 'title' param)."}
        proj = db.get_project(name)
        if not proj:
            return {"error": f"Project '{name}' not found. Create it first."}
        issue = db.create_issue(
            project_name=name,
            title=title,
            description=description,
            priority=priority or "P2",
            labels=labels,
        )
        return {"result": f"Issue #{issue['id']} created in '{name}'.", "issue": issue}

    elif action == "issue-list":
        if not name:
            return {"error": "Project name is required (use 'name' param)."}
        proj = db.get_project(name)
        if not proj:
            return {"error": f"Project '{name}' not found."}
        issues = db.list_issues(name, status=status)
        return {
            "result": f"{len(issues)} issue(s) in '{name}'.",
            "issues": issues,
        }

    elif action == "issue-update":
        if not issue_id:
            return {"error": "issue_id is required."}
        updated = db.update_issue(
            issue_id=issue_id,
            title=title if title is not None else ...,
            description=description if description is not None else ...,
            status=status if status is not None else ...,
            priority=priority if priority is not None else ...,
            labels=labels if labels is not None else ...,
            assignee_session_id=assignee_session_id if assignee_session_id is not None else ...,
        )
        if not updated:
            return {"error": f"Issue #{issue_id} not found."}
        return {"result": f"Issue #{issue_id} updated.", "issue": updated}

    elif action == "issue-close":
        if not issue_id:
            return {"error": "issue_id is required."}
        updated = db.update_issue(issue_id, status="closed")
        if not updated:
            return {"error": f"Issue #{issue_id} not found."}
        return {"result": f"Issue #{issue_id} closed.", "issue": updated}

    elif action == "issue-assign":
        if not issue_id:
            return {"error": "issue_id is required."}
        sid = assignee_session_id or _get_session_id(session_id)
        if not sid:
            return {"error": "No session ID to assign. Provide assignee_session_id."}
        updated = db.assign_issue(issue_id, sid)
        if not updated:
            return {"error": f"Issue #{issue_id} not found."}
        return {"result": f"Issue #{issue_id} assigned to session {sid[:8]}…", "issue": updated}

    elif action == "issue-delete":
        if not issue_id:
            return {"error": "issue_id is required."}
        ok = db.delete_issue(issue_id)
        if ok:
            return {"result": f"Issue #{issue_id} deleted."}
        return {"error": f"Issue #{issue_id} not found."}

    elif action == "issues":
        """List open issues for current project (shortcut)."""
        sid = _get_session_id(session_id)
        if not sid:
            return {"error": "No active session ID."}
        # Find current project for this session
        sessions = db.search_sessions(limit=1, query=sid)
        proj_name = None
        # Try to get project from session
        try:
            row = db._conn.execute(
                "SELECT project_id FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            if row:
                proj_name = row["project_id"] if isinstance(row, dict) else row[0]
        except Exception:
            pass
        if not proj_name and name:
            proj_name = name
        if not proj_name:
            return {"error": "No project associated with current session. Use action='set' first or provide name."}
        issues = db.get_open_issues_for_project(proj_name)
        return {
            "result": f"{len(issues)} open issue(s) in '{proj_name}'.",
            "project": proj_name,
            "issues": issues,
        }

    else:
        return {"error": f"Unknown action '{action}'. Valid: list, create, update, delete, info, set, unset, issue-create, issue-list, issue-update, issue-close, issue-assign, issue-delete, issues."}


# ── Schema & Registry ──────────────────────────────────────────────────────

PROJECT_SCHEMA = {
    "name": "project",
    "description": (
        "Manage projects and their issues — named contexts that provide working "
        "directory, description, and cross-session task tracking.\n\n"
        "Projects are NOT isolated: global memory, tools, and skills remain fully accessible.\n\n"
        "PROJECT ACTIONS:\n"
        "- list: List all projects.\n"
        "- create: Create a new project (name required, path and description optional).\n"
        "- update: Update project fields (name required, path/description optional).\n"
        "- delete: Delete a project.\n"
        "- info: Get detailed info about a project.\n"
        "- set: Associate the current session with a project. Updates CWD if project has a path.\n"
        "- unset: Remove project association from current session.\n\n"
        "ISSUE ACTIONS:\n"
        "- issue-create: Create an issue in a project (name + title required).\n"
        "- issue-list: List issues for a project (name required, filter by status).\n"
        "- issue-update: Update an issue (issue_id required, other fields optional).\n"
        "- issue-close: Close an issue (issue_id required).\n"
        "- issue-assign: Assign an issue to the current session (issue_id required).\n"
        "- issue-delete: Delete an issue (issue_id required).\n"
        "- issues: List open/in_progress issues for current project (shortcut).\n\n"
        "WHEN TO USE:\n"
        "- User mentions working on a specific project/codebase → create or set project.\n"
        "- Starting work in a specific directory → set project to update CWD.\n"
        "- User asks 'what projects do we have' → list.\n"
        "- Switching between different work contexts → set/unset.\n"
        "- Planning multi-session work → issue-create.\n"
        "- Picking up work from a previous session → issues."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list", "create", "update", "delete", "info", "set", "unset",
                    "issue-create", "issue-list", "issue-update", "issue-close",
                    "issue-assign", "issue-delete", "issues",
                ],
                "description": "The action to perform.",
            },
            "name": {
                "type": "string",
                "description": "Project name. Required for create, update, delete, info, set, issue-create, issue-list.",
            },
            "path": {
                "type": "string",
                "description": "Working directory path for the project. Optional. Expanduser is applied automatically.",
            },
            "description": {
                "type": "string",
                "description": "Description for project or issue. Optional.",
            },
            "issue_id": {
                "type": "integer",
                "description": "Issue ID. Required for issue-update, issue-close, issue-assign, issue-delete.",
            },
            "title": {
                "type": "string",
                "description": "Issue title. Required for issue-create.",
            },
            "priority": {
                "type": "string",
                "enum": ["P0", "P1", "P2", "P3"],
                "description": "Priority level for issue. Default: P2.",
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Labels/tags for issue. Optional.",
            },
            "status": {
                "type": "string",
                "description": "Filter or set status. For issue-list: filter. For issue-update: new status.",
            },
            "assignee_session_id": {
                "type": "string",
                "description": "Session ID to assign issue to. For issue-assign. Defaults to current session.",
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
    handler=lambda args, **kw: json.dumps(
        project_tool(
            action=args.get("action", ""),
            name=args.get("name"),
            path=args.get("path"),
            description=args.get("description"),
            session_id=kw.get("session_id"),
            issue_id=args.get("issue_id"),
            title=args.get("title"),
            priority=args.get("priority"),
            labels=args.get("labels"),
            status=args.get("status"),
            assignee_session_id=args.get("assignee_session_id"),
        ),
        ensure_ascii=False,
    ),
    check_fn=lambda: True,
    emoji="📁",
    max_result_size_chars=50_000,
)
