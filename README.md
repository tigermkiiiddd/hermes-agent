# hermes-agent (fork)

This is a fork of [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) with additional features developed locally.

## What's Changed

### Project System

- New `project` tool for managing named project contexts (name, working directory, description)
- DB schema v7 — `projects` table in `state.db`
- CLI `/project` commands (create, set, unset, list, info, update, delete)
- LLM-facing tool registered in `memory` toolset — agent can switch projects mid-conversation

### Memory Partitions

- Structured memory storage with named partitions (e.g. `programming`, `game-dev`, `fiction-writing`, `environment`)
- System prompt automatically displays loaded partition topics
- Custom partitions supported alongside built-in ones

### Todo Persistence & Priority

- `TodoStore` persists to `state.db` (schema v8) via `session_todos` table
- Priority levels: `P0` (critical) → `P3` (low), default `P2`
- Severity levels: `S1` (blocker) → `S3` (minor), default `S2`
- DB-first restore on session resume, falls back to history scan
- TUI activity feed shows per-item status markers

### Plugin System Enhancements

- **Claude plugin compatibility layer** — load Claude-format plugins directly
- **Plugin MCP servers** — plugins can declare MCP servers in `plugin.yaml` or register them at runtime via `register_mcp_server()`. Auto-merged into MCP discovery alongside `config.yaml` entries.
- **Plugin TUI widgets** — `register_tui_widget()` lets plugins inject prompt_toolkit widgets into the CLI layout
- Plugin skills auto-categorized from directory name; recursive symlink scanning

### delegate_task Skills Injection

- `delegate_task` supports `skills` parameter — subagents preload skill files before executing their task

### Project Tool Registration Fix

- `project` tool registered in all relevant toolsets (`memory`, `api_server`, etc.)
- Proper JSON serialization for both tool handler and `run_agent` dispatch

## Syncing with Upstream

```bash
git remote add upstream https://github.com/NousResearch/hermes-agent.git
git fetch upstream
git rebase upstream/main
```

## License

This fork follows the same license as the upstream project.

**MIT License** — Copyright (c) 2025 Nous Research

See [LICENSE](./LICENSE) for the full text.
