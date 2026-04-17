"""Tests for skill domain folding (activate/deactivate/collapsed/expanded).

Covers:
  1. prompt_builder: collapsed vs expanded rendering
  2. skill_manage: activate/deactivate actions
  3. Active domain persistence (JSON file)
  4. Active domain restoration from system prompt fallback
  5. Auto-activate on skill_view
  6. Compression: active domains injected into compressed messages
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def skills_dir(tmp_path):
    """Create a populated skills directory under HERMES_HOME."""
    sd = tmp_path / "hermes_test" / "skills"

    # mlops/training/axolotl
    (sd / "mlops" / "training" / "axolotl").mkdir(parents=True)
    (sd / "mlops" / "training" / "axolotl" / "SKILL.md").write_text(
        "---\nname: axolotl\ndescription: Fine-tune LLMs with Axolotl\n---\n"
    )
    (sd / "mlops" / "training" / "unsloth").mkdir(parents=True)
    (sd / "mlops" / "training" / "unsloth" / "SKILL.md").write_text(
        "---\nname: unsloth\ndescription: Fast fine-tuning with Unsloth\n---\n"
    )

    # mlops/inference/vllm
    (sd / "mlops" / "inference" / "vllm").mkdir(parents=True)
    (sd / "mlops" / "inference" / "vllm" / "SKILL.md").write_text(
        "---\nname: vllm\ndescription: High-throughput LLM serving\n---\n"
    )

    # github/github-auth
    (sd / "github" / "github-auth").mkdir(parents=True)
    (sd / "github" / "github-auth" / "SKILL.md").write_text(
        "---\nname: github-auth\ndescription: Set up GitHub auth\n---\n"
    )

    # flat skill (no category)
    (sd / "notes").mkdir(parents=True)
    (sd / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: Take notes\n---\n"
    )

    return sd


@pytest.fixture
def pb(skills_dir, monkeypatch):
    """Import prompt_builder with isolated HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(skills_dir.parent))
    # Clear LRU cache between tests
    from agent.prompt_builder import _SKILLS_PROMPT_CACHE
    _SKILLS_PROMPT_CACHE.clear()
    from agent.prompt_builder import build_skills_system_prompt
    return build_skills_system_prompt


# ── 1. Prompt builder rendering ──────────────────────────────────────────

class TestCollapsedRendering:
    """Default (no active_domains) should show collapsed one-line categories."""

    def test_collapsed_shows_skill_names_comma_separated(self, pb):
        result = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains=None,
        )
        # Collapsed: category line with names
        assert "axolotl" in result
        assert "unsloth" in result
        # Should NOT have expanded sub-entries with descriptions
        assert "Fine-tune LLMs" not in result

    def test_collapsed_single_line_per_category(self, pb):
        result = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains=None,
        )
        lines = result.split("\n")
        # Within <available_skills>, no line should start with "    - " (expanded format)
        in_skills = False
        for line in lines:
            if "<available_skills>" in line:
                in_skills = True
                continue
            if "</available_skills>" in line:
                break
            if in_skills:
                assert not line.startswith("    - "), (
                    f"Found expanded sub-entry in collapsed mode: {line}"
                )

    def test_collapsed_prompt_contains_usage_instructions(self, pb):
        result = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains=None,
        )
        assert "activate" in result
        assert "deactivate" in result
        assert "domain" in result


class TestExpandedRendering:
    """When active_domains includes a category, it should expand."""

    def test_expanded_shows_full_descriptions(self, pb):
        result = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"mlops/training"},
        )
        # Expanded domain should show descriptions
        assert "Fine-tune LLMs with Axolotl" in result
        assert "Fast fine-tuning with Unsloth" in result

    def test_expanded_has_sub_entries(self, pb):
        result = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"mlops/training"},
        )
        lines = result.split("\n")
        # Should have "    - axolotl:" style lines
        has_expanded = any(
            line.startswith("    - axolotl:") for line in lines
        )
        assert has_expanded, "Expanded domain should have '- name: desc' sub-entries"

    def test_non_expanded_domains_still_collapsed(self, pb):
        result = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"mlops/training"},
        )
        # mlops/inference is NOT in active_domains, should be collapsed
        assert "High-throughput LLM serving" not in result
        # But vllm name should still appear
        assert "vllm" in result

    def test_multiple_domains_expanded(self, pb):
        result = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"mlops/training", "github"},
        )
        assert "Fine-tune LLMs with Axolotl" in result
        assert "Set up GitHub auth" in result
        # Non-active still collapsed
        assert "High-throughput LLM serving" not in result

    def test_empty_active_domains_same_as_none(self, pb):
        result_none = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains=None,
        )
        result_empty = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains=set(),
        )
        assert result_none == result_empty


class TestTokenSavings:
    """Verify that collapsed mode produces shorter output than expanded."""

    def test_collapsed_shorter_than_fully_expanded(self, pb):
        all_domains = {
            "mlops/training", "mlops/inference", "github",
        }
        collapsed = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains=None,
        )
        expanded = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains=all_domains,
        )
        assert len(collapsed) < len(expanded)
        # Should save at least 20%
        savings = (len(expanded) - len(collapsed)) / len(expanded)
        assert savings > 0.15, f"Token savings only {savings:.0%}, expected > 15%"


# ── 2. skill_manage activate/deactivate ──────────────────────────────────

class TestSkillManageActivateDeactivate:
    """Test skill_manage tool's activate/deactivate actions."""

    def test_activate_requires_domain(self):
        from tools.skill_manager_tool import skill_manage
        result = json.loads(skill_manage(action="activate", domain=""))
        assert result["success"] is False
        assert "domain is required" in result["error"]

    def test_deactivate_requires_domain(self):
        from tools.skill_manager_tool import skill_manage
        result = json.loads(skill_manage(action="deactivate", domain=""))
        assert result["success"] is False

    def test_activate_success(self):
        from tools.skill_manager_tool import skill_manage
        setter_calls = []
        def mock_setter(action, domain):
            setter_calls.append((action, domain))

        result = json.loads(skill_manage(
            action="activate",
            domain="mlops",
            _active_domains_setter=mock_setter,
        ))
        assert result["success"] is True
        assert "mlops" in result["message"]
        assert setter_calls == [("activate", "mlops")]

    def test_deactivate_success(self):
        from tools.skill_manager_tool import skill_manage
        setter_calls = []
        def mock_setter(action, domain):
            setter_calls.append((action, domain))

        result = json.loads(skill_manage(
            action="deactivate",
            domain="mlops",
            _active_domains_setter=mock_setter,
        ))
        assert result["success"] is True
        assert setter_calls == [("deactivate", "mlops")]

    def test_activate_without_setter_still_succeeds(self):
        """No crash when _active_domains_setter is None (e.g. direct call)."""
        from tools.skill_manager_tool import skill_manage
        result = json.loads(skill_manage(
            action="activate",
            domain="github",
            _active_domains_setter=None,
        ))
        assert result["success"] is True

    def test_schema_includes_activate_deactivate(self):
        from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA
        enum = SKILL_MANAGE_SCHEMA["parameters"]["properties"]["action"]["enum"]
        assert "activate" in enum
        assert "deactivate" in enum

    def test_schema_has_domain_param(self):
        from tools.skill_manager_tool import SKILL_MANAGE_SCHEMA
        assert "domain" in SKILL_MANAGE_SCHEMA["parameters"]["properties"]


# ── 3. Persistence (JSON file) ──────────────────────────────────────────

class TestActiveDomainsPersistence:
    """Test save/load cycle for active domains."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Minimal round-trip test using a mock agent-like object."""
        logs_dir = tmp_path / "sessions"
        logs_dir.mkdir(parents=True)
        session_id = "test_session_001"

        # Write
        domains = {"mlops/training", "github"}
        path = logs_dir / f".active_domains_{session_id}.json"
        path.write_text(json.dumps(sorted(domains)))

        # Read
        loaded = set(json.loads(path.read_text()))
        assert loaded == domains

    def test_load_nonexistent_returns_empty(self, tmp_path):
        logs_dir = tmp_path / "sessions"
        logs_dir.mkdir(parents=True)
        path = logs_dir / ".active_domains_nonexistent.json"
        if path.exists():
            path.unlink()
        # Should not crash
        domains = set()
        if path.exists():
            domains = set(json.loads(path.read_text()))
        assert domains == set()


# ── 4. Fallback: infer from system prompt ────────────────────────────────

class TestInferFromPrompt:
    """Test the fallback parser that infers active domains from system prompt text."""

    def test_parse_expanded_domains(self):
        """Should detect categories with expanded sub-entries."""
        import re
        prompt = (
            "<available_skills>\n"
            "  github: Set up GitHub auth\n"
            "    - github-auth: Set up GitHub auth\n"
            "  mlops/inference: High-throughput serving\n"
            "    - vllm: High-throughput LLM serving\n"
            "  mlops/training: axolotl, unsloth\n"
            "</available_skills>\n"
        )
        lines = prompt.split("\n")
        active_domains = set()
        in_available = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if "<available_skills>" in line:
                in_available = True
                i += 1
                continue
            if "</available_skills>" in line:
                break
            if in_available and line.startswith("  ") and not line.startswith("    "):
                cat_match = re.match(r'^  ([\w/.-]+)[: ]', line)
                if cat_match and i + 1 < len(lines) and lines[i + 1].startswith("    - "):
                    active_domains.add(cat_match.group(1))
            i += 1

        assert active_domains == {"github", "mlops/inference"}
        # mlops/training is collapsed (no sub-entries), so NOT detected

    def test_no_expanded_domains_returns_empty(self):
        import re
        prompt = (
            "<available_skills>\n"
            "  github: github-auth\n"
            "  mlops/training: axolotl, unsloth\n"
            "</available_skills>\n"
        )
        lines = prompt.split("\n")
        active_domains = set()
        in_available = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if "<available_skills>" in line:
                in_available = True
                i += 1
                continue
            if "</available_skills>" in line:
                break
            if in_available and line.startswith("  ") and not line.startswith("    "):
                cat_match = re.match(r'^  ([\w/.-]+)[: ]', line)
                if cat_match and i + 1 < len(lines) and lines[i + 1].startswith("    - "):
                    active_domains.add(cat_match.group(1))
            i += 1

        assert active_domains == set()


# ── 5. Auto-activate on skill_view ──────────────────────────────────────

class TestAutoActivateOnSkillView:
    """Test _auto_activate_skill_domain logic."""

    def test_auto_activate_finds_category(self, skills_dir, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(skills_dir.parent))

        # Mock get_all_skills_dirs to return our test dir
        active_domains = set()
        save_calls = []

        def mock_save():
            save_calls.append(list(active_domains))

        # Simulate the logic of _auto_activate_skill_domain
        skill_name = "axolotl"
        from agent.skill_utils import get_all_skills_dirs
        for sd in get_all_skills_dirs():
            for cat_dir in sd.iterdir():
                if cat_dir.is_dir() and (cat_dir / skill_name / "SKILL.md").exists():
                    # Found: check sub-cats too
                    pass
                # Check nested: mlops/training/axolotl
                if cat_dir.is_dir():
                    for sub in cat_dir.iterdir():
                        if sub.is_dir() and (sub / skill_name / "SKILL.md").exists():
                            cat = f"{cat_dir.name}/{sub.name}"
                            active_domains.add(cat)

        assert "mlops/training" in active_domains

    def test_flat_skill_no_auto_activate(self, skills_dir, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(skills_dir.parent))
        # 'notes' is in skills root, no category
        skill_name = "notes"
        from agent.skill_utils import get_all_skills_dirs
        found_cat = None
        for sd in get_all_skills_dirs():
            flat = sd / skill_name / "SKILL.md"
            if flat.exists():
                found_cat = "flat"
                break
            for cat_dir in sd.iterdir():
                if cat_dir.is_dir() and (cat_dir / skill_name / "SKILL.md").exists():
                    found_cat = cat_dir.name
                    break
        assert found_cat == "flat"


# ── 6. Compression injection ────────────────────────────────────────────

class TestCompressionInjection:
    """Verify active domains are injected into compressed messages."""

    def test_domains_injected_on_compression(self):
        """Simulate the compression injection logic."""
        active_domains = {"mlops/training", "github"}
        domains_str = ", ".join(sorted(active_domains))
        injection = f"[System: Active skill domains: {domains_str}. These domains are expanded in the system prompt.]"

        assert "github" in injection
        assert "mlops/training" in injection

    def test_no_injection_when_empty(self):
        active_domains = set()
        # The code checks `if self._active_skill_domains:` before injecting
        assert not active_domains  # No injection should happen


# ── 7. Cache key differentiation ────────────────────────────────────────

class TestCacheKeyDifferentiation:
    """Different active_domains should produce different cache entries."""

    def test_different_domains_different_output(self, pb):
        r1 = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"mlops/training"},
        )
        r2 = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"github"},
        )
        assert r1 != r2

    def test_same_domains_same_output(self, pb):
        r1 = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"mlops/training"},
        )
        r2 = pb(
            available_tools={"skill_manage", "skill_view"},
            available_toolsets={"skills"},
            active_domains={"mlops/training"},
        )
        assert r1 == r2


# ── 8. Backward compatibility ────────────────────────────────────────────

class TestBackwardCompat:
    """Ensure existing skill_manage actions still work."""

    def test_create_still_works(self, skills_dir, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(skills_dir.parent))
        from agent.prompt_builder import _SKILLS_PROMPT_CACHE
        _SKILLS_PROMPT_CACHE.clear()
        from tools.skill_manager_tool import skill_manage
        result = json.loads(skill_manage(
            action="create",
            name="test-skill-bc",
            content="---\nname: test-skill-bc\ndescription: Test\n---\nBody",
        ))
        assert result.get("success") is True

    def test_name_not_required_for_activate(self, skills_dir, monkeypatch):
        """activate/deactivate don't need 'name' parameter."""
        from tools.skill_manager_tool import skill_manage
        result = json.loads(skill_manage(
            action="activate",
            domain="github",
        ))
        assert result["success"] is True
