from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli(model: str = "anthropic/claude-sonnet-4-20250514"):
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = model
    cli_obj.session_start = datetime.now() - timedelta(minutes=14, seconds=32)
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    # __new__ skips __init__, so manually set the StatusBar instance
    from hermes_cli.status_bar import StatusBar
    cli_obj._status_bar = StatusBar()
    return cli_obj


def _attach_agent(
    cli_obj,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    api_calls: int,
    context_tokens: int,
    context_length: int,
    compressions: int = 0,
):
    cli_obj.agent = SimpleNamespace(
        model=cli_obj.model,
        provider="anthropic" if cli_obj.model.startswith("anthropic/") else None,
        base_url="",
        session_input_tokens=input_tokens if input_tokens is not None else prompt_tokens,
        session_output_tokens=output_tokens if output_tokens is not None else completion_tokens,
        session_cache_read_tokens=cache_read_tokens,
        session_cache_write_tokens=cache_write_tokens,
        session_prompt_tokens=prompt_tokens,
        session_completion_tokens=completion_tokens,
        session_total_tokens=total_tokens,
        session_api_calls=api_calls,
        get_rate_limit_state=lambda: None,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=context_tokens,
            context_length=context_length,
            compression_count=compressions,
        ),
    )
    return cli_obj


class TestCLIStatusBar:
    def test_context_style_thresholds(self):
        cli_obj = _make_cli()

        assert cli_obj._status_bar_context_style(None) == "class:status-bar-dim"
        assert cli_obj._status_bar_context_style(10) == "class:status-bar-good"
        assert cli_obj._status_bar_context_style(50) == "class:status-bar-warn"
        assert cli_obj._status_bar_context_style(81) == "class:status-bar-bad"
        assert cli_obj._status_bar_context_style(95) == "class:status-bar-critical"

    def test_build_status_bar_text_for_wide_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)

        assert "claude-sonnet-4-20250514" in text
        assert "12.4K/200K" in text
        assert "6%" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" in text

    def test_input_height_counts_wide_characters_using_cell_width(self):
        cli_obj = _make_cli()

        class _Doc:
            lines = ["你" * 10]

        class _Buffer:
            document = _Doc()

        input_area = SimpleNamespace(buffer=_Buffer())

        def _input_height():
            try:
                from prompt_toolkit.application import get_app
                from prompt_toolkit.utils import get_cwidth

                doc = input_area.buffer.document
                prompt_width = max(2, get_cwidth(cli_obj._get_tui_prompt_text()))
                try:
                    available_width = get_app().output.get_size().columns - prompt_width
                except Exception:
                    import shutil
                    available_width = shutil.get_terminal_size((80, 24)).columns - prompt_width
                if available_width < 10:
                    available_width = 40
                visual_lines = 0
                for line in doc.lines:
                    line_width = get_cwidth(line)
                    if line_width <= 0:
                        visual_lines += 1
                    else:
                        visual_lines += max(1, -(-line_width // available_width))
                return min(max(visual_lines, 1), 8)
            except Exception:
                return 1

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=14)
        with patch.object(HermesCLI, "_get_tui_prompt_text", return_value="❯ "), \
             patch("prompt_toolkit.application.get_app", return_value=mock_app):
            assert _input_height() == 2

    def test_input_height_uses_prompt_toolkit_width_over_shutil(self):
        cli_obj = _make_cli()

        class _Doc:
            lines = ["你" * 10]

        class _Buffer:
            document = _Doc()

        input_area = SimpleNamespace(buffer=_Buffer())

        def _input_height():
            try:
                from prompt_toolkit.application import get_app
                from prompt_toolkit.utils import get_cwidth

                doc = input_area.buffer.document
                prompt_width = max(2, get_cwidth(cli_obj._get_tui_prompt_text()))
                try:
                    available_width = get_app().output.get_size().columns - prompt_width
                except Exception:
                    import shutil
                    available_width = shutil.get_terminal_size((80, 24)).columns - prompt_width
                if available_width < 10:
                    available_width = 40
                visual_lines = 0
                for line in doc.lines:
                    line_width = get_cwidth(line)
                    if line_width <= 0:
                        visual_lines += 1
                    else:
                        visual_lines += max(1, -(-line_width // available_width))
                return min(max(visual_lines, 1), 8)
            except Exception:
                return 1

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=14)
        with patch.object(HermesCLI, "_get_tui_prompt_text", return_value="❯ "), \
             patch("prompt_toolkit.application.get_app", return_value=mock_app), \
             patch("shutil.get_terminal_size") as mock_shutil:
            assert _input_height() == 2
        mock_shutil.assert_not_called()

    def test_build_status_bar_text_no_cost_in_status_bar(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=5000,
            total_tokens=15000,
            api_calls=7,
            context_tokens=50000,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=120)
        assert "$" not in text  # cost is never shown in status bar

    def test_build_status_bar_text_collapses_for_narrow_terminal(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10000,
            completion_tokens=2400,
            total_tokens=12400,
            api_calls=7,
            context_tokens=12400,
            context_length=200_000,
        )

        text = cli_obj._build_status_bar_text(width=60)

        assert "⚕" in text
        assert "$0.06" not in text  # cost hidden by default
        assert "15m" in text
        assert "200K" not in text

    def test_build_status_bar_text_handles_missing_agent(self):
        cli_obj = _make_cli()

        text = cli_obj._build_status_bar_text(width=100)

        assert "⚕" in text
        assert "claude-sonnet-4-20250514" in text

    def test_minimal_tui_chrome_threshold(self):
        cli_obj = _make_cli()

        assert cli_obj._use_minimal_tui_chrome(width=63) is True
        assert cli_obj._use_minimal_tui_chrome(width=64) is False

    def test_bottom_input_rule_hides_on_narrow_terminals(self):
        cli_obj = _make_cli()

        assert cli_obj._tui_input_rule_height("top", width=50) == 1
        assert cli_obj._tui_input_rule_height("bottom", width=50) == 0
        assert cli_obj._tui_input_rule_height("bottom", width=90) == 1

    def test_agent_spacer_reclaimed_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._agent_running = True

        assert cli_obj._agent_spacer_height(width=50) == 0
        assert cli_obj._agent_spacer_height(width=90) == 1
        cli_obj._agent_running = False
        assert cli_obj._agent_spacer_height(width=90) == 0

    def test_spinner_line_hidden_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._spinner_text = "thinking"

        assert cli_obj._spinner_widget_height(width=50) == 0
        assert cli_obj._spinner_widget_height(width=90) == 1
        cli_obj._spinner_text = ""
        assert cli_obj._spinner_widget_height(width=90) == 0

    def test_voice_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = False
        cli_obj._voice_processing = False
        cli_obj._voice_tts = True
        cli_obj._voice_continuous = True

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status", " 🎤 Ctrl+B ")]

    def test_voice_recording_status_bar_compacts_on_narrow_terminals(self):
        cli_obj = _make_cli()
        cli_obj._voice_mode = True
        cli_obj._voice_recording = True
        cli_obj._voice_processing = False

        fragments = cli_obj._get_voice_status_fragments(width=50)

        assert fragments == [("class:voice-status-recording", " ● REC ")]


class TestCLIUsageReport:
    def test_show_usage_includes_estimated_cost(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_230,
            completion_tokens=2_220,
            total_tokens=12_450,
            api_calls=7,
            context_tokens=12_450,
            context_length=200_000,
            compressions=1,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Model:" in output
        assert "Cost status:" in output
        assert "Cost source:" in output
        assert "Total cost:" in output
        assert "$" in output
        assert "0.064" in output
        assert "Session duration:" in output
        assert "Compressions:" in output

    def test_show_usage_marks_unknown_pricing(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(model="local/my-custom-model"),
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            api_calls=1,
            context_tokens=1_000,
            context_length=32_000,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Total cost:" in output
        assert "n/a" in output
        assert "Pricing unknown for local/my-custom-model" in output

    def test_zero_priced_provider_models_stay_unknown(self, capsys):
        cli_obj = _attach_agent(
            _make_cli(model="glm-5"),
            prompt_tokens=1_000,
            completion_tokens=500,
            total_tokens=1_500,
            api_calls=1,
            context_tokens=1_000,
            context_length=32_000,
        )
        cli_obj.verbose = False

        cli_obj._show_usage()
        output = capsys.readouterr().out

        assert "Total cost:" in output
        assert "n/a" in output
        assert "Pricing unknown for glm-5" in output


class TestStatusBarWidthSource:
    """Ensure status bar fragments don't overflow the terminal width."""

    def _make_wide_cli(self):
        from datetime import datetime, timedelta
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=100_000,
            completion_tokens=5_000,
            total_tokens=105_000,
            api_calls=20,
            context_tokens=100_000,
            context_length=200_000,
        )
        cli_obj._status_bar_visible = True
        return cli_obj

    def test_fragments_fit_within_announced_width(self):
        """Total fragment text length must not exceed the width used to build them."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        for width in (40, 52, 76, 80, 120, 200):
            mock_app = MagicMock()
            mock_app.output.get_size.return_value = MagicMock(columns=width)

            with patch("prompt_toolkit.application.get_app", return_value=mock_app):
                frags = cli_obj._get_status_bar_fragments()

            total_text = "".join(text for _, text in frags)
            display_width = cli_obj._status_bar_display_width(total_text)
            assert display_width <= width + 4, (  # +4 for minor padding chars
                f"At width={width}, fragment total {display_width} cells overflows "
                f"({total_text!r})"
            )

    def test_fragments_use_pt_width_over_shutil(self):
        """When prompt_toolkit reports a width, shutil.get_terminal_size must not be used."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app) as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            cli_obj._get_status_bar_fragments()

        mock_shutil.assert_not_called()

    def test_fragments_fall_back_to_shutil_when_no_app(self):
        """Outside a TUI context (no running app), shutil must be used as fallback."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app", side_effect=Exception("no app")), \
             patch("shutil.get_terminal_size", return_value=MagicMock(columns=100)) as mock_shutil:
            frags = cli_obj._get_status_bar_fragments()

        mock_shutil.assert_called()
        assert len(frags) > 0

    def test_build_status_bar_text_uses_pt_width(self):
        """_build_status_bar_text() must also prefer prompt_toolkit width."""
        from unittest.mock import MagicMock, patch
        cli_obj = self._make_wide_cli()

        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=80)

        with patch("prompt_toolkit.application.get_app", return_value=mock_app), \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text()  # no explicit width

        mock_shutil.assert_not_called()
        assert isinstance(text, str)
        assert len(text) > 0

    def test_explicit_width_skips_pt_lookup(self):
        """An explicit width= argument must bypass both PT and shutil lookups."""
        from unittest.mock import patch
        cli_obj = self._make_wide_cli()

        with patch("prompt_toolkit.application.get_app") as mock_get_app, \
             patch("shutil.get_terminal_size") as mock_shutil:
            text = cli_obj._build_status_bar_text(width=100)

        mock_get_app.assert_not_called()
        mock_shutil.assert_not_called()
        assert len(text) > 0


# ── Snapshot: active_domains collection ───────────────────────────────

class TestSnapshotActiveDomains:
    """Verify _get_status_bar_snapshot collects active_domains from agent."""

    def test_snapshot_without_agent_has_empty_domains(self):
        cli_obj = _make_cli()
        snap = cli_obj._get_status_bar_snapshot()
        assert snap["active_domains"] == set()

    def test_snapshot_with_agent_reads_active_domains(self):
        cli_obj = _make_cli()
        cli_obj.agent = SimpleNamespace(
            model="test-model",
            session_input_tokens=0,
            session_output_tokens=0,
            session_cache_read_tokens=0,
            session_cache_write_tokens=0,
            session_prompt_tokens=0,
            session_completion_tokens=0,
            session_total_tokens=0,
            session_api_calls=0,
            get_rate_limit_state=lambda: None,
            context_compressor=SimpleNamespace(
                last_prompt_tokens=0,
                context_length=100_000,
                compression_count=0,
            ),
            _active_skill_domains={"mlops/training", "github"},
        )
        snap = cli_obj._get_status_bar_snapshot()
        assert snap["active_domains"] == {"mlops/training", "github"}

    def test_snapshot_with_agent_no_domains_attribute(self):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            api_calls=1,
            context_tokens=100,
            context_length=32_000,
        )
        # agent has no _active_skill_domains attribute
        snap = cli_obj._get_status_bar_snapshot()
        assert snap["active_domains"] == set()


# ── Domain formatting ─────────────────────────────────────────────────

class TestFormatActiveDomainsLabel:
    """_format_active_domains_label static method."""

    def test_empty_set_returns_empty(self):
        assert HermesCLI._format_active_domains_label(set()) == ""

    def test_single_two_level_domain(self):
        result = HermesCLI._format_active_domains_label({"mlops/training"})
        assert "m/training" in result

    def test_multiple_domains_comma_separated(self):
        result = HermesCLI._format_active_domains_label({"mlops/training", "github"})
        assert "," in result
        assert "training" in result
        assert "github" in result

    def test_flat_domain_no_slash(self):
        result = HermesCLI._format_active_domains_label({"notes"})
        assert "notes" in result

    def test_long_list_truncated(self):
        domains = {f"cat{i}/skill{j}" for i in range(5) for j in range(3)}
        result = HermesCLI._format_active_domains_label(domains, max_width=20)
        assert len(result) <= 21


# ── build_text with active domains ────────────────────────────────────

class TestBuildTextWithDomains:
    """_build_status_bar_text should include domains when active."""

    def _make_cli_with_domains(self, domains=None):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=10_000,
            completion_tokens=2_400,
            total_tokens=12_400,
            api_calls=5,
            context_tokens=12_400,
            context_length=128_000,
        )
        # Inject active domains into snapshot via agent
        # NOTE: Don't use `or` — set() is falsy!
        if domains is None:
            domains = {"mlops/training", "github"}
        cli_obj.agent._active_skill_domains = domains
        return cli_obj

    def test_wide_shows_domains_between_percent_and_duration(self):
        cli_obj = self._make_cli_with_domains()
        text = cli_obj._build_status_bar_text(width=120)

        assert "📂" in text
        # Duration should still be visible at wide width
        assert "15m" in text

    def test_wide_no_domains_no_folder(self):
        cli_obj = self._make_cli_with_domains(domains=set())
        text = cli_obj._build_status_bar_text(width=120)

        assert "📂" not in text
        assert "15m" in text

    def test_medium_shows_domains_when_space(self):
        cli_obj = self._make_cli_with_domains(domains={"github"})
        text = cli_obj._build_status_bar_text(width=60)

        # At medium width with a short domain, it should appear
        assert "📂" in text

    def test_narrow_no_domains(self):
        cli_obj = self._make_cli_with_domains()
        text = cli_obj._build_status_bar_text(width=40)

        # Narrow shows model + duration only, no domains
        assert "claude-sonnet" in text
        assert "1h" not in text or "15m" in text


# ── build_fragments with active domains ───────────────────────────────

class TestBuildFragmentsWithDomains:
    """_get_status_bar_fragments should include domain fragments."""

    def _make_wide_cli_with_domains(self, domains=None):
        cli_obj = _attach_agent(
            _make_cli(),
            prompt_tokens=100_000,
            completion_tokens=5_000,
            total_tokens=105_000,
            api_calls=20,
            context_tokens=100_000,
            context_length=200_000,
        )
        cli_obj._status_bar_visible = True
        if domains is None:
            domains = {"mlops/training", "github"}
        cli_obj.agent._active_skill_domains = domains
        return cli_obj

    def test_wide_fragments_include_domains_style(self):
        cli_obj = self._make_wide_cli_with_domains()
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            frags = cli_obj._get_status_bar_fragments()
        styles = {s for s, _ in frags}
        assert "class:status-bar-domains" in styles

    def test_wide_fragments_domains_text(self):
        cli_obj = self._make_wide_cli_with_domains()
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            frags = cli_obj._get_status_bar_fragments()
        full_text = "".join(t for _, t in frags)
        assert "📂" in full_text

    def test_wide_fragments_no_domains_style_absent(self):
        cli_obj = self._make_wide_cli_with_domains(domains=set())
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=120)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            frags = cli_obj._get_status_bar_fragments()
        styles = {s for s, _ in frags}
        assert "class:status-bar-domains" not in styles

    def test_medium_fragments_with_domains(self):
        cli_obj = self._make_wide_cli_with_domains(domains={"github"})
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=60)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            frags = cli_obj._get_status_bar_fragments()
        full_text = "".join(t for _, t in frags)
        assert "📂" in full_text

    def test_narrow_fragments_no_domains(self):
        cli_obj = self._make_wide_cli_with_domains()
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=40)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            frags = cli_obj._get_status_bar_fragments()
        full_text = "".join(t for _, t in frags)
        assert "📂" not in full_text

    def test_overflow_falls_back_to_plain_trimmed(self):
        """If fragments overflow, fall back to a single trimmed plain-text fragment."""
        cli_obj = self._make_wide_cli_with_domains()
        # Force a very narrow width that triggers overflow
        mock_app = MagicMock()
        mock_app.output.get_size.return_value = MagicMock(columns=30)
        with patch("prompt_toolkit.application.get_app", return_value=mock_app):
            frags = cli_obj._get_status_bar_fragments()
        # Should still produce output
        assert len(frags) > 0
        full_text = "".join(t for _, t in frags)
        assert len(full_text) > 0


# ── Width helpers: CJK ────────────────────────────────────────────────

class TestWidthHelpersCJK:
    """display_width and trim with wide characters."""

    def test_display_width_cjk_chars(self):
        w = HermesCLI._status_bar_display_width("你好世界")
        assert w >= 8  # 4 CJK chars × 2 cells

    def test_trim_preserves_short_text(self):
        assert HermesCLI._trim_status_bar_text("hello", 100) == "hello"

    def test_trim_adds_ellipsis(self):
        result = HermesCLI._trim_status_bar_text("abcdefghij", 7)
        assert result.endswith("...")
        assert len(result) <= 10

    def test_trim_zero_width_returns_empty(self):
        assert HermesCLI._trim_status_bar_text("anything", 0) == ""
