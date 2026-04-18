"""Tests for auxiliary failover chain: _read_failover_chain, call_llm_failover,
and integration with _generate_summary and flush_memories routing.

TDD flow:
  RED   — these tests define expected behaviour *before* the feature existed.
  GREEN — the implementation in auxiliary_client.py makes them pass.
  REFACTOR — no structural changes needed beyond what's already there.
"""

import os
import time
from unittest.mock import patch, MagicMock, call

import pytest

from agent.auxiliary_client import (
    _read_failover_chain,
    call_llm_failover,
    _normalize_aux_provider,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _make_config(failover_chain):
    """Return a config dict with the given failover_chain."""
    return {"auxiliary": {"failover_chain": failover_chain}}


MOCK_CONFIG_3 = _make_config([
    {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free"},
    {"provider": "openrouter", "model": "glm-4.5-air:free"},
    {"provider": "zai", "model": "glm-5.1"},
])

MOCK_CONFIG_WITH_BASE_URL = _make_config([
    {"provider": "openrouter", "model": "glm-4.5-air:free"},
    {"provider": "custom", "model": "glm-4.5-air",
     "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "api_key": "GLM_API_KEY"},
    {"provider": "zai", "model": "glm-5.1"},
])


# ── _read_failover_chain ─────────────────────────────────────────────

class TestReadFailoverChain:
    """Unit tests for _read_failover_chain config parsing."""

    @patch("hermes_cli.config.load_config")
    def test_returns_empty_when_no_chain(self, mock_load):
        mock_load.return_value = {"auxiliary": {}}
        assert _read_failover_chain() == []

    @patch("hermes_cli.config.load_config")
    def test_returns_entries(self, mock_load):
        mock_load.return_value = MOCK_CONFIG_3
        chain = _read_failover_chain()
        assert len(chain) == 3
        assert chain[0] == {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free"}
        assert chain[2] == {"provider": "zai", "model": "glm-5.1"}

    @patch("hermes_cli.config.load_config")
    def test_includes_base_url_and_api_key(self, mock_load):
        mock_load.return_value = MOCK_CONFIG_WITH_BASE_URL
        chain = _read_failover_chain()
        assert len(chain) == 3
        custom_entry = chain[1]
        assert custom_entry["provider"] == "custom"
        assert custom_entry["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert "api_key" in custom_entry

    @patch("hermes_cli.config.load_config")
    def test_resolves_env_var_api_key(self, mock_load, monkeypatch):
        monkeypatch.setenv("GLM_API_KEY", "real-key-12345")
        mock_load.return_value = _make_config([
            {"provider": "custom", "model": "glm-4.5-air",
             "base_url": "https://open.bigmodel.cn/api/paas/v4",
             "api_key": "GLM_API_KEY"},
        ])
        chain = _read_failover_chain()
        assert chain[0]["api_key"] == "real-key-12345"

    @patch("hermes_cli.config.load_config")
    def test_skips_non_dict_entries(self, mock_load):
        mock_load.return_value = {"auxiliary": {"failover_chain": [
            "not a dict",
            {"provider": "openrouter", "model": "test"},
            42,
        ]}}
        chain = _read_failover_chain()
        assert len(chain) == 1
        assert chain[0]["provider"] == "openrouter"

    @patch("hermes_cli.config.load_config")
    def test_skips_entries_without_provider(self, mock_load):
        mock_load.return_value = {"auxiliary": {"failover_chain": [
            {"model": "no-provider"},
            {"provider": "openrouter", "model": "valid"},
        ]}}
        chain = _read_failover_chain()
        assert len(chain) == 1

    @patch("hermes_cli.config.load_config")
    def test_handles_malformed_config(self, mock_load):
        mock_load.return_value = None
        assert _read_failover_chain() == []

        mock_load.return_value = "not a dict"
        assert _read_failover_chain() == []

        mock_load.return_value = {"auxiliary": "not a dict"}
        assert _read_failover_chain() == []

        mock_load.return_value = {"auxiliary": {"failover_chain": "not a list"}}
        assert _read_failover_chain() == []


# ── _normalize_aux_provider ──────────────────────────────────────────

class TestNormalizeFailover:
    """Ensure 'failover' passes through normalization unchanged."""

    def test_failover_passthrough(self):
        assert _normalize_aux_provider("failover") == "failover"

    def test_failover_case_insensitive(self):
        assert _normalize_aux_provider("FAILOVER") == "failover"

    def test_failover_not_aliased_to_other(self):
        """failover must NOT be normalized to auto, custom, etc."""
        result = _normalize_aux_provider("failover")
        assert result not in ("auto", "custom", "openrouter", "main")


# ── call_llm_failover ────────────────────────────────────────────────

class TestCallLlmFailover:
    """Unit tests for call_llm_failover routing logic."""

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_first_provider_succeeds(self, mock_call, mock_chain):
        mock_chain.return_value = [
            {"provider": "openrouter", "model": "a"},
            {"provider": "zai", "model": "b"},
        ]
        mock_resp = MagicMock()
        mock_call.return_value = mock_resp

        result = call_llm_failover(task="compression", messages=[{"role": "user", "content": "hi"}])

        assert result is mock_resp
        mock_call.assert_called_once()
        # Verify it passed provider and model
        kwargs = mock_call.call_args
        assert kwargs[1]["provider"] == "openrouter"
        assert kwargs[1]["model"] == "a"

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_falls_through_on_failure(self, mock_call, mock_chain):
        mock_chain.return_value = [
            {"provider": "openrouter", "model": "a"},
            {"provider": "zai", "model": "b"},
        ]
        # First call fails, second succeeds
        mock_resp = MagicMock()
        mock_call.side_effect = [Exception("429 rate limited"), mock_resp]

        result = call_llm_failover(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            failover_delay=0,  # no sleep in tests
        )

        assert result is mock_resp
        assert mock_call.call_count == 2

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_raises_last_exception_when_all_fail(self, mock_call, mock_chain):
        mock_chain.return_value = [
            {"provider": "openrouter", "model": "a"},
            {"provider": "zai", "model": "b"},
        ]
        mock_call.side_effect = [
            Exception("429 rate limited"),
            RuntimeError("no provider"),
        ]

        with pytest.raises(RuntimeError, match="no provider"):
            call_llm_failover(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
                failover_delay=0,
            )

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_raises_when_chain_empty(self, mock_call, mock_chain):
        mock_chain.return_value = []

        with pytest.raises(RuntimeError, match="failover_chain is empty"):
            call_llm_failover(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        mock_call.assert_not_called()

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_passes_base_url_and_api_key(self, mock_call, mock_chain):
        mock_chain.return_value = [
            {"provider": "custom", "model": "glm-4.5-air",
             "base_url": "https://open.bigmodel.cn/api/paas/v4",
             "api_key": "real-key"},
        ]
        mock_resp = MagicMock()
        mock_call.return_value = mock_resp

        call_llm_failover(task="compression", messages=[{"role": "user", "content": "hi"}])

        kwargs = mock_call.call_args[1]
        assert kwargs["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert kwargs["api_key"] == "real-key"

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_full_chain_three_failures_then_success(self, mock_call, mock_chain):
        """Realistic scenario: 2 free models fail, 3rd succeeds."""
        mock_chain.return_value = [
            {"provider": "openrouter", "model": "nvidia/nemotron-3-super-120b-a12b:free"},
            {"provider": "openrouter", "model": "glm-4.5-air:free"},
            {"provider": "zai", "model": "glm-5.1"},
        ]
        mock_resp = MagicMock()
        mock_call.side_effect = [
            Exception("Connection error"),
            Exception("429 rate limited"),
            mock_resp,
        ]

        result = call_llm_failover(
            task="compression",
            messages=[{"role": "user", "content": "summarize this"}],
            failover_delay=0,
        )

        assert result is mock_resp
        assert mock_call.call_count == 3
        # Verify provider progression
        providers_called = [c[1]["provider"] for c in mock_call.call_args_list]
        assert providers_called == ["openrouter", "openrouter", "zai"]

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_forwards_all_call_llm_kwargs(self, mock_call, mock_chain):
        """Ensure temperature, max_tokens, tools, etc. are forwarded."""
        mock_chain.return_value = [
            {"provider": "openrouter", "model": "test"},
        ]
        mock_call.return_value = MagicMock()

        call_llm_failover(
            task="flush_memories",
            messages=[{"role": "user", "content": "save"}],
            temperature=0.3,
            max_tokens=5120,
            tools=[{"type": "function", "function": {"name": "memory"}}],
            timeout=30,
            extra_body={"foo": "bar"},
            main_runtime={"model": "glm-5.1", "provider": "zai"},
        )

        kwargs = mock_call.call_args[1]
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 5120
        assert kwargs["tools"] is not None
        assert kwargs["timeout"] == 30
        assert kwargs["extra_body"] == {"foo": "bar"}
        assert kwargs["main_runtime"]["model"] == "glm-5.1"


# ── Integration: _generate_summary routing ───────────────────────────

class TestGenerateSummaryFailoverRouting:
    """Verify _generate_summary routes to call_llm_failover when
    auxiliary.compression.provider is set to 'failover'."""

    @patch("agent.context_compressor._resolve_task_provider_model")
    @patch("agent.context_compressor.call_llm_failover")
    @patch("agent.context_compressor.call_llm")
    def test_uses_failover_when_configured(self, mock_call_llm, mock_failover, mock_resolve):
        from agent.context_compressor import ContextCompressor

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "## Summary\nTest summary content"
        mock_failover.return_value = mock_resp

        comp = ContextCompressor(
            model="test-model",
            config_context_length=128000,
            threshold_percent=0.5,
        )
        result = comp._generate_summary([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ])

        assert result is not None
        mock_failover.assert_called_once()
        mock_call_llm.assert_not_called()

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    @patch("agent.auxiliary_client.call_llm")
    def test_uses_normal_call_llm_when_not_failover(self, mock_call_llm, mock_resolve):
        from agent.context_compressor import ContextCompressor

        mock_resolve.return_value = ("openrouter", None, None, None, None)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "## Summary\nNormal path"
        mock_call_llm.return_value = mock_resp

        comp = ContextCompressor(
            model="test-model",
            config_context_length=128000,
            threshold_percent=0.5,
        )
        result = comp._generate_summary([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ])

        assert result is not None
        mock_call_llm.assert_called_once()
