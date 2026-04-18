"""Tests for TUI status display during compression and memory flush.

Covers:
  - status_fn callback in call_llm_failover
  - status_fn propagation through ContextCompressor.compress → _generate_summary
  - _compress_context emits status messages with provider/model info
  - Non-failover path also shows provider/model

TDD flow: RED → GREEN → REFACTOR
"""

from unittest.mock import patch, MagicMock, call

import pytest


# ── call_llm_failover status_fn ──────────────────────────────────────

class TestFailoverStatusCallback:
    """Verify call_llm_failover calls status_fn before each attempt."""

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_status_fn_called_on_first_attempt(self, mock_call, mock_chain):
        from agent.auxiliary_client import call_llm_failover

        mock_chain.return_value = [
            {"provider": "openrouter", "model": "test-model"},
        ]
        mock_call.return_value = MagicMock()
        status_fn = MagicMock()

        call_llm_failover(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            status_fn=status_fn,
        )

        status_fn.assert_called_once_with("openrouter", "test-model", 1, 1)

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_status_fn_called_on_each_failover(self, mock_call, mock_chain):
        from agent.auxiliary_client import call_llm_failover

        mock_chain.return_value = [
            {"provider": "openrouter", "model": "model-a"},
            {"provider": "zai", "model": "model-b"},
        ]
        mock_call.side_effect = [Exception("fail"), MagicMock()]
        status_fn = MagicMock()

        call_llm_failover(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            failover_delay=0,
            status_fn=status_fn,
        )

        assert status_fn.call_count == 2
        status_fn.assert_any_call("openrouter", "model-a", 1, 2)
        status_fn.assert_any_call("zai", "model-b", 2, 2)

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_status_fn_exception_does_not_break_failover(self, mock_call, mock_chain):
        from agent.auxiliary_client import call_llm_failover

        mock_chain.return_value = [
            {"provider": "openrouter", "model": "test"},
        ]
        mock_call.return_value = MagicMock()
        status_fn = MagicMock(side_effect=RuntimeError("TUI crashed"))

        # Should NOT raise — status_fn errors are swallowed
        result = call_llm_failover(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            status_fn=status_fn,
        )
        assert result is not None

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client.call_llm")
    def test_status_fn_none_is_safe(self, mock_call, mock_chain):
        from agent.auxiliary_client import call_llm_failover

        mock_chain.return_value = [
            {"provider": "openrouter", "model": "test"},
        ]
        mock_call.return_value = MagicMock()

        # Should not crash with status_fn=None (default)
        result = call_llm_failover(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert result is not None


# ── ContextCompressor.compress status_fn ─────────────────────────────

class TestCompressorStatusPropagation:
    """Verify compress() forwards status_fn to _generate_summary."""

    @patch("agent.context_compressor._resolve_task_provider_model")
    @patch("agent.context_compressor.call_llm_failover")
    @patch("agent.context_compressor.call_llm")
    def test_failover_path_receives_status_fn(self, mock_call_llm, mock_failover, mock_resolve):
        from agent.context_compressor import ContextCompressor

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "## Summary\nTest"
        mock_failover.return_value = mock_resp
        status_fn = MagicMock()

        comp = ContextCompressor(
            model="test-model",
            config_context_length=128000,
            threshold_percent=0.5,
        )

        # Build messages long enough to trigger compression
        messages = [{"role": "system", "content": "sys"}]
        messages.append({"role": "user", "content": "hello"})
        messages.append({"role": "assistant", "content": "hi"})
        # Add enough messages to have middle turns to compress
        for i in range(10):
            messages.append({"role": "user", "content": f"msg {i}"})
            messages.append({"role": "assistant", "content": f"reply {i}"})

        comp.compress(messages, status_fn=status_fn)

        # status_fn should have been called via call_llm_failover
        mock_failover.assert_called_once()
        call_kwargs = mock_failover.call_args[1]
        assert call_kwargs.get("status_fn") is status_fn

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    @patch("agent.auxiliary_client.call_llm")
    def test_normal_path_calls_status_fn(self, mock_call_llm, mock_resolve):
        from agent.context_compressor import ContextCompressor

        mock_resolve.return_value = ("openrouter", None, None, None, None)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "## Summary\nTest"
        mock_call_llm.return_value = mock_resp
        status_fn = MagicMock()

        comp = ContextCompressor(
            model="test-model",
            config_context_length=128000,
            threshold_percent=0.5,
        )

        messages = [{"role": "system", "content": "sys"}]
        messages.append({"role": "user", "content": "hello"})
        messages.append({"role": "assistant", "content": "hi"})
        for i in range(10):
            messages.append({"role": "user", "content": f"msg {i}"})
            messages.append({"role": "assistant", "content": f"reply {i}"})

        comp.compress(messages, status_fn=status_fn)

        # Non-failover path should call status_fn with resolved provider/model
        status_fn.assert_called_once()
        args = status_fn.call_args[0]
        assert args[0] == "openrouter"  # provider
        assert args[2] == 1  # attempt
        assert args[3] == 1  # total


# ── _compress_context emits status ───────────────────────────────────

class TestCompressContextStatus:
    """Verify _compress_context calls _emit_status with model info."""

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    @patch("agent.auxiliary_client.call_llm")
    def test_emits_compacting_status(self, mock_call_llm, mock_resolve):
        """_compress_context should emit status showing compression in progress."""
        from agent.context_compressor import ContextCompressor

        mock_resolve.return_value = ("openrouter", None, None, None, None)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "## Summary\nTest"
        mock_call_llm.return_value = mock_resp

        # We test at the compress() level — _compress_context wraps it
        # The key assertion: status_fn receives (provider, model, attempt, total)
        received = []
        status_fn = lambda prov, mdl, att, tot: received.append((prov, mdl, att, tot))

        comp = ContextCompressor(
            model="test-model",
            config_context_length=128000,
            threshold_percent=0.5,
        )

        messages = [{"role": "system", "content": "sys"}]
        messages.append({"role": "user", "content": "hello"})
        messages.append({"role": "assistant", "content": "hi"})
        for i in range(10):
            messages.append({"role": "user", "content": f"msg {i}"})
            messages.append({"role": "assistant", "content": f"reply {i}"})

        comp.compress(messages, status_fn=status_fn)

        assert len(received) >= 1
        prov, mdl, att, tot = received[0]
        assert prov is not None
        assert att == 1
        assert tot == 1
