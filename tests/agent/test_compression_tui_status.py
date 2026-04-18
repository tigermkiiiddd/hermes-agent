"""Tests for TUI status display during compression and memory flush.

Covers:
  - status_fn callback in call_llm_failover
  - status_fn propagation through ContextCompressor.compress → call_llm
  - _compress_context emits status messages with provider/model info
  - Both failover and non-failover paths show provider/model

After refactor: context_compressor no longer branches on failover.
It passes status_fn to call_llm which handles routing internally.

TDD flow: RED -> GREEN -> REFACTOR
"""

from unittest.mock import patch, MagicMock, call

import pytest


# -- call_llm_failover status_fn -----------------------------------------

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
        bad_fn = MagicMock(side_effect=ValueError("boom"))

        result = call_llm_failover(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            status_fn=bad_fn,
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


# -- ContextCompressor.compress status_fn (unified via call_llm) ---------

class TestCompressorStatusPropagation:
    """Verify compress() forwards status_fn to call_llm."""

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    @patch("agent.auxiliary_client.call_llm_failover")
    def test_failover_path_receives_status_fn(self, mock_failover, mock_resolve):
        """When provider=failover, status_fn should reach call_llm_failover."""
        from agent.context_compressor import ContextCompressor

        # First call resolves compression provider to failover
        # Second call resolves again inside call_llm
        mock_resolve.side_effect = [
            ("failover", None, None, None, None),
            ("failover", None, None, None, None),
        ]
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

        messages = [{"role": "system", "content": "sys"}]
        messages.append({"role": "user", "content": "hello"})
        messages.append({"role": "assistant", "content": "hi"})
        for i in range(10):
            messages.append({"role": "user", "content": f"msg {i}"})
            messages.append({"role": "assistant", "content": f"reply {i}"})

        comp.compress(messages, status_fn=status_fn)

        # status_fn should have been forwarded through call_llm -> call_llm_failover
        mock_failover.assert_called_once()
        call_kwargs = mock_failover.call_args[1]
        assert call_kwargs.get("status_fn") is status_fn

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    @patch("agent.auxiliary_client._get_cached_client")
    def test_normal_path_calls_status_fn(self, mock_client, mock_resolve):
        """Non-failover path: call_llm calls status_fn internally."""
        from agent.context_compressor import ContextCompressor

        mock_resolve.return_value = ("openrouter", "test-model", None, None, None)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "## Summary\nTest"

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_resp
        mock_client.return_value = (mock_openai, "test-model")

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

        # status_fn called by call_llm for non-failover path
        status_fn.assert_called()
        args = status_fn.call_args[0]
        assert args[0] is not None  # provider
        assert args[2] == 1  # attempt
        assert args[3] == 1  # total


# -- _compress_context emits status --------------------------------------

class TestCompressContextStatus:
    """Verify _compress_context calls status_fn with model info."""

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    @patch("agent.auxiliary_client._get_cached_client")
    def test_emits_compacting_status(self, mock_client, mock_resolve):
        """_compress_context should emit status showing compression in progress."""
        from agent.context_compressor import ContextCompressor

        mock_resolve.return_value = ("openrouter", "test-model", None, None, None)
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "## Summary\nTest"

        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = mock_resp
        mock_client.return_value = (mock_openai, "test-model")

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
