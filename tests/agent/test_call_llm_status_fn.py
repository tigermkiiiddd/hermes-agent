"""Tests for call_llm status_fn parameter and unified failover routing.

Refactor plan:
  1. call_llm / async_call_llm gain status_fn parameter
  2. When provider=failover, call_llm forwards status_fn to call_llm_failover
  3. When provider!=failover, call_llm calls status_fn directly (single attempt)
  4. context_compressor no longer needs manual failover branching

TDD flow: RED -> GREEN -> REFACTOR
"""

from unittest.mock import patch, MagicMock, call
import pytest


class TestCallLlmStatusFn:
    """call_llm should accept and use status_fn for both failover and
    non-failover providers."""

    # ── Non-failover path: status_fn called once ───────────────────

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_status_fn_called_for_non_failover(self, mock_resolve):
        """Non-failover path should call status_fn(provider, model, 1, 1)."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("zai", "glm-5.1", None, None, None)
        status_fn = MagicMock()

        with patch("agent.auxiliary_client._get_cached_client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_client.return_value = (MagicMock(), "glm-5.1")

            # Need to mock the actual API call
            with patch.object(mock_client.return_value[0].chat.completions, "create") as mock_create:
                mock_create.return_value = mock_resp
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                    status_fn=status_fn,
                )

        status_fn.assert_called_once_with("zai", "glm-5.1", 1, 1)

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_status_fn_none_is_safe_non_failover(self, mock_resolve):
        """status_fn=None should not crash for non-failover path."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("zai", "glm-5.1", None, None, None)

        with patch("agent.auxiliary_client._get_cached_client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_client.return_value = (MagicMock(), "glm-5.1")

            with patch.object(mock_client.return_value[0].chat.completions, "create") as mock_create:
                mock_create.return_value = mock_resp
                # Should not raise
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                    status_fn=None,
                )

    # ── Failover path: status_fn forwarded to call_llm_failover ────

    @patch("agent.auxiliary_client.call_llm_failover")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_status_fn_forwarded_to_failover(self, mock_resolve, mock_failover):
        """When provider=failover, status_fn should be forwarded to call_llm_failover."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_failover.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        status_fn = MagicMock()

        call_llm(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            status_fn=status_fn,
        )

        mock_failover.assert_called_once()
        kw = mock_failover.call_args[1]
        assert kw["status_fn"] is status_fn

    @patch("agent.auxiliary_client.call_llm_failover")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_status_fn_none_forwarded_to_failover(self, mock_resolve, mock_failover):
        """status_fn=None with failover should pass None through."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_failover.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        call_llm(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            status_fn=None,
        )

        mock_failover.assert_called_once()
        kw = mock_failover.call_args[1]
        assert kw.get("status_fn") is None

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_status_fn_exception_suppressed_non_failover(self, mock_resolve):
        """If status_fn raises, call_llm should still succeed."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("zai", "glm-5.1", None, None, None)
        bad_fn = MagicMock(side_effect=ValueError("boom"))

        with patch("agent.auxiliary_client._get_cached_client") as mock_client:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_client.return_value = (MagicMock(), "glm-5.1")

            with patch.object(mock_client.return_value[0].chat.completions, "create") as mock_create:
                mock_create.return_value = mock_resp
                result = call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hi"}],
                    status_fn=bad_fn,
                )
                assert result is not None


class TestAsyncCallLlmStatusFn:
    """async_call_llm should also support status_fn."""

    @pytest.mark.asyncio
    @patch("agent.auxiliary_client.call_llm_failover")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    async def test_status_fn_forwarded_to_failover(self, mock_resolve, mock_failover):
        from agent.auxiliary_client import async_call_llm

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_failover.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )
        status_fn = MagicMock()

        await async_call_llm(
            task="compression",
            messages=[{"role": "user", "content": "hi"}],
            status_fn=status_fn,
        )

        mock_failover.assert_called_once()
        kw = mock_failover.call_args[1]
        assert kw["status_fn"] is status_fn
