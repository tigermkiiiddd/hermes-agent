"""Tests for call_llm auto-routing when provider resolves to 'failover'.

Bug: call_llm(task="compression") with auxiliary.compression.provider=failover
     passes "failover" to resolve_provider_client which doesn't know it,
     causing "unknown provider 'failover'" warnings and failed compression.

Fix: call_llm should detect resolved_provider=="failover" and automatically
     delegate to call_llm_failover instead of trying to create a client for
     the literal "failover" provider.

TDD flow: RED -> GREEN -> REFACTOR
"""

from unittest.mock import patch, MagicMock, call
import pytest


class TestCallLlmFailoverAutoRouting:
    """call_llm should transparently route to call_llm_failover when
    the resolved provider is 'failover'.
    """

    @patch("agent.auxiliary_client.call_llm_failover")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_routes_to_call_llm_failover_when_provider_is_failover(
        self, mock_resolve, mock_failover
    ):
        """call_llm(task="compression") with failover provider should
        delegate to call_llm_failover, NOT to resolve_provider_client."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_failover.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="summary text"))]
        )

        msgs = [{"role": "user", "content": "summarize this"}]
        result = call_llm(task="compression", messages=msgs, temperature=0.3)

        # Should have called call_llm_failover, not resolve_provider_client
        mock_failover.assert_called_once()
        call_kwargs = mock_failover.call_args
        assert call_kwargs[1]["task"] == "compression" or call_kwargs[0][0] == "compression" if call_kwargs[0] else call_kwargs[1].get("task") == "compression"

    @patch("agent.auxiliary_client.call_llm_failover")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_forwards_all_kwargs_to_failover(self, mock_resolve, mock_failover):
        """All original kwargs should be forwarded to call_llm_failover."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_failover.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        msgs = [{"role": "user", "content": "test"}]
        call_llm(
            task="compression",
            messages=msgs,
            temperature=0.5,
            max_tokens=2048,
        )

        mock_failover.assert_called_once()
        kw = mock_failover.call_args[1]
        assert kw["messages"] == msgs
        assert kw["temperature"] == 0.5
        assert kw["max_tokens"] == 2048

    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_does_not_route_when_provider_is_not_failover(self, mock_resolve):
        """Normal providers should NOT be routed to call_llm_failover."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("zai", "glm-5.1", None, None, None)

        # We need to mock _get_cached_client to avoid actually hitting APIs
        with patch("agent.auxiliary_client._get_cached_client") as mock_client:
            mock_client.return_value = (
                MagicMock(),
                "glm-5.1",
            )
            call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
            # Should have called _get_cached_client, not call_llm_failover
            mock_client.assert_called_once()

    @patch("agent.auxiliary_client.call_llm_failover")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_failover_error_propagates(self, mock_resolve, mock_failover):
        """If call_llm_failover raises, the error should propagate."""
        from agent.auxiliary_client import call_llm

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_failover.side_effect = RuntimeError("All providers in chain failed")

        with pytest.raises(RuntimeError, match="All providers in chain failed"):
            call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])


class TestAsyncCallLlmFailoverAutoRouting:
    """async_call_llm should also route to call_llm_failover when
    provider is 'failover'.
    """

    @pytest.mark.asyncio
    @patch("agent.auxiliary_client.call_llm_failover")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    async def test_async_routes_to_failover(self, mock_resolve, mock_failover):
        from agent.auxiliary_client import async_call_llm

        mock_resolve.return_value = ("failover", None, None, None, None)
        mock_failover.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="async summary"))]
        )

        msgs = [{"role": "user", "content": "summarize"}]
        result = await async_call_llm(
            task="session_search", messages=msgs, temperature=0.1
        )

        mock_failover.assert_called_once()
