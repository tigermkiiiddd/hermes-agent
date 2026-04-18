"""Tests for resolve_provider_client handling 'failover' provider.

Bug: resolve_provider_client("failover", ...) logs "unknown provider 'failover'"
     and returns (None, None), which causes:
     - Startup warning about no auxiliary provider
     - get_text_auxiliary_client / get_async_text_auxiliary_client returning None
     - trajectory_compressor failing to detect provider

Fix: resolve_provider_client should recognize "failover" and resolve the first
     entry in the failover chain, so callers get a real client back.

TDD flow: RED -> GREEN -> REFACTOR
"""

from unittest.mock import patch, MagicMock
import pytest


class TestResolveProviderClientFailover:
    """resolve_provider_client should handle 'failover' by resolving
    the first valid entry in the failover chain."""

    @patch("agent.auxiliary_client._read_failover_chain")
    def test_failover_resolves_first_chain_entry(self, mock_chain):
        """When provider='failover', should resolve to the first chain entry's
        provider and return a valid client."""
        from agent.auxiliary_client import resolve_provider_client

        mock_chain.return_value = [
            {"provider": "openrouter", "model": "test-model-a"},
            {"provider": "zai", "model": "test-model-b"},
        ]

        # Mock the actual client creation for "openrouter"
        with patch("agent.auxiliary_client._try_openrouter") as mock_try:
            mock_client = MagicMock()
            mock_client.base_url = "https://openrouter.ai/api/v1"
            mock_try.return_value = (mock_client, "test-model-a")

            client, model = resolve_provider_client("failover", "test-model-a")

            assert client is not None
            assert model == "test-model-a"

    @patch("agent.auxiliary_client._read_failover_chain")
    def test_failover_empty_chain_returns_none(self, mock_chain):
        """Empty failover chain should return (None, None), not crash."""
        from agent.auxiliary_client import resolve_provider_client

        mock_chain.return_value = []
        client, model = resolve_provider_client("failover", "some-model")

        assert client is None
        assert model is None

    @patch("agent.auxiliary_client._read_failover_chain")
    def test_failover_first_entry_fails_tries_second(self, mock_chain):
        """If first chain entry has no credentials, try the next one."""
        from agent.auxiliary_client import resolve_provider_client

        mock_chain.return_value = [
            {"provider": "openrouter", "model": "model-a"},
            {"provider": "nous", "model": "model-b"},
        ]

        with patch("agent.auxiliary_client._try_openrouter") as mock_or, \
             patch("agent.auxiliary_client._try_nous") as mock_nous:
            mock_or.return_value = (None, None)  # no credentials
            mock_nous.return_value = (MagicMock(), "model-b")

            client, model = resolve_provider_client("failover", "model-a")

            assert client is not None
            assert model == "model-b"

    @patch("agent.auxiliary_client._read_failover_chain")
    def test_failover_all_entries_fail_returns_none(self, mock_chain):
        """All chain entries unavailable → return (None, None)."""
        from agent.auxiliary_client import resolve_provider_client

        mock_chain.return_value = [
            {"provider": "openrouter", "model": "model-a"},
            {"provider": "nous", "model": "model-b"},
        ]

        with patch("agent.auxiliary_client._try_openrouter") as mock_or, \
             patch("agent.auxiliary_client._try_nous") as mock_nous:
            mock_or.return_value = (None, None)
            mock_nous.return_value = (None, None)

            client, model = resolve_provider_client("failover", "model-a")

            assert client is None
            assert model is None


class TestGetTextAuxiliaryClientFailover:
    """get_text_auxiliary_client should return a valid client when
    compression.provider=failover instead of (None, None)."""

    @patch("agent.auxiliary_client._read_failover_chain")
    @patch("agent.auxiliary_client._resolve_task_provider_model")
    def test_returns_client_for_failover_task(self, mock_resolve, mock_chain):
        from agent.auxiliary_client import get_text_auxiliary_client

        mock_resolve.return_value = ("failover", "some-model", None, None, None)
        mock_chain.return_value = [
            {"provider": "openrouter", "model": "test-model"},
        ]

        with patch("agent.auxiliary_client._try_openrouter") as mock_try:
            mock_client = MagicMock()
            mock_client.base_url = "https://openrouter.ai/api/v1"
            mock_try.return_value = (mock_client, "test-model")

            client, model = get_text_auxiliary_client("compression")

            assert client is not None
            assert model is not None
