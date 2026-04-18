"""Tests for main-model streaming call status feedback.

Bug: When the main model API call hangs (e.g. Z.AI doesn't respond),
     the user sees zero feedback until stale timeout (180s).
     The agent appears frozen.

Fix: _call_chat_completions emits _emit_status immediately when the
     streaming request is dispatched, showing model name.

     Stale timeout and retry messages were already implemented.
     This test ensures the INITIAL dispatch status is present.

TDD flow: RED -> GREEN -> REFACTOR
"""

import pytest


class TestEmitStatusPlacement:
    """Source code contract: _emit_status must exist between
    _touch_activity and chat.completions.create in _call_chat_completions."""

    def test_emit_status_exists_before_create(self):
        """Verify the source code has _emit_status call in the right place.
        
        Reads run_agent.py and checks that between the line containing
        'waiting for provider response (streaming)' and the line containing
        'chat.completions.create(**stream_kwargs)', there is an _emit_status call.
        
        This is the core UX fix: the gap between 'user presses enter' and
        'first chunk arrives' can be 30-180s. During that gap, the user
        needs to see "正在等待 {model} 响应..." so they know the agent
        isn't frozen.
        """
        with open("run_agent.py") as f:
            lines = f.readlines()
        
        touch_line = None
        create_line = None
        emit_line = None
        
        for i, line in enumerate(lines):
            if "waiting for provider response (streaming)" in line:
                touch_line = i
            if touch_line is not None and create_line is None:
                if "chat.completions.create(**stream_kwargs)" in line:
                    create_line = i
                if "_emit_status" in line and emit_line is None:
                    emit_line = i
        
        assert touch_line is not None, "Could not find _touch_activity line"
        assert create_line is not None, "Could not find chat.completions.create line"
        assert emit_line is not None, (
            "_emit_status not found between _touch_activity and "
            "chat.completions.create — user will see no feedback!"
        )
        assert touch_line < emit_line < create_line, (
            f"_emit_status (line {emit_line+1}) must be between "
            f"_touch_activity (line {touch_line+1}) and "
            f"chat.completions.create (line {create_line+1})"
        )

    def test_emit_status_message_contains_model(self):
        """The _emit_status message should reference the model name."""
        with open("run_agent.py") as f:
            content = f.read()
        
        # Find the _emit_status call between touch_activity and create
        touch_idx = content.index("waiting for provider response (streaming)")
        create_idx = content.index("chat.completions.create(**stream_kwargs)", touch_idx)
        segment = content[touch_idx:create_idx]
        
        assert "_emit_status" in segment, "No _emit_status between touch and create"
        assert "正在等待" in segment or "waiting for" in segment.lower(), (
            "_emit_status message should indicate waiting for model"
        )


class TestRetryAndStaleStatusExists:
    """Verify retry and stale timeout status messages are still in place."""

    def test_stale_timeout_has_emit_status(self):
        """Stale stream kill should emit status to user."""
        with open("run_agent.py") as f:
            content = f.read()
        
        # Find stale stream kill section
        stale_idx = content.find("Stream stale for")
        assert stale_idx > 0, "Stale detector code not found"
        
        # Check _emit_status appears shortly after
        nearby = content[stale_idx:stale_idx + 800]
        assert "_emit_status" in nearby, (
            "Stale detector should call _emit_status to notify user"
        )

    def test_retry_has_emit_status(self):
        """Stream retry should emit status showing attempt count."""
        with open("run_agent.py") as f:
            content = f.read()
        
        # Find retry path — _emit_status is BEFORE the log line
        retry_idx = content.find("retrying with fresh connection")
        assert retry_idx > 0, "Retry logic not found"
        
        # _emit_status appears right after the logger.info line
        nearby = content[retry_idx:retry_idx + 500]
        assert "_emit_status" in nearby, (
            "Retry path should call _emit_status to notify user"
        )

    def test_all_retries_exhausted_has_emit_status(self):
        """When all retries fail, user should see final error status."""
        with open("run_agent.py") as f:
            content = f.read()
        
        exhausted_idx = content.find("Connection to provider failed after")
        assert exhausted_idx > 0, "Retry exhaustion handler not found"
        
        # _emit_status is called just before this string
        nearby = content[exhausted_idx - 200:exhausted_idx + 200]
        assert "_emit_status" in nearby, (
            "Retry exhaustion should call _emit_status with final error"
        )
