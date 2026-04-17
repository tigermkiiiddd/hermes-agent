"""StatusBar — standalone TUI status bar renderer.

Extracted from cli.py HermesCLI methods.  Renders session info (model,
context usage, active skill domains, duration) as either plain text or
prompt_toolkit fragments, adapting to terminal width.

Three width breakpoints:
  - narrow  (< 52):  model + duration only
  - medium  (< 76):  model + percent + domains (if any) + duration
  - wide    (>= 76): model + ctx_used/ctx_total + [████░░] percent
                      + domains (if any) + duration

All methods accept a *snapshot* dict rather than reading state directly,
so the class has zero dependency on HermesCLI or AIAgent internals.
"""

from __future__ import annotations

from typing import Any, Optional


# ── Helpers also used by cli.py ────────────────────────────────────────

def _format_context_length(tokens: int) -> str:
    """Re-export from hermes_cli.banner for use inside StatusBar."""
    from hermes_cli.banner import _format_context_length as _real
    return _real(tokens)


def _format_token_count_compact(value: int) -> str:
    """Re-export from agent.usage_pricing for use inside StatusBar."""
    from agent.usage_pricing import format_token_count_compact as _real
    return _real(value)


# ── StatusBar class ───────────────────────────────────────────────────

class StatusBar:
    """Pure-rendering TUI status bar.

    Usage::

        sb = StatusBar()
        text  = sb.build_text(snapshot, width=120)
        frags = sb.build_fragments(snapshot, width=120)
    """

    # Style dict for prompt_toolkit
    STYLES: dict[str, str] = {
        "status-bar": "bg:#1a1a2e #C0C0C0",
        "status-bar-strong": "bg:#1a1a2e #FFD700 bold",
        "status-bar-dim": "bg:#1a1a2e #8B8682",
        "status-bar-good": "bg:#1a1a2e #8FBC8F bold",
        "status-bar-warn": "bg:#1a1a2e #FFD700 bold",
        "status-bar-bad": "bg:#1a1a2e #FF8C00 bold",
        "status-bar-critical": "bg:#1a1a2e #FF6B6B bold",
        "status-bar-domains": "bg:#1a1a2e #87CEEB",
    }

    # ── Static helpers (pure functions) ────────────────────────────────

    @staticmethod
    def context_style(percent_used: Optional[int]) -> str:
        """Return prompt_toolkit style class for a context usage percent."""
        if percent_used is None:
            return "class:status-bar-dim"
        if percent_used >= 95:
            return "class:status-bar-critical"
        if percent_used > 80:
            return "class:status-bar-bad"
        if percent_used >= 50:
            return "class:status-bar-warn"
        return "class:status-bar-good"

    @staticmethod
    def build_context_bar(percent_used: Optional[int], width: int = 10) -> str:
        """Return a visual usage bar like ``[██████░░░░]``."""
        safe_percent = max(0, min(100, percent_used or 0))
        filled = round((safe_percent / 100) * width)
        return f"[{('█' * filled) + ('░' * max(0, width - filled))}]"

    @staticmethod
    def format_domains(domains: set[str], max_width: int = 24) -> str:
        """Format active skill domains for the TUI status bar.

        Shows short domain names, e.g. ``m/training`` instead of
        ``mlops/training``, and truncates if too many domains.
        """
        if not domains:
            return ""
        short: list[str] = []
        for d in sorted(domains):
            parts = d.split("/")
            if len(parts) >= 2:
                s = f"{parts[-2][0]}/{parts[-1][:8]}"
            else:
                s = d[:10]
            short.append(s)
        label = ", ".join(short)
        if len(label) > max_width:
            label = label[: max_width - 1] + "…"
        return label

    @staticmethod
    def display_width(text: str) -> int:
        """Return terminal cell width (handles CJK / wide glyphs)."""
        try:
            from prompt_toolkit.utils import get_cwidth
            return get_cwidth(text or "")
        except Exception:
            return len(text or "")

    @classmethod
    def trim(cls, text: str, max_width: int) -> str:
        """Trim status-bar text to a single terminal row."""
        if max_width <= 0:
            return ""
        try:
            from prompt_toolkit.utils import get_cwidth
        except Exception:
            get_cwidth = None

        if cls.display_width(text) <= max_width:
            return text

        ellipsis = "..."
        ellipsis_width = cls.display_width(ellipsis)
        if max_width <= ellipsis_width:
            return ellipsis[:max_width]

        out: list[str] = []
        width = 0
        for ch in text:
            ch_width = get_cwidth(ch) if get_cwidth else len(ch)
            if width + ch_width + ellipsis_width > max_width:
                break
            out.append(ch)
            width += ch_width
        return "".join(out).rstrip() + ellipsis

    # ── build_text ─────────────────────────────────────────────────────

    def build_text(self, snapshot: dict[str, Any], width: int, *, fallback_model: str = "Hermes") -> str:
        """Return a compact one-line session status string for the TUI footer.

        Parameters
        ----------
        snapshot : dict
            Produced by ``HermesCLI._get_status_bar_snapshot()``.
        width : int
            Terminal width in cells.
        fallback_model : str
            Model name to show on exception.
        """
        try:
            percent = snapshot["context_percent"]
            percent_label = f"{percent}%" if percent is not None else "--"
            duration_label = snapshot["duration"]
            domains_label = self.format_domains(snapshot.get("active_domains") or set())

            if width < 52:
                text = f"⚕ {snapshot['model_short']} · {duration_label}"
                return self.trim(text, width)
            if width < 76:
                parts = [f"⚕ {snapshot['model_short']}", percent_label]
                if domains_label:
                    parts.append(f"📂{domains_label}")
                parts.append(duration_label)
                return self.trim(" · ".join(parts), width)

            if snapshot["context_length"]:
                ctx_total = _format_context_length(snapshot["context_length"])
                ctx_used = _format_token_count_compact(snapshot["context_tokens"])
                context_label = f"{ctx_used}/{ctx_total}"
            else:
                context_label = "ctx --"

            parts = [f"⚕ {snapshot['model_short']}", context_label, percent_label]
            if domains_label:
                parts.append(f"📂{domains_label}")
            parts.append(duration_label)
            return self.trim(" │ ".join(parts), width)
        except Exception:
            return f"⚕ {fallback_model}"

    # ── build_fragments ────────────────────────────────────────────────

    def build_fragments(
        self, snapshot: dict[str, Any], width: int,
        *, fallback_text: str = "",
    ) -> list[tuple[str, str]]:
        """Return prompt_toolkit formatted-text fragments.

        Parameters
        ----------
        snapshot : dict
            Produced by ``HermesCLI._get_status_bar_snapshot()``.
        width : int
            Terminal width in cells.
        fallback_text : str
            Pre-built plain text to use on exception (avoids re-entering cli).
        """
        try:
            duration_label = snapshot["duration"]
            domains_label = self.format_domains(snapshot.get("active_domains") or set())

            if width < 52:
                frags: list[tuple[str, str]] = [
                    ("class:status-bar", " ⚕ "),
                    ("class:status-bar-strong", snapshot["model_short"]),
                    ("class:status-bar-dim", " · "),
                    ("class:status-bar-dim", duration_label),
                    ("class:status-bar", " "),
                ]
            else:
                percent = snapshot["context_percent"]
                percent_label = f"{percent}%" if percent is not None else "--"
                if width < 76:
                    frags = [
                        ("class:status-bar", " ⚕ "),
                        ("class:status-bar-strong", snapshot["model_short"]),
                        ("class:status-bar-dim", " · "),
                        (self.context_style(percent), percent_label),
                    ]
                    if domains_label:
                        frags += [
                            ("class:status-bar-dim", " · "),
                            ("class:status-bar-domains", f"📂{domains_label}"),
                        ]
                    frags += [
                        ("class:status-bar-dim", " · "),
                        ("class:status-bar-dim", duration_label),
                        ("class:status-bar", " "),
                    ]
                else:
                    if snapshot["context_length"]:
                        ctx_total = _format_context_length(snapshot["context_length"])
                        ctx_used = _format_token_count_compact(snapshot["context_tokens"])
                        context_label = f"{ctx_used}/{ctx_total}"
                    else:
                        context_label = "ctx --"

                    bar_style = self.context_style(percent)
                    frags = [
                        ("class:status-bar", " ⚕ "),
                        ("class:status-bar-strong", snapshot["model_short"]),
                        ("class:status-bar-dim", " │ "),
                        ("class:status-bar-dim", context_label),
                        ("class:status-bar-dim", " │ "),
                        (bar_style, self.build_context_bar(percent)),
                        ("class:status-bar-dim", " "),
                        (bar_style, percent_label),
                    ]
                    if domains_label:
                        frags += [
                            ("class:status-bar-dim", " │ "),
                            ("class:status-bar-domains", f"📂{domains_label}"),
                        ]
                    frags += [
                        ("class:status-bar-dim", " │ "),
                        ("class:status-bar-dim", duration_label),
                        ("class:status-bar", " "),
                    ]

            total_width = sum(self.display_width(text) for _, text in frags)
            if total_width > width:
                plain_text = "".join(text for _, text in frags)
                trimmed = self.trim(plain_text, width)
                return [("class:status-bar", trimmed)]
            return frags
        except Exception:
            if fallback_text:
                return [("class:status-bar", f" {fallback_text} ")]
            return [("class:status-bar", f" ⚕ Hermes ")]
