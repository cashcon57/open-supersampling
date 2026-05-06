from __future__ import annotations

from scripts.training_dashboard import HTML


def test_codex_log_animation_stub_is_rendered() -> None:
    assert 'id="codex-thinking"' in HTML
    assert 'class="codex-thinking-spinner"' in HTML
    assert 'codex thinking...' in HTML
    assert ".codex-log .codex-fade-in" in HTML
    assert "@keyframes codexFadeIn" in HTML
