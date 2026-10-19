"""Read-only mode: the deployment that cannot spend the operator's quota.

The guarantee is worth testing precisely because it is about something NOT
happening. A demo that quietly falls back to calling the provider would be
indistinguishable from a working one until the bill arrived.
"""

from __future__ import annotations

import pytest

from arxivbot.llm import CacheMiss, LLMConfig, complete, stream


def cache_only() -> LLMConfig:
    config = LLMConfig(provider="gemini", api_key="not-a-real-key")
    config.cache_only = True
    return config


# A prompt no run has produced, so it cannot be in anyone's cache.
NOVEL = "an unseen prompt reserved for tests: 9f3c1a7e-cache-only"


def test_a_completion_refuses_rather_than_calls_out():
    with pytest.raises(CacheMiss):
        complete(NOVEL, config=cache_only())


def test_a_stream_refuses_rather_than_calls_out():
    with pytest.raises(CacheMiss):
        list(stream(NOVEL + " streamed", config=cache_only()))


def test_cache_miss_is_an_llm_error_so_callers_already_handle_it():
    """Existing except LLMError blocks must not start leaking this."""
    from arxivbot.llm import LLMError

    assert issubclass(CacheMiss, LLMError)


def test_cache_only_is_off_by_default():
    """A flag that defends the quota must never be the thing you forgot."""
    assert LLMConfig().cache_only is False


def test_a_cached_reply_is_still_served(monkeypatch):
    """Read-only is not the same as broken: what is paid for still works."""
    import arxivbot.llm as llm

    monkeypatch.setattr(llm, "_cached", lambda key: "a remembered answer")
    assert llm.complete(NOVEL, config=cache_only()) == "a remembered answer"
