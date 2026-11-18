"""Tests for how both analysts read Claude's replies: tolerant JSON parsing and a
single corrective retry. Fake clients only - NO network, NO LLM. Run: ``pytest -q``
"""

from __future__ import annotations

import logging
import types

import pytest

from config.settings import Settings
from src.analysis.analyst_agent import (
    AnalystAgent,
    AnalystError,
    _extract_json_object,
    _repair_json,
)
from src.analysis.options_agent import OptionDecision, OptionsAnalystAgent
from src.ingestion.data_fetcher import MarketSnapshot
from tests.test_options import SNAPSHOT, make_chain


class ScriptedClient:
    """Fake Anthropic client that replies with the given texts in order (or raises)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        outer = self

        class _Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                reply = outer.replies.pop(0)
                if isinstance(reply, Exception):
                    raise reply
                return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=reply)])

        self.messages = _Messages()


def _settings():
    return Settings(_env_file=None, ANTHROPIC_API_KEY="test-key")


# The exact shape that broke the first live options run: a trailing comma.
TRAILING_COMMA_REPLY = """{
  "action": "BUY_PUT",
  "ticker": "RELIANCE",
  "expiry": "2026-10-27",
  "strike": 1180,
  "underlying_target": 1150,
  "reasoning": "Bearish below SMA50 and SMA200, put wall at 1200, target 1150.",
}"""

VALID_OPTION_REPLY = ('{"action": "BUY_PUT", "ticker": "RELIANCE", "expiry": "2026-10-27", '
                      '"strike": 1200, "underlying_target": 1150, "reasoning": "r"}')


# --- Tolerant parsing --------------------------------------------------------- #

def test_trailing_comma_reply_now_parses():
    decision = OptionDecision.from_model_text(TRAILING_COMMA_REPLY, "RELIANCE")
    assert (decision.action, decision.strike) == ("BUY_PUT", 1180.0)
    assert decision.reasoning == "Bearish below SMA50 and SMA200, put wall at 1200, target 1150."


def test_nested_trailing_commas_are_removed():
    assert _extract_json_object('{"a": [1, 2,], "b": {"c": 1,},}') == {"a": [1, 2], "b": {"c": 1}}


def test_commas_inside_strings_are_untouched():
    text = '{"reasoning": "odd text ,} and ,] stays", "n": 1,}'
    assert _repair_json(text) == '{"reasoning": "odd text ,} and ,] stays", "n": 1}'
    assert _extract_json_object(text)["reasoning"] == "odd text ,} and ,] stays"


def test_escaped_quotes_do_not_confuse_the_cleaner():
    text = '{"reasoning": "he said \\"sell,\\" then left", "n": 2,}'
    assert _extract_json_object(text) == {"reasoning": 'he said "sell," then left', "n": 2}


def test_invisible_characters_between_keys_are_forgiven():
    # A non-breaking space or zero-width character after a comma is not JSON
    # whitespace; text inside strings must stay exactly as written.
    text = ('{\n  "action": "SELL",\n  "ticker": "RELIANCE.NS",\u00a0\n'
            '  "target_price": 1150,\u200b\n  "reasoning": "keep\u00a0this",\u00a0}')
    parsed = _extract_json_object(text)
    assert parsed["target_price"] == 1150 and parsed["reasoning"] == "keep\u00a0this"


def test_non_json_still_fails_loudly():
    with pytest.raises(AnalystError):
        _extract_json_object("no json here")


# --- One corrective retry ----------------------------------------------------- #

def test_options_analyst_retries_once_with_the_error():
    client = ScriptedClient('{"action": "BUY_PUT", "strike": 1200', VALID_OPTION_REPLY)
    decision = OptionsAnalystAgent(_settings(), client=client).analyze(SNAPSHOT, make_chain())
    assert decision.action == "BUY_PUT" and decision.strike == 1200.0
    assert len(client.calls) == 2
    retry_messages = client.calls[1]["messages"]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert retry_messages[1]["content"] == '{"action": "BUY_PUT", "strike": 1200'
    assert "could not be used" in retry_messages[2]["content"]


def test_gives_up_after_one_retry():
    client = ScriptedClient("not json", "still not json")
    with pytest.raises(AnalystError, match="after one retry"):
        OptionsAnalystAgent(_settings(), client=client).analyze(SNAPSHOT, make_chain())
    assert len(client.calls) == 2


def test_equity_analyst_retries_an_invalid_action():
    snapshot = MarketSnapshot(ticker="TCS.NS", as_of="2026-09-29T00:00:00+00:00", current_price=100.0)
    client = ScriptedClient(
        '{"action": "YOLO", "ticker": "TCS.NS", "target_price": 110, "reasoning": "r"}',
        '{"action": "BUY", "ticker": "TCS.NS", "target_price": 110, "reasoning": "r",}',
    )
    decision = AnalystAgent(_settings(), client=client).analyze(snapshot)
    assert decision.action == "BUY" and len(client.calls) == 2


def test_api_failures_are_not_retried():
    client = ScriptedClient(RuntimeError("network down"))
    with pytest.raises(AnalystError, match="Anthropic API call failed"):
        OptionsAnalystAgent(_settings(), client=client).analyze(SNAPSHOT, make_chain())
    assert len(client.calls) == 1


def test_the_retry_warning_shows_where_the_reply_broke(caplog):
    snapshot = MarketSnapshot(ticker="TCS.NS", as_of="2026-09-29T00:00:00+00:00", current_price=100.0)
    broken = '{\n  "action": "BUY",\n  "ticker": "TCS.NS",// note\n  "target_price": 110\n}'
    valid = '{"action": "BUY", "ticker": "TCS.NS", "target_price": 110, "reasoning": "r"}'
    client = ScriptedClient(broken, valid)
    with caplog.at_level(logging.WARNING):
        AnalystAgent(_settings(), client=client).analyze(snapshot)
    assert "asking once more" in caplog.text and "near " in caplog.text and "// note" in caplog.text


def test_stray_quotes_inside_the_reasoning_are_forgiven():
    """29 Oct 2026: the reasoning quoted a phrase and the whole pick was lost."""
    reply = ('{"action": "BUY", "ticker": "HDFCBANK.NS", "target_price": 1650,\n'
             ' "reasoning": "A "fresh break" above the range, with the Nifty."}')
    parsed = _extract_json_object(reply)
    assert parsed["ticker"] == "HDFCBANK.NS" and parsed["reasoning"] == 'A "fresh break" above the range, with the Nifty.'
    assert _extract_json_object('{"a": "x", "b": ["y", "z"], "c": {"d": "e"}}') == {"a": "x", "b": ["y", "z"], "c": {"d": "e"}}


def test_raw_line_breaks_inside_the_reasoning_are_forgiven():
    """3 Nov 2026: the futures pick was lost because the reasoning held a raw line break."""
    reply = '{"action": "BUY", "ticker": "NIFTY-FUT.NS", "target_price": 23800,\n "reasoning": "Breakout.\nRoom to run.\t"}'
    parsed = _extract_json_object(reply)
    assert parsed["ticker"] == "NIFTY-FUT.NS" and parsed["reasoning"] == "Breakout.\nRoom to run.\t"
