"""Tests for the non-stream proxy server."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from server import app, parse_sse, reassemble_message


# ---------------------------------------------------------------------------
# Helpers — build mock SSE event streams
# ---------------------------------------------------------------------------


def _sse_block(data: dict) -> str:
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"


def _make_stream(events: list[dict]) -> str:
    return "".join(_sse_block(e) for e in events)


def _msg_start(msg_id="msg_01", model="claude-sonnet-4-20250514", usage=None):
    return {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 10, "output_tokens": 0},
        },
    }


def _block_start(index, content_block):
    return {"type": "content_block_start", "index": index, "content_block": content_block}


def _block_delta(index, delta):
    return {"type": "content_block_delta", "index": index, "delta": delta}


def _block_stop(index):
    return {"type": "content_block_stop", "index": index}


def _msg_delta(stop_reason="end_turn", stop_sequence=None, usage=None):
    return {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
        "usage": usage or {"output_tokens": 5},
    }


def _msg_stop():
    return {"type": "message_stop"}


# ---------------------------------------------------------------------------
# Pre-built event sequences
# ---------------------------------------------------------------------------

SIMPLE_TEXT_EVENTS = [
    _msg_start(),
    _block_start(0, {"type": "text", "text": ""}),
    _block_delta(0, {"type": "text_delta", "text": "Hello"}),
    _block_delta(0, {"type": "text_delta", "text": " world"}),
    _block_stop(0),
    _msg_delta("end_turn"),
    _msg_stop(),
]


TOOL_USE_EVENTS = [
    _msg_start("msg_02", usage={"input_tokens": 20, "output_tokens": 0}),
    _block_start(0, {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}),
    _block_delta(0, {"type": "input_json_delta", "partial_json": '{"loc'}),
    _block_delta(0, {"type": "input_json_delta", "partial_json": 'ation": "NYC"}'}),
    _block_stop(0),
    _msg_delta("tool_use", usage={"output_tokens": 10}),
    _msg_stop(),
]


THINKING_EVENTS = [
    _msg_start("msg_03", usage={"input_tokens": 15, "output_tokens": 0}),
    _block_start(0, {"type": "thinking", "thinking": ""}),
    _block_delta(0, {"type": "thinking_delta", "thinking": "Let me think"}),
    _block_stop(0),
    _block_start(1, {"type": "text", "text": ""}),
    _block_delta(1, {"type": "text_delta", "text": "Answer"}),
    _block_stop(1),
    _msg_delta("end_turn", usage={"output_tokens": 8}),
    _msg_stop(),
]


# ---------------------------------------------------------------------------
# 1. parse_sse
# ---------------------------------------------------------------------------


class TestParseSSE:
    def test_parses_simple_events(self):
        text = _make_stream(SIMPLE_TEXT_EVENTS)
        events = parse_sse(text)
        types = [e["type"] for e in events]
        assert "message_start" in types
        assert "content_block_delta" in types

    def test_skips_done_marker(self):
        text = "data: [DONE]\n\n"
        assert parse_sse(text) == []

    def test_skips_invalid_json(self):
        text = "data: not json\n\n"
        assert parse_sse(text) == []

    def test_empty_input(self):
        assert parse_sse("") == []

    def test_ignores_event_line_only_parses_data(self):
        text = "event: message_start\ndata: {\"type\": \"message_start\", \"message\": {}}\n\n"
        events = parse_sse(text)
        assert len(events) == 1
        assert events[0]["type"] == "message_start"

    def test_multiple_data_lines_in_one_block(self):
        """Each data: line in a single SSE block becomes a separate event."""
        text = 'data: {"type": "ping"}\ndata: {"type": "ping"}\n\n'
        events = parse_sse(text)
        assert len(events) == 2

    def test_done_with_whitespace(self):
        text = "data:   [DONE]  \n\n"
        assert parse_sse(text) == []

    def test_preserves_event_order(self):
        text = _make_stream(SIMPLE_TEXT_EVENTS)
        events = parse_sse(text)
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"


# ---------------------------------------------------------------------------
# 2. reassemble_message — text responses
# ---------------------------------------------------------------------------


class TestReassembleText:
    def test_simple_text(self):
        events = parse_sse(_make_stream(SIMPLE_TEXT_EVENTS))
        msg = reassemble_message(events)
        assert msg["type"] == "message"
        assert msg["id"] == "msg_01"
        assert msg["role"] == "assistant"
        assert msg["model"] == "claude-sonnet-4-20250514"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Hello world"
        assert msg["stop_reason"] == "end_turn"
        assert msg["usage"]["output_tokens"] == 5
        assert msg["usage"]["input_tokens"] == 10

    def test_many_small_text_deltas(self):
        """Accumulate text across many small delta chunks."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
        ]
        words = ["The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog"]
        for w in words:
            events.append(_block_delta(0, {"type": "text_delta", "text": w}))
        events += [_block_stop(0), _msg_delta("end_turn"), _msg_stop()]

        msg = reassemble_message(events)
        assert msg["content"][0]["text"] == "The quick brown fox jumps over the lazy dog"

    def test_single_delta(self):
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "Done."}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["text"] == "Done."

    def test_empty_text_deltas(self):
        """Empty text deltas should not corrupt the result."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "hi"}),
            _block_delta(0, {"type": "text_delta", "text": ""}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["text"] == "hi"

    def test_unicode_text(self):
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "こんにちは"}),
            _block_delta(0, {"type": "text_delta", "text": " 🌍"}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["text"] == "こんにちは 🌍"


# ---------------------------------------------------------------------------
# 3. reassemble_message — stop reasons
# ---------------------------------------------------------------------------


class TestReassembleStopReasons:
    def test_end_turn(self):
        events = parse_sse(_make_stream(SIMPLE_TEXT_EVENTS))
        msg = reassemble_message(events)
        assert msg["stop_reason"] == "end_turn"

    def test_max_tokens(self):
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "Truncat"}),
            _block_stop(0),
            _msg_delta("max_tokens", usage={"output_tokens": 4096}),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["stop_reason"] == "max_tokens"
        assert msg["content"][0]["text"] == "Truncat"

    def test_stop_sequence(self):
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "some output"}),
            _block_stop(0),
            _msg_delta("stop_sequence", stop_sequence="\n\nHuman:"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["stop_reason"] == "stop_sequence"
        assert msg["stop_sequence"] == "\n\nHuman:"

    def test_tool_use_stop_reason(self):
        events = parse_sse(_make_stream(TOOL_USE_EVENTS))
        msg = reassemble_message(events)
        assert msg["stop_reason"] == "tool_use"


# ---------------------------------------------------------------------------
# 4. reassemble_message — tool use
# ---------------------------------------------------------------------------


class TestReassembleToolUse:
    def test_simple_tool_call(self):
        events = parse_sse(_make_stream(TOOL_USE_EVENTS))
        msg = reassemble_message(events)
        block = msg["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "toolu_01"
        assert block["name"] == "get_weather"
        assert block["input"] == {"location": "NYC"}

    def test_empty_input(self):
        """Tool with no input JSON deltas should get empty dict."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "tool_use", "id": "t1", "name": "noop", "input": {}}),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["input"] == {}

    def test_complex_nested_input(self):
        """Tool input with nested objects and arrays."""
        input_json = json.dumps({
            "query": "SELECT * FROM users",
            "params": [1, "hello", None],
            "options": {"timeout": 30, "retry": True},
        })
        mid = len(input_json) // 2
        events = [
            _msg_start(),
            _block_start(0, {"type": "tool_use", "id": "t1", "name": "db_query", "input": {}}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": input_json[:mid]}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": input_json[mid:]}),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["input"]["query"] == "SELECT * FROM users"
        assert msg["content"][0]["input"]["params"] == [1, "hello", None]
        assert msg["content"][0]["input"]["options"]["retry"] is True

    def test_invalid_json_input_falls_back_to_empty_dict(self):
        events = [
            _msg_start(),
            _block_start(0, {"type": "tool_use", "id": "t1", "name": "broken", "input": {}}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"bad json'}),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["input"] == {}

    def test_text_then_tool_use(self):
        """Assistant writes text then makes a tool call."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "Let me check the weather."}),
            _block_stop(0),
            _block_start(1, {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}),
            _block_delta(1, {"type": "input_json_delta", "partial_json": '{"city": "London"}'}),
            _block_stop(1),
            _msg_delta("tool_use", usage={"output_tokens": 20}),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == "Let me check the weather."
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["input"] == {"city": "London"}
        assert msg["stop_reason"] == "tool_use"

    def test_multiple_parallel_tool_calls(self):
        """Multiple tool calls in a single response (parallel tool use)."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"city": "NYC"}'}),
            _block_stop(0),
            _block_start(1, {"type": "tool_use", "id": "toolu_02", "name": "get_time", "input": {}}),
            _block_delta(1, {"type": "input_json_delta", "partial_json": '{"tz": "EST"}'}),
            _block_stop(1),
            _block_start(2, {"type": "tool_use", "id": "toolu_03", "name": "get_news", "input": {}}),
            _block_delta(2, {"type": "input_json_delta", "partial_json": '{"topic": "tech"}'}),
            _block_stop(2),
            _msg_delta("tool_use", usage={"output_tokens": 30}),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 3
        assert msg["content"][0]["name"] == "get_weather"
        assert msg["content"][0]["input"] == {"city": "NYC"}
        assert msg["content"][1]["name"] == "get_time"
        assert msg["content"][1]["input"] == {"tz": "EST"}
        assert msg["content"][2]["name"] == "get_news"
        assert msg["content"][2]["input"] == {"topic": "tech"}

    def test_text_then_multiple_tool_calls(self):
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "I'll look that up."}),
            _block_stop(0),
            _block_start(1, {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {}}),
            _block_delta(1, {"type": "input_json_delta", "partial_json": '{"q": "a"}'}),
            _block_stop(1),
            _block_start(2, {"type": "tool_use", "id": "toolu_02", "name": "search", "input": {}}),
            _block_delta(2, {"type": "input_json_delta", "partial_json": '{"q": "b"}'}),
            _block_stop(2),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 3
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][2]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# 5. reassemble_message — server_tool_use (built-in server tools)
# ---------------------------------------------------------------------------


class TestReassembleServerToolUse:
    def test_server_tool_use_input_parsed(self):
        """server_tool_use blocks use input_json_delta just like tool_use."""
        events = [
            _msg_start(),
            _block_start(0, {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {},
            }),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"query"'}),
            _block_delta(0, {"type": "input_json_delta", "partial_json": ': "python docs"}'}),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        block = msg["content"][0]
        assert block["type"] == "server_tool_use"
        assert block["id"] == "srvtoolu_01"
        assert block["name"] == "web_search"
        assert block["input"] == {"query": "python docs"}

    def test_server_tool_use_empty_input(self):
        events = [
            _msg_start(),
            _block_start(0, {
                "type": "server_tool_use",
                "id": "srvtoolu_02",
                "name": "code_execution",
                "input": {},
            }),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["input"] == {}

    def test_web_search_tool_result_passthrough(self):
        """web_search_tool_result blocks arrive complete in content_block_start."""
        events = [
            _msg_start(),
            _block_start(0, {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {},
            }),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "test"}'}),
            _block_stop(0),
            _block_start(1, {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://example.com",
                        "title": "Example",
                        "encrypted_content": "abc123",
                    }
                ],
            }),
            _block_stop(1),
            _block_start(2, {"type": "text", "text": ""}),
            _block_delta(2, {"type": "text_delta", "text": "Based on my search..."}),
            _block_stop(2),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 3
        assert msg["content"][0]["type"] == "server_tool_use"
        assert msg["content"][0]["input"] == {"query": "test"}
        assert msg["content"][1]["type"] == "web_search_tool_result"
        assert msg["content"][1]["tool_use_id"] == "srvtoolu_01"
        assert msg["content"][1]["content"][0]["url"] == "https://example.com"
        assert msg["content"][2]["type"] == "text"
        assert msg["content"][2]["text"] == "Based on my search..."


# ---------------------------------------------------------------------------
# 6. reassemble_message — mcp_tool_use / mcp_tool_result
# ---------------------------------------------------------------------------


class TestReassembleMcpToolUse:
    def test_mcp_tool_use_input_parsed(self):
        """mcp_tool_use blocks use input_json_delta like regular tool_use."""
        events = [
            _msg_start(),
            _block_start(0, {
                "type": "mcp_tool_use",
                "id": "mcptoolu_01",
                "name": "mcp__myserver__query",
                "server_name": "myserver",
                "input": {},
            }),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"sql": "SELECT 1"}'}),
            _block_stop(0),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        block = msg["content"][0]
        assert block["type"] == "mcp_tool_use"
        assert block["name"] == "mcp__myserver__query"
        assert block["server_name"] == "myserver"
        assert block["input"] == {"sql": "SELECT 1"}

    def test_mcp_tool_result_passthrough(self):
        """mcp_tool_result blocks arrive complete in content_block_start."""
        events = [
            _msg_start(),
            _block_start(0, {
                "type": "mcp_tool_use",
                "id": "mcptoolu_01",
                "name": "mcp__db__query",
                "server_name": "db",
                "input": {},
            }),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"q": "x"}'}),
            _block_stop(0),
            _block_start(1, {
                "type": "mcp_tool_result",
                "tool_use_id": "mcptoolu_01",
                "is_error": False,
                "content": [{"type": "text", "text": "result data"}],
            }),
            _block_stop(1),
            _block_start(2, {"type": "text", "text": ""}),
            _block_delta(2, {"type": "text_delta", "text": "The result is..."}),
            _block_stop(2),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 3
        assert msg["content"][0]["type"] == "mcp_tool_use"
        assert msg["content"][0]["input"] == {"q": "x"}
        assert msg["content"][1]["type"] == "mcp_tool_result"
        assert msg["content"][1]["is_error"] is False
        assert msg["content"][2]["text"] == "The result is..."


# ---------------------------------------------------------------------------
# 7. reassemble_message — thinking / extended thinking
# ---------------------------------------------------------------------------


class TestReassembleThinking:
    def test_thinking_then_text(self):
        events = parse_sse(_make_stream(THINKING_EVENTS))
        msg = reassemble_message(events)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "Let me think"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "Answer"

    def test_thinking_with_signature(self):
        """Thinking blocks include a signature_delta for verification."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "Step 1. "}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "Step 2."}),
            _block_delta(0, {"type": "signature_delta", "signature": "abc123sig"}),
            _block_stop(0),
            _block_start(1, {"type": "text", "text": ""}),
            _block_delta(1, {"type": "text_delta", "text": "Result"}),
            _block_stop(1),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "Step 1. Step 2."
        assert msg["content"][0]["signature"] == "abc123sig"
        assert msg["content"][1]["text"] == "Result"

    def test_redacted_thinking(self):
        """Redacted thinking blocks pass through as-is (no deltas)."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "visible thought"}),
            _block_delta(0, {"type": "signature_delta", "signature": "sig1"}),
            _block_stop(0),
            _block_start(1, {"type": "redacted_thinking", "data": "base64encodeddata=="}),
            _block_stop(1),
            _block_start(2, {"type": "text", "text": ""}),
            _block_delta(2, {"type": "text_delta", "text": "Final answer"}),
            _block_stop(2),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 3
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "visible thought"
        assert msg["content"][0]["signature"] == "sig1"
        assert msg["content"][1]["type"] == "redacted_thinking"
        assert msg["content"][1]["data"] == "base64encodeddata=="
        assert msg["content"][2]["type"] == "text"
        assert msg["content"][2]["text"] == "Final answer"

    def test_thinking_then_tool_use(self):
        """Extended thinking followed by a tool call."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "I need to search"}),
            _block_delta(0, {"type": "signature_delta", "signature": "sig"}),
            _block_stop(0),
            _block_start(1, {"type": "tool_use", "id": "toolu_01", "name": "search", "input": {}}),
            _block_delta(1, {"type": "input_json_delta", "partial_json": '{"q": "test"}'}),
            _block_stop(1),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "I need to search"
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["input"] == {"q": "test"}
        assert msg["stop_reason"] == "tool_use"

    def test_thinking_many_deltas(self):
        """Long thinking split across many deltas."""
        events = [_msg_start(), _block_start(0, {"type": "thinking", "thinking": ""})]
        for i in range(100):
            events.append(_block_delta(0, {"type": "thinking_delta", "thinking": f"chunk{i} "}))
        events.append(_block_delta(0, {"type": "signature_delta", "signature": "longsig"}))
        events.append(_block_stop(0))
        events += [
            _block_start(1, {"type": "text", "text": ""}),
            _block_delta(1, {"type": "text_delta", "text": "done"}),
            _block_stop(1),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        expected = "".join(f"chunk{i} " for i in range(100))
        assert msg["content"][0]["thinking"] == expected
        assert msg["content"][0]["signature"] == "longsig"

    def test_interleaved_thinking(self):
        """With interleaved-thinking beta, thinking can appear between text blocks."""
        events = [
            _msg_start(),
            # First thinking block
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "First thought"}),
            _block_delta(0, {"type": "signature_delta", "signature": "sig1"}),
            _block_stop(0),
            # First text block
            _block_start(1, {"type": "text", "text": ""}),
            _block_delta(1, {"type": "text_delta", "text": "Part one."}),
            _block_stop(1),
            # Second thinking block (interleaved)
            _block_start(2, {"type": "thinking", "thinking": ""}),
            _block_delta(2, {"type": "thinking_delta", "thinking": "Second thought"}),
            _block_delta(2, {"type": "signature_delta", "signature": "sig2"}),
            _block_stop(2),
            # Second text block
            _block_start(3, {"type": "text", "text": ""}),
            _block_delta(3, {"type": "text_delta", "text": "Part two."}),
            _block_stop(3),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 4
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "First thought"
        assert msg["content"][0]["signature"] == "sig1"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "Part one."
        assert msg["content"][2]["type"] == "thinking"
        assert msg["content"][2]["thinking"] == "Second thought"
        assert msg["content"][2]["signature"] == "sig2"
        assert msg["content"][3]["type"] == "text"
        assert msg["content"][3]["text"] == "Part two."

    def test_interleaved_thinking_with_tool_use(self):
        """Interleaved thinking with tool calls."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "Need data"}),
            _block_delta(0, {"type": "signature_delta", "signature": "sig1"}),
            _block_stop(0),
            _block_start(1, {"type": "text", "text": ""}),
            _block_delta(1, {"type": "text_delta", "text": "Let me look that up."}),
            _block_stop(1),
            _block_start(2, {"type": "thinking", "thinking": ""}),
            _block_delta(2, {"type": "thinking_delta", "thinking": "Call the API"}),
            _block_delta(2, {"type": "signature_delta", "signature": "sig2"}),
            _block_stop(2),
            _block_start(3, {"type": "tool_use", "id": "toolu_01", "name": "api_call", "input": {}}),
            _block_delta(3, {"type": "input_json_delta", "partial_json": '{"endpoint": "/data"}'}),
            _block_stop(3),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 4
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][2]["type"] == "thinking"
        assert msg["content"][3]["type"] == "tool_use"
        assert msg["content"][3]["input"] == {"endpoint": "/data"}
        assert msg["stop_reason"] == "tool_use"

    def test_multiple_redacted_thinking_blocks(self):
        """Multiple redacted thinking blocks in sequence."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "visible"}),
            _block_delta(0, {"type": "signature_delta", "signature": "sig1"}),
            _block_stop(0),
            _block_start(1, {"type": "redacted_thinking", "data": "redacted1=="}),
            _block_stop(1),
            _block_start(2, {"type": "redacted_thinking", "data": "redacted2=="}),
            _block_stop(2),
            _block_start(3, {"type": "text", "text": ""}),
            _block_delta(3, {"type": "text_delta", "text": "Response"}),
            _block_stop(3),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 4
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][1]["type"] == "redacted_thinking"
        assert msg["content"][1]["data"] == "redacted1=="
        assert msg["content"][2]["type"] == "redacted_thinking"
        assert msg["content"][2]["data"] == "redacted2=="
        assert msg["content"][3]["type"] == "text"


# ---------------------------------------------------------------------------
# 8. reassemble_message — citations
# ---------------------------------------------------------------------------


class TestReassembleCitations:
    def test_text_with_single_citation(self):
        """Text block with a single citation from document source."""
        citation = {
            "type": "char_location",
            "cited_text": "The answer is 42.",
            "document_index": 0,
            "document_title": "Guide",
            "start_char_index": 0,
            "end_char_index": 17,
        }
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "The answer is 42."}),
            _block_delta(0, {"type": "citations_delta", "citation": citation}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["text"] == "The answer is 42."
        assert len(msg["content"][0]["citations"]) == 1
        assert msg["content"][0]["citations"][0]["type"] == "char_location"
        assert msg["content"][0]["citations"][0]["cited_text"] == "The answer is 42."

    def test_text_with_multiple_citations(self):
        """Text block with multiple citations accumulated from deltas."""
        cit1 = {
            "type": "char_location",
            "cited_text": "fact one",
            "document_index": 0,
            "document_title": "Doc A",
            "start_char_index": 0,
            "end_char_index": 8,
        }
        cit2 = {
            "type": "char_location",
            "cited_text": "fact two",
            "document_index": 1,
            "document_title": "Doc B",
            "start_char_index": 10,
            "end_char_index": 18,
        }
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "fact one, fact two"}),
            _block_delta(0, {"type": "citations_delta", "citation": cit1}),
            _block_delta(0, {"type": "citations_delta", "citation": cit2}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"][0]["citations"]) == 2
        assert msg["content"][0]["citations"][0]["document_title"] == "Doc A"
        assert msg["content"][0]["citations"][1]["document_title"] == "Doc B"

    def test_text_without_citations_has_no_citations_key(self):
        """Text blocks without citations should not have a citations field."""
        events = parse_sse(_make_stream(SIMPLE_TEXT_EVENTS))
        msg = reassemble_message(events)
        assert "citations" not in msg["content"][0]

    def test_citations_with_page_location(self):
        """Page-location citation from a PDF source."""
        citation = {
            "type": "page_location",
            "cited_text": "Important finding",
            "document_index": 0,
            "document_title": "Report.pdf",
            "start_page_number": 5,
            "end_page_number": 5,
        }
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "Important finding"}),
            _block_delta(0, {"type": "citations_delta", "citation": citation}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        cit = msg["content"][0]["citations"][0]
        assert cit["type"] == "page_location"
        assert cit["start_page_number"] == 5

    def test_citations_with_web_search_result(self):
        """Citation referencing a web search result source."""
        citation = {
            "type": "web_search_result_location",
            "cited_text": "Python is a language",
            "url": "https://python.org",
            "title": "Python.org",
            "encrypted_index": "enc123",
        }
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "Python is a language"}),
            _block_delta(0, {"type": "citations_delta", "citation": citation}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        cit = msg["content"][0]["citations"][0]
        assert cit["type"] == "web_search_result_location"
        assert cit["url"] == "https://python.org"


# ---------------------------------------------------------------------------
# 9. reassemble_message — usage fields
# ---------------------------------------------------------------------------


class TestReassembleUsage:
    def test_basic_usage(self):
        events = parse_sse(_make_stream(SIMPLE_TEXT_EVENTS))
        msg = reassemble_message(events)
        assert msg["usage"]["input_tokens"] == 10
        assert msg["usage"]["output_tokens"] == 5

    def test_cache_usage_fields(self):
        """Cache tokens from message_start usage are preserved."""
        events = [
            _msg_start(usage={
                "input_tokens": 100,
                "output_tokens": 0,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 30,
            }),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "cached"}),
            _block_stop(0),
            _msg_delta("end_turn", usage={"output_tokens": 2}),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["usage"]["input_tokens"] == 100
        assert msg["usage"]["cache_creation_input_tokens"] == 50
        assert msg["usage"]["cache_read_input_tokens"] == 30
        assert msg["usage"]["output_tokens"] == 2

    def test_usage_merges_message_start_and_delta(self):
        """output_tokens from message_delta overrides the initial 0."""
        events = [
            _msg_start(usage={"input_tokens": 50, "output_tokens": 0}),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "x"}),
            _block_stop(0),
            _msg_delta("end_turn", usage={"output_tokens": 42}),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["usage"]["input_tokens"] == 50
        assert msg["usage"]["output_tokens"] == 42


# ---------------------------------------------------------------------------
# 10. reassemble_message — edge cases
# ---------------------------------------------------------------------------


class TestReassembleEdgeCases:
    def test_content_blocks_ordered_by_index(self):
        events = parse_sse(_make_stream(THINKING_EVENTS))
        msg = reassemble_message(events)
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][1]["type"] == "text"

    def test_ping_events_ignored(self):
        """ping events in the stream should not break reassembly."""
        events = [
            {"type": "ping"},
            _msg_start(),
            {"type": "ping"},
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "hello"}),
            {"type": "ping"},
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["text"] == "hello"

    def test_error_event_in_stream(self):
        """An error event mid-stream should not crash reassembly."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "partial"}),
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        ]
        msg = reassemble_message(events)
        # Should still produce a message with whatever was accumulated
        assert msg["content"][0]["text"] == "partial"

    def test_message_stop_is_noop(self):
        """message_stop event should not affect the assembled message."""
        events = [
            _msg_start("msg_99"),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "test"}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["id"] == "msg_99"
        assert msg["content"][0]["text"] == "test"

    def test_no_content_blocks(self):
        """Message with no content blocks produces empty content list."""
        events = [
            _msg_start(),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"] == []

    def test_unknown_delta_type_ignored(self):
        """Unknown delta types should not crash reassembly."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "hi"}),
            _block_delta(0, {"type": "some_future_delta", "data": "x"}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["text"] == "hi"

    def test_unknown_content_block_type_passthrough(self):
        """Unknown block types should pass through unchanged."""
        events = [
            _msg_start(),
            _block_start(0, {"type": "future_block_type", "data": "opaque"}),
            _block_stop(0),
            _block_start(1, {"type": "text", "text": ""}),
            _block_delta(1, {"type": "text_delta", "text": "after"}),
            _block_stop(1),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["content"][0]["type"] == "future_block_type"
        assert msg["content"][0]["data"] == "opaque"
        assert msg["content"][1]["text"] == "after"

    def test_model_field_preserved(self):
        events = [
            _msg_start(model="claude-opus-4-20250514"),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "x"}),
            _block_stop(0),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["model"] == "claude-opus-4-20250514"

    def test_type_field_set_to_message(self):
        """The 'type' field from message_start is 'message' but gets popped;
        reassemble should re-add it."""
        events = parse_sse(_make_stream(SIMPLE_TEXT_EVENTS))
        msg = reassemble_message(events)
        assert msg["type"] == "message"

    def test_stop_sequence_none_when_not_set(self):
        events = [
            _msg_start(),
            _block_start(0, {"type": "text", "text": ""}),
            _block_delta(0, {"type": "text_delta", "text": "x"}),
            _block_stop(0),
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["stop_reason"] == "end_turn"
        assert msg.get("stop_sequence") is None


# ---------------------------------------------------------------------------
# 11. reassemble_message — complex real-world scenarios
# ---------------------------------------------------------------------------


class TestReassembleComplexScenarios:
    def test_thinking_text_tool_use_full_flow(self):
        """Full agentic flow: thinking → text → tool_use."""
        events = [
            _msg_start(usage={"input_tokens": 500, "output_tokens": 0}),
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "The user wants weather. "}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "I should use the tool."}),
            _block_delta(0, {"type": "signature_delta", "signature": "thinking_sig_abc"}),
            _block_stop(0),
            _block_start(1, {"type": "text", "text": ""}),
            _block_delta(1, {"type": "text_delta", "text": "I'll check the weather for you."}),
            _block_stop(1),
            _block_start(2, {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {}}),
            _block_delta(2, {"type": "input_json_delta", "partial_json": '{"location"'}),
            _block_delta(2, {"type": "input_json_delta", "partial_json": ': "San Francisco"'}),
            _block_delta(2, {"type": "input_json_delta", "partial_json": ', "units": "celsius"}'}),
            _block_stop(2),
            _msg_delta("tool_use", usage={"output_tokens": 150}),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert msg["type"] == "message"
        assert len(msg["content"]) == 3
        # Thinking
        assert msg["content"][0]["type"] == "thinking"
        assert "weather" in msg["content"][0]["thinking"]
        assert msg["content"][0]["signature"] == "thinking_sig_abc"
        # Text
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "I'll check the weather for you."
        # Tool use
        assert msg["content"][2]["type"] == "tool_use"
        assert msg["content"][2]["id"] == "toolu_abc"
        assert msg["content"][2]["input"] == {"location": "San Francisco", "units": "celsius"}
        # Message-level fields
        assert msg["stop_reason"] == "tool_use"
        assert msg["usage"]["input_tokens"] == 500
        assert msg["usage"]["output_tokens"] == 150

    def test_web_search_full_flow(self):
        """Full web search flow: server_tool_use → result → text with citations."""
        citation = {
            "type": "web_search_result_location",
            "cited_text": "Python 3.13 was released",
            "url": "https://python.org/downloads/",
            "title": "Downloads",
            "encrypted_index": "enc_idx",
        }
        events = [
            _msg_start(),
            _block_start(0, {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {},
            }),
            _block_delta(0, {"type": "input_json_delta", "partial_json": '{"query": "latest python version"}'}),
            _block_stop(0),
            _block_start(1, {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [
                    {
                        "type": "web_search_result",
                        "url": "https://python.org/downloads/",
                        "title": "Downloads",
                        "encrypted_content": "enc_content",
                    }
                ],
            }),
            _block_stop(1),
            _block_start(2, {"type": "text", "text": ""}),
            _block_delta(2, {"type": "text_delta", "text": "Python 3.13 was released"}),
            _block_delta(2, {"type": "citations_delta", "citation": citation}),
            _block_delta(2, {"type": "text_delta", "text": " in October 2024."}),
            _block_stop(2),
            _msg_delta("end_turn"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 3
        assert msg["content"][0]["type"] == "server_tool_use"
        assert msg["content"][0]["input"] == {"query": "latest python version"}
        assert msg["content"][1]["type"] == "web_search_tool_result"
        assert msg["content"][2]["type"] == "text"
        assert msg["content"][2]["text"] == "Python 3.13 was released in October 2024."
        assert len(msg["content"][2]["citations"]) == 1
        assert msg["content"][2]["citations"][0]["url"] == "https://python.org/downloads/"

    def test_interleaved_thinking_multi_tool(self):
        """Interleaved thinking with multiple tool calls."""
        events = [
            _msg_start(),
            # Think about first tool
            _block_start(0, {"type": "thinking", "thinking": ""}),
            _block_delta(0, {"type": "thinking_delta", "thinking": "First, check weather"}),
            _block_delta(0, {"type": "signature_delta", "signature": "s1"}),
            _block_stop(0),
            # First tool call
            _block_start(1, {"type": "tool_use", "id": "t1", "name": "weather", "input": {}}),
            _block_delta(1, {"type": "input_json_delta", "partial_json": '{"city": "NYC"}'}),
            _block_stop(1),
            # Think about second tool
            _block_start(2, {"type": "thinking", "thinking": ""}),
            _block_delta(2, {"type": "thinking_delta", "thinking": "Also check calendar"}),
            _block_delta(2, {"type": "signature_delta", "signature": "s2"}),
            _block_stop(2),
            # Second tool call
            _block_start(3, {"type": "tool_use", "id": "t2", "name": "calendar", "input": {}}),
            _block_delta(3, {"type": "input_json_delta", "partial_json": '{"date": "today"}'}),
            _block_stop(3),
            _msg_delta("tool_use"),
            _msg_stop(),
        ]
        msg = reassemble_message(events)
        assert len(msg["content"]) == 4
        assert [b["type"] for b in msg["content"]] == [
            "thinking", "tool_use", "thinking", "tool_use"
        ]
        assert msg["content"][1]["input"] == {"city": "NYC"}
        assert msg["content"][3]["input"] == {"date": "today"}


# ---------------------------------------------------------------------------
# 12. parse_sse + reassemble round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_full_round_trip_text(self):
        text = _make_stream(SIMPLE_TEXT_EVENTS)
        msg = reassemble_message(parse_sse(text))
        assert msg["type"] == "message"
        assert msg["id"] == "msg_01"
        assert msg["role"] == "assistant"
        assert msg["content"][0]["text"] == "Hello world"
        assert msg["stop_reason"] == "end_turn"

    def test_full_round_trip_tool_use(self):
        text = _make_stream(TOOL_USE_EVENTS)
        msg = reassemble_message(parse_sse(text))
        assert msg["content"][0]["type"] == "tool_use"
        assert msg["content"][0]["input"] == {"location": "NYC"}

    def test_full_round_trip_thinking(self):
        text = _make_stream(THINKING_EVENTS)
        msg = reassemble_message(parse_sse(text))
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][1]["type"] == "text"


# ---------------------------------------------------------------------------
# 13. /v1/messages proxy endpoint
# ---------------------------------------------------------------------------


class TestProxyEndpoint:
    def _mock_response(self, status_code, text):
        resp = MagicMock()
        resp.status_code = status_code
        resp.text = text
        if status_code != 200:
            resp.json.return_value = json.loads(text)
        return resp

    def _patch_upstream(self, mock_resp):
        """Context manager that patches httpx.AsyncClient to return mock_resp."""
        mock_cls = patch("server.httpx.AsyncClient")

        class _Ctx:
            def __init__(self):
                self.patcher = mock_cls
                self.mock_ctx = None

            def __enter__(self):
                cls = self.patcher.__enter__()
                self.mock_ctx = AsyncMock()
                self.mock_ctx.__aenter__ = AsyncMock(return_value=self.mock_ctx)
                self.mock_ctx.__aexit__ = AsyncMock(return_value=False)
                self.mock_ctx.post = AsyncMock(return_value=mock_resp)
                self.mock_ctx.request = AsyncMock(return_value=mock_resp)
                cls.return_value = self.mock_ctx
                return self.mock_ctx

            def __exit__(self, *args):
                return self.patcher.__exit__(*args)

        return _Ctx()

    def test_reassembles_text_response(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "hi"}]},
                headers={"x-api-key": "test-key", "anthropic-version": "2023-06-01"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["content"][0]["text"] == "Hello world"

    def test_reassembles_tool_use_response(self):
        mock_resp = self._mock_response(200, _make_stream(TOOL_USE_EVENTS))
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        body = resp.json()
        assert body["stop_reason"] == "tool_use"
        assert body["content"][0]["type"] == "tool_use"
        assert body["content"][0]["input"] == {"location": "NYC"}

    def test_reassembles_thinking_response(self):
        mock_resp = self._mock_response(200, _make_stream(THINKING_EVENTS))
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        body = resp.json()
        assert len(body["content"]) == 2
        assert body["content"][0]["type"] == "thinking"
        assert body["content"][1]["type"] == "text"

    def test_forces_stream_true_for_non_stream_request(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-20250514", "stream": False, "messages": []},
            )
            sent_body = mock_ctx.post.call_args.kwargs.get("json") or mock_ctx.post.call_args[1].get("json")
            assert sent_body["stream"] is True

    def test_stream_true_passthrough(self):
        """When caller sets stream=True, proxy passes SSE through directly."""
        sse_text = _make_stream(SIMPLE_TEXT_EVENTS)
        sse_bytes = sse_text.encode()

        async def fake_aiter_bytes():
            yield sse_bytes

        mock_stream_resp = MagicMock()
        mock_stream_resp.aiter_bytes = fake_aiter_bytes

        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            # mock client.stream() as an async context manager
            mock_stream_cm = AsyncMock()
            mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_resp)
            mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.stream = MagicMock(return_value=mock_stream_cm)

            mock_cls.return_value = mock_ctx

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-20250514", "stream": True, "messages": []},
                headers={"x-api-key": "sk-test"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        # Response body should contain raw SSE data, not reassembled JSON
        assert "event: message_start" in resp.text

    def test_stream_true_forwards_body_as_is(self):
        """When stream=True, the body is forwarded unchanged (stream stays True)."""
        async def fake_aiter_bytes():
            yield b"data: {}\n\n"

        mock_stream_resp = MagicMock()
        mock_stream_resp.aiter_bytes = fake_aiter_bytes

        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)

            mock_stream_cm = AsyncMock()
            mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_resp)
            mock_stream_cm.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.stream = MagicMock(return_value=mock_stream_cm)

            mock_cls.return_value = mock_ctx

            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-20250514", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
            )

            call_args = mock_ctx.stream.call_args
            sent_body = call_args.kwargs.get("json") or call_args[1].get("json")
            assert sent_body["stream"] is True

    def test_stream_false_reassembles(self):
        """When stream=False, proxy reassembles into non-streaming response."""
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-20250514", "stream": False, "messages": []},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["content"][0]["text"] == "Hello world"

    def test_stream_absent_reassembles(self):
        """When stream is not set, proxy reassembles into non-streaming response."""
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/v1/messages",
                json={"model": "claude-sonnet-4-20250514", "messages": []},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"

    def test_forwards_upstream_400_error(self):
        error_body = json.dumps({"error": {"type": "invalid_request_error", "message": "bad request"}})
        mock_resp = self._mock_response(400, error_body)
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"

    def test_forwards_upstream_401_auth_error(self):
        error_body = json.dumps({"error": {"type": "authentication_error", "message": "invalid api key"}})
        mock_resp = self._mock_response(401, error_body)
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"

    def test_forwards_upstream_403_permission_error(self):
        error_body = json.dumps({"error": {"type": "permission_error", "message": "forbidden"}})
        mock_resp = self._mock_response(403, error_body)
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 403
        assert resp.json()["error"]["type"] == "permission_error"

    def test_forwards_upstream_404_not_found(self):
        error_body = json.dumps({"error": {"type": "not_found_error", "message": "not found"}})
        mock_resp = self._mock_response(404, error_body)
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 404

    def test_forwards_upstream_429_rate_limit(self):
        error_body = json.dumps({"error": {"type": "rate_limit_error", "message": "rate limited"}})
        mock_resp = self._mock_response(429, error_body)
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 429

    def test_forwards_upstream_500_api_error(self):
        error_body = json.dumps({"error": {"type": "api_error", "message": "internal error"}})
        mock_resp = self._mock_response(500, error_body)
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 500

    def test_forwards_upstream_529_overloaded(self):
        error_body = json.dumps({"error": {"type": "overloaded_error", "message": "overloaded"}})
        mock_resp = self._mock_response(529, error_body)
        with self._patch_upstream(mock_resp):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/messages", json={"messages": []})
        assert resp.status_code == 529

    def test_upstream_non_json_error(self):
        """When upstream returns non-JSON error body, wrap it."""
        resp = MagicMock()
        resp.status_code = 502
        resp.text = "Bad Gateway"
        resp.json.side_effect = Exception("not json")
        with self._patch_upstream(resp):
            client = TestClient(app, raise_server_exceptions=False)
            result = client.post("/v1/messages", json={"messages": []})
        assert result.status_code == 502
        assert result.json()["error"] == "Bad Gateway"

    def test_upstream_empty_error_body(self):
        """When upstream returns empty error body."""
        resp = MagicMock()
        resp.status_code = 503
        resp.text = ""
        resp.json.side_effect = Exception("empty")
        with self._patch_upstream(resp):
            client = TestClient(app, raise_server_exceptions=False)
            result = client.post("/v1/messages", json={"messages": []})
        assert result.status_code == 503
        assert result.json()["error"] == "upstream error"

    def test_forwards_x_api_key(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"messages": []},
                headers={"x-api-key": "sk-ant-test123"},
            )
            sent_headers = mock_ctx.post.call_args.kwargs.get("headers") or mock_ctx.post.call_args[1].get("headers")
            assert sent_headers["x-api-key"] == "sk-ant-test123"

    def test_forwards_authorization_header(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"messages": []},
                headers={"authorization": "Bearer sk-ant-test123"},
            )
            sent_headers = mock_ctx.post.call_args.kwargs.get("headers") or mock_ctx.post.call_args[1].get("headers")
            assert sent_headers["authorization"] == "Bearer sk-ant-test123"

    def test_forwards_anthropic_version(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"messages": []},
                headers={"anthropic-version": "2023-06-01"},
            )
            sent_headers = mock_ctx.post.call_args.kwargs.get("headers") or mock_ctx.post.call_args[1].get("headers")
            assert sent_headers["anthropic-version"] == "2023-06-01"

    def test_forwards_anthropic_beta(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"messages": []},
                headers={"anthropic-beta": "interleaved-thinking-2025-05-14"},
            )
            sent_headers = mock_ctx.post.call_args.kwargs.get("headers") or mock_ctx.post.call_args[1].get("headers")
            assert sent_headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"

    def test_strips_hop_by_hop_headers(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"messages": []},
                headers={"connection": "keep-alive", "transfer-encoding": "chunked"},
            )
            sent_headers = mock_ctx.post.call_args.kwargs.get("headers") or mock_ctx.post.call_args[1].get("headers")
            assert "host" not in sent_headers
            assert "content-length" not in sent_headers
            assert "transfer-encoding" not in sent_headers
            assert "connection" not in sent_headers

    def test_forwards_all_headers(self):
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={"messages": []},
                headers={"x-custom-header": "value", "x-api-key": "sk-test"},
            )
            sent_headers = mock_ctx.post.call_args.kwargs.get("headers") or mock_ctx.post.call_args[1].get("headers")
            assert sent_headers["x-custom-header"] == "value"
            assert sent_headers["x-api-key"] == "sk-test"

    def test_original_body_fields_preserved(self):
        """Non-stream fields like model, max_tokens pass through to upstream."""
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={
                    "model": "claude-opus-4-20250514",
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "system": "You are helpful.",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            sent_body = mock_ctx.post.call_args.kwargs.get("json") or mock_ctx.post.call_args[1].get("json")
            assert sent_body["model"] == "claude-opus-4-20250514"
            assert sent_body["max_tokens"] == 1024
            assert sent_body["temperature"] == 0.7
            assert sent_body["system"] == "You are helpful."
            assert sent_body["stream"] is True

    def test_tools_field_preserved(self):
        """Tool definitions in request body pass through to upstream."""
        mock_resp = self._mock_response(200, _make_stream(SIMPLE_TEXT_EVENTS))
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ]
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post("/v1/messages", json={"messages": [], "tools": tools})
            sent_body = mock_ctx.post.call_args.kwargs.get("json") or mock_ctx.post.call_args[1].get("json")
            assert sent_body["tools"] == tools

    def test_thinking_config_preserved(self):
        """thinking parameter in request body passes through."""
        mock_resp = self._mock_response(200, _make_stream(THINKING_EVENTS))
        with self._patch_upstream(mock_resp) as mock_ctx:
            client = TestClient(app, raise_server_exceptions=False)
            client.post(
                "/v1/messages",
                json={
                    "messages": [],
                    "model": "claude-sonnet-4-20250514",
                    "thinking": {"type": "enabled", "budget_tokens": 10000},
                },
            )
            sent_body = mock_ctx.post.call_args.kwargs.get("json") or mock_ctx.post.call_args[1].get("json")
            assert sent_body["thinking"] == {"type": "enabled", "budget_tokens": 10000}


# ---------------------------------------------------------------------------
# 14. Passthrough endpoint
# ---------------------------------------------------------------------------


class TestPassthrough:
    def test_passthrough_get(self):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {"status": "ok"}

        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.request = AsyncMock(return_value=resp_mock)
            mock_cls.return_value = mock_ctx

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_passthrough_strips_hop_by_hop_headers(self):
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {}

        with patch("server.httpx.AsyncClient") as mock_cls:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.request = AsyncMock(return_value=resp_mock)
            mock_cls.return_value = mock_ctx

            client = TestClient(app, raise_server_exceptions=False)
            client.get("/some/path")

            call_kwargs = mock_ctx.request.call_args
            sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
            assert "host" not in sent_headers
