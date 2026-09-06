import pytest

from baton.stream_translate import translate_event

# Raw event shapes below are trimmed captures from a real
# `claude -p --output-format stream-json --include-partial-messages --verbose` run.

SYSTEM_INIT = {
    "type": "system",
    "subtype": "init",
    "cwd": "D:\\GitHub Repos\\Baton",
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
    "tools": ["Bash", "Read"],
    "model": "claude-sonnet-5",
}

SYSTEM_HOOK_STARTED = {
    "type": "system",
    "subtype": "hook_started",
    "hook_id": "564f0ea4-d3ff-4967-8ca0-e73342ec532a",
    "hook_name": "SessionStart:startup",
    "hook_event": "SessionStart",
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
}

TEXT_DELTA = {
    "type": "stream_event",
    "event": {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hi"},
    },
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
    "parent_tool_use_id": None,
}

THINKING_DELTA = {
    "type": "stream_event",
    "event": {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "", "estimated_tokens": 50},
    },
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
}

THINKING_BLOCK_START = {
    "type": "stream_event",
    "event": {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
}

ASSISTANT_WITH_READ_TOOL_USE = {
    "type": "assistant",
    "message": {
        "model": "claude-sonnet-5",
        "id": "msg_011CeezeM7gi68atQZVC8SsE",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_018SFjLEyKF31Lkqc6b1fPC9",
                "name": "Read",
                "input": {"file_path": "D:\\GitHub Repos\\Baton\\baton\\db.py"},
            }
        ],
    },
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
}

ASSISTANT_WITH_BASH_TOOL_USE = {
    "type": "assistant",
    "message": {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "Bash",
                "input": {"command": "npm test"},
            }
        ],
    },
}

ASSISTANT_WITH_GREP_TOOL_USE = {
    "type": "assistant",
    "message": {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_def",
                "name": "Grep",
                "input": {"pattern": "def foo"},
            }
        ],
    },
}

ASSISTANT_WITH_OTHER_TOOL_USE = {
    "type": "assistant",
    "message": {
        "content": [{"type": "tool_use", "id": "toolu_ghi", "name": "WebSearch", "input": {"query": "x"}}],
    },
}

ASSISTANT_TEXT_ONLY = {
    "type": "assistant",
    "message": {
        "content": [{"type": "text", "text": "Hi"}],
    },
}

RESULT_SUCCESS = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "I've read `baton/db.py` — it's the SQLite persistence layer.",
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
    "total_cost_usd": 0.0711224,
    "duration_ms": 5381,
}

TERMINAL_OUTPUT = {
    "type": "terminal_output",
    "data": "\x1b[32mHello\x1b[0m\r\n",
}

RATE_LIMIT_EVENT = {
    "type": "rate_limit_event",
    "rate_limit_info": {
        "status": "allowed",
        "rateLimitType": "five_hour",
        "unifiedWindows": {
            "five_hour": {"utilization": 0.42, "resetsAt": 1788379800},
            "seven_day": {"utilization": 0.04, "resetsAt": 1788440400},
        },
    },
    "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
}


@pytest.mark.parametrize(
    "raw_event, expected",
    [
        (SYSTEM_INIT, None),
        (SYSTEM_HOOK_STARTED, None),
        (TEXT_DELTA, {"type": "text", "text": "Hi"}),
        (THINKING_DELTA, None),
        (THINKING_BLOCK_START, {"type": "action", "summary": "thinking…"}),
        (
            ASSISTANT_WITH_READ_TOOL_USE,
            {"type": "action", "summary": "Read D:\\GitHub Repos\\Baton\\baton\\db.py"},
        ),
        (ASSISTANT_WITH_BASH_TOOL_USE, {"type": "action", "summary": "$ npm test"}),
        (ASSISTANT_WITH_GREP_TOOL_USE, {"type": "action", "summary": "Grep def foo"}),
        (ASSISTANT_WITH_OTHER_TOOL_USE, {"type": "action", "summary": "WebSearch"}),
        (ASSISTANT_TEXT_ONLY, None),
        (
            RESULT_SUCCESS,
            {
                "type": "turn",
                "result": "I've read `baton/db.py` — it's the SQLite persistence layer.",
                "session_id": "7d373094-4d99-499c-8766-c6c9914b44b7",
                "is_error": False,
            },
        ),
        (RATE_LIMIT_EVENT, {"type": "usage", "five_hour_pct": 0.42, "seven_day_pct": 0.04}),
        (
            TERMINAL_OUTPUT,
            {"type": "terminal_output", "data": "\x1b[32mHello\x1b[0m\r\n"},
        ),
    ],
)
def test_translate_event(raw_event, expected):
    assert translate_event(raw_event) == expected


def test_translate_event_never_includes_a_dollar_cost_figure():
    for raw_event in (SYSTEM_INIT, TEXT_DELTA, ASSISTANT_WITH_READ_TOOL_USE, RATE_LIMIT_EVENT, RESULT_SUCCESS):
        translated = translate_event(raw_event)
        if translated is not None:
            assert "cost" not in translated
            assert "total_cost_usd" not in translated
