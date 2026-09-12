"""Translate raw Claude CLI NDJSON events into human-readable app-level events.

Pure and stateless: one raw event dict in, one app-level event dict (or
`None` for filtered noise) out. No dollar-cost figure is ever produced --
`result` events carry `total_cost_usd` and `rate_limit_event`/usage fields
are deliberately not forwarded verbatim.
"""

_FILE_TOOLS = {"Read", "Edit", "Write"}
_PATTERN_TOOLS = {"Grep", "Glob"}


def _action_summary(tool_use: dict) -> dict:
    name = tool_use.get("name", "")
    tool_input = tool_use.get("input", {})

    if name == "Bash":
        summary = f"$ {tool_input.get('command', '')}"
    elif name in _FILE_TOOLS:
        summary = f"{name} {tool_input.get('file_path', '')}"
    elif name in _PATTERN_TOOLS:
        summary = f"{name} {tool_input.get('pattern', '')}"
    else:
        summary = name

    return {"type": "action", "summary": summary}


def translate_event(raw_event: dict) -> dict | None:
    kind = raw_event.get("type")

    if kind == "system":
        return None

    if kind == "terminal_output":
        # Passed straight through from `PtyEngine.stream_turn` (issue #88) --
        # the raw PTY chunk, unmodified, for a live-terminal-view consumer.
        # Not filtered/parsed like every other event here since there is
        # nothing to translate: `data` already IS what the frontend renders.
        return {"type": "terminal_output", "data": raw_event.get("data", "")}

    if kind == "stall":
        # Passed straight through from `PtyEngine.stream_turn` (issue #169):
        # `data` is the buffer-so-far, already rendered via
        # `_render_terminal_text` -- `session_runner._run_turn` folds this
        # into the existing `turn` event shape (`stalled`/`stalled_context`)
        # rather than publishing this raw shape directly.
        return {"type": "stall", "data": raw_event.get("data", "")}

    if kind == "stream_event":
        event = raw_event.get("event", {})
        event_type = event.get("type")

        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                return {"type": "text", "text": delta.get("text", "")}
            return None

        if event_type == "content_block_start":
            content_block = event.get("content_block", {})
            if content_block.get("type") == "thinking":
                return {"type": "action", "summary": "thinking…"}
            return None

        return None

    if kind == "assistant":
        content = raw_event.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "tool_use":
                return _action_summary(block)
        return None

    if kind == "result":
        return {
            "type": "turn",
            "result": raw_event.get("result", ""),
            "session_id": raw_event.get("session_id"),
            "is_error": raw_event.get("is_error", False),
        }

    if kind == "rate_limit_event":
        windows = raw_event.get("rate_limit_info", {}).get("unifiedWindows", {})
        return {
            "type": "usage",
            "five_hour_pct": windows.get("five_hour", {}).get("utilization"),
            "seven_day_pct": windows.get("seven_day", {}).get("utilization"),
        }

    return None
