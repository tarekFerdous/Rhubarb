from rhubarb.ollama_rescue import classify_turn_needs_input


def _generate_response(payload: dict) -> dict:
    """Shape Ollama's `/api/generate` actually returns: the schema-
    constrained JSON comes back as a STRING inside the `response` field."""
    import json

    return {"response": json.dumps(payload)}


# ---------------------------------------------------------------------------
# Console logging (issue #118)
#
# Issue #230 deleted `rescue_grilling_response`/`rescue_qa_response` (and the
# rescue-trigger heuristics/rescue-call/content-validation tests that used to
# exercise them here) -- the regex-parser + Ollama-rescue extraction chain
# they belonged to is fully retired; every phase now extracts via the
# parser-session/`/rhubarb:parse-interview` skill pipeline instead (see
# `tests/test_parser_session.py`/`tests/test_sessions.py`). `_call_ollama`'s
# own prompt-logging behavior (still shared with `classify_turn_needs_input`
# below) is exercised here via that still-live function instead.
# ---------------------------------------------------------------------------


def test_call_ollama_prints_the_fully_formatted_prompt_before_the_http_call(capsys):
    calls = []

    def fake_post(url, body, *, timeout):
        calls.append(body["prompt"])
        return _generate_response({"needs_input": False, "reason": None})

    classify_turn_needs_input("some raw turn text", "grilling", http_post=fake_post)

    captured = capsys.readouterr()
    assert calls, "http_post was never called"
    assert calls[0] in captured.out
    assert "some raw turn text" in captured.out


def test_call_ollama_prints_prompt_even_when_the_call_ultimately_fails(capsys):
    def timing_out_post(url, body, *, timeout):
        raise TimeoutError("Ollama took too long")

    classify_turn_needs_input("malformed text that will time out", "grilling", http_post=timing_out_post)

    captured = capsys.readouterr()
    assert "malformed text that will time out" in captured.out


# ---------------------------------------------------------------------------
# Needs-input classification (issue #175, child of PRD #174)
# ---------------------------------------------------------------------------


def test_classify_turn_needs_input_returns_valid_shape_from_ollama():
    payload = {"needs_input": True, "reason": "Asked the user to pick an option."}

    def fake_post(url, body, *, timeout):
        assert url == "http://localhost:11434/api/generate"
        assert body["model"] == "llama3.2:1b"
        assert body["format"]  # a schema was passed
        assert body["stream"] is False
        assert "grilling" in body["prompt"]
        return _generate_response(payload)

    result = classify_turn_needs_input("Which database should we use?", "grilling", http_post=fake_post)

    assert result == payload


def test_classify_turn_needs_input_accepts_null_reason():
    payload = {"needs_input": False, "reason": None}

    def fake_post(url, body, *, timeout):
        return _generate_response(payload)

    assert classify_turn_needs_input("All done, PRD published.", "publishing", http_post=fake_post) == payload


def test_classify_turn_needs_input_returns_none_when_needs_input_is_missing():
    def fake_post(url, body, *, timeout):
        return _generate_response({"reason": "no needs_input key"})

    assert classify_turn_needs_input("text", "grilling", http_post=fake_post) is None


def test_classify_turn_needs_input_returns_none_when_needs_input_is_wrong_type():
    def fake_post(url, body, *, timeout):
        return _generate_response({"needs_input": "yes", "reason": None})

    assert classify_turn_needs_input("text", "grilling", http_post=fake_post) is None


def test_classify_turn_needs_input_returns_none_when_reason_is_wrong_type():
    def fake_post(url, body, *, timeout):
        return _generate_response({"needs_input": True, "reason": 123})

    assert classify_turn_needs_input("text", "grilling", http_post=fake_post) is None


def test_classify_turn_needs_input_returns_none_on_unparseable_json():
    def fake_post(url, body, *, timeout):
        return {"response": "not valid json at all {{{"}

    assert classify_turn_needs_input("text", "grilling", http_post=fake_post) is None


def test_classify_turn_needs_input_returns_none_on_timeout():
    def timing_out_post(url, body, *, timeout):
        raise TimeoutError("Ollama took too long")

    assert classify_turn_needs_input("text", "grilling", http_post=timing_out_post) is None


def test_classify_turn_needs_input_returns_none_on_connection_failure():
    def unreachable_post(url, body, *, timeout):
        raise ConnectionError("nobody home")

    assert classify_turn_needs_input("text", "grilling", http_post=unreachable_post) is None
