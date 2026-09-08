from rhubarb.ollama_rescue import (
    rescue_grilling_response,
    rescue_qa_response,
    should_attempt_grilling_rescue,
    should_attempt_qa_rescue,
)


def _generate_response(payload: dict) -> dict:
    """Shape Ollama's `/api/generate` actually returns: the schema-
    constrained JSON comes back as a STRING inside the `response` field."""
    import json

    return {"response": json.dumps(payload)}


# ---------------------------------------------------------------------------
# Rescue-trigger heuristic
# ---------------------------------------------------------------------------


def test_grilling_rescue_triggers_when_parser_empty_and_text_looks_like_questions():
    parsed = {"header": "", "questions": [], "footer": ""}
    raw = 'Question 1: "Should this be Python or Node' + " (malformed, missing closing quote)\n"

    assert should_attempt_grilling_rescue(parsed, raw) is True


def test_grilling_rescue_does_not_trigger_on_genuine_wrap_up():
    parsed = {"header": "Thanks, that's everything I need.", "questions": [], "footer": ""}

    assert should_attempt_grilling_rescue(parsed, "Thanks, that's everything I need.") is False


def test_grilling_rescue_does_not_trigger_when_parser_already_found_questions():
    parsed = {
        "header": "",
        "questions": [{"id": "q1", "text": "Python or Node?", "kind": "open", "options": None, "recommended": None, "recommended_text": None}],
        "footer": "",
    }
    raw = 'Question 1: "Python or Node?"\n'

    assert should_attempt_grilling_rescue(parsed, raw) is False


def test_qa_rescue_triggers_when_parser_empty_and_text_looks_like_a_qa_session():
    parsed = {"prd": None, "issues": []}
    raw = 'QA session for PRD 98: this is malformed and missing the quoted title\n'

    assert should_attempt_qa_rescue(parsed, raw) is True


def test_qa_rescue_does_not_trigger_on_unrelated_empty_text():
    parsed = {"prd": None, "issues": []}

    assert should_attempt_qa_rescue(parsed, "Implemented PRD #5, nothing to verify yet.") is False


# ---------------------------------------------------------------------------
# Rescue call: success, invalid shape, timeout/connection failure
# ---------------------------------------------------------------------------


def test_rescue_grilling_response_returns_valid_shape_from_ollama():
    payload = {
        "header": "",
        "footer": "",
        "questions": [
            {
                "id": "q1",
                "text": "Should this be Python or Node?",
                "kind": "single",
                "options": ["Python", "Node"],
                "recommended": [1],
                "recommended_text": None,
            }
        ],
    }

    def fake_post(url, body, *, timeout):
        assert url == "http://localhost:11434/api/generate"
        assert body["model"] == "llama3.2:1b"
        assert body["format"]  # a schema was passed
        assert body["stream"] is False
        return _generate_response(payload)

    result = rescue_grilling_response("some malformed text", http_post=fake_post)

    assert result == payload


def test_rescue_grilling_response_returns_none_when_ollama_output_is_missing_required_keys():
    def fake_post(url, body, *, timeout):
        return _generate_response({"header": "", "questions": []})  # missing "footer"

    assert rescue_grilling_response("text", http_post=fake_post) is None


def test_rescue_grilling_response_returns_none_when_a_question_has_wrong_types():
    payload = {
        "header": "",
        "footer": "",
        "questions": [{"id": "q1", "text": "X?", "kind": "bogus-kind", "options": None, "recommended": None, "recommended_text": None}],
    }

    def fake_post(url, body, *, timeout):
        return _generate_response(payload)

    assert rescue_grilling_response("text", http_post=fake_post) is None


def test_rescue_grilling_response_returns_none_on_unparseable_json():
    def fake_post(url, body, *, timeout):
        return {"response": "not valid json at all {{{"}

    assert rescue_grilling_response("text", http_post=fake_post) is None


def test_rescue_grilling_response_returns_none_on_timeout():
    def timing_out_post(url, body, *, timeout):
        raise TimeoutError("Ollama took too long")

    assert rescue_grilling_response("text", http_post=timing_out_post) is None


def test_rescue_grilling_response_returns_none_on_connection_failure():
    def unreachable_post(url, body, *, timeout):
        raise ConnectionError("nobody home")

    assert rescue_grilling_response("text", http_post=unreachable_post) is None


def test_rescue_qa_response_returns_valid_shape_from_ollama():
    payload = {
        "prd": {"number": 98, "title": "Structured question format"},
        "issues": [
            {
                "number": 99,
                "title": "Fix textarea auto-grow bug",
                "questions": [{"id": "issue99-q1", "text": "Does it resize?", "recommended_text": "Yes."}],
            }
        ],
    }

    def fake_post(url, body, *, timeout):
        return _generate_response(payload)

    assert rescue_qa_response("malformed qa text", http_post=fake_post) == payload


def test_rescue_qa_response_returns_none_when_issue_missing_required_keys():
    payload = {"prd": {"number": 98, "title": "X"}, "issues": [{"number": 99, "questions": []}]}  # missing "title"

    def fake_post(url, body, *, timeout):
        return _generate_response(payload)

    assert rescue_qa_response("text", http_post=fake_post) is None


def test_rescue_qa_response_returns_none_when_prd_is_wrong_type():
    payload = {"prd": "not an object", "issues": []}

    def fake_post(url, body, *, timeout):
        return _generate_response(payload)

    assert rescue_qa_response("text", http_post=fake_post) is None


def test_rescue_qa_response_accepts_null_prd():
    payload = {"prd": None, "issues": []}

    def fake_post(url, body, *, timeout):
        return _generate_response(payload)

    assert rescue_qa_response("text", http_post=fake_post) == payload
