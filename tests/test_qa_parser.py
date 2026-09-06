from rhubarb.qa_parser import parse_grilling_response


def test_question_with_explicit_options_list():
    text = (
        "❓ **Q1** - **Language/runtime**: Should this be Python or Node?\n"
        "Options: Python, Node\n"
        "\n"
        "➡️ Python, since the rest of the codebase is Python.\n"
    )

    result = parse_grilling_response(text)

    assert result["preamble"] == ""
    assert len(result["sections"]) == 1
    assert result["sections"][0]["heading"] is None
    questions = result["sections"][0]["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["id"] == "q1"
    assert q["title"] == "Language/runtime"
    assert q["text"] == "Should this be Python or Node?"
    assert q["recommendation"] == "Python, since the rest of the codebase is Python."
    assert q["options"] == ["Python", "Node"]


def test_question_falls_back_to_inline_or_heuristic_without_options_line():
    text = (
        "❓ **Q1** - **Storage**: SQLite or Postgres?\n"
        "\n"
        "➡️ SQLite, matching the existing db module.\n"
    )

    result = parse_grilling_response(text)

    questions = result["sections"][0]["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["title"] == "Storage"
    assert q["text"] == "SQLite or Postgres?"
    assert q["recommendation"] == "SQLite, matching the existing db module."
    assert q["options"] == ["SQLite", "Postgres"]


def test_question_with_no_recommendation_line():
    text = "❓ **Q1** - **Deployment**: Where should this run?\n"

    result = parse_grilling_response(text)

    questions = result["sections"][0]["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["title"] == "Deployment"
    assert q["text"] == "Where should this run?"
    assert q["recommendation"] is None
    assert q["options"] is None


def test_wrap_up_message_with_no_questions_is_pure_preamble():
    text = "Thanks, that's everything I need."

    result = parse_grilling_response(text)

    assert result["sections"] == []
    assert result["preamble"] == "Thanks, that's everything I need."


def test_multi_question_round_separated_by_dashes():
    text = (
        "Here's what I need to nail down before we proceed.\n"
        "\n"
        "❓ **Q1** - **Language/runtime**: Should this be Python or Node?\n"
        "Options: Python, Node\n"
        "\n"
        "➡️ Python, since the rest of the codebase is Python.\n"
        "\n"
        "---\n"
        "\n"
        "❓ **Q2** - **Storage**: Where should results be persisted?\n"
        "\n"
        "➡️ SQLite, matching the existing db module.\n"
    )

    result = parse_grilling_response(text)

    assert result["preamble"] == "Here's what I need to nail down before we proceed."
    assert len(result["sections"]) == 1
    questions = result["sections"][0]["questions"]
    assert len(questions) == 2

    q1, q2 = questions
    assert q1["id"] == "q1"
    assert q1["title"] == "Language/runtime"
    assert q1["text"] == "Should this be Python or Node?"
    assert q1["recommendation"] == "Python, since the rest of the codebase is Python."
    assert q1["options"] == ["Python", "Node"]

    assert q2["id"] == "q2"
    assert q2["title"] == "Storage"
    assert q2["text"] == "Where should results be persisted?"
    assert q2["recommendation"] == "SQLite, matching the existing db module."
    assert q2["options"] is None

    assert q1["id"] != q2["id"]


def test_old_numbered_and_bulleted_format_is_no_longer_detected():
    text = "1. What should it do?\n2. Who is it for?\n- Some bullet point?\n"

    result = parse_grilling_response(text)

    assert result["sections"] == []
    assert result["preamble"] == "1. What should it do? 2. Who is it for? - Some bullet point?"


def test_question_without_bold_title_shape_has_none_title():
    text = "❓ **Q1** - Should this run in Docker?\n\n➡️ Yes, Docker.\n"

    result = parse_grilling_response(text)

    questions = result["sections"][0]["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["title"] is None
    assert q["text"] == "Should this run in Docker?"
    assert q["recommendation"] == "Yes, Docker."
