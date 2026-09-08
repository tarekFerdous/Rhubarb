from rhubarb.qa_parser import parse_grilling_response, parse_qa_response


# ---------------------------------------------------------------------------
# Grilling/do format
# ---------------------------------------------------------------------------


def test_single_select_question_with_options_and_recommendation():
    text = (
        'Question 1: "Should this be Python or Node?"\n'
        "Options:\n"
        'Option 1: "Python"\n'
        'Option 2: "Node"\n'
        "Recommended: [1]\n"
    )

    result = parse_grilling_response(text)

    assert result["header"] == ""
    assert result["footer"] == ""
    assert len(result["questions"]) == 1
    q = result["questions"][0]
    assert q["id"] == "q1"
    assert q["text"] == "Should this be Python or Node?"
    assert q["kind"] == "single"
    assert q["options"] == ["Python", "Node"]
    assert q["recommended"] == [1]
    assert q["recommended_text"] is None


def test_multi_select_question_with_multi_value_recommendation():
    text = (
        'Question 2 (select multiple): "Which environments should this support?"\n'
        "Options:\n"
        'Option 1: "Dev"\n'
        'Option 2: "Staging"\n'
        'Option 3: "Prod"\n'
        "Recommended: [1, 3]\n"
    )

    result = parse_grilling_response(text)

    questions = result["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["id"] == "q2"
    assert q["kind"] == "multi"
    assert q["options"] == ["Dev", "Staging", "Prod"]
    assert q["recommended"] == [1, 3]
    assert q["recommended_text"] is None


def test_open_ended_question_with_recommended_text():
    text = 'Question 3: "Where should this run?"\nRecommended text: "On the existing droplet, matching current infra."\n'

    result = parse_grilling_response(text)

    questions = result["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["id"] == "q3"
    assert q["kind"] == "open"
    assert q["options"] is None
    assert q["recommended"] is None
    assert q["recommended_text"] == "On the existing droplet, matching current infra."


def test_open_ended_question_without_recommended_text():
    text = 'Question 1: "Where should this run?"\n'

    result = parse_grilling_response(text)

    questions = result["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["kind"] == "open"
    assert q["options"] is None
    assert q["recommended"] is None
    assert q["recommended_text"] is None


def test_header_and_footer_preserved_verbatim_around_multi_question_round():
    text = (
        "Here's what I need to nail down before we proceed.\n"
        "\n"
        'Question 1: "Should this be Python or Node?"\n'
        "Options:\n"
        'Option 1: "Python"\n'
        'Option 2: "Node"\n'
        "Recommended: [1]\n"
        "\n"
        'Question 2: "Where should results be persisted?"\n'
        'Recommended text: "SQLite, matching the existing db module."\n'
        "\n"
        "Based on your answers, a second wave might be needed.\n"
    )

    result = parse_grilling_response(text)

    assert result["header"] == "Here's what I need to nail down before we proceed."
    assert result["footer"] == "Based on your answers, a second wave might be needed."

    questions = result["questions"]
    assert len(questions) == 2

    q1, q2 = questions
    assert q1["id"] == "q1"
    assert q1["kind"] == "single"
    assert q1["options"] == ["Python", "Node"]
    assert q1["recommended"] == [1]

    assert q2["id"] == "q2"
    assert q2["kind"] == "open"
    assert q2["recommended_text"] == "SQLite, matching the existing db module."

    assert q1["id"] != q2["id"]


def test_bare_open_question_with_no_structured_lines_treats_trailing_prose_as_footer():
    text = 'Question 1: "Where should this run?"\n\nThanks, that answers it for now.\n'

    result = parse_grilling_response(text)

    questions = result["questions"]
    assert len(questions) == 1
    assert questions[0]["recommended_text"] is None
    assert result["footer"] == "Thanks, that answers it for now."


def test_old_and_plain_text_formats_are_no_longer_detected():
    text = "1. What should it do?\n2. Who is it for?\n- Some bullet point?\n"

    result = parse_grilling_response(text)

    assert result["questions"] == []
    assert result["footer"] == ""
    assert result["header"] == "1. What should it do? 2. Who is it for? - Some bullet point?"


def test_wrap_up_message_with_no_questions_is_pure_header():
    text = "Thanks, that's everything I need."

    result = parse_grilling_response(text)

    assert result["questions"] == []
    assert result["header"] == "Thanks, that's everything I need."
    assert result["footer"] == ""


# ---------------------------------------------------------------------------
# QA format
# ---------------------------------------------------------------------------


def test_qa_single_issue_single_question_with_recommended_text():
    text = (
        'QA session for PRD 98: "Structured question format"\n'
        "\n"
        'Issue 99: "Fix textarea auto-grow bug"\n'
        'Question 1: "Does a pre-filled recommendation box resize immediately?"\n'
        'Recommended text: "Yes, confirmed in the browser."\n'
    )

    result = parse_qa_response(text)

    assert result["prd"] == {"number": 98, "title": "Structured question format"}
    assert len(result["issues"]) == 1
    issue = result["issues"][0]
    assert issue["number"] == 99
    assert issue["title"] == "Fix textarea auto-grow bug"
    assert len(issue["questions"]) == 1
    q = issue["questions"][0]
    assert q["id"] == "issue99-q1"
    assert q["text"] == "Does a pre-filled recommendation box resize immediately?"
    assert q["recommended_text"] == "Yes, confirmed in the browser."


def test_qa_multiple_issues_multiple_questions_grouped_correctly():
    text = (
        'QA session for PRD 98: "Structured question format"\n'
        "\n"
        'Issue 99: "Fix textarea auto-grow bug"\n'
        'Question 1: "Does a pre-filled recommendation box resize immediately?"\n'
        'Recommended text: "Yes, confirmed in the browser."\n'
        'Question 2: "Does typing still auto-grow the box?"\n'
        "\n"
        'Issue 100: "Rewrite qa_parser.py"\n'
        'Question 1: "Do the new unit tests cover multi-select?"\n'
        'Recommended text: "Yes, see test_multi_select_question_with_multi_value_recommendation."\n'
    )

    result = parse_qa_response(text)

    assert result["prd"] == {"number": 98, "title": "Structured question format"}
    assert len(result["issues"]) == 2

    issue_99, issue_100 = result["issues"]
    assert issue_99["number"] == 99
    assert len(issue_99["questions"]) == 2
    assert issue_99["questions"][0]["id"] == "issue99-q1"
    assert issue_99["questions"][1]["id"] == "issue99-q2"
    assert issue_99["questions"][1]["recommended_text"] is None

    assert issue_100["number"] == 100
    assert len(issue_100["questions"]) == 1
    assert issue_100["questions"][0]["id"] == "issue100-q1"
    assert issue_100["questions"][0]["recommended_text"] == (
        "Yes, see test_multi_select_question_with_multi_value_recommendation."
    )

    all_ids = [q["id"] for issue in result["issues"] for q in issue["questions"]]
    assert len(all_ids) == len(set(all_ids))


def test_qa_response_without_prd_header_returns_empty_result():
    assert parse_qa_response("Implemented PRD #5, nothing to verify yet.") == {"prd": None, "issues": []}
    assert parse_qa_response("") == {"prd": None, "issues": []}


# ---------------------------------------------------------------------------
# Reflow: tolerance for terminal word-wrap (issue #109/#111)
# ---------------------------------------------------------------------------


def test_grilling_question_header_wrapped_across_multiple_physical_lines_still_parses():
    text = (
        'Question 2: "The ~/.baton/db.py legacy path constant in rhubarb/db.py\n'
        "(and its tests) is intentional backward-compat migration code, not\n"
        'user-facing text. What should happen to it?"\n'
        "Options:\n"
        'Option 1: "Leave it untouched."\n'
        'Option 2: "Rename/remove it anyway."\n'
        "Recommended: [1]\n"
    )

    result = parse_grilling_response(text)

    questions = result["questions"]
    assert len(questions) == 1
    q = questions[0]
    assert q["id"] == "q2"
    assert q["text"] == (
        "The ~/.baton/db.py legacy path constant in rhubarb/db.py (and its tests) "
        "is intentional backward-compat migration code, not user-facing text. "
        "What should happen to it?"
    )
    assert q["options"] == ["Leave it untouched.", "Rename/remove it anyway."]
    assert q["recommended"] == [1]


def test_grilling_option_wrapped_across_multiple_physical_lines_still_parses():
    text = (
        'Question 1: "Should this be Python or Node?"\n'
        "Options:\n"
        'Option 1: "Leave it untouched — it\'s functional legacy-path handling,\n'
        'correctly named for what it does."\n'
        'Option 2: "Rename/remove it anyway as part of this cleanup."\n'
        "Recommended: [1]\n"
    )

    result = parse_grilling_response(text)

    q = result["questions"][0]
    assert q["options"] == [
        "Leave it untouched — it's functional legacy-path handling, correctly named for what it does.",
        "Rename/remove it anyway as part of this cleanup.",
    ]


def test_reflow_does_not_merge_across_a_blank_line_boundary():
    """A blank line between a round's footer prose and free-standing text
    (or between two rounds/questions) is a genuine separator, not a wrap
    artifact, and must survive the reflow pass untouched."""
    text = (
        'Question 1: "Should this be Python or Node?"\n'
        "Options:\n"
        'Option 1: "Python"\n'
        'Option 2: "Node"\n'
        "Recommended: [1]\n"
        "\n"
        "Based on your answers, a second wave might be needed.\n"
    )

    result = parse_grilling_response(text)

    assert len(result["questions"]) == 1
    assert result["footer"] == "Based on your answers, a second wave might be needed."


def test_qa_question_wrapped_across_multiple_physical_lines_still_parses():
    text = (
        'QA session for PRD 98: "Structured question format"\n'
        "\n"
        'Issue 99: "Fix textarea auto-grow bug"\n'
        'Question 1: "Does a pre-filled recommendation box resize immediately,\n'
        'without requiring the user to touch it first?"\n'
        'Recommended text: "Yes, confirmed in the browser after checking\n'
        'multiple question kinds."\n'
    )

    result = parse_qa_response(text)

    q = result["issues"][0]["questions"][0]
    assert q["text"] == (
        "Does a pre-filled recommendation box resize immediately, without requiring the user to touch it first?"
    )
    assert q["recommended_text"] == "Yes, confirmed in the browser after checking multiple question kinds."


def test_reflow_does_not_merge_across_a_blank_line_between_qa_issues():
    text = (
        'QA session for PRD 98: "Structured question format"\n'
        "\n"
        'Issue 99: "Fix textarea auto-grow bug"\n'
        'Question 1: "Does it resize?"\n'
        "\n"
        'Issue 100: "Rewrite qa_parser.py"\n'
        'Question 1: "Do the new tests pass?"\n'
    )

    result = parse_qa_response(text)

    assert len(result["issues"]) == 2
    assert result["issues"][0]["number"] == 99
    assert result["issues"][1]["number"] == 100
