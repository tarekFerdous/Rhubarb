from rhubarb.question_files import delete_question_file, read_question_file


def test_read_question_file_returns_exact_content_when_present(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / "rhubarb_question.md").write_text('Question 1: "Python or Node?"', encoding="utf-8")

    assert read_question_file(str(tmp_path), "rhubarb_question.md") == 'Question 1: "Python or Node?"'


def test_read_question_file_returns_none_when_absent(tmp_path):
    assert read_question_file(str(tmp_path), "rhubarb_question.md") is None


def test_read_question_file_returns_none_for_unset_cwd():
    assert read_question_file(None, "rhubarb_question.md") is None


def test_delete_question_file_removes_an_existing_file(tmp_path):
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    target = claude_dir / "rhubarb_question.md"
    target.write_text("content", encoding="utf-8")

    delete_question_file(str(tmp_path), "rhubarb_question.md")

    assert not target.exists()


def test_delete_question_file_does_not_raise_when_absent(tmp_path):
    delete_question_file(str(tmp_path), "rhubarb_question.md")  # no .claude dir at all


def test_delete_question_file_does_not_raise_for_unset_cwd():
    delete_question_file(None, "rhubarb_question.md")
