def test_no_duplicate_questions_across_papers():
    paper_a = parse_pdf("fixtures/paper_2022.pdf", type="past_paper")
    paper_b = parse_pdf("fixtures/paper_2023.pdf", type="past_paper")
    flagged = flag_duplicate_questions(paper_a + paper_b)
    # nothing removed — same count in, same count out
    assert len(flagged) == len(paper_a) + len(paper_b)
    # but at least one entry is marked as a repeat
    assert any(q["is_repeat"] for q in flagged)

def test_format_tries_pdf_date_before_filename():
    # PDF text contains "Academic Year 2022/2023" but filename has no date
    fmt = extract_format(["fixtures/paper_no_date_filename.pdf"])
    assert fmt["source_year"] == 2023
    assert fmt["date_source"] == "pdf_text"

def test_format_falls_back_to_filename_if_no_pdf_date():
    fmt = extract_format(["fixtures/2021_paper.pdf"])  # no date inside PDF text
    assert fmt["source_year"] == 2021
    assert fmt["date_source"] == "filename"

def test_pdf_date_takes_precedence_over_filename():
    fmt = extract_format(["fixtures/2022_conflict.pdf"])
    assert fmt["source_year"] == 2024
    assert fmt["date_source"] == "pdf_text"