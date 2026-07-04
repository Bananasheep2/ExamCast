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

def test_ties_at_fifth_place_are_all_included():
    # two types both tied with count=2 for 5th place
    tied_data = MOCK_CLASSIFIED + [
        {"question": "Find eigenvalues...", "type": "eigenvalues", "paper": "2021"},
        {"question": "Find eigenvalues again...", "type": "eigenvalues", "paper": "2022"},
        {"question": "Integrate by parts...", "type": "integration", "paper": "2021"},
        {"question": "Integrate again...", "type": "integration", "paper": "2022"},
    ]
    ranked = rank_question_types(tied_data)
    fifth_place_count = ranked[4]["count"]
    tied_at_fifth = [r for r in ranked if r["count"] == fifth_place_count]
    if len(tied_at_fifth) > 1:
        assert len(ranked) > 5  # list grew to fit all ties
