from caa_perturbation.local_judge import parse_score


def test_parse_tagged_score():
    assert parse_score("<score>7</score>") == 7
    assert parse_score("<score>10</score>") == 10


def test_parse_score_rejects_missing_number():
    assert parse_score("No score available") is None
