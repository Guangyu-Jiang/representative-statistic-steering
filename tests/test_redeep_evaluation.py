from repstat_steering.redeep_evaluation import (
    parse_local_pairwise_output,
    parse_local_judge_output,
    passage_grounding_ratio,
    reference_token_recall,
    token_f1,
)


def test_parse_local_judge_json_and_fallback() -> None:
    label, reason = parse_local_judge_output(
        '{"label":"SUPPORTED","reason":"grounded"}'
    )
    assert label == "SUPPORTED"
    assert reason == "grounded"
    label, _ = parse_local_judge_output("The answer is UNSUPPORTED.")
    assert label == "UNSUPPORTED"


def test_parse_local_pairwise_output() -> None:
    winner, reason = parse_local_pairwise_output(
        '{"winner":"B","reason":"more faithful"}'
    )
    assert winner == "B"
    assert reason == "more faithful"


def test_reference_metrics_reward_matching_content() -> None:
    response = "Secret of the Incas and The Rose Tattoo"
    reference = "Secret of the Incas\nThe Rose Tattoo\nLawman"
    assert token_f1(response, reference) > 0.7
    assert reference_token_recall(response, reference) > 0.5


def test_passage_grounding_ratio_uses_question_and_passage() -> None:
    grounded = passage_grounding_ratio(
        "The album includes Hey June", "Track listing: Hey June", "Which songs?"
    )
    unrelated = passage_grounding_ratio(
        "Saturn has crystal oceans", "Track listing: Hey June", "Which songs?"
    )
    assert grounded > unrelated
