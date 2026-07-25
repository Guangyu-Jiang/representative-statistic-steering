from __future__ import annotations

import json
from pathlib import Path

from aen_replication.data.situatedqa import build_situatedqa_pairs


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def test_builder_keeps_geo_records_without_is_dependent(tmp_path: Path) -> None:
    temp_path = tmp_path / "temp.jsonl"
    geo_path = tmp_path / "geo.jsonl"

    _write_jsonl(
        temp_path,
        [
            {
                "id": "temp-1",
                "question": "when did x happen",
                "is_dependent": True,
                "context_answer_pairs": [
                    {
                        "edited_question": "when did x happen as of 2020",
                    }
                ],
            },
            {
                "id": "temp-2",
                "question": "when did y happen",
                "is_dependent": False,
                "context_answer_pairs": [
                    {
                        "edited_question": "when did y happen as of 2020",
                    }
                ],
            },
        ],
    )
    _write_jsonl(
        geo_path,
        [
            {
                "id": "geo-1",
                "question": "who is the prime minister",
                "context_answer_pairs": [
                    {
                        "edited_question": "who is the prime minister in canada",
                    }
                ],
            }
        ],
    )

    df = build_situatedqa_pairs(
        temp_paths=[str(temp_path)],
        geo_paths=[str(geo_path)],
        seed=13,
        selection_strategy="random_seeded",
    )

    assert set(df["pair_id"]) == {"situatedqa__temp__temp-1", "situatedqa__geo__geo-1"}
    assert "situatedqa__temp__temp-2" not in set(df["pair_id"])
    assert set(df["context_type"]) == {"temp", "geo"}


def test_builder_supports_first_selection_strategy(tmp_path: Path) -> None:
    temp_path = tmp_path / "temp.jsonl"
    geo_path = tmp_path / "geo.jsonl"

    _write_jsonl(
        temp_path,
        [
            {
                "id": "temp-1",
                "question": "when did x happen",
                "is_dependent": True,
                "context_answer_pairs": [
                    {"edited_question": "when did x happen as of 2001"},
                    {"edited_question": "when did x happen as of 2002"},
                ],
            }
        ],
    )
    _write_jsonl(
        geo_path,
        [
            {
                "id": "geo-1",
                "question": "who is the prime minister",
                "context_answer_pairs": [
                    {"edited_question": "who is the prime minister in canada"},
                    {"edited_question": "who is the prime minister in australia"},
                ],
            }
        ],
    )

    df = build_situatedqa_pairs(
        temp_paths=[str(temp_path)],
        geo_paths=[str(geo_path)],
        seed=13,
        selection_strategy="first",
    )

    clear_rows = df.loc[df["label_ambiguous"].eq(0)].sort_values("pair_id").reset_index(drop=True)
    assert clear_rows["text"].tolist() == [
        "who is the prime minister in canada",
        "when did x happen as of 2001",
    ]
