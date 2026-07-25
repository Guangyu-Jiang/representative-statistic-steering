import pandas as pd

from validation.summarize_causal_head_results import summarize_frames


def test_summary_distinguishes_fold_ids_from_source_runs():
    frames = []
    for seed in (1, 2, 3):
        for fold in (0, 1):
            frame = pd.DataFrame(
                {
                    "setting": ["candidate"],
                    "fold": [fold],
                    "n": [10],
                    "mc1": [0.5 + seed / 100],
                    "mc2": [0.6],
                    "relative_action_norm": [1.0],
                    "intervention_rate": [1.0],
                    "pre_target_error": [0.0],
                    "post_target_error": [0.0],
                    "source": [f"seed{seed}-fold{fold}"],
                }
            )
            frames.append(frame)

    summary = summarize_frames(frames, min_folds=2, min_sources=6).iloc[0]

    assert summary["folds"] == 2
    assert summary["sources"] == 6
    assert summary["n"] == 60
