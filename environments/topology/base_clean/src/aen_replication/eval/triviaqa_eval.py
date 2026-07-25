"""TriviaQA false-positive evaluation for AEN classifiers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset

from aen_replication.models.generation import render_prompts
from aen_replication.models.hidden_state_extractor import HiddenStateExtractor, load_hidden_state_table
from aen_replication.models.hf_model import HFModelBundle
from aen_replication.train.aen import evaluate_full_probe, evaluate_sparse_probe
from aen_replication.utils.io_utils import ensure_dir, read_json, slugify, write_json, write_parquet

LOGGER = logging.getLogger(__name__)


def load_or_prepare_triviaqa_questions(config: dict[str, Any]) -> pd.DataFrame:
    """Load a deterministic slice of TriviaQA questions and cache them locally."""

    eval_cfg = config["triviaqa_eval"]
    processed_path = Path(eval_cfg["processed_path"])
    if processed_path.exists():
        return pd.read_parquet(processed_path)

    dataset = load_dataset(
        str(eval_cfg.get("dataset_id", "trivia_qa")),
        str(eval_cfg.get("config_name", "rc.nocontext")),
        split=f"{eval_cfg.get('split', 'validation')}[:{int(eval_cfg.get('sample_n', 1000))}]",
    )
    df = pd.DataFrame(
        {
            "example_id": [f"triviaqa_{idx:04d}" for idx in range(len(dataset))],
            "raw_question_id": dataset["question_id"],
            "text": dataset["question"],
            "dataset": "triviaqa",
            "split": "eval",
            "label_ambiguous": 0,
        }
    )
    write_parquet(df, processed_path)
    return df


def _extract_triviaqa_matrix(
    bundle: HFModelBundle,
    trivia_df: pd.DataFrame,
    extraction_cfg: dict[str, Any],
    layer: int,
    readout: str,
) -> np.ndarray:
    extractor = HiddenStateExtractor(
        bundle=bundle,
        batch_size=int(extraction_cfg["batch_size"]),
        max_length=int(extraction_cfg["max_length"]),
        use_mixed_precision=bool(extraction_cfg.get("use_mixed_precision", False)),
    )
    extraction_df = trivia_df.copy()
    text_column = "text"
    use_chat_template = bool(extraction_cfg.get("use_chat_template", False))
    system_prompt = extraction_cfg.get("system_prompt")
    if use_chat_template or system_prompt:
        extraction_df["_rendered_text"] = render_prompts(
            bundle=bundle,
            prompt_texts=trivia_df["text"].astype(str).tolist(),
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            add_generation_prompt=False,
        )
        text_column = "_rendered_text"
    vectors = extractor.extract(
        df=extraction_df,
        layers=[layer],
        readouts=[readout],
        text_column=text_column,
    )
    return vectors[(layer, readout)]


def _apply_sparse_probe_to_matrix(
    matrix: np.ndarray,
    full_probe: dict[str, Any],
    sparse_probe: dict[str, Any],
    indices: list[int],
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = matrix
    if full_probe["scaler"] is not None:
        transformed = full_probe["scaler"].transform(transformed)
    transformed = transformed[:, indices]
    if sparse_probe["scaler"] is not None:
        transformed = sparse_probe["scaler"].transform(transformed)
    scores = np.asarray(sparse_probe["classifier"].decision_function(transformed), dtype=float)
    predictions = (scores >= threshold).astype(int)
    return scores, predictions


def evaluate_triviaqa_false_positives(
    bundle: HFModelBundle,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate AEN classifiers on mostly clear TriviaQA questions."""

    eval_cfg = config["triviaqa_eval"]
    project_root = Path(config["_meta"]["project_root"])
    model_slug = slugify(config["model"]["name"])
    output_root = ensure_dir(Path(config["reports"]["output_dir"]) / model_slug)
    summary_path = output_root / str(eval_cfg.get("summary_filename", "triviaqa_aen_false_positive.json"))
    predictions_path = output_root / str(eval_cfg.get("predictions_filename", "triviaqa_aen_false_positive.parquet"))

    if summary_path.exists() and predictions_path.exists():
        LOGGER.info("Skipping TriviaQA evaluation for %s because artifacts already exist.", config["model"]["name"])
        return read_json(summary_path)

    trivia_df = load_or_prepare_triviaqa_questions(config)
    default_layer = int(config["extraction"]["default_layer"])
    readout = str(eval_cfg.get("readout", "mean_pool"))
    trivia_matrix = _extract_triviaqa_matrix(
        bundle=bundle,
        trivia_df=trivia_df,
        extraction_cfg=config["extraction"],
        layer=default_layer,
        readout=readout,
    )

    prediction_frames: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        "model_name": config["model"]["name"],
        "layer": default_layer,
        "readout": readout,
        "n_eval": int(len(trivia_df)),
        "train_datasets": {},
    }

    for train_dataset in eval_cfg.get("train_datasets", ["ambigqa", "situatedqa"]):
        hidden_state_path = project_root / "artifacts" / "hidden_states" / model_slug / f"{train_dataset}__layer_{default_layer:02d}__{readout}.parquet"
        report_path = project_root / "artifacts" / "probes" / model_slug / f"{train_dataset}_default_layer_report.json"
        metadata, matrix = load_hidden_state_table(hidden_state_path)
        default_report = read_json(report_path)
        full_probe = evaluate_full_probe(
            metadata=metadata,
            matrix=matrix,
            probe_cfg=config["probe"],
            seed=int(config["seed"]),
        )
        indices = [int(value) for value in default_report["aen_selection"]["aen_indices"]]
        sparse_probe = evaluate_sparse_probe(
            full_probe=full_probe,
            indices=indices,
            probe_cfg=config["probe"],
            seed=int(config["seed"]),
        )
        scores, predictions = _apply_sparse_probe_to_matrix(
            matrix=trivia_matrix,
            full_probe=full_probe,
            sparse_probe=sparse_probe,
            indices=indices,
        )
        frame = trivia_df.copy()
        frame["model_name"] = config["model"]["name"]
        frame["train_dataset"] = train_dataset
        frame["decision_value"] = scores
        frame["predicted_ambiguous"] = predictions
        prediction_frames.append(frame)

        summary["train_datasets"][train_dataset] = {
            "accuracy": float((predictions == 0).mean()),
            "false_positive_rate": float((predictions == 1).mean()),
            "n_eval": int(len(predictions)),
            "n_predicted_ambiguous": int(predictions.sum()),
            "aen_indices": indices,
            "aen_k": int(default_report["aen_selection"]["aen_k"]),
        }

    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    write_parquet(predictions_df, predictions_path)
    write_json(summary_path, summary)
    LOGGER.info("Saved TriviaQA evaluation to %s", summary_path)
    return summary
