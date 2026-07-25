"""Extract hidden states for one prepared dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from aen_replication.config import load_config
from aen_replication.models.generation import render_prompts
from aen_replication.models.hidden_state_extractor import HiddenStateExtractor
from aen_replication.models.hf_model import load_hf_model
from aen_replication.utils.io_utils import append_command_history, ensure_dir, slugify, write_json
from aen_replication.utils.logging_utils import setup_logging
from aen_replication.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config["runtime"]["log_level"], Path(config["runtime"]["log_dir"]) / f"extract_{args.dataset}.log")
    set_global_seed(int(config["seed"]))
    append_command_history(config["runtime"]["command_history_path"], sys.argv)

    pairs_path = Path(config["data"]["pair_output_dir"]) / f"{args.dataset}_pairs.parquet"
    df = pd.read_parquet(pairs_path)
    bundle = load_hf_model(config["model"], config["extraction"])
    extraction_cfg = config["extraction"]
    extractor = HiddenStateExtractor(
        bundle=bundle,
        batch_size=int(extraction_cfg["batch_size"]),
        max_length=int(extraction_cfg["max_length"]),
        use_mixed_precision=bool(extraction_cfg.get("use_mixed_precision", False)),
    )
    model_depth = int(getattr(bundle.model.config, "num_hidden_layers"))
    layers = list(range(model_depth)) if extraction_cfg["layers"] == "auto" else list(extraction_cfg["layers"])
    output_dir = Path(extraction_cfg["cache_dir"])
    text_column = extraction_cfg["text_column"]
    use_chat_template = bool(extraction_cfg.get("use_chat_template", False))
    system_prompt = extraction_cfg.get("system_prompt")
    extraction_df = df
    if use_chat_template or system_prompt:
        extraction_df = df.copy()
        extraction_df["_rendered_text"] = render_prompts(
            bundle=bundle,
            prompt_texts=df[text_column].astype(str).tolist(),
            use_chat_template=use_chat_template,
            system_prompt=system_prompt,
            add_generation_prompt=False,
        )
        text_column = "_rendered_text"
    records = extractor.extract_and_save(
        df=extraction_df,
        layers=layers,
        readouts=list(extraction_cfg["readouts"]),
        text_column=text_column,
        output_dir=output_dir,
        project_root=config["_meta"]["project_root"],
        file_prefix=f"{args.dataset}__",
        metadata_overrides={
            "input_text_column": extraction_cfg["text_column"],
            "input_use_chat_template": use_chat_template,
            "input_system_prompt": system_prompt,
            "input_add_generation_prompt": False,
        },
    )
    dataset_manifest = {
        "dataset": args.dataset,
        "model_name": config["model"]["name"],
        "input_rendering": {
            "text_column": extraction_cfg["text_column"],
            "use_chat_template": use_chat_template,
            "system_prompt": system_prompt,
            "add_generation_prompt": False,
        },
        "files": [
            {
                "dataset": args.dataset,
                "layer": record.layer,
                "readout": record.readout,
                "parquet_path": record.parquet_path,
                "metadata_path": record.metadata_path,
            }
            for record in records
        ],
    }
    manifest_path = ensure_dir(output_dir) / slugify(config["model"]["name"]) / f"{args.dataset}_manifest.json"
    write_json(manifest_path, dataset_manifest)


if __name__ == "__main__":
    main()
