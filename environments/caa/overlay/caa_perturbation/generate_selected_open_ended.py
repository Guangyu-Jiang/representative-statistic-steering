from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from caa_perturbation.core import BehaviorStatistics
from caa_perturbation.run_experiment import (
    InterventionConfig,
    InterventionController,
    behavior_dir,
    decoder_layers,
    dump_json,
    fit_stats_for_layer,
    generation_input,
    load_json,
    load_model,
    test_data_path,
)


DEFAULT_SELECTION = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "caa_perturbation_all_behaviors"
    / "reports"
    / "all_behaviors_split_selected.csv"
)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def truncate_stats(stats: BehaviorStatistics, components: int) -> BehaviorStatistics:
    state = stats.state_dict()
    for key in (
        "components",
        "component_scale",
        "positive_centroid",
        "negative_centroid",
        "pca_margin_positive_targets",
        "explained_variance_ratio",
    ):
        state[key] = state[key][:components]
    return BehaviorStatistics.from_state_dict(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/caa_perturbation_all_behaviors"),
    )
    parser.add_argument("--selection-csv", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Optional selected method names to generate (for example unsteered fixed_caa)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    selected = [
        row for row in read_csv(args.selection_csv) if row["behavior"] == args.behavior
    ]
    if args.methods:
        selected = [row for row in selected if row["method"] in set(args.methods)]
    if not selected:
        raise ValueError(f"No selected methods found for {args.behavior}")

    destination_dir = behavior_dir(args.output_root, args.model, args.behavior)
    summaries = {
        row["setting_id"]: row for row in read_csv(destination_dir / "mc_summary.csv")
    }
    artifact = torch.load(
        destination_dir / "train_activations.pt", map_location="cpu", weights_only=False
    )
    layer = int(selected[0]["layer"])
    max_components = max(
        1, max(int(summaries[row["setting_id"]]["components"]) for row in selected)
    )
    maximum_stats = fit_stats_for_layer(
        artifact, layer, max_components, args.seed
    )

    model, tokenizer = load_model(args.model, args.device, args.local_files_only)
    model_dtype = next(model.parameters()).dtype
    controller = InterventionController(maximum_stats.to(args.device, model_dtype))
    hook = decoder_layers(model)[layer].register_forward_hook(controller.hook)
    data = load_json(test_data_path(args.behavior, open_ended=True))
    if args.max_examples is not None:
        data = data[: args.max_examples]

    try:
        for selected_row in selected:
            setting_id = selected_row["setting_id"]
            summary = summaries[setting_id]
            component_count = max(1, int(summary["components"]))
            controller.stats = truncate_stats(maximum_stats, component_count).to(
                args.device, model_dtype
            )
            config = InterventionConfig(
                method=summary["method"],
                strength=float(summary["strength"]),
                ridge=float(summary["ridge"]),
                max_relative_norm=float(summary["max_relative_norm"]),
                target_quantile=float(summary.get("target_quantile", 0.75)),
            )
            controller.reset_metrics()
            rows = []
            for index, item in enumerate(
                tqdm(data, desc=f"{args.behavior}/{selected_row['method']}")
            ):
                input_ids, attention_mask, patch_mask = generation_input(
                    tokenizer, item["question"], args.device
                )
                controller.configure(config, patch_mask=patch_mask, decode_all=True)
                with torch.inference_mode():
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                continuation = generated[0, input_ids.shape[1] :]
                rows.append(
                    {
                        "index": index,
                        "question": item["question"],
                        "response": tokenizer.decode(
                            continuation, skip_special_tokens=True
                        ).strip(),
                    }
                )

            output_path = (
                destination_dir / "open_ended_split_selected" / f"{setting_id}.json"
            )
            dump_json(output_path, rows)
            dump_json(
                output_path.with_suffix(".metadata.json"),
                {
                    "model": args.model,
                    "behavior": args.behavior,
                    "layer": layer,
                    "method": summary["method"],
                    "components": int(summary["components"]),
                    "strength": float(summary["strength"]),
                    "ridge": float(summary["ridge"]),
                    "target_quantile": float(
                        summary.get("target_quantile", 0.75)
                    ),
                    "mean_action_relative_norm": controller.mean_relative_norm,
                    "selection": "stratified held-out tuning half",
                    "external_api_used": False,
                },
            )
            print(output_path)
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
