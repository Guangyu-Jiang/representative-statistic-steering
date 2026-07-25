"""Forward-only hidden-state extraction and cache writing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from aen_replication.models.hf_model import HFModelBundle
from aen_replication.models.readout import apply_readout
from aen_replication.utils.io_utils import ensure_dir, get_git_commit, slugify, utc_now_iso, write_json, write_parquet

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class HiddenStateCacheRecord:
    """Metadata for one saved hidden-state cache table."""

    layer: int
    readout: str
    parquet_path: str
    metadata_path: str


class HiddenStateExtractor:
    """Extract fixed-size vectors from selected layers and readout positions."""

    def __init__(
        self,
        bundle: HFModelBundle,
        batch_size: int,
        max_length: int,
        use_mixed_precision: bool = False,
    ) -> None:
        self.bundle = bundle
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_mixed_precision = use_mixed_precision and bundle.device.type == "cuda"

    def _iter_batches(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        return [df.iloc[start : start + self.batch_size] for start in range(0, len(df), self.batch_size)]

    def extract(
        self,
        df: pd.DataFrame,
        layers: list[int],
        readouts: list[str],
        text_column: str,
    ) -> dict[tuple[int, str], np.ndarray]:
        """Extract layer/readout vectors for each input example."""

        n_examples = len(df)
        # Preallocate contiguous output buffers so large CLAMBER runs do not spend
        # minutes concatenating hundreds of tiny batch arrays at the end.
        outputs: dict[tuple[int, str], np.ndarray] = {}
        offsets: dict[tuple[int, str], int] = {}
        encoder = self.bundle.tokenizer
        model = self.bundle.model
        device = self.bundle.device

        if not layers:
            raise ValueError("No extraction layers configured.")

        max_layer = max(layers)
        total_layers = getattr(model.config, "num_hidden_layers", None)
        if total_layers is not None and max_layer >= total_layers:
            raise ValueError(f"Configured layer {max_layer} exceeds model depth {total_layers}")

        batches = self._iter_batches(df)
        autocast_enabled = self.use_mixed_precision

        for batch_df in tqdm(batches, desc="extract_hidden_states", leave=False):
            encoded = encoder(
                batch_df[text_column].tolist(),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            attention_mask = encoded["attention_mask"].to(device)
            model_inputs = {key: value.to(device) for key, value in encoded.items()}

            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                    model_outputs = model(**model_inputs, output_hidden_states=True, use_cache=False)
            hidden_states = model_outputs.hidden_states
            if hidden_states is None:
                raise RuntimeError("Model did not return hidden states.")

            for layer in layers:
                layer_output = hidden_states[layer + 1]
                for readout in readouts:
                    vectors = apply_readout(
                        strategy=readout,
                        hidden_state=layer_output,
                        attention_mask=attention_mask,
                    )
                    batch_vectors = vectors.detach().float().cpu().numpy()
                    key = (layer, readout)
                    if key not in outputs:
                        outputs[key] = np.empty((n_examples, batch_vectors.shape[1]), dtype=batch_vectors.dtype)
                        offsets[key] = 0
                    start = offsets[key]
                    end = start + batch_vectors.shape[0]
                    outputs[key][start:end] = batch_vectors
                    offsets[key] = end

        for key, end in offsets.items():
            if end != n_examples:
                raise RuntimeError(f"Incomplete hidden-state extraction for {key}: expected {n_examples}, got {end}")

        return outputs

    def extract_and_save(
        self,
        df: pd.DataFrame,
        layers: list[int],
        readouts: list[str],
        text_column: str,
        output_dir: str | Path,
        project_root: str | Path,
        file_prefix: str = "",
        metadata_overrides: dict[str, Any] | None = None,
    ) -> list[HiddenStateCacheRecord]:
        """Extract hidden states and save one cache table per layer/readout."""

        vectors_by_key = self.extract(df=df, layers=layers, readouts=readouts, text_column=text_column)
        output_root = ensure_dir(output_dir) / slugify(self.bundle.model_name)
        git_commit = get_git_commit(project_root)
        records: list[HiddenStateCacheRecord] = []

        metadata_columns = [
            "example_id",
            "pair_id",
            "dataset",
            "label_ambiguous",
            "split",
            "text",
            "source_id",
            "context_type",
        ]
        optional_metadata_columns = [
            "source_question",
            "resolved_from",
            "notes",
            "context",
            "clarifying_question",
            "category",
            "subclass",
            "require_clarification",
        ]
        available_columns = metadata_columns + [
            column for column in optional_metadata_columns if column in df.columns
        ]
        base_table = df.loc[:, available_columns].reset_index(drop=True)

        for (layer, readout), vectors in vectors_by_key.items():
            filename = f"{file_prefix}layer_{layer:02d}__{readout}"
            parquet_path = output_root / f"{filename}.parquet"
            metadata_path = output_root / f"{filename}.metadata.json"

            cache_table = base_table.copy()
            cache_table["vector"] = vectors.tolist()
            write_parquet(cache_table, parquet_path)

            metadata = {
                "model_name": self.bundle.model_name,
                "tokenizer_name": self.bundle.tokenizer_name,
                "readout": readout,
                "layer": layer,
                "hidden_size": int(vectors.shape[1]),
                "n_examples": int(vectors.shape[0]),
                "extraction_date": utc_now_iso(),
                "git_commit": git_commit,
                "parquet_path": str(parquet_path),
            }
            if metadata_overrides:
                metadata.update(metadata_overrides)
            write_json(metadata_path, metadata)
            records.append(
                HiddenStateCacheRecord(
                    layer=layer,
                    readout=readout,
                    parquet_path=str(parquet_path),
                    metadata_path=str(metadata_path),
                )
            )
            LOGGER.info("Saved hidden-state cache %s", parquet_path)

        manifest_name = f"{file_prefix}manifest.json" if file_prefix else "manifest.json"
        manifest_path = output_root / manifest_name
        write_json(
            manifest_path,
            {
                "model_name": self.bundle.model_name,
                "tokenizer_name": self.bundle.tokenizer_name,
                "layers": layers,
                "readouts": readouts,
                "metadata_overrides": metadata_overrides or {},
                "files": [
                    {
                        "layer": record.layer,
                        "readout": record.readout,
                        "parquet_path": record.parquet_path,
                        "metadata_path": record.metadata_path,
                    }
                    for record in records
                ],
            },
        )
        LOGGER.info("Saved hidden-state manifest to %s", manifest_path)
        return records


def load_hidden_state_table(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a cached hidden-state parquet table into metadata and a matrix."""

    df = pd.read_parquet(path)
    if "vector" not in df.columns:
        raise ValueError(f"Hidden-state cache is missing vector column: {path}")
    matrix = np.vstack(df["vector"].apply(np.asarray).to_list())
    return df, matrix
