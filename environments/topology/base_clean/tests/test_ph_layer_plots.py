from __future__ import annotations

from pathlib import Path

import pandas as pd

from aen_replication.features.ph_layer_plots import run_ph_layer_plot_analysis
from aen_replication.utils.io_utils import write_parquet


def test_run_ph_layer_plot_analysis_writes_layer_figures(tmp_path: Path) -> None:
    model_name = "unit/test-model"
    rows = []
    for dataset in ["ambigqa"]:
        for layer in [0, 1]:
            for subspace_name in ["top_20"]:
                for label_group, delta in [("ambiguous", 0.1), ("clear", 0.0)]:
                    for homology_dim in [0, 1]:
                        for birth, death in [(0.1, 0.4 + delta), (0.2, 0.7 + delta)]:
                            rows.append(
                                {
                                    "model_name": model_name,
                                    "dataset": dataset,
                                    "layer": layer,
                                    "readout": "mean_pool",
                                    "subspace_name": subspace_name,
                                    "label_group": label_group,
                                    "point_count": 8,
                                    "selected_count": 20,
                                    "homology_dim": homology_dim,
                                    "birth": birth,
                                    "death": death,
                                    "persistence": death - birth,
                                }
                            )

    diagrams_path = tmp_path / "diagrams.parquet"
    write_parquet(pd.DataFrame(rows), diagrams_path)

    outputs = run_ph_layer_plot_analysis(
        model_name=model_name,
        diagrams_path=diagrams_path,
        layer_plot_config={
            "output_dir": str(tmp_path / "layer_plots"),
            "datasets": ["ambigqa"],
            "subspaces": ["top_20"],
            "homology_dims": [0, 1],
            "barcode_top_n": 10,
            "report_filename": "report.md",
            "metadata_filename": "metadata.json",
        },
    )

    plots_root = Path(outputs["plots_root"])
    assert (plots_root / "ambigqa" / "top_20" / "h0" / "layer_00.png").exists()
    assert (plots_root / "ambigqa" / "top_20" / "h1" / "layer_01.png").exists()
