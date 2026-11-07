"""Loading the curated dataset and the one frozen local validation split.

The split is grouped by InChIKey skeleton (``scaffold_key``), so stereoisomers
and tautomers of the same structure cannot land on both sides. Every
comparison between model variants uses this same split and this same seed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import featurize_many, standardize

log = logging.getLogger("molecula.dataset")

RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2


@dataclass
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame

    def describe(self) -> dict:
        return {
            "n_train": int(len(self.train)),
            "n_validation": int(len(self.validation)),
            "train_scaffolds": int(self.train["scaffold_key"].nunique()),
            "validation_scaffolds": int(self.validation["scaffold_key"].nunique()),
            "shared_scaffolds": int(
                len(set(self.train["scaffold_key"]) & set(self.validation["scaffold_key"]))
            ),
            "shared_inchikeys": int(len(set(self.train["inchikey"]) & set(self.validation["inchikey"]))),
            "seed": RANDOM_SEED,
            "validation_fraction": VALIDATION_FRACTION,
        }


def load_dataset(path: str | Path = "data/processed/dataset.csv") -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"inchikey", "scaffold_key", "canonical_smiles", "log10_si", "si"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["canonical_smiles", "log10_si"]).reset_index(drop=True)
    return frame


def make_split(frame: pd.DataFrame, seed: int = RANDOM_SEED) -> Split:
    """Grouped split: whole scaffold groups go to one side or the other."""
    groups = (
        frame.groupby("scaffold_key").size().reset_index(name="size").sort_values("scaffold_key")
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    groups = groups.iloc[order]

    target = int(round(len(frame) * VALIDATION_FRACTION))
    validation_groups: set[str] = set()
    accumulated = 0
    for row in groups.to_dict("records"):
        if accumulated >= target:
            break
        validation_groups.add(row["scaffold_key"])
        accumulated += row["size"]

    mask = frame["scaffold_key"].isin(validation_groups)
    split = Split(
        train=frame[~mask].reset_index(drop=True),
        validation=frame[mask].reset_index(drop=True),
    )
    assert not (set(split.train["scaffold_key"]) & set(split.validation["scaffold_key"])), (
        "scaffold leaked across the split"
    )
    return split


def featurize_frame(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Features and target for a dataset frame, in row order."""
    mols = [standardize(s).mol for s in frame["canonical_smiles"]]
    X = featurize_many(mols)
    y = frame["log10_si"].to_numpy(dtype=np.float64)
    return X, y
