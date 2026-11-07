"""The saved model artefact: what is in it, how it is written, how it predicts.

A single joblib file holds the fitted estimator, the training fingerprints
(needed for the applicability-domain check and for nearest-neighbour lookups),
the feature contract and the metrics measured on the frozen validation split.
The HTTP API and the submission CLI both load this one file - neither
re-implements any part of the prediction path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .chem import FP_BITS, FP_RADIUS, N_FEATURES, StandardMolecule, featurize, standardize

DEFAULT_MODEL_PATH = Path("models/molecula_si.joblib")

_BIT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


def binary_fingerprint(mol) -> Any:
    """Binary Morgan fingerprint, used only for Tanimoto similarity."""
    return _BIT_GENERATOR.GetFingerprint(mol)


@dataclass
class TrainingNeighbour:
    inchikey: str
    smiles: str
    si: float
    similarity: float
    dois: str | None
    documents: str | None


@dataclass
class Prediction:
    canonical_smiles: str
    inchikey: str
    log10_si: float
    si: float
    nearest_similarity: float
    in_domain: bool
    neighbours: list[TrainingNeighbour] = field(default_factory=list)


def _serve_single_threaded(estimator: Any, depth: int = 0) -> None:
    """Set n_jobs to 1 on every fitted estimator inside, however it is wrapped."""
    if estimator is None or depth > 5:
        return
    try:
        if "n_jobs" in estimator.get_params(deep=False):
            estimator.set_params(n_jobs=1)
    except (AttributeError, ValueError, TypeError):
        pass

    children: list[Any] = []
    fitted = getattr(estimator, "fitted_", None)
    if isinstance(fitted, tuple):
        children.append(fitted[0])
    for attribute in ("base_", "regressor_", "classifier_", "candidate"):
        children.append(getattr(estimator, attribute, None))
    steps = getattr(estimator, "steps", None)
    if steps:
        children.extend(step for _, step in steps)
    estimator_attribute = getattr(estimator, "estimator", None)
    if estimator_attribute is not None and estimator_attribute is not estimator:
        children.append(estimator_attribute)

    for child in children:
        if child is not None:
            _serve_single_threaded(child, depth + 1)


@dataclass
class SavedModel:
    estimator: Any
    model_version: str
    dataset_version: str
    feature_contract: dict
    metrics: dict
    training_rows: list[dict]
    training_fingerprints: list[Any]
    domain_threshold: float
    trained_at: str
    config: dict
    # What the estimator's own output means. The competition scores MSE against
    # the raw selectivity index, so the shipped model predicts "si" directly;
    # "log10_si" is kept for models trained against the log target.
    target_kind: str = "log10_si"

    # ----------------------------------------------------------------- io
    def save(self, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path = DEFAULT_MODEL_PATH) -> "SavedModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"no trained model at {path}. Run `make train` (see README) before starting the service."
            )
        model: SavedModel = joblib.load(path)
        # A forest fitted with several jobs spawns a thread pool per single-row
        # call, which costs far more than the prediction inside a request
        # handler. The shipped estimator wraps the forest in a transform and
        # sometimes in a two-part model, so this walks down to it.
        _serve_single_threaded(model.estimator)
        if model.feature_contract.get("n_features") != N_FEATURES:
            raise RuntimeError(
                "the saved model was trained against a different feature contract "
                f"({model.feature_contract.get('n_features')} features, code expects {N_FEATURES}). "
                "Retrain before serving."
            )
        return model

    # ------------------------------------------------------------ predict
    def predict_molecule(self, molecule: StandardMolecule, n_neighbours: int = 3) -> Prediction:
        features = featurize(molecule.mol).reshape(1, -1)
        raw = float(self.estimator.predict(features)[0])
        if getattr(self, "target_kind", "log10_si") == "si":
            si = max(0.0, raw)
            log10_si = float(np.log10(si + 0.1))
        else:
            log10_si = raw
            si = float(10.0**raw)

        query = binary_fingerprint(molecule.mol)
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(query, self.training_fingerprints), dtype=np.float64
        )
        order = np.argsort(-similarities, kind="stable")[:n_neighbours]
        neighbours = [
            TrainingNeighbour(
                inchikey=self.training_rows[i]["inchikey"],
                smiles=self.training_rows[i]["canonical_smiles"],
                si=float(self.training_rows[i]["si"]),
                similarity=float(similarities[i]),
                dois=self.training_rows[i].get("dois"),
                documents=self.training_rows[i].get("documents"),
            )
            for i in order
        ]
        nearest = float(similarities.max()) if similarities.size else 0.0
        return Prediction(
            canonical_smiles=molecule.canonical_smiles,
            inchikey=molecule.inchikey,
            log10_si=log10_si,
            si=si,
            nearest_similarity=nearest,
            in_domain=nearest >= self.domain_threshold,
            neighbours=neighbours,
        )

    def predict_smiles(self, smiles: str, n_neighbours: int = 3) -> Prediction:
        return self.predict_molecule(standardize(smiles), n_neighbours=n_neighbours)

    # --------------------------------------------------------------- info
    def info(self) -> dict:
        return {
            "model_version": self.model_version,
            "dataset_version": self.dataset_version,
            "trained_at": self.trained_at,
            "feature_contract": self.feature_contract,
            "metrics": self.metrics,
            "domain_threshold": self.domain_threshold,
            "target_kind": getattr(self, "target_kind", "log10_si"),
            "config": self.config,
            "n_training_molecules": len(self.training_rows),
        }

    def info_json(self) -> str:
        return json.dumps(self.info(), indent=2, ensure_ascii=False)
