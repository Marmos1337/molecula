"""The comparative study: one set of folds, many approaches, honest numbers.

Everything that decides which model ships is measured here, and it is measured
the same way for every candidate: the folds are built once per (scheme, seed),
and every candidate sees exactly those folds. Nothing is fitted outside a
training fold - not the scaler, not the smearing correction, not the external
data we add to training.

Three things are worth stating up front.

*The metric is the competition's.* It is MSE against the raw selectivity
index, so that is the number that ranks the candidates. RMSE on log10(SI) is
reported next to it because it is what a chemist can read, but it never
decides anything.

*The schemes answer different questions.* ``molecule`` is a plain random split
and matches how the competition itself was divided (53 skeletons appear on
both sides of the official train/test boundary). ``scaffold`` and ``cluster``
hold out whole structural families, and ``publication`` holds out whole source
papers; those three say how far the model carries to something genuinely new.
The first is the honest estimate of the leaderboard, the others are the honest
estimate of the science.

*External data is added inside the fold.* When a candidate uses the collected
ChEMBL set, those rows join the training part only, and any ChEMBL molecule
whose skeleton sits in the validation part of that fold is dropped first.

    python -m molecula.experiments --out reports/
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .chem import FP_BITS, featurize_many, standardize
from .official import CULTURE_VOCABULARY, load_train

log = logging.getLogger("molecula.experiments")

SEEDS = (42, 2026, 7)
N_FOLDS = 5
SCHEMES = ("molecule", "scaffold", "cluster", "publication")
CLUSTER_CUTOFF = 0.4  # Tanimoto distance for the Butina chemical-family grouping

# How the collected ChEMBL rows are filtered before they join a training fold.
# Under "molecule" they are left alone on purpose: the real test set overlaps
# the collected set about as much as the training set does (84 of 597 test
# skeletons against 41 of 320 training ones), so filtering would understate
# what the external data does on the leaderboard. The three transfer schemes
# ask a different question - how far the model carries to chemistry it has not
# seen - and there an external analogue of a held-out molecule is a leak.
EXTERNAL_SIMILARITY_CUTOFF = 0.6
SCHEMES_FILTERING_EXTERNAL = ("scaffold", "cluster", "publication")


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------


def base_features(frame: pd.DataFrame) -> np.ndarray:
    """Morgan count fingerprint plus RDKit descriptors, from SMILES alone."""
    mols = [standardize(s).mol for s in frame["canonical_smiles"]]
    return featurize_many(mols)


def culture_columns(frame: pd.DataFrame) -> np.ndarray:
    """One-hot of the cell culture, from a fixed vocabulary (fitted nowhere)."""
    vocabulary = (*CULTURE_VOCABULARY, "other")
    out = np.zeros((len(frame), len(vocabulary)), dtype=np.float64)
    for row, value in enumerate(frame["culture"]):
        out[row, vocabulary.index(value if value in vocabulary else "other")] = 1.0
    return out


def build_features(frame: pd.DataFrame, kind: str) -> np.ndarray:
    """``kind`` is part of a candidate's identity, so it is never guessed."""
    base = base_features(frame)
    if kind == "smiles":
        return base
    if kind == "smiles+culture":
        return np.hstack([base, culture_columns(frame)])
    if kind == "fingerprint":
        return base[:, :FP_BITS]
    if kind == "descriptors":
        return base[:, FP_BITS:]
    raise ValueError(f"unknown feature set {kind!r}")


# --------------------------------------------------------------------------
# Tanimoto-based estimators
# --------------------------------------------------------------------------


def _binary(X: np.ndarray) -> np.ndarray:
    return (X[:, :FP_BITS] > 0).astype(np.float64)


def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    intersection = A @ B.T
    a = A.sum(axis=1)[:, None]
    b = B.sum(axis=1)[None, :]
    union = a + b - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(union > 0, intersection / union, 0.0)
    return out


class TanimotoKernelRidge(BaseEstimator, RegressorMixin):
    """Kernel ridge regression on the Tanimoto kernel of the fingerprints."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.X_ = _binary(np.asarray(X))
        y = np.asarray(y, dtype=np.float64)
        self.offset_ = float(y.mean())
        K = tanimoto(self.X_, self.X_)
        n = K.shape[0]
        self.dual_ = np.linalg.solve(K + self.alpha * np.eye(n), y - self.offset_)
        return self

    def predict(self, X):
        K = tanimoto(_binary(np.asarray(X)), self.X_)
        return K @ self.dual_ + self.offset_


class TunedTanimotoKernelRidge(BaseEstimator, RegressorMixin):
    """Kernel ridge whose regularisation is chosen inside the training fold.

    The search never sees the validation part of the outer fold, so the outer
    number stays an honest estimate of a procedure that includes its own
    tuning - which is the only way a tuned model can be reported.
    """

    def __init__(self, alphas: tuple[float, ...] = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0), cv: int = 3):
        self.alphas = alphas
        self.cv = cv

    def fit(self, X, y):
        from sklearn.model_selection import GridSearchCV, KFold

        search = GridSearchCV(
            TanimotoKernelRidge(),
            {"alpha": list(self.alphas)},
            cv=KFold(n_splits=self.cv, shuffle=True, random_state=0),
            scoring="neg_mean_squared_error",
        )
        search.fit(np.asarray(X), np.asarray(y, dtype=np.float64))
        self.best_alpha_ = float(search.best_params_["alpha"])
        self.estimator_ = search.best_estimator_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class TanimotoKNN(BaseEstimator, RegressorMixin):
    """Similarity-weighted mean of the k most similar training molecules."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y):
        self.X_ = _binary(np.asarray(X))
        self.y_ = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, X):
        similarity = tanimoto(_binary(np.asarray(X)), self.X_)
        k = min(self.k, similarity.shape[1])
        neighbours = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
        out = np.empty(similarity.shape[0], dtype=np.float64)
        for row in range(similarity.shape[0]):
            index = neighbours[row]
            weights = similarity[row, index]
            out[row] = (
                float(np.average(self.y_[index], weights=weights))
                if weights.sum() > 0
                else float(self.y_.mean())
            )
        return out


class StructureLookup(BaseEstimator, RegressorMixin):
    """Answer from the training table when the very same structure is in it.

    53 skeletons sit on both sides of the official train/test boundary, so for
    part of the test set the measured index is already known. Where the query
    fingerprint matches a training fingerprint exactly, that measurement is a
    better answer than any regression; everywhere else the base model speaks.
    Identical fingerprints are not quite identical molecules, which is why the
    fallback exists and why this is measured rather than assumed.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        X = np.asarray(X)
        self.X_ = _binary(X)
        self.y_ = np.asarray(y, dtype=np.float64)
        self.base_ = clone(self.base)
        self.base_.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        prediction = self.base_.predict(X)
        similarity = tanimoto(_binary(X), self.X_)
        for row in range(similarity.shape[0]):
            exact = np.flatnonzero(similarity[row] >= 0.999)
            if exact.size:
                prediction[row] = float(self.y_[exact].mean())
        return prediction


class ConstantRegressor(BaseEstimator, RegressorMixin):
    """The reference points: always zero, always the mean, always the median."""

    def __init__(self, strategy: str = "mean"):
        self.strategy = strategy

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        self.value_ = {
            "zero": 0.0,
            "mean": float(y.mean()),
            "median": float(np.median(y)),
        }[self.strategy]
        return self

    def predict(self, X):
        return np.full(len(X), self.value_, dtype=np.float64)


class TwoStage(BaseEstimator, RegressorMixin):
    """P(the compound has a measured IC50) times its expected index.

    A quarter of the training rows carry SI = 0 by the organisers' convention.
    A single regressor has to straddle that spike and the long tail at once;
    this splits the job in two and multiplies the parts back together, which
    is what an expectation under MSE actually asks for.
    """

    def __init__(self, classifier=None, regressor=None, smear: bool = True):
        self.classifier = classifier
        self.regressor = regressor
        self.smear = smear

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)
        positive = y > 0
        self.classifier_ = clone(self.classifier)
        self.regressor_ = clone(self.regressor)
        self.fallback_ = float(y.mean())

        if positive.all() or not positive.any():
            self.classifier_ = None
        else:
            self.classifier_.fit(X, positive.astype(int))

        if positive.sum() >= 3:
            target = np.log1p(y[positive])
            if self.smear:
                self.residuals_ = target - _held_out_predictions(self.regressor_, X[positive], target)
            else:
                self.residuals_ = np.zeros(1)
            self.regressor_.fit(X[positive], target)
        else:
            self.regressor_ = None
            self.residuals_ = np.zeros(1)
        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.regressor_ is None:
            return np.full(len(X), self.fallback_)
        expected = _expm1_smeared(self.regressor_.predict(X), self.residuals_)
        if self.classifier_ is None:
            return expected
        probability = self.classifier_.predict_proba(X)[:, 1]
        return probability * expected


SMEARING_INNER_FOLDS = 3


def _held_out_predictions(estimator, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Predictions for every row from a model that did not see that row."""
    from sklearn.model_selection import KFold, cross_val_predict

    if len(y) < SMEARING_INNER_FOLDS * 2:
        # Too few rows to hold any out. Zero residuals means no smearing at
        # all, which is honest; in-sample residuals would pretend to correct.
        return y.copy()
    splitter = KFold(n_splits=SMEARING_INNER_FOLDS, shuffle=True, random_state=0)
    return cross_val_predict(clone(estimator), X, y, cv=splitter)


def _expm1_smeared(prediction: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Back-transform a log1p prediction to an expectation, not a median.

    exp of the mean is not the mean of exp. Duan's smearing estimator puts the
    training residuals back in, which is what MSE is scored against.
    """
    if residuals.size <= 1:
        return np.expm1(prediction)
    grid = np.expm1(prediction[:, None] + residuals[None, :])
    return grid.mean(axis=1)


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    estimator: object
    target: str = "raw"  # raw | log1p | log1p-smear
    features: str = "smiles"
    augment: bool = False
    clip_quantile: float | None = None
    note: str = ""

    def fitted(self, X: np.ndarray, y: np.ndarray):
        estimator = clone(self.estimator)
        if self.clip_quantile is not None:
            # Hold the training tail back so one molecule cannot drag the fit.
            # Whether that helps or hurts under MSE is an empirical question.
            y = np.minimum(y, np.quantile(y, self.clip_quantile))
        if self.target == "raw":
            estimator.fit(X, y)
            return estimator, None
        target = np.log1p(y)
        residuals = None
        if self.target == "log1p-smear":
            # Residuals taken where the model has not seen the row. In-sample
            # residuals of a forest are far too small - it nearly memorises the
            # training rows - and the smearing factor built from them barely
            # corrects anything, which is the opposite of the point.
            residuals = target - _held_out_predictions(estimator, X, target)
        estimator.fit(X, target)
        return estimator, residuals

    def predicted(self, fitted, X: np.ndarray) -> np.ndarray:
        estimator, residuals = fitted
        raw = estimator.predict(X)
        if self.target == "raw":
            return raw
        if residuals is None:
            return np.expm1(raw)
        return _expm1_smeared(raw, residuals)


class FittedCandidate(BaseEstimator, RegressorMixin):
    """One candidate, fitted and persistable: transform, estimator, inverse.

    The shipped model is this object, so the service and the submission CLI run
    exactly the pipeline the comparison measured - including the smearing
    correction and, where the candidate uses it, the structure lookup.
    Its ``predict`` returns the raw selectivity index, never a log.
    """

    def __init__(self, candidate: "Candidate | None" = None):
        self.candidate = candidate

    def fit(self, X, y):
        self.fitted_ = self.candidate.fitted(np.asarray(X), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X):
        prediction = self.candidate.predicted(self.fitted_, np.asarray(X))
        return np.clip(np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def candidate_by_name(name: str, seed: int) -> Candidate:
    for candidate in candidates(seed):
        if candidate.name == name:
            return candidate
    raise SystemExit(f"unknown candidate {name!r}; see reports/experiments_summary.csv")


def candidates(seed: int) -> list[Candidate]:
    forest = lambda: RandomForestRegressor(  # noqa: E731
        n_estimators=400, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=4
    )
    boosting = lambda: HistGradientBoostingRegressor(  # noqa: E731
        max_iter=400, learning_rate=0.05, min_samples_leaf=5, l2_regularization=1.0,
        early_stopping=False, random_state=seed,
    )
    ridge = lambda: Pipeline(  # noqa: E731
        [("scale", StandardScaler(with_mean=False)), ("model", RidgeCV(alphas=np.logspace(-2, 4, 25)))]
    )

    return [
        Candidate("const-zero", ConstantRegressor("zero"), note="всегда ноль"),
        Candidate("const-mean", ConstantRegressor("mean"), note="среднее обучающей части"),
        Candidate("const-median", ConstantRegressor("median"), note="медиана обучающей части"),
        Candidate("ridge-raw", ridge(), "raw"),
        Candidate("ridge-log1p", ridge(), "log1p"),
        Candidate("ridge-log1p-smear", ridge(), "log1p-smear"),
        Candidate("krr-tanimoto-log1p", TanimotoKernelRidge(alpha=1.0), "log1p"),
        Candidate("krr-tanimoto-log1p-smear", TanimotoKernelRidge(alpha=1.0), "log1p-smear"),
        Candidate("krr-tanimoto-log1p-smear-a0.1", TanimotoKernelRidge(alpha=0.1), "log1p-smear"),
        Candidate("krr-tanimoto-log1p-smear-a0.3", TanimotoKernelRidge(alpha=0.3), "log1p-smear"),
        Candidate("krr-tanimoto-log1p-smear-a3", TanimotoKernelRidge(alpha=3.0), "log1p-smear"),
        Candidate("krr-tanimoto-log1p-smear-a10", TanimotoKernelRidge(alpha=10.0), "log1p-smear"),
        Candidate("krr-tanimoto-log1p-smear-tuned", TunedTanimotoKernelRidge(), "log1p-smear",
                  note="параметр подбирается внутри обучающей части фолда"),
        Candidate("krr-tanimoto-log1p-smear-tuned+external", TunedTanimotoKernelRidge(),
                  "log1p-smear", augment=True,
                  note="то же, плюс собранный внешний набор в обучающей части фолда"),
        Candidate("knn-tanimoto-raw", TanimotoKNN(k=5), "raw"),
        Candidate("knn-tanimoto-log1p", TanimotoKNN(k=5), "log1p"),
        Candidate("svr-log1p", Pipeline([("scale", StandardScaler(with_mean=False)), ("model", SVR(C=10.0))]), "log1p"),
        Candidate("random-forest-raw", forest(), "raw"),
        Candidate("random-forest-log1p", forest(), "log1p"),
        Candidate("random-forest-log1p-smear", forest(), "log1p-smear"),
        Candidate("extra-trees-raw", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=4), "raw"),
        Candidate("hist-boosting-raw", boosting(), "raw"),
        Candidate("hist-boosting-log1p-smear", boosting(), "log1p-smear"),
        Candidate(
            "two-stage-forest",
            TwoStage(
                classifier=RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                    random_state=seed, n_jobs=4,
                ),
                regressor=forest(),
            ),
            "raw",
            note="P(IC50 измерена) × ожидаемый SI",
        ),
        Candidate("random-forest-log1p-smear+culture", forest(), "log1p-smear", features="smiles+culture"),
        Candidate("random-forest-log1p-smear+chembl", forest(), "log1p-smear", augment=True),
        Candidate("lookup+random-forest-raw", StructureLookup(base=forest()), "raw",
                  note="точное совпадение структуры берётся из таблицы, остальное — лес"),
        Candidate("lookup+random-forest-raw+chembl", StructureLookup(base=forest()), "raw", augment=True),
        Candidate("random-forest-raw-clip99", forest(), "raw", clip_quantile=0.99,
                  note="обучающая цель обрезана по 99-му процентилю"),
        Candidate("random-forest-raw-clip95", forest(), "raw", clip_quantile=0.95),
        Candidate("two-stage-forest+chembl", TwoStage(
            classifier=RandomForestClassifier(
                n_estimators=400, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=4
            ),
            regressor=forest(),
        ), "raw", augment=True),
    ]


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------


def butina_labels(X: np.ndarray, cutoff: float = CLUSTER_CUTOFF) -> np.ndarray:
    """Chemical families by Butina clustering on Tanimoto distance."""
    from rdkit.ML.Cluster import Butina

    similarity = tanimoto(_binary(X), _binary(X))
    n = similarity.shape[0]
    distances: list[float] = []
    for i in range(1, n):
        distances.extend((1.0 - similarity[i, :i]).tolist())
    clusters = Butina.ClusterData(distances, n, cutoff, isDistData=True)
    labels = np.empty(n, dtype=object)
    for number, cluster in enumerate(clusters):
        for member in cluster:
            labels[member] = f"cluster_{number}"
    return labels


def grouping(frame: pd.DataFrame, scheme: str, X: np.ndarray) -> np.ndarray:
    if scheme == "molecule":
        return frame["inchikey"].to_numpy()
    if scheme == "scaffold":
        return frame["murcko"].to_numpy()
    if scheme == "cluster":
        return butina_labels(X)
    if scheme == "publication":
        return frame["publication"].to_numpy()
    raise ValueError(f"unknown split scheme {scheme!r}")


def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """Balanced grouped k-fold: whole groups, folds of comparable size."""
    unique, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    unique, counts = unique[order], counts[order]
    # biggest groups first, each to the lightest fold - keeps folds even
    order = np.argsort(-counts, kind="stable")
    sizes = np.zeros(n_folds, dtype=int)
    assignment: dict = {}
    for position in order:
        fold = int(np.argmin(sizes))
        assignment[unique[position]] = fold
        sizes[fold] += counts[position]
    fold_of_row = np.array([assignment[g] for g in groups])
    return [np.where(fold_of_row == fold)[0] for fold in range(n_folds)]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    error = y_true - y_pred
    out = {
        "mse": float(np.mean(error**2)),  # the competition's metric
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "medae": float(np.median(np.abs(error))),
        "n": int(len(y_true)),
    }
    log_true = np.log10(y_true + 0.1)
    log_pred = np.log10(y_pred + 0.1)
    out["rmse_log10"] = float(np.sqrt(np.mean((log_true - log_pred) ** 2)))
    if len(y_true) > 2 and len(np.unique(y_pred)) > 1 and len(np.unique(y_true)) > 1:
        out["spearman"] = float(stats.spearmanr(y_true, y_pred).statistic)
    return out


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    frame: pd.DataFrame
    y: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)
    external: pd.DataFrame | None = None
    external_features: dict[str, np.ndarray] = field(default_factory=dict)
    external_y: np.ndarray | None = None


def load_everything(train_path: str, chembl_path: str | None) -> Dataset:
    official = load_train(train_path)
    frame = official.frame.copy()
    # Grouping by source paper needs a source; the official file has none, so
    # the strain/culture description stands in for "where this row came from".
    frame["publication"] = frame["strain_raw"].astype(str).str.slice(0, 60)
    data = Dataset(frame=frame, y=frame["si"].to_numpy(dtype=np.float64))
    for kind in ("smiles", "smiles+culture"):
        data.features[kind] = build_features(frame, kind)

    if chembl_path and Path(chembl_path).exists():
        external = pd.read_csv(chembl_path)
        external = external.dropna(subset=["canonical_smiles", "si"]).reset_index(drop=True)
        external["culture"] = "other"
        data.external = external
        data.external_y = external["si"].to_numpy(dtype=np.float64)
        for kind in ("smiles", "smiles+culture"):
            data.external_features[kind] = build_features(external, kind)
    return data


def external_rows_to_keep(
    data: Dataset, scheme: str, held_out_keys: set, validation_features: np.ndarray
) -> np.ndarray:
    """Which collected rows may join this fold's training part.

    Always drops an external row that is the very same structure as a held-out
    one. Under the transfer schemes it also drops anything merely similar to a
    held-out molecule, because an analogue in training is exactly what those
    schemes are built to exclude.
    """
    keep = ~data.external["scaffold_key"].isin(held_out_keys).to_numpy()
    if scheme in SCHEMES_FILTERING_EXTERNAL:
        similarity = tanimoto(
            _binary(data.external_features["smiles"]), _binary(validation_features)
        )
        keep &= similarity.max(axis=1) < EXTERNAL_SIMILARITY_CUTOFF
    return keep


def run(
    data: Dataset,
    schemes: tuple[str, ...],
    seeds: tuple[int, ...],
    n_folds: int,
    only: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    oof: list[dict] = []

    for scheme in schemes:
        groups = grouping(data.frame, scheme, data.features["smiles"])
        n_groups = len(np.unique(groups))
        log.info("scheme %s: %d groups", scheme, n_groups)
        for seed in seeds:
            folds = grouped_folds(groups, n_folds, seed)
            for fold_number, validation_index in enumerate(folds):
                train_index = np.setdiff1d(np.arange(len(data.frame)), validation_index)
                held_out_keys = set(data.frame["scaffold_key"].to_numpy()[validation_index])

                for candidate in candidates(seed):
                    if only is not None and candidate.name not in only:
                        continue
                    X_all = data.features[candidate.features]
                    X_train = X_all[train_index]
                    y_train = data.y[train_index]

                    if candidate.augment and data.external is not None:
                        keep = external_rows_to_keep(
                            data, scheme, held_out_keys, data.features["smiles"][validation_index]
                        )
                        X_train = np.vstack([X_train, data.external_features[candidate.features][keep]])
                        y_train = np.concatenate([y_train, data.external_y[keep]])

                    started = time.time()
                    fitted = candidate.fitted(X_train, y_train)
                    prediction = candidate.predicted(fitted, X_all[validation_index])
                    # Clipped once, here, so the scored numbers and the saved
                    # out-of-fold predictions can never describe different runs.
                    prediction = np.clip(
                        np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None
                    )
                    elapsed = time.time() - started

                    metrics = score(data.y[validation_index], prediction)
                    rows.append({
                        "scheme": scheme, "seed": seed, "fold": fold_number,
                        "candidate": candidate.name, "target": candidate.target,
                        "features": candidate.features, "augmented": candidate.augment,
                        "n_train": int(len(y_train)), "n_validation": int(len(validation_index)),
                        "seconds": round(elapsed, 2), **metrics,
                    })
                    for position, row_index in enumerate(validation_index):
                        oof.append({
                            "scheme": scheme, "seed": seed, "candidate": candidate.name,
                            "row": int(data.frame["row"].iloc[row_index]),
                            "inchikey": data.frame["inchikey"].iloc[row_index],
                            "si": float(data.y[row_index]),
                            "prediction": float(max(0.0, prediction[position])),
                        })
            log.info("  seed %s done", seed)
    return pd.DataFrame(rows), pd.DataFrame(oof)


def summarise(experiments: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate and scheme: mean MSE over folds, and its spread."""
    grouped = experiments.groupby(["candidate", "scheme"], as_index=False).agg(
        mse_mean=("mse", "mean"),
        mse_sd=("mse", "std"),
        mse_worst_fold=("mse", "max"),
        rmse_log10_mean=("rmse_log10", "mean"),
        mae_mean=("mae", "mean"),
        spearman_mean=("spearman", "mean"),
        seconds=("seconds", "mean"),
    )
    return grouped.sort_values(["scheme", "mse_mean"]).reset_index(drop=True)


def paired(experiments: pd.DataFrame, reference: str = "const-zero") -> pd.DataFrame:
    """Compare candidates to a reference on the folds they actually share.

    The fold-to-fold spread of MSE here is far larger than the gap between any
    two candidates, because a single molecule with SI in the tens of thousands
    decides a fold. Averaging MSE across folds therefore says almost nothing.
    Differences taken fold by fold, where both candidates saw the same
    molecules, do: the shared variation cancels.
    """
    key = ["scheme", "seed", "fold"]
    base = experiments[experiments["candidate"] == reference].set_index(key)["mse"]
    rows = []
    for (candidate, scheme), part in experiments.groupby(["candidate", "scheme"]):
        aligned = part.set_index(key)["mse"]
        difference = (aligned - base.reindex(aligned.index)).dropna()
        if difference.empty:
            continue
        rows.append({
            "candidate": candidate,
            "scheme": scheme,
            "reference": reference,
            "mean_delta_mse": float(difference.mean()),
            "sd_delta_mse": float(difference.std()),
            "fits_better": int((difference < 0).sum()),
            "fits": int(len(difference)),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["scheme", "mean_delta_mse"]).reset_index(drop=True)


def paired_by_molecule(oof: pd.DataFrame, reference: str = "const-zero") -> pd.DataFrame:
    """The same comparison, tested where the units really are independent.

    Folds are not a sample of 15: the three seeds re-partition the same 320
    molecules, so a signed-rank test over folds counts each molecule three
    times. Molecules are a sample of 320. For each one the squared error is
    averaged over the seeds, and the test runs over those paired per-molecule
    errors.

    Two honest caveats on the p-value, which is descriptive and uncorrected
    across the candidates compared. Under the transfer schemes the molecules
    of one series are not independent of each other, so it stays optimistic.
    And a signed-rank test asks whether the typical molecule improves, while
    the competition's MSE is a mean that a few molecules decide: a candidate
    can win this test and still have the worse MSE, which is why the mean and
    median deltas sit next to it.
    """
    oof = oof.copy()
    oof["squared_error"] = (oof["si"] - oof["prediction"]) ** 2
    per_molecule = (
        oof.groupby(["scheme", "candidate", "inchikey"], as_index=False)["squared_error"].mean()
    )
    rows = []
    for scheme, scope in per_molecule.groupby("scheme"):
        base = scope[scope["candidate"] == reference].set_index("inchikey")["squared_error"]
        for candidate, part in scope.groupby("candidate"):
            if candidate == reference:
                continue
            aligned = part.set_index("inchikey")["squared_error"]
            difference = (aligned - base.reindex(aligned.index)).dropna()
            if difference.empty:
                continue
            row = {
                "candidate": candidate,
                "scheme": scheme,
                "reference": reference,
                "molecules": int(len(difference)),
                "molecules_better": int((difference < 0).sum()),
                "mean_delta_squared_error": float(difference.mean()),
                "median_delta_squared_error": float(difference.median()),
            }
            if difference.abs().sum() > 0:
                test = stats.wilcoxon(difference, zero_method="zsplit")
                row["wilcoxon_p"] = float(test.pvalue)
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["scheme", "mean_delta_squared_error"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare approaches on identical folds")
    parser.add_argument("--train", default="data/kaggle/train_v2.csv")
    parser.add_argument("--chembl", default="data/processed/dataset.csv")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument(
        "--only",
        default=None,
        help="comma-separated candidate names to run instead of all of them",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    started = time.time()
    data = load_everything(args.train, args.chembl)
    log.info("official rows %d, external rows %s", len(data.frame), len(data.external) if data.external is not None else 0)

    schemes = tuple(s for s in args.schemes.split(",") if s)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s)
    only = {name for name in (args.only or '').split(',') if name} or None
    if only:
        log.info('running only: %s', sorted(only))
    experiments, oof = run(data, schemes, seeds, args.folds, only)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    experiments.to_csv(out / "experiments.csv", index=False)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    summary = summarise(experiments)
    summary.to_csv(out / "experiments_summary.csv", index=False)
    paired(experiments).to_csv(out / "experiments_paired.csv", index=False)
    paired_by_molecule(oof).to_csv(out / "experiments_paired_by_molecule.csv", index=False)

    (out / "experiments_config.json").write_text(json.dumps({
        "metric": "MSE против сырого SI (официальная метрика соревнования)",
        "schemes": list(schemes),
        "seeds": list(seeds),
        "folds": args.folds,
        "train": args.train,
        "external": args.chembl,
        "n_official_rows": int(len(data.frame)),
        "n_external_rows": int(len(data.external)) if data.external is not None else 0,
        "candidates": [c.name for c in candidates(seeds[0])],
        "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2, ensure_ascii=False))

    for scheme in schemes:
        part = summary[summary["scheme"] == scheme].head(6)
        log.info("\n=== %s ===\n%s", scheme, part.to_string(index=False))
    log.info("done in %.0f s", time.time() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
