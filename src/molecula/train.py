"""Fit the model that ships, using the approach the comparison picked.

    python -m molecula.train --candidate lookup+random-forest-raw

This script does not choose anything. The choice is made in
:mod:`molecula.experiments`, on folds that every candidate shared, and written
to ``reports/experiments_summary.csv``; here the winner is simply refitted on
every training row the competition allows. The quality numbers copied into the
artefact are the cross-validated ones, so they describe the procedure, not the
rows this particular fit happened to see.

The saved file also carries the reference molecules used for the result
passport: the organisers' training rows (no DOI - the file has none) and the
ChEMBL rows collected by ``molecula.collect`` (with DOI). Each is flagged with
whether it took part in training, so the interface can say which is which.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .chem import FEATURE_NAMES, FP_BITS, FP_RADIUS, N_FEATURES, standardize
from .experiments import (
    FittedCandidate,
    build_features,
    candidate_by_name,
)
from .model import SavedModel, binary_fingerprint
from .official import assert_clean, load_train

log = logging.getLogger("molecula.train")

# Chosen by molecula.experiments, not by hand. A fixed alpha of 0.3 scored
# slightly better on average, but picking it would mean choosing a
# hyperparameter by looking at the very evaluation we then report; the tuned
# variant chooses alpha inside each training fold, so its number describes a
# procedure that can be handed over whole. The gap between them is well inside
# the fold-to-fold spread.
#
# The two ideas that looked most promising beforehand - adding the collected
# ChEMBL rows, and answering from the table when the same structure is already
# in it - both measured worse, so neither ships.
DEFAULT_CANDIDATE = "krr-tanimoto-log1p-smear-tuned"
DOMAIN_THRESHOLD = 0.35


def _reference_rows(
    official: pd.DataFrame, chembl: pd.DataFrame | None, augmented: bool
) -> tuple[list[dict], list]:
    """Molecules the passport can point at, with their provenance."""
    rows: list[dict] = []
    fingerprints: list = []

    for record in official.to_dict("records"):
        molecule = standardize(record["canonical_smiles"])
        rows.append({
            "inchikey": record["inchikey"],
            "canonical_smiles": record["canonical_smiles"],
            "si": float(record["si"]),
            "log10_si": float(np.log10(float(record["si"]) + 0.1)),
            "source": "kaggle-train",
            "title": record.get("title"),
            "culture": record.get("culture"),
            "dois": None,
            "documents": None,
            "used_in_training": True,
        })
        fingerprints.append(binary_fingerprint(molecule.mol))

    if chembl is not None:
        for record in chembl.to_dict("records"):
            molecule = standardize(record["canonical_smiles"])
            rows.append({
                "inchikey": record["inchikey"],
                "canonical_smiles": record["canonical_smiles"],
                "si": float(record["si"]),
                "log10_si": float(record["log10_si"]),
                "source": "chembl",
                "title": record.get("molecule_chembl_id"),
                "culture": record.get("cell_lines"),
                "dois": record.get("dois"),
                "documents": record.get("documents"),
                "used_in_training": bool(augmented),
            })
            fingerprints.append(binary_fingerprint(molecule.mol))
    return rows, fingerprints


def _measured_quality(summary_path: Path, paired_path: Path, candidate: str) -> dict:
    """Copy this candidate's cross-validated numbers, if the study has run."""
    out: dict = {}
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        mine = summary[summary["candidate"] == candidate]
        out["cross_validation"] = {
            str(row["scheme"]): {
                "mse_mean": float(row["mse_mean"]),
                "mse_sd": float(row["mse_sd"]),
                "rmse_log10_mean": float(row["rmse_log10_mean"]),
                "mae_mean": float(row["mae_mean"]),
                "spearman_mean": None if pd.isna(row["spearman_mean"]) else float(row["spearman_mean"]),
            }
            for row in mine.to_dict("records")
        }
        best = summary[summary["scheme"] == "molecule"].sort_values("mse_mean").head(1)
        if len(best):
            out["best_on_molecule_scheme"] = str(best.iloc[0]["candidate"])
    if paired_path.exists():
        comparison = pd.read_csv(paired_path)
        mine = comparison[comparison["candidate"] == candidate]
        out["versus_constant_zero"] = {
            str(row["scheme"]): {
                "mean_delta_mse": float(row["mean_delta_mse"]),
                "sd_delta_mse": float(row["sd_delta_mse"]),
                "fits_better": int(row["fits_better"]),
                "fits": int(row["fits"]),
            }
            for row in mine.to_dict("records")
        }

    # The applicability indicator is measured on out-of-fold predictions by
    # molecula.analysis, which runs before training. Copy its verdict in, so
    # the service can say plainly whether reliability is assessed at all.
    analysis_path = paired_path.parent / "error_analysis.json"
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text())
        if analysis.get("candidate") == candidate:
            out["applicability_domain"] = analysis.get("applicability_domain", {})
            out["error_analysis_scheme"] = analysis.get("scheme")
        else:
            out["applicability_domain_note"] = (
                f"разбор ошибок посчитан для {analysis.get('candidate')!r}, "
                f"а обучается {candidate!r}; показатель применимости для этой модели не измерен"
            )

    by_molecule_path = paired_path.parent / "experiments_paired_by_molecule.csv"
    if by_molecule_path.exists():
        by_molecule = pd.read_csv(by_molecule_path)
        mine = by_molecule[by_molecule["candidate"] == candidate]
        out["versus_constant_zero_by_molecule"] = {
            str(row["scheme"]): {
                "molecules": int(row["molecules"]),
                "molecules_better": int(row["molecules_better"]),
                "mean_delta_squared_error": float(row["mean_delta_squared_error"]),
                "wilcoxon_p_descriptive": (
                    None if pd.isna(row.get("wilcoxon_p")) else float(row["wilcoxon_p"])
                ),
            }
            for row in mine.to_dict("records")
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit the shipped Molecula model")
    parser.add_argument("--train", default="data/kaggle/train_v2.csv")
    parser.add_argument("--chembl", default="data/processed/dataset.csv")
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--out", default="models/molecula_si.joblib")
    parser.add_argument("--report", default="reports/training_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    official = load_train(args.train)
    frame = official.frame
    assert_clean(frame)
    log.info("official training rows: %d (%d excluded)", len(frame), len(official.excluded))

    chembl = None
    if args.chembl and Path(args.chembl).exists():
        chembl = pd.read_csv(args.chembl).dropna(subset=["canonical_smiles", "si"]).reset_index(drop=True)
        chembl["culture"] = "other"
        log.info("collected ChEMBL reference rows: %d", len(chembl))

    candidate = candidate_by_name(args.candidate, args.seed)
    log.info("candidate %s (target %s, features %s, augmented %s)",
             candidate.name, candidate.target, candidate.features, candidate.augment)

    if candidate.features != "smiles":
        raise SystemExit(
            f"{candidate.name} needs the {candidate.features!r} feature set, but the service and "
            "the submission CLI are given a SMILES and nothing else. Shipping it would mean two "
            "different preprocessing paths; keep such variants to the comparison table."
        )

    X = build_features(frame, candidate.features)
    y = frame["si"].to_numpy(dtype=np.float64)
    if candidate.augment:
        if chembl is None:
            raise SystemExit(f"{candidate.name} needs the collected ChEMBL set; run `make collect` first")
        X = np.vstack([X, build_features(chembl, candidate.features)])
        y = np.concatenate([y, chembl["si"].to_numpy(dtype=np.float64)])
    log.info("fitting on %d rows x %d features", X.shape[0], X.shape[1])

    started = time.time()
    estimator = FittedCandidate(candidate).fit(X, y)
    log.info("fitted in %.1f s", time.time() - started)

    rows, fingerprints = _reference_rows(frame, chembl, candidate.augment)
    dataset_version = "kaggle-train-v2"
    if candidate.augment and chembl is not None and "dataset_version" in chembl:
        dataset_version = f"kaggle-train-v2+{chembl['dataset_version'].iloc[0]}"
    model_version = f"molecula-{candidate.name}-{dataset_version}-s{args.seed}-v{__version__}"

    saved = SavedModel(
        estimator=estimator,
        model_version=model_version,
        dataset_version=dataset_version,
        feature_contract={
            "n_features": N_FEATURES if candidate.features == "smiles" else X.shape[1],
            "feature_set": candidate.features,
            "fingerprint": f"счётный Morgan fingerprint, радиус {FP_RADIUS}, {FP_BITS} бит",
            "n_descriptors": len(FEATURE_NAMES) - FP_BITS,
            "target": "SI = CC50 / IC50, сырое значение (официальная цель соревнования)",
            "forbidden_inputs": (
                "колонки CC50_uM и IC50_uM_VACV не читаются ни на одном шаге: условие задачи "
                "запрещает предсказывать SI по двум величинам, из которых он вычисляется"
            ),
        },
        metrics={
            "selected_candidate": candidate.name,
            "selection": (
                "Выбран сравнением на одних и тех же фолдах (molecula.experiments), "
                "метрика — MSE против сырого SI, как на Kaggle."
            ),
            **_measured_quality(Path("reports/experiments_summary.csv"),
                                Path("reports/experiments_paired.csv"), candidate.name),
        },
        training_rows=rows,
        training_fingerprints=fingerprints,
        domain_threshold=DOMAIN_THRESHOLD,
        trained_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        config={
            "seed": args.seed,
            "train": str(args.train),
            "chembl": str(args.chembl) if chembl is not None else None,
            "candidate": candidate.name,
            "target": candidate.target,
            "features": candidate.features,
            "augmented": candidate.augment,
            "n_training_rows": int(X.shape[0]),
            "python_packages": _package_versions(),
        },
        target_kind="si",
    )
    path = saved.save(args.out)
    log.info("saved model: %s (%.1f MB)", path, path.stat().st_size / 1e6)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(
        {"model_version": model_version, "config": saved.config, **saved.metrics},
        indent=2, ensure_ascii=False,
    ))
    log.info("saved report: %s", report)
    return 0


def _package_versions() -> dict:
    import numpy
    import pandas
    import rdkit
    import sklearn

    return {
        "python": __import__("sys").version.split()[0],
        "rdkit": rdkit.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
