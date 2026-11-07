"""What the errors look like once the comparison is over.

Reads the out-of-fold predictions written by :mod:`molecula.experiments` and
answers the questions a reviewer will ask: where does the metric actually come
from, which molecules does the model get wrong, and does the similarity
indicator deserve to be shown next to a prediction.

The applicability check is the reason this file exists at all. Measured on a
single 45-molecule holdout it was indistinguishable from noise; measured on
every out-of-fold prediction across three seeds it rests on nearly a thousand
points, which is enough to answer honestly either way.

    python -m molecula.analysis --candidate lookup+random-forest-raw
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .experiments import _binary, build_features, tanimoto
from .official import load_train

log = logging.getLogger("molecula.analysis")

SI_BUCKETS = (
    ("= 0 (IC50 не установлена)", lambda s: s == 0),
    ("(0, 1]", lambda s: (s > 0) & (s <= 1)),
    ("(1, 10]", lambda s: (s > 1) & (s <= 10)),
    ("(10, 100]", lambda s: (s > 10) & (s <= 100)),
    ("(100, 1000]", lambda s: (s > 100) & (s <= 1000)),
    ("> 1000", lambda s: s > 1000),
)


def concentration(si: np.ndarray) -> dict:
    """How much of the metric a handful of molecules decides."""
    squared = np.sort(si**2)[::-1]
    total = float(squared.sum())
    return {
        "note": (
            "MSE считается на сыром SI, поэтому вклад молекулы в метрику растёт как квадрат "
            "её значения. Ниже доля всей суммы квадратов, приходящаяся на k самых больших."
        ),
        "share_from_top": {
            str(k): round(float(squared[:k].sum() / total), 4) for k in (1, 2, 5, 10, 20)
        },
        "largest_si": [round(float(v), 1) for v in np.sort(si)[::-1][:5]],
    }


def leave_one_out_similarity(frame: pd.DataFrame) -> np.ndarray:
    """Max Tanimoto of each molecule to every structurally different molecule.

    A molecule's own duplicate or stereoisomer is excluded, otherwise half the
    set would score 1.0 against itself and the bins would be meaningless. It
    ignores which fold a molecule sat in, so it stays slightly optimistic; it
    is used to bin molecules, never to pick a model.
    """
    binary = _binary(build_features(frame, "smiles"))
    similarity = tanimoto(binary, binary)
    keys = frame["scaffold_key"].to_numpy()
    same_structure = keys[:, None] == keys[None, :]
    similarity = np.where(same_structure, -1.0, similarity)
    return similarity.max(axis=1)


def by_group(frame: pd.DataFrame, labels: pd.Series, name: str) -> list[dict]:
    rows = []
    for value, part in frame.groupby(labels, observed=True):
        error = part["si"] - part["prediction"]
        log_error = np.abs(np.log10(part["si"] + 0.1) - np.log10(part["prediction"] + 0.1))
        rows.append({
            name: str(value),
            "n": int(len(part)),
            "mse": float(np.mean(error**2)),
            "mae": float(np.mean(np.abs(error))),
            "median_abs_log10_error": float(np.median(log_error)),
            "mean_si": float(part["si"].mean()),
            "mean_prediction": float(part["prediction"].mean()),
        })
    return sorted(rows, key=lambda r: -r["n"])


def applicability(frame: pd.DataFrame, similarity: pd.Series, threshold: float) -> dict:
    """Does similarity to the training set track the size of the error?"""
    log_error = np.abs(np.log10(frame["si"] + 0.1) - np.log10(frame["prediction"] + 0.1))
    correlation = stats.spearmanr(similarity, log_error)
    inside = similarity >= threshold
    result = {
        "indicator": "максимальное сходство Танимото с обучающим набором (Morgan, радиус 2, 2048 бит)",
        "threshold": threshold,
        "n_points": int(len(frame)),  # molecules, one row each
        "share_below_threshold": float(np.mean(~inside)),
        "spearman_similarity_vs_abs_log_error": float(correlation.statistic),
        "spearman_p_value": float(correlation.pvalue),
    }
    if inside.sum() >= 10 and (~inside).sum() >= 10:
        result["median_abs_log10_error_inside"] = float(np.median(log_error[inside]))
        result["median_abs_log10_error_outside"] = float(np.median(log_error[~inside]))
        result["separates_error"] = bool(
            correlation.statistic < 0
            and correlation.pvalue < 0.05
            and result["median_abs_log10_error_outside"] > result["median_abs_log10_error_inside"]
        )
    else:
        result["separates_error"] = None
        result["note"] = "по одну сторону порога слишком мало молекул"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Error analysis on out-of-fold predictions")
    parser.add_argument("--oof", default="reports/oof_predictions.csv")
    parser.add_argument("--train", default="data/kaggle/train_v2.csv")
    parser.add_argument("--candidate", default=None, help="default: best by mean MSE on the molecule scheme")
    parser.add_argument("--scheme", default="molecule")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--out", default="reports/error_analysis.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    oof = pd.read_csv(args.oof)
    official = load_train(args.train).frame

    candidate = args.candidate
    if candidate is None:
        scope = oof[oof["scheme"] == args.scheme]
        means = scope.groupby("candidate").apply(
            lambda p: np.mean((p["si"] - p["prediction"]) ** 2), include_groups=False
        )
        candidate = str(means.idxmin())
    log.info("analysing %s on the %s scheme", candidate, args.scheme)

    chosen = oof[(oof["candidate"] == candidate) & (oof["scheme"] == args.scheme)].copy()
    if chosen.empty:
        raise SystemExit(f"no out-of-fold rows for {candidate!r} on scheme {args.scheme!r}")

    similarity = pd.Series(leave_one_out_similarity(official), index=official["row"].to_numpy())
    chosen["similarity"] = chosen["row"].map(similarity)
    chosen["culture"] = chosen["row"].map(official.set_index("row")["culture"])

    # One row per molecule. The seeds re-partition the same molecules, so
    # pooling their predictions would count every molecule three times and
    # make any correlation look better supported than it is.
    pooled = chosen
    chosen = (
        chosen.groupby("row", as_index=False)
        .agg(
            inchikey=("inchikey", "first"),
            si=("si", "first"),
            prediction=("prediction", "mean"),
            similarity=("similarity", "first"),
            culture=("culture", "first"),
            seed=("seed", "count"),
        )
        .rename(columns={"seed": "seeds_averaged"})
    )

    bucket = pd.Series("", index=chosen.index, dtype=object)
    for label, test in SI_BUCKETS:
        bucket[test(chosen["si"])] = label
    chosen["bucket"] = bucket

    similarity_bin = pd.cut(
        chosen["similarity"], [0, 0.2, 0.35, 0.5, 0.7, 1.01],
        labels=["< 0.2", "0.2-0.35", "0.35-0.5", "0.5-0.7", "> 0.7"],
    )

    report = {
        "candidate": candidate,
        "scheme": args.scheme,
        "seeds_averaged": sorted(int(s) for s in pooled["seed"].unique()),
        "n_molecules": int(len(chosen)),
        "note_on_counting": (
            "Каждая молекула входит один раз: прогнозы по трём зёрнам усреднены. "
            "Складывать их в одну выборку нельзя — это те же молекулы, а не новые."
        ),
        "overall": {
            "note": (
                "Считается на прогнозах, усреднённых по зёрнам, поэтому MSE здесь немного "
                "ниже, чем в таблице сравнения: усреднение трёх прогнозов — это уже маленький "
                "ансамбль. Сопоставимое с таблицей число — mse_per_fit."
            ),
            "mse_seed_averaged": float(np.mean((chosen["si"] - chosen["prediction"]) ** 2)),
            "mse_per_fit": float(np.mean((pooled["si"] - pooled["prediction"]) ** 2)),
            "mse": float(np.mean((chosen["si"] - chosen["prediction"]) ** 2)),
            "mae": float(np.mean(np.abs(chosen["si"] - chosen["prediction"]))),
            "median_abs_log10_error": float(np.median(
                np.abs(np.log10(chosen["si"] + 0.1) - np.log10(chosen["prediction"] + 0.1))
            )),
        },
        "metric_concentration": concentration(official["si"].to_numpy()),
        "by_si_range": by_group(chosen, chosen["bucket"], "range"),
        "by_culture": by_group(chosen, chosen["culture"], "culture"),
        "by_similarity_to_training": by_group(chosen, similarity_bin, "similarity"),
        "applicability_domain": applicability(chosen, chosen["similarity"], args.threshold),
        "worst_molecules": (
            chosen.assign(squared_error=(chosen["si"] - chosen["prediction"]) ** 2)
            .nlargest(10, "squared_error")[["row", "inchikey", "si", "prediction", "similarity"]]
            .round(3)
            .to_dict("records")
        ),
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("wrote %s", path)
    log.info(json.dumps(report["applicability_domain"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
