"""The competition's own data: loading, validation and the one rule about it.

The organisers ship ``CC50_uM`` and ``IC50_uM_VACV`` in both train and test,
and the task statement forbids using them to build the model or to predict,
because ``SI = CC50 / IC50`` is computable from the pair. Test rows carry both
numbers, so the answer is literally in the file; we do not take it. Every
feature path in this project goes through :func:`features`, which is built
from the SMILES (and, optionally, the cell culture) alone. :func:`assert_clean`
is the guard that keeps a forbidden column from sneaking back in.

The second thing worth knowing: 83 of the 320 training rows carry ``SI = 0``
with a missing IC50. That is the organisers' encoding for a compound whose
inhibitory concentration was not established, not a measured ratio of zero.
They stay in the training set, because the metric is MSE against exactly this
convention, but they are flagged so an experiment can ask what they do.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import InvalidSmilesError, murcko_key, standardize

log = logging.getLogger("molecula.official")

FORBIDDEN_COLUMNS = ("CC50_uM", "IC50_uM_VACV")
TARGET_COLUMN = "SI VACV"
ID_COLUMN = "index"

# Cultures that appear often enough in the training data to be worth a column
# of their own; everything rarer becomes "other". Fitted nowhere - this is a
# fixed vocabulary read off the training file, so train and test agree.
# Order matters: "hel" is a substring of "hela", so the longer name is tried
# first. HeLa (cervical carcinoma) and HEL (embryonic lung fibroblasts) are
# different cell lines and must not be merged.
CULTURE_VOCABULARY = (
    "vero",
    "hela",
    "hel",
    "human foreskin fibroblast",
    "prk",
    "bsc",
    "lung fibroblast",
    "mrc",
    "rk13",
)


@dataclass
class OfficialData:
    frame: pd.DataFrame
    excluded: list[dict]
    audit: dict


def _normalise_culture(value) -> str:
    """Map the free-text culture field onto the fixed vocabulary."""
    text = str(value).strip().lower() if isinstance(value, str) else ""
    for known in CULTURE_VOCABULARY:
        if known in text:
            return known
    return "other"


def _prepare(frame: pd.DataFrame, *, has_target: bool, source: str) -> OfficialData:
    rows: list[dict] = []
    excluded: list[dict] = []

    for position, record in enumerate(frame.to_dict("records")):
        smiles = record.get("SMILES")
        try:
            molecule = standardize(smiles)
        except InvalidSmilesError as error:
            excluded.append(
                {
                    "source": source,
                    "row": position,
                    "title": record.get("Title"),
                    "smiles": smiles,
                    "reason": f"SMILES not readable: {error}",
                }
            )
            continue

        row = {
            "row": position,
            "title": record.get("Title"),
            "input_smiles": smiles,
            "canonical_smiles": molecule.canonical_smiles,
            "inchikey": molecule.inchikey,
            # Two different keys, two different questions: the connectivity
            # block keeps one structure (and its stereoisomers) together, the
            # Murcko framework keeps a whole structural series together.
            "scaffold_key": molecule.inchikey.split("-")[0],
            "murcko": murcko_key(molecule.mol),
            "culture": _normalise_culture(record.get("Culture")),
            "culture_raw": record.get("Culture"),
            "strain_raw": record.get("strain"),
        }
        if ID_COLUMN in record:
            row[ID_COLUMN] = record[ID_COLUMN]
        if has_target:
            target = record.get(TARGET_COLUMN)
            if target is None or not np.isfinite(float(target)):
                excluded.append(
                    {
                        "source": source,
                        "row": position,
                        "title": record.get("Title"),
                        "smiles": smiles,
                        "reason": "target is missing or not a number",
                    }
                )
                continue
            if float(target) < 0:
                excluded.append(
                    {
                        "source": source,
                        "row": position,
                        "title": record.get("Title"),
                        "smiles": smiles,
                        "reason": f"negative selectivity index ({target})",
                    }
                )
                continue
            row["si"] = float(target)
            row["si_is_zero"] = float(target) == 0.0
        rows.append(row)

    prepared = pd.DataFrame(rows)
    audit: dict = {
        "source": source,
        "rows_in_file": int(len(frame)),
        "rows_kept": int(len(prepared)),
        "rows_excluded": int(len(excluded)),
        "unique_structures": int(prepared["inchikey"].nunique()) if len(prepared) else 0,
        "unique_skeletons": int(prepared["scaffold_key"].nunique()) if len(prepared) else 0,
        "unique_murcko_frameworks": int(prepared["murcko"].nunique()) if len(prepared) else 0,
        "cultures": prepared["culture"].value_counts().to_dict() if len(prepared) else {},
    }
    if has_target and len(prepared):
        si = prepared["si"]
        audit["target"] = {
            "mean": float(si.mean()),
            "median": float(si.median()),
            "max": float(si.max()),
            "share_zero": float((si == 0).mean()),
            "share_above_1000": float((si > 1000).mean()),
            "mse_of_constant_zero": float(np.mean(si**2)),
            "mse_of_constant_mean": float(np.mean((si - si.mean()) ** 2)),
        }
        if "IC50_uM_VACV" in frame.columns:
            # Not a feature and not a column of the prepared frame: whether an
            # IC50 was reported is almost exactly whether SI > 0, so carrying
            # that flag around would smuggle a forbidden column back in.
            audit["forbidden_missingness_channel"] = {
                "share_of_rows_with_ic50": float(frame["IC50_uM_VACV"].notna().mean()),
                "agreement_with_si_above_zero": float(
                    (frame["IC50_uM_VACV"].notna().to_numpy() == (prepared["si"] > 0).to_numpy()).mean()
                ) if len(prepared) == len(frame) else None,
                "note": (
                    "Признак «IC50 указана» почти совпадает с «SI > 0», поэтому он не попадает "
                    "ни в подготовленную таблицу, ни в признаки модели."
                ),
            }
        duplicated = prepared["inchikey"].duplicated(keep=False)
        audit["duplicate_structures"] = int(duplicated.sum())
        if duplicated.any():
            spread = (
                prepared[duplicated]
                .groupby("inchikey")["si"]
                .agg(["count", "min", "max"])
                .reset_index()
            )
            audit["duplicate_detail"] = spread.to_dict("records")
    return OfficialData(frame=prepared, excluded=excluded, audit=audit)


def load_train(path: str | Path = "data/kaggle/train_v2.csv") -> OfficialData:
    """The competition training set, standardised and validated."""
    return _prepare(pd.read_csv(path), has_target=True, source=str(path))


def load_test(path: str | Path = "data/kaggle/test.csv") -> OfficialData:
    """The competition test set. The target columns in it are never read."""
    return _prepare(pd.read_csv(path), has_target=False, source=str(path))


def assert_clean(frame: pd.DataFrame) -> None:
    """Fail loudly if a forbidden column reached a frame used for features."""
    present = [column for column in FORBIDDEN_COLUMNS if column in frame.columns]
    if present:
        raise RuntimeError(
            f"{present} must not take part in modelling: the task statement forbids "
            "predicting the selectivity index from the two concentrations it is computed from"
        )


def audit_report(
    train_path: str | Path = "data/kaggle/train_v2.csv",
    test_path: str | Path = "data/kaggle/test.csv",
    out: str | Path = "reports/official_data_audit.json",
) -> dict:
    """Write down what the competition files contain and what we refuse to use."""
    train = load_train(train_path)
    test = load_test(test_path)

    train_keys = set(train.frame["scaffold_key"])
    test_keys = set(test.frame["scaffold_key"])
    report = {
        "train": train.audit,
        "test": test.audit,
        "shared_skeletons_train_test": len(train_keys & test_keys),
        "excluded": train.excluded + test.excluded,
        "forbidden_columns": {
            "columns": list(FORBIDDEN_COLUMNS),
            "present_in_test_file": True,
            "note": (
                "SI = CC50 / IC50, поэтому обе колонки в test.csv дают ответ напрямую. "
                "Условие задачи запрещает их использовать, и ни один признак модели их не читает."
            ),
        },
        "zero_target_convention": (
            "SI = 0 в обучающем наборе означает, что IC50 не установлена, а не измеренное "
            "нулевое отношение. Такие строки остаются в обучении, потому что метрика "
            "считается против этой же записи."
        ),
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("wrote %s", path)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(audit_report(), indent=2, ensure_ascii=False)[:4000])
