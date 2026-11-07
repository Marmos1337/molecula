"""Look every molecule we are asked about up in ChEMBL, by structure.

``molecula.collect`` works outwards from the virus: it finds vaccinia assays,
takes the papers they came from, and reads everything in those papers. That is
the right shape for building a training set, but it covers only the molecules
those particular papers happen to contain - 133 of the 597 test structures.

This module works the other way round: it starts from the structures in the
competition files and asks ChEMBL what is published about each one. Molecules
are matched on the InChIKey connectivity block, so a stereoisomer of a known
compound still finds its literature.

What it does NOT do is read the competition's own CC50 and IC50 columns. Those
are forbidden, and nothing here touches them: the only input is the SMILES.

    python -m molecula.collect_targeted --out data/processed/targeted.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .chem import InvalidSmilesError, standardize
from .collect import (
    ANTIVIRAL_PATTERN,
    CYTOTOX_PATTERN,
    CYTOTOX_TYPES,
    NOT_VACV_PATTERN,
    POTENCY_TYPES,
    VACV_PATTERN,
    ChemblClient,
    _clean_cell,
    _to_micromolar,
)

log = logging.getLogger("molecula.collect_targeted")

BATCH = 40
WORKERS = 8


def structures(paths: list[str]) -> dict[str, str]:
    """InChIKey connectivity block -> canonical SMILES, for every input file."""
    out: dict[str, str] = {}
    for path in paths:
        frame = pd.read_csv(path)
        column = "SMILES" if "SMILES" in frame.columns else "canonical_smiles"
        for smiles in frame[column].dropna():
            try:
                molecule = standardize(smiles)
            except InvalidSmilesError:
                continue
            out.setdefault(molecule.inchikey.split("-")[0], molecule.canonical_smiles)
    return out


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def find_molecules(client: ChemblClient, keys: list[str]) -> dict[str, list[str]]:
    """Connectivity block -> ChEMBL molecule ids that share it."""
    found: dict[str, list[str]] = defaultdict(list)

    def one_batch(batch: list[str]) -> list[dict]:
        return list(client.paginate(
            "molecule", "molecules",
            molecule_structures__standard_inchi_key__in=",".join(batch),
            only="molecule_chembl_id,molecule_structures",
        ))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for rows in pool.map(one_batch, _chunks(keys, BATCH)):
            for row in rows:
                structure = row.get("molecule_structures") or {}
                key = str(structure.get("standard_inchi_key") or "")
                if key:
                    found[key.split("-")[0]].append(row["molecule_chembl_id"])
    return dict(found)


def fetch_activities(client: ChemblClient, molecule_ids: list[str]) -> pd.DataFrame:
    """Every activity record ChEMBL holds for these molecules."""
    fields = (
        "molecule_chembl_id,canonical_smiles,standard_type,standard_relation,standard_value,"
        "standard_units,assay_chembl_id,assay_description,assay_type,document_chembl_id,"
        "target_organism,data_validity_comment,potential_duplicate"
    )

    def one_batch(batch: list[str]) -> list[dict]:
        return list(client.paginate(
            "activity", "activities",
            molecule_chembl_id__in=",".join(batch),
            only=fields,
        ))

    collected: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for rows in pool.map(one_batch, _chunks(molecule_ids, BATCH)):
            collected.extend(rows)
    return pd.DataFrame(collected)


def classify(activities: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Split into vaccinia potency and host-cell cytotoxicity.

    Looser than ``molecula.collect`` in one way only: the cell line does not
    have to be stated, because 13,214 records were dropped for that alone and
    the organisers plainly did not apply that rule. Everything else holds -
    the virus must be vaccinia, the endpoint families must match, and a
    censored value is still refused.
    """
    kept: list[dict] = []
    excluded: list[dict] = []

    for row in activities.to_dict("records"):
        description = _clean_cell(row.get("assay_description")) or ""
        organism = _clean_cell(row.get("target_organism")) or ""
        text = f"{description} {organism}"
        standard_type = _clean_cell(row.get("standard_type")) or ""
        relation = _clean_cell(row.get("standard_relation"))
        validity = _clean_cell(row.get("data_validity_comment"))
        reason = None

        if validity:
            reason = f"data validity flag: {validity}"
        elif relation is not None and relation not in {"=", "~"}:
            reason = f"censored measurement (relation {relation!r})"

        kind = None
        family = None
        # An IC50 measured on uninfected cells is a cytotoxicity reading that
        # happens to carry a potency name. Dividing it into a CC50 from the
        # same paper would give a selectivity index of about one out of thin
        # air, so the wording of the assay decides, not the endpoint name.
        looks_antiviral = bool(ANTIVIRAL_PATTERN.search(description))
        looks_cytotoxic = bool(CYTOTOX_PATTERN.search(description)) and not looks_antiviral

        if standard_type in POTENCY_TYPES and not looks_cytotoxic:
            if VACV_PATTERN.search(text) and not NOT_VACV_PATTERN.search(text):
                kind, family = "potency", POTENCY_TYPES[standard_type]
            else:
                reason = reason or "potency measured against something other than vaccinia"
        elif standard_type in CYTOTOX_TYPES or looks_cytotoxic:
            family = CYTOTOX_TYPES.get(standard_type) or POTENCY_TYPES.get(standard_type)
            if family is None:
                reason = reason or f"endpoint {standard_type!r} is neither potency nor cytotoxicity"
            else:
                kind = "cytotoxicity"
        else:
            reason = reason or f"endpoint {standard_type!r} is neither potency nor cytotoxicity"

        if reason or kind is None:
            excluded.append({
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "standard_type": standard_type,
                "reason": reason or "unclassified",
            })
            continue

        # The API hands numbers back as strings on this endpoint.
        try:
            raw_value = float(row.get("standard_value"))
        except (TypeError, ValueError):
            excluded.append({
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "standard_type": standard_type,
                "reason": "no numeric value",
            })
            continue

        value, note = _to_micromolar(
            raw_value, row.get("standard_units"), row.get("canonical_smiles")
        )
        if value is None or not np.isfinite(value) or value <= 0:
            excluded.append({
                "molecule_chembl_id": row.get("molecule_chembl_id"),
                "standard_type": standard_type,
                "reason": note or "value not usable",
            })
            continue

        kept.append({
            "molecule_chembl_id": row.get("molecule_chembl_id"),
            "smiles": row.get("canonical_smiles"),
            "document_chembl_id": row.get("document_chembl_id"),
            "assay_chembl_id": row.get("assay_chembl_id"),
            "standard_type": standard_type,
            "measurement_kind": kind,
            "endpoint_family": family,
            "value_um": float(value),
            "relation_assumed_exact": relation is None,
        })
    return pd.DataFrame(kept), excluded


def pair(measurements: pd.DataFrame) -> pd.DataFrame:
    """One selectivity index per molecule, from matched endpoint families.

    Both sides must come from the same publication and the same endpoint
    family. Repeats are reduced with a geometric median before the ratio, so a
    molecule measured five times does not get five votes.
    """
    rows: list[dict] = []
    group = ["molecule_chembl_id", "document_chembl_id", "endpoint_family"]
    for (molecule, document, family), part in measurements.groupby(group):
        potency = part[part.measurement_kind == "potency"]["value_um"]
        cytotox = part[part.measurement_kind == "cytotoxicity"]["value_um"]
        if potency.empty or cytotox.empty:
            continue
        ic50 = float(np.exp(np.median(np.log(potency))))
        cc50 = float(np.exp(np.median(np.log(cytotox))))
        if ic50 <= 0:
            continue
        rows.append({
            "molecule_chembl_id": molecule,
            "document_chembl_id": document,
            "endpoint_family": family,
            "smiles": part["smiles"].iloc[0],
            "ic50_um": ic50,
            "cc50_um": cc50,
            "si": cc50 / ic50,
            "n_potency": int(len(potency)),
            "n_cytotox": int(len(cytotox)),
        })
    return pd.DataFrame(rows)


def aggregate(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per structure: the median index across its publications."""
    rows: list[dict] = []
    for smiles, part in pairs.groupby("smiles"):
        try:
            molecule = standardize(smiles)
        except InvalidSmilesError:
            continue
        values = part["si"].to_numpy(dtype=float)
        rows.append({
            "inchikey": molecule.inchikey,
            "scaffold_key": molecule.inchikey.split("-")[0],
            "canonical_smiles": molecule.canonical_smiles,
            "si": float(np.exp(np.median(np.log(values)))),
            "log10_si": float(np.log10(float(np.exp(np.median(np.log(values)))) + 0.1)),
            "n_pairs": int(len(values)),
            "si_min": float(values.min()),
            "si_max": float(values.max()),
            "documents": "; ".join(sorted(set(part["document_chembl_id"].dropna().astype(str)))),
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Look the competition's molecules up in ChEMBL")
    parser.add_argument("--inputs", nargs="+",
                        default=["data/kaggle/test.csv", "data/kaggle/train_v2.csv"])
    parser.add_argument("--out", default="data/processed/targeted.csv")
    parser.add_argument("--report", default="reports/targeted_collection.json")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    started = time.time()
    wanted = structures(args.inputs)
    log.info("structures to look up: %d", len(wanted))

    # ChEMBL matches whole InChIKeys, so ask with the keys from the files.
    keys: list[str] = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        column = "SMILES" if "SMILES" in frame.columns else "canonical_smiles"
        for smiles in frame[column].dropna():
            try:
                keys.append(standardize(smiles).inchikey)
            except InvalidSmilesError:
                continue
    keys = sorted(set(keys))

    client = ChemblClient(Path("data/raw/chembl_cache"))
    found = find_molecules(client, keys)
    molecule_ids = sorted({m for ids in found.values() for m in ids})
    log.info("matched %d of %d structures, %d ChEMBL molecules",
             len(found), len(wanted), len(molecule_ids))

    activities = fetch_activities(client, molecule_ids)
    log.info("activity records: %d", len(activities))

    measurements, excluded = classify(activities)
    log.info("usable measurements: %d (excluded %d)", len(measurements), len(excluded))

    pairs = pair(measurements) if len(measurements) else pd.DataFrame()
    dataset = aggregate(pairs) if len(pairs) else pd.DataFrame()
    log.info("molecules with a selectivity index: %d", len(dataset))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    if len(pairs):
        pairs.to_csv(out.with_name(out.stem + "_pairs.csv"), index=False)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "structures_requested": len(wanted),
        "structures_matched_in_chembl": len(found),
        "chembl_molecules": len(molecule_ids),
        "activity_records": int(len(activities)),
        "usable_measurements": int(len(measurements)),
        "molecules_with_si": int(len(dataset)),
        "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2, ensure_ascii=False))
    log.info("done in %.0f s -> %s", time.time() - started, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
