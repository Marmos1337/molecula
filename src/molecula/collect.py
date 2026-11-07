"""Reproducible collection of VACV/VV selectivity data from ChEMBL.

What this does, in order:

1. find every ChEMBL assay whose organism or description points at vaccinia
   virus (VACV / VV / Orthopoxvirus vaccinia);
2. take the publications those assays come from, and pull every activity
   recorded in those publications;
3. split activities into antiviral potency (IC50/EC50/MIC...) measured in a
   vaccinia assay, and cytotoxicity (CC50/TC50/MCC/CD50...) measured on host
   cells;
4. normalise concentrations to micromolar, converting ug/mL with the molecular
   weight that RDKit computes from the recorded SMILES;
5. pair potency with cytotoxicity only inside one publication and one cell
   line, then compute SI = CC50 / IC50;
6. write the curated dataset, a source list with DOIs, and an exclusions
   report that says why every rejected record was rejected.

Everything that is thrown away lands in the exclusions report - nothing is
silently dropped.

    python -m molecula.collect --out data

Raw API pages are cached under data/raw/chembl_cache so a re-run is offline
and byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .chem import InvalidSmilesError, molecular_weight, standardize

log = logging.getLogger("molecula.collect")

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
PAGE_SIZE = 200
WORKERS = 8
USER_AGENT = "molecula-hackathon/1.0 (chemoinformatics coursework; contact via repository)"

DATASET_VERSION = "chembl-vacv-v1"

# A ratio is only meaningful between two endpoints of the same kind. Half-maximal
# values (IC50 and its cytotoxic counterparts) and "minimum visible effect"
# values (the De Clercq MIC/MCC pair) are both used in this literature, but a
# MIC divided by a CC50 is not a selectivity index - so each readout carries the
# family it belongs to, and only same-family measurements are ever paired.
POTENCY_TYPES = {
    "IC50": "half-maximal",
    "EC50": "half-maximal",
    "ID50": "half-maximal",
    "MIC": "minimum-effect",
    "MIC50": "half-maximal",
}
CYTOTOX_TYPES = {
    "CC50": "half-maximal",
    "TC50": "half-maximal",
    "CD50": "half-maximal",
    "TD50": "half-maximal",
    "MCC": "minimum-effect",
    "MTC": "minimum-effect",
}

# Finding candidate assays is deliberately wide; deciding that a measurement is
# vaccinia potency is deliberately narrow. Cowpox and unspecified poxviruses
# bring in assays that are not the target of this task.
POXVIRUS_PATTERN = re.compile(
    r"vaccinia|\bvacv\b|\bvv\b|orthopoxvirus|cowpox|poxvirus", re.IGNORECASE
)
VACV_PATTERN = re.compile(r"vaccinia|\bvacv\b|\bvv\b|orthopoxvirus vaccinia", re.IGNORECASE)
NOT_VACV_PATTERN = re.compile(r"cowpox|monkeypox|\bcpxv\b|\bmpxv\b|camelpox|ectromelia", re.IGNORECASE)
CYTOTOX_PATTERN = re.compile(
    r"cytotox|cytostat|cell viability|minimum cytotoxic|morphology of|"
    r"host cell|uninfected|cell growth|cytopathic.*absence",
    re.IGNORECASE,
)
ANTIVIRAL_PATTERN = re.compile(
    r"antivir|virus-induced|virus induced|cytopathogenicity|plaque|virus yield|"
    r"virus replication|viral replication|infected",
    re.IGNORECASE,
)

# Units we accept and how to reach micromolar.
MOLAR_TO_UM = {
    "M": 1e6,
    "mM": 1e3,
    "uM": 1.0,
    "nM": 1e-3,
    "pM": 1e-6,
}
MASS_TO_UG_PER_ML = {
    "ug.mL-1": 1.0,
    "ug ml-1": 1.0,  # same unit, older spelling in ChEMBL
    "ug/mL": 1.0,
    "mg.L-1": 1.0,
    "ng.mL-1": 1e-3,
    "mg.mL-1": 1e3,
}


# --------------------------------------------------------------------------
# cached HTTP
# --------------------------------------------------------------------------


class ChemblClient:
    def __init__(self, cache_dir: Path, offline: bool = False, pause: float = 0.0):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.pause = pause
        self.hits = 0
        self.misses = 0

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:40]
        return self.cache_dir / f"{digest}.json"

    def get(self, path: str, **params: Any) -> dict:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{CHEMBL_BASE}/{path}.json?{query}"
        cached = self._cache_path(url)
        if cached.exists():
            self.hits += 1
            return json.loads(cached.read_text())
        if self.offline:
            raise RuntimeError(f"offline mode but {url} is not cached")

        last_error: Exception | None = None
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = json.load(response)
                break
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(1.5 * (attempt + 1))
        else:
            raise RuntimeError(f"ChEMBL request failed after retries: {url}") from last_error

        # write via a temporary file: an interrupted run must not leave a
        # truncated page that every later run then fails to parse
        temporary = cached.with_suffix(".part")
        temporary.write_text(json.dumps(payload))
        temporary.replace(cached)
        self.misses += 1
        time.sleep(self.pause)
        return payload

    def paginate(self, path: str, collection: str, **params: Any) -> Iterator[dict]:
        offset = 0
        while True:
            page = self.get(path, limit=PAGE_SIZE, offset=offset, **params)
            rows = page.get(collection, [])
            yield from rows
            meta = page.get("page_meta", {})
            if not meta.get("next") or not rows:
                return
            offset += PAGE_SIZE


# --------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------


def find_vacv_assays(client: ChemblClient) -> pd.DataFrame:
    """Assays whose organism names vaccinia, plus assays whose text does."""
    rows: dict[str, dict] = {}

    for assay in client.paginate("assay", "assays", assay_organism__icontains="vaccinia"):
        rows[assay["assay_chembl_id"]] = assay
    for term in ("vaccinia", "VACV"):
        for assay in client.paginate("assay", "assays", description__icontains=term):
            rows.setdefault(assay["assay_chembl_id"], assay)

    records = []
    for assay in rows.values():
        text = " ".join(str(assay.get(k) or "") for k in ("assay_organism", "description"))
        if not POXVIRUS_PATTERN.search(text):
            continue
        records.append(
            {
                "assay_chembl_id": assay["assay_chembl_id"],
                "document_chembl_id": assay.get("document_chembl_id"),
                "assay_cell_type": assay.get("assay_cell_type"),
                "assay_organism": assay.get("assay_organism"),
                "assay_description": assay.get("description"),
                "assay_type": assay.get("assay_type"),
            }
        )
    frame = pd.DataFrame(records)
    log.info("vaccinia assays: %d in %d publications", len(frame), frame["document_chembl_id"].nunique())
    return frame


def fetch_candidate_assays(client: ChemblClient, document_ids: list[str]) -> pd.DataFrame:
    """Every assay of the given publications, narrowed to the ones we can use.

    Going through assays rather than through "all activities of a publication" is
    what makes a full run take minutes: a medicinal-chemistry paper can carry
    tens of thousands of unrelated measurements.
    """
    batches = [document_ids[i : i + 25] for i in range(0, len(document_ids), 25)]

    def one_batch(batch: list[str]) -> list[dict]:
        return list(
            client.paginate("assay", "assays", document_chembl_id__in=",".join(batch))
        )

    assays: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for rows in pool.map(one_batch, batches):
            assays.extend(rows)
    log.info("assays in those publications: %d", len(assays))

    kept = []
    for assay in assays:
        text = " ".join(str(assay.get(k) or "") for k in ("assay_organism", "description"))
        is_vacv = bool(VACV_PATTERN.search(text)) and not NOT_VACV_PATTERN.search(text)
        is_cytotox = bool(CYTOTOX_PATTERN.search(text))
        if not (is_vacv or is_cytotox):
            continue
        kept.append(
            {
                "assay_chembl_id": assay["assay_chembl_id"],
                "document_chembl_id": assay.get("document_chembl_id"),
                "assay_cell_type": assay.get("assay_cell_type"),
                "assay_organism": assay.get("assay_organism"),
                "assay_description": assay.get("description"),
                "assay_type": assay.get("assay_type"),
                "looks_vacv": is_vacv,
                "looks_cytotoxic": is_cytotox,
            }
        )
    frame = pd.DataFrame(kept)
    log.info("assays kept as antiviral or cytotoxicity readouts: %d", len(frame))
    return frame


def fetch_assay_activities(client: ChemblClient, assay_ids: list[str]) -> pd.DataFrame:
    """Measurements recorded in the given assays, only for the readout types we use."""
    keep = (
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "activity_comment",
        "assay_chembl_id",
        "assay_description",
        "assay_type",
        "document_chembl_id",
        "target_organism",
        "data_validity_comment",
        "potential_duplicate",
    )
    wanted_types = ",".join(sorted(POTENCY_TYPES | CYTOTOX_TYPES))
    batches = [assay_ids[i : i + 20] for i in range(0, len(assay_ids), 20)]

    def one_batch(batch: list[str]) -> list[dict]:
        return [
            {k: activity.get(k) for k in keep}
            for activity in client.paginate(
                "activity",
                "activities",
                assay_chembl_id__in=",".join(batch),
                standard_type__in=wanted_types,
            )
        ]

    records: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for rows in pool.map(one_batch, batches):
            records.extend(rows)
            done += 1
            if done % 10 == 0:
                log.info("  assay batches pulled: %d/%d", done, len(batches))
    frame = pd.DataFrame(records)
    log.info("usable measurements pulled: %d", len(frame))
    return frame


def fetch_documents(client: ChemblClient, document_ids: list[str]) -> pd.DataFrame:
    def one(document_id: str) -> dict | None:
        page = client.get("document", document_chembl_id=document_id, limit=1)
        docs = page.get("documents", [])
        if not docs:
            return None
        doc = docs[0]
        return (
            {
                "document_chembl_id": doc.get("document_chembl_id"),
                "doi": doc.get("doi"),
                "pubmed_id": doc.get("pubmed_id"),
                "year": doc.get("year"),
                "journal": doc.get("journal"),
                "title": doc.get("title"),
            }
        )

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        records = [r for r in pool.map(one, document_ids) if r]
    return pd.DataFrame(records)


def _to_micromolar(value: float, units: str | None, smiles: str | None) -> tuple[float | None, str]:
    """Return (micromolar value, note). note explains a conversion or refusal."""
    if units in MOLAR_TO_UM:
        return value * MOLAR_TO_UM[units], f"molar:{units}"
    if units in MASS_TO_UG_PER_ML:
        weight = molecular_weight(smiles)
        if not weight:
            return None, "mass units but molecular weight unavailable"
        # ug/mL -> uM : (ug/mL) / (g/mol) * 1000
        return value * MASS_TO_UG_PER_ML[units] / weight * 1000.0, f"converted from {units} using MW={weight:.1f}"
    return None, f"unsupported units: {units!r}"


def classify_measurements(
    activities: pd.DataFrame, vacv_assays: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """Turn raw activities into normalised potency/cytotoxicity measurements."""
    vacv_assay_ids = set(vacv_assays["assay_chembl_id"])
    cell_by_assay = dict(zip(vacv_assays["assay_chembl_id"], vacv_assays["assay_cell_type"]))

    measurements: list[dict] = []
    exclusions: list[dict] = []

    for row in activities.to_dict("records"):
        description = str(row.get("assay_description") or "")
        standard_type = (row.get("standard_type") or "").strip()
        assay_id = row.get("assay_chembl_id")
        smiles = row.get("canonical_smiles")

        # Potency has to be vaccinia specifically - a cowpox or monkeypox assay in
        # the same paper is a different target, and the task asks for VACV/VV.
        names_vacv = bool(VACV_PATTERN.search(description)) and not NOT_VACV_PATTERN.search(description)
        is_vacv_assay = (assay_id in vacv_assay_ids and not NOT_VACV_PATTERN.search(description)) or names_vacv

        looks_antiviral = bool(ANTIVIRAL_PATTERN.search(description))
        # "inhibition of virus-induced cytotoxicity" is an antiviral readout that
        # happens to contain the word cytotoxicity, so an antiviral description
        # wins unless the type itself is a cytotoxicity endpoint.
        looks_cytotoxic = standard_type in CYTOTOX_TYPES or (
            bool(CYTOTOX_PATTERN.search(description)) and not looks_antiviral
        )

        if standard_type in POTENCY_TYPES and is_vacv_assay and not looks_cytotoxic:
            kind = "potency"
            endpoint_family = POTENCY_TYPES[standard_type]
        elif looks_cytotoxic and standard_type in CYTOTOX_TYPES:
            kind = "cytotoxicity"
            endpoint_family = CYTOTOX_TYPES[standard_type]
        elif looks_cytotoxic and standard_type in POTENCY_TYPES:
            # A potency-named type used as a cytotoxicity readout (an IC50 measured
            # on uninfected cells). Keep it, in its own family.
            kind = "cytotoxicity"
            endpoint_family = POTENCY_TYPES[standard_type]
        else:
            continue  # unrelated readout in the same paper, not a candidate at all

        # ChEMBL leaves these fields empty as NaN, which is truthy - check for text.
        validity = row.get("data_validity_comment")
        validity = validity.strip() if isinstance(validity, str) else ""
        relation = row.get("standard_relation")
        relation = relation.strip() if isinstance(relation, str) else ""
        # An empty relation is how ChEMBL stores a plain number from an older
        # paper. We accept it as exact and mark the record, rather than either
        # dropping a third of the corpus or hiding the assumption.
        relation_assumed_exact = relation == ""

        reason = None
        if not smiles:
            reason = "no SMILES recorded for the molecule"
        elif row.get("standard_value") in (None, ""):
            reason = "no numeric value"
        elif relation not in ("", "="):
            reason = f"censored measurement (relation {relation!r})"
        elif validity:
            reason = f"ChEMBL data validity flag: {validity}"

        micromolar = None
        note = ""
        if reason is None:
            try:
                raw_value = float(row["standard_value"])
            except (TypeError, ValueError):
                reason = "value is not a number"
                raw_value = None
            if reason is None:
                if raw_value <= 0:
                    reason = "non-positive concentration"
                else:
                    micromolar, note = _to_micromolar(raw_value, row.get("standard_units"), smiles)
                    if micromolar is None:
                        reason = note

        if reason is not None:
            exclusions.append(
                {
                    "stage": "measurement",
                    "molecule_chembl_id": row.get("molecule_chembl_id"),
                    "smiles": smiles,
                    "document_chembl_id": row.get("document_chembl_id"),
                    "assay_chembl_id": assay_id,
                    "measurement_kind": kind,
                    "standard_type": standard_type,
                    "standard_relation": relation or None,
                    "standard_value": row.get("standard_value"),
                    "standard_units": row.get("standard_units"),
                    "reason": reason,
                }
            )
            continue

        cell = row.get("assay_cell_type") if "assay_cell_type" in row else cell_by_assay.get(assay_id)
        measurements.append(
            {
                "molecule_chembl_id": row["molecule_chembl_id"],
                "smiles": smiles,
                "document_chembl_id": row.get("document_chembl_id"),
                "assay_chembl_id": assay_id,
                "assay_description": description,
                "cell_line": cell,
                "measurement_kind": kind,
                "standard_type": standard_type,
                "raw_value": float(row["standard_value"]),
                "raw_units": row.get("standard_units"),
                "value_um": float(micromolar),
                "unit_note": note,
                "endpoint_family": endpoint_family,
                "relation_assumed_exact": relation_assumed_exact,
            }
        )

    return pd.DataFrame(measurements), exclusions


def _clean_cell(value: Any) -> str | None:
    """ChEMBL writes an unknown cell line as NaN; NaN is truthy and compares equal
    to itself, so an unguarded comparison silently calls two unknowns a match."""
    return value.strip() if isinstance(value, str) and value.strip() else None


def pair_into_si(
    measurements: pd.DataFrame, assay_cells: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict]]:
    """Build one SI per (molecule, publication, cell line, endpoint family).

    Rules that keep a ratio meaningful:
      * both sides come from the same publication;
      * both sides are measured on the same, named cell line;
      * both sides are the same kind of endpoint - half-maximal with half-maximal,
        "minimum visible effect" with "minimum visible effect";
      * where a publication reports several assays of one kind for a molecule,
        their values are averaged first (median), so one potency value cannot be
        reused across several pairs and inflate its own weight.
    """
    import numpy as np

    exclusions: list[dict] = []
    pairs: list[dict] = []

    if measurements.empty:
        return pd.DataFrame(), exclusions

    measurements = measurements.copy()
    measurements["cell_line"] = [
        _clean_cell(assay_cells.get(assay_id)) or _clean_cell(fallback)
        for assay_id, fallback in zip(measurements["assay_chembl_id"], measurements["cell_line"])
    ]

    known_cell = measurements["cell_line"].notna()
    for row in measurements[~known_cell].to_dict("records"):
        exclusions.append(
            {
                "stage": "pairing",
                "molecule_chembl_id": row["molecule_chembl_id"],
                "smiles": row["smiles"],
                "document_chembl_id": row["document_chembl_id"],
                "assay_chembl_id": row["assay_chembl_id"],
                "reason": "cell line not stated for this assay - comparability cannot be checked",
            }
        )
    measurements = measurements[known_cell]

    grouped = measurements.groupby(
        ["molecule_chembl_id", "document_chembl_id", "cell_line", "endpoint_family"], dropna=False
    )
    for (molecule_id, document_id, cell_line, family), group in grouped:
        potency = group[group["measurement_kind"] == "potency"]
        cytotox = group[group["measurement_kind"] == "cytotoxicity"]

        if potency.empty or cytotox.empty:
            missing = "cytotoxicity" if cytotox.empty else "antiviral potency"
            exclusions.append(
                {
                    "stage": "pairing",
                    "molecule_chembl_id": molecule_id,
                    "smiles": group["smiles"].iloc[0],
                    "document_chembl_id": document_id,
                    "reason": (
                        f"publication reports no comparable {missing} for this molecule "
                        f"on {cell_line} with a {family} endpoint"
                    ),
                }
            )
            continue

        # Average each side first: several assays of the same kind in one paper
        # (different exposure times, repeats) describe the same quantity.
        ic50 = float(np.exp(np.median(np.log(potency["value_um"].to_numpy()))))
        cc50 = float(np.exp(np.median(np.log(cytotox["value_um"].to_numpy()))))
        si = cc50 / ic50

        if not (1e-4 < si < 1e6):
            exclusions.append(
                {
                    "stage": "pairing",
                    "molecule_chembl_id": molecule_id,
                    "smiles": group["smiles"].iloc[0],
                    "document_chembl_id": document_id,
                    "reason": f"implausible SI value {si:.3g}",
                }
            )
            continue

        pairs.append(
            {
                "molecule_chembl_id": molecule_id,
                "smiles": group["smiles"].iloc[0],
                "document_chembl_id": document_id,
                "ic50_um": ic50,
                "ic50_type": "; ".join(sorted(set(potency["standard_type"]))),
                "ic50_raw": "; ".join(
                    f"{v} {u}" for v, u in zip(potency["raw_value"], potency["raw_units"])
                ),
                "cc50_um": cc50,
                "cc50_type": "; ".join(sorted(set(cytotox["standard_type"]))),
                "cc50_raw": "; ".join(
                    f"{v} {u}" for v, u in zip(cytotox["raw_value"], cytotox["raw_units"])
                ),
                "si": si,
                "cell_line": cell_line,
                "endpoint_family": family,
                "n_potency_assays": int(potency["assay_chembl_id"].nunique()),
                "n_cytotox_assays": int(cytotox["assay_chembl_id"].nunique()),
                "relation_assumed_exact": bool(
                    potency["relation_assumed_exact"].any() or cytotox["relation_assumed_exact"].any()
                ),
                "potency_assays": "; ".join(sorted(set(potency["assay_chembl_id"]))),
                "cytotox_assays": "; ".join(sorted(set(cytotox["assay_chembl_id"]))),
            }
        )

    return pd.DataFrame(pairs), exclusions


def aggregate_per_molecule(pairs: pd.DataFrame, documents: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """One row per distinct structure: median log10(SI) over its measurements."""
    import numpy as np

    exclusions: list[dict] = []
    standardised: list[dict] = []

    for row in pairs.to_dict("records"):
        try:
            molecule = standardize(row["smiles"])
        except InvalidSmilesError as exc:
            exclusions.append(
                {
                    "stage": "standardisation",
                    "molecule_chembl_id": row["molecule_chembl_id"],
                    "smiles": row["smiles"],
                    "document_chembl_id": row["document_chembl_id"],
                    "reason": f"SMILES rejected: {exc}",
                }
            )
            continue
        row = dict(row)
        row["canonical_smiles"] = molecule.canonical_smiles
        row["inchikey"] = molecule.inchikey
        row["scaffold_key"] = molecule.scaffold_key
        standardised.append(row)

    if not standardised:
        return pd.DataFrame(), exclusions

    frame = pd.DataFrame(standardised)
    doi_by_doc = dict(zip(documents["document_chembl_id"], documents["doi"])) if not documents.empty else {}

    out_rows = []
    for inchikey, group in frame.groupby("inchikey"):
        log_si = np.log10(group["si"].to_numpy())
        spread = float(log_si.max() - log_si.min()) if len(log_si) > 1 else 0.0
        documents_used = sorted(set(group["document_chembl_id"].dropna()))
        dois = sorted({doi_by_doc.get(d) for d in documents_used if doi_by_doc.get(d)})
        out_rows.append(
            {
                "inchikey": inchikey,
                "scaffold_key": group["scaffold_key"].iloc[0],
                "canonical_smiles": group["canonical_smiles"].iloc[0],
                "molecule_chembl_id": group["molecule_chembl_id"].iloc[0],
                "si": float(10 ** np.median(log_si)),
                "log10_si": float(np.median(log_si)),
                "n_measurements": int(len(group)),
                "log10_si_spread": spread,
                "ic50_um_median": float(np.median(group["ic50_um"])),
                "cc50_um_median": float(np.median(group["cc50_um"])),
                "cell_lines": "; ".join(sorted({str(c) for c in group["cell_line"].dropna()})) or None,
                "endpoint_families": "; ".join(sorted(set(group["endpoint_family"]))),
                "relation_assumed_exact": bool(group["relation_assumed_exact"].any()),
                "documents": "; ".join(documents_used),
                "dois": "; ".join(d for d in dois if isinstance(d, str)) or None,
            }
        )

    result = pd.DataFrame(out_rows).sort_values("inchikey").reset_index(drop=True)
    log.info("curated molecules: %d", len(result))
    return result, exclusions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect VACV selectivity data from ChEMBL")
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument("--offline", action="store_true", help="use only the cached API pages")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="skip the network entirely and rebuild the dataset from the raw extracts "
        "committed in data/raw (activities.csv.gz, candidate_assays.csv, vacv_assays.csv)",
    )
    parser.add_argument("--limit-documents", type=int, default=None, help="debug: cap publications")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = Path(args.out)
    raw = out / "raw"
    processed = out / "processed"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    if args.from_raw:
        vacv_assays = pd.read_csv(raw / "vacv_assays.csv")
        candidate_assays = pd.read_csv(raw / "candidate_assays.csv")
        activities = pd.read_csv(raw / "activities.csv.gz")
        documents = pd.read_csv(processed / "sources.csv")
        log.info(
            "rebuilding from committed extracts: %d activities, %d assays, %d publications",
            len(activities),
            len(candidate_assays),
            len(documents),
        )
        assay_cells = dict(
            zip(candidate_assays["assay_chembl_id"], candidate_assays["assay_cell_type"])
        )
        client = None
        return _build(
            processed, vacv_assays, candidate_assays, activities, documents, assay_cells, client, len(documents)
        )

    client = ChemblClient(raw / "chembl_cache", offline=args.offline)

    vacv_assays = find_vacv_assays(client)
    vacv_assays.to_csv(raw / "vacv_assays.csv", index=False)

    document_ids = sorted({d for d in vacv_assays["document_chembl_id"].dropna()})
    if args.limit_documents:
        document_ids = document_ids[: args.limit_documents]

    candidate_assays = fetch_candidate_assays(client, document_ids)
    candidate_assays.to_csv(raw / "candidate_assays.csv", index=False)

    activities = fetch_assay_activities(client, sorted(candidate_assays["assay_chembl_id"]))
    activities.to_csv(raw / "activities.csv.gz", index=False, compression="gzip")

    documents = fetch_documents(client, document_ids)
    documents.to_csv(processed / "sources.csv", index=False)

    # Cell lines come straight from the assay records we already have.
    assay_cells = dict(
        zip(candidate_assays["assay_chembl_id"], candidate_assays["assay_cell_type"])
    )

    return _build(
        processed, vacv_assays, candidate_assays, activities, documents, assay_cells, client, len(document_ids)
    )


def _build(
    processed: Path,
    vacv_assays: pd.DataFrame,
    candidate_assays: pd.DataFrame,
    activities: pd.DataFrame,
    documents: pd.DataFrame,
    assay_cells: dict,
    client: "ChemblClient | None",
    publications: int,
) -> int:
    """Everything after the download: classify, pair, aggregate, write."""
    measurements, measurement_exclusions = classify_measurements(activities, vacv_assays)
    measurements.to_csv(processed / "measurements.csv", index=False)

    pairs, pairing_exclusions = pair_into_si(measurements, assay_cells)
    pairs.to_csv(processed / "pairs.csv", index=False)

    dataset, standardisation_exclusions = aggregate_per_molecule(pairs, documents)
    dataset.insert(0, "dataset_version", DATASET_VERSION)
    dataset.to_csv(processed / "dataset.csv", index=False)

    exclusions = pd.DataFrame(
        measurement_exclusions + pairing_exclusions + standardisation_exclusions
    )
    exclusions.to_csv(processed / "exclusions.csv", index=False)

    summary = {
        "dataset_version": DATASET_VERSION,
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "ChEMBL REST API",
        "vacv_assays": int(len(vacv_assays)),
        "candidate_assays": int(len(candidate_assays)),
        "publications_scanned": publications,
        "activities_scanned": int(len(activities)),
        "normalised_measurements": int(len(measurements)),
        "si_pairs": int(len(pairs)),
        "molecules": int(len(dataset)),
        "excluded_records": int(len(exclusions)),
        "exclusion_reasons": (
            exclusions["reason"].str.replace(r"\d+\.?\d*", "N", regex=True).value_counts().head(20).to_dict()
            if not exclusions.empty
            else {}
        ),
        "cache_hits": client.hits if client else None,
        "cache_misses": client.misses if client else None,
    }
    (processed / "collection_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("%s", json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
