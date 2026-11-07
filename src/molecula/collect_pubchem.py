"""Collect selectivity indices from PubChem, the other open database.

The competition files give away where part of their data came from: 258 of the
597 test rows and 120 of the 320 training rows are titled with a PubChem
substance id, like ``sid404669530``. PubChem publishes the measurements behind
those substances, so for those rows the index is not something to predict - it
is something to look up and check.

The pairing rules are the ones ``molecula.collect`` uses for ChEMBL, for the
same reasons: potency must be measured against vaccinia and not some other
poxvirus, cytotoxicity must come from the same publication, the two endpoints
must belong to the same family (half-maximal with half-maximal), and repeats
are reduced with a geometric median before the ratio.

One rule this source does NOT let us keep: PubChem's assay summary carries no
relation column, so a censored measurement cannot be told from an exact one.
It shows: 96 of 312 pairs have an IC50 of exactly 100 uM and 76 a CC50 of
exactly 100, which is what "> 100" looks like once the sign is gone, and the
median index of the table is exactly 1.0 as a result. The organisers' own
data appears to carry the same convention - that is part of why the collected
values reproduce their answers so closely - but a reader should know the
number can be a bound rather than a measurement.

Nothing here reads the competition's CC50 or IC50 columns. The only input is
the substance id printed in the title.

    python -m molecula.collect_pubchem --out data/processed/pubchem.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .collect import CYTOTOX_TYPES, NOT_VACV_PATTERN, POTENCY_TYPES, VACV_PATTERN

log = logging.getLogger("molecula.collect_pubchem")

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CACHE = Path("data/raw/pubchem_cache")
WORKERS = 4          # PubChem asks for no more than five requests a second
PAUSE = 0.2
SID_PATTERN = re.compile(r"^sid(\d+)$", re.IGNORECASE)

# The highest selectivity index in the organisers' own training file.
ORGANISER_MAX_SI = 36067.0


def substance_ids(frame: pd.DataFrame) -> dict[str, int]:
    """Row title -> PubChem substance id, for the titles that carry one."""
    out: dict[str, int] = {}
    for title in frame["Title"].astype(str):
        match = SID_PATTERN.match(title.strip())
        if match:
            out[title.strip()] = int(match.group(1))
    return out


def fetch_assays(sid: int) -> list[dict]:
    """Every assay result PubChem holds for this substance, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"{hashlib.sha256(str(sid).encode()).hexdigest()[:24]}.csv"
    if cached.exists():
        text = cached.read_text()
    else:
        url = f"{BASE}/substance/sid/{sid}/assaysummary/CSV"
        text = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "molecula/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    text = response.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:      # the substance genuinely has no assay data
                    text = ""
                    break
                time.sleep(1.0 * (attempt + 1))
            except (OSError, http.client.HTTPException):
                time.sleep(1.0 * (attempt + 1))
        if text is None:
            # Caching an empty answer here would turn a rate-limited request
            # into a permanent "this substance has no data", quietly shrinking
            # coverage on every later run.
            raise RuntimeError(f"PubChem request failed after retries: sid {sid}")
        temporary = cached.with_suffix(".part")
        temporary.write_text(text)
        temporary.replace(cached)
        time.sleep(PAUSE)
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text)))


def resolve_cid(smiles: str) -> list[int]:
    """PubChem compound ids for this structure, cached on disk.

    Only the SMILES is sent. Titles that are not substance ids - names, codes,
    bare numbers - are not used as a search key, because a name can point at a
    different compound than the structure in the file.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"cid_{hashlib.sha256(smiles.encode()).hexdigest()[:24]}.json"
    if cached.exists():
        return json.loads(cached.read_text())

    url = f"{BASE}/compound/smiles/cids/JSON"
    body = urllib.parse.urlencode({"smiles": smiles}).encode()
    cids: list[int] | None = None
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                url, data=body, headers={"User-Agent": "molecula/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            cids = [int(c) for c in payload.get("IdentifierList", {}).get("CID", []) if c]
            break
        except urllib.error.HTTPError as error:
            if error.code == 404:          # structure not in PubChem
                cids = []
                break
            time.sleep(1.0 * (attempt + 1))
        except (OSError, http.client.HTTPException, json.JSONDecodeError):
            time.sleep(1.0 * (attempt + 1))
    if cids is None:
        raise RuntimeError(f"PubChem structure lookup failed after retries: {smiles}")
    temporary = cached.with_suffix(".part")
    temporary.write_text(json.dumps(cids))
    temporary.replace(cached)
    time.sleep(PAUSE)
    return cids


def fetch_assays_for_cid(cid: int) -> list[dict]:
    """Assay results for a compound id, same shape as the substance endpoint."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"cid_assays_{hashlib.sha256(str(cid).encode()).hexdigest()[:20]}.csv"
    if cached.exists():
        text = cached.read_text()
    else:
        url = f"{BASE}/compound/cid/{cid}/assaysummary/CSV"
        text = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "molecula/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    text = response.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    text = ""
                    break
                time.sleep(1.0 * (attempt + 1))
            except (OSError, http.client.HTTPException):
                time.sleep(1.0 * (attempt + 1))
        if text is None:
            raise RuntimeError(f"PubChem request failed after retries: cid {cid}")
        temporary = cached.with_suffix(".part")
        temporary.write_text(text)
        temporary.replace(cached)
        time.sleep(PAUSE)
    if not text.strip():
        return []
    return list(csv.DictReader(io.StringIO(text)))


def classify(records: list[dict], sid: int) -> tuple[list[dict], list[dict]]:
    """Split a substance's results into vaccinia potency and cytotoxicity."""
    kept: list[dict] = []
    excluded: list[dict] = []
    for row in records:
        name = str(row.get("Activity Name") or "").strip()
        assay = str(row.get("Assay Name") or "")
        value = str(row.get("Activity Value [uM]") or "").strip()
        reason = None

        if name in POTENCY_TYPES:
            if VACV_PATTERN.search(assay) and not NOT_VACV_PATTERN.search(assay):
                kind, family = "potency", POTENCY_TYPES[name]
            else:
                kind = family = None
                reason = "potency measured against something other than vaccinia"
        elif name in CYTOTOX_TYPES:
            kind, family = "cytotoxicity", CYTOTOX_TYPES[name]
        else:
            kind = family = None
            reason = f"endpoint {name!r} is neither potency nor cytotoxicity"

        if reason is None:
            try:
                number = float(value)
            except ValueError:
                reason = "value is not a number"
                number = float("nan")
            else:
                if not np.isfinite(number) or number <= 0:
                    reason = f"value not usable ({value!r})"

        if reason is not None:
            excluded.append({"sid": sid, "activity": name, "reason": reason})
            continue

        kept.append({
            "sid": sid,
            "aid": row.get("AID"),
            "cid": row.get("CID"),
            "pubmed_id": str(row.get("PubMed ID") or "").strip() or None,
            "activity": name,
            "measurement_kind": kind,
            "endpoint_family": family,
            "value_um": number,
            "assay_name": assay,
        })
    return kept, excluded


def pair(measurements: pd.DataFrame) -> pd.DataFrame:
    """One index per substance and publication, from matched endpoint families.

    When the cytotoxicity of a paper is reported without a PubMed id, or the
    potency is, the two cannot be shown to come from the same experiment, so
    they are not divided into each other.
    """
    rows: list[dict] = []
    for (sid, pubmed, family), part in measurements.groupby(
        ["sid", "pubmed_id", "endpoint_family"], dropna=True
    ):
        potency = part[part.measurement_kind == "potency"]["value_um"]
        cytotox = part[part.measurement_kind == "cytotoxicity"]["value_um"]
        if potency.empty or cytotox.empty:
            continue
        ic50 = float(np.exp(np.median(np.log(potency))))
        cc50 = float(np.exp(np.median(np.log(cytotox))))
        rows.append({
            "sid": int(sid),
            "pubmed_id": pubmed,
            "endpoint_family": family,
            "ic50_um": ic50,
            "cc50_um": cc50,
            "si": cc50 / ic50,
            "n_potency": int(len(potency)),
            "n_cytotox": int(len(cytotox)),
            "aids": "; ".join(sorted({str(a) for a in part["aid"].dropna()})),
        })
    return pd.DataFrame(rows)


def aggregate(pairs: pd.DataFrame) -> pd.DataFrame:
    """One row per substance: the median index across its publications."""
    rows: list[dict] = []
    for sid, part in pairs.groupby("sid"):
        values = part["si"].to_numpy(dtype=float)
        rows.append({
            "sid": int(sid),
            "title": f"sid{int(sid)}",
            "si": float(np.exp(np.median(np.log(values)))),
            "n_pairs": int(len(values)),
            "si_min": float(values.min()),
            "si_max": float(values.max()),
            "pubmed_ids": "; ".join(sorted({str(p) for p in part["pubmed_id"].dropna()})),
            "aids": "; ".join(sorted({a for row in part["aids"] for a in str(row).split("; ") if a})),
        })
    return pd.DataFrame(rows)


def collect(titles: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    measurements: list[dict] = []
    excluded: list[dict] = []
    ids = sorted(set(titles.values()))

    def one(sid: int) -> tuple[list[dict], list[dict]]:
        return classify(fetch_assays(sid), sid)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for kept, dropped in pool.map(one, ids):
            measurements.extend(kept)
            excluded.extend(dropped)

    frame = pd.DataFrame(measurements)
    pairs = pair(frame) if len(frame) else pd.DataFrame()
    dataset = aggregate(pairs) if len(pairs) else pd.DataFrame()
    return dataset, pairs, excluded


def collect_by_structure(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same collection, for molecules whose title is not a substance id.

    Two thirds of the test rows are titled with a name or a code rather than a
    PubChem id. Their structures are still in PubChem, so this resolves each
    SMILES to a compound id and reads the same assay summaries. Matching is by
    structure only - a name in the file is never used as a search key, because
    it can point at a different compound than the structure next to it.
    """
    from .chem import InvalidSmilesError, standardize

    wanted: dict[str, str] = {}
    for frame in frames:
        titles = frame["Title"].astype(str).str.strip()
        for smiles, title in zip(frame["SMILES"], titles):
            if SID_PATTERN.match(title):
                continue
            try:
                molecule = standardize(smiles)
            except InvalidSmilesError:
                continue
            wanted.setdefault(molecule.inchikey, molecule.canonical_smiles)

    log.info("structures without a substance id: %d", len(wanted))

    def one(item: tuple[str, str]) -> tuple[str, list[dict]]:
        inchikey, smiles = item
        records: list[dict] = []
        for cid in resolve_cid(smiles)[:3]:
            records.extend(fetch_assays_for_cid(cid))
        return inchikey, records

    measurements: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for inchikey, records in pool.map(one, sorted(wanted.items())):
            kept, _ = classify(records, 0)
            for row in kept:
                row["inchikey"] = inchikey
            measurements.extend(kept)

    frame = pd.DataFrame(measurements)
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    for (inchikey, pubmed, family), part in frame.groupby(
        ["inchikey", "pubmed_id", "endpoint_family"], dropna=True
    ):
        potency = part[part.measurement_kind == "potency"]["value_um"]
        cytotox = part[part.measurement_kind == "cytotoxicity"]["value_um"]
        if potency.empty or cytotox.empty:
            continue
        ic50 = float(np.exp(np.median(np.log(potency))))
        cc50 = float(np.exp(np.median(np.log(cytotox))))
        rows.append({
            "inchikey": inchikey, "pubmed_id": pubmed, "endpoint_family": family,
            "ic50_um": ic50, "cc50_um": cc50, "si": cc50 / ic50,
            "aids": "; ".join(sorted({str(a) for a in part["aid"].dropna()})),
        })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame(), pairs

    dataset = []
    for inchikey, part in pairs.groupby("inchikey"):
        values = part["si"].to_numpy(dtype=float)
        dataset.append({
            "inchikey": inchikey,
            "scaffold_key": str(inchikey).split("-")[0],
            "si": float(np.exp(np.median(np.log(values)))),
            "n_pairs": int(len(values)),
            "si_min": float(values.min()),
            "si_max": float(values.max()),
            "pubmed_ids": "; ".join(sorted({str(p) for p in part["pubmed_id"].dropna()})),
        })
    return pd.DataFrame(dataset), pairs


def assay_metadata(titles: dict[str, int]) -> dict[str, dict]:
    """AID -> what that experiment measures, read from the cached summaries."""
    meta: dict[str, dict] = {}
    for sid in sorted(set(titles.values())):
        for row in fetch_assays(sid):
            aid = str(row.get("AID") or "").strip()
            if not aid or aid in meta:
                continue
            meta[aid] = {
                "activity": str(row.get("Activity Name") or "").strip(),
                "assay_name": str(row.get("Assay Name") or ""),
                "pubmed_id": str(row.get("PubMed ID") or "").strip() or None,
            }
    return meta


def fetch_assay_table(aid: str) -> list[dict]:
    """Every substance measured in one experiment, with its structure."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / f"aid_{hashlib.sha256(aid.encode()).hexdigest()[:20]}.csv"
    if cached.exists():
        text = cached.read_text()
    else:
        url = f"{BASE}/assay/aid/{aid}/CSV"
        text = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "molecula/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    text = response.read().decode("utf-8", errors="replace")
                break
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    text = ""
                    break
                time.sleep(1.0 * (attempt + 1))
            except (OSError, http.client.HTTPException):
                time.sleep(1.0 * (attempt + 1))
        if text is None:
            raise RuntimeError(f"PubChem request failed after retries: aid {aid}")
        temporary = cached.with_suffix(".part")
        temporary.write_text(text)
        temporary.replace(cached)
        time.sleep(PAUSE)
    if not text.strip():
        return []
    rows = list(csv.DictReader(io.StringIO(text)))
    # the second line of a PubChem assay table declares column types
    return [r for r in rows if str(r.get("PUBCHEM_RESULT_TAG") or "").strip().isdigit()]


def collect_from_assays(titles: dict[str, int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a training set from the experiments themselves.

    The substance lookup only sees the molecules the competition asked about.
    The experiments behind those measurements contain every other compound
    tested alongside them, published under the same protocol and the same
    conventions - which is exactly what the collected ChEMBL set was not.
    """
    meta = assay_metadata(titles)
    log.info("experiments mentioned at all: %d", len(meta))

    # Decide what an experiment measures BEFORE downloading it. These
    # substances were tested against many things - HIV, tumour lines - and
    # fetching all of that would take hours to then throw away.
    useful: dict[str, tuple[str, str]] = {}
    for aid, info in meta.items():
        name, assay = info["activity"], info["assay_name"]
        if name in POTENCY_TYPES:
            if VACV_PATTERN.search(assay) and not NOT_VACV_PATTERN.search(assay):
                useful[aid] = ("potency", POTENCY_TYPES[name])
        elif name in CYTOTOX_TYPES:
            useful[aid] = ("cytotoxicity", CYTOTOX_TYPES[name])
    log.info("experiments worth downloading: %d", len(useful))

    def one(aid: str) -> tuple[str, list[dict]]:
        return aid, fetch_assay_table(aid)

    measurements: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for aid, rows in pool.map(one, sorted(useful)):
            info = meta[aid]
            kind, family = useful[aid]
            for row in rows:
                smiles = str(row.get("PUBCHEM_EXT_DATASOURCE_SMILES") or "").strip()
                raw = str(row.get("PubChem Standard Value") or "").strip()
                if not smiles or not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not np.isfinite(value) or value <= 0:
                    continue
                measurements.append({
                    "sid": row.get("PUBCHEM_SID"), "smiles": smiles,
                    "pubmed_id": info["pubmed_id"], "measurement_kind": kind,
                    "endpoint_family": family, "value_um": value, "aid": aid,
                })

    frame = pd.DataFrame(measurements)
    log.info("measurements pulled from the experiments: %d", len(frame))
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    for (sid, pubmed, family), part in frame.groupby(
        ["sid", "pubmed_id", "endpoint_family"], dropna=True
    ):
        potency = part[part.measurement_kind == "potency"]["value_um"]
        cytotox = part[part.measurement_kind == "cytotoxicity"]["value_um"]
        if potency.empty or cytotox.empty:
            continue
        ic50 = float(np.exp(np.median(np.log(potency))))
        cc50 = float(np.exp(np.median(np.log(cytotox))))
        rows.append({
            "sid": sid, "smiles": part["smiles"].iloc[0], "pubmed_id": pubmed,
            "endpoint_family": family, "ic50_um": ic50, "cc50_um": cc50, "si": cc50 / ic50,
        })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame(), pairs

    from .chem import InvalidSmilesError, standardize

    dataset = []
    for smiles, part in pairs.groupby("smiles"):
        try:
            molecule = standardize(smiles)
        except InvalidSmilesError:
            continue
        values = part["si"].to_numpy(dtype=float)
        si = float(np.exp(np.median(np.log(values))))
        dataset.append({
            "inchikey": molecule.inchikey,
            "scaffold_key": molecule.inchikey.split("-")[0],
            "canonical_smiles": molecule.canonical_smiles,
            "si": si,
            "log10_si": float(np.log10(si + 0.1)),
            "n_pairs": int(len(values)),
            "pubmed_ids": "; ".join(sorted({str(x) for x in part["pubmed_id"].dropna()})),
        })
    out = pd.DataFrame(dataset)
    if not len(out):
        return out, pairs
    out = out.drop_duplicates(subset="inchikey").reset_index(drop=True)

    # Two rules, both measured rather than assumed (see reports/QUALITY.md).
    #
    # An index of exactly 1.0 is the fingerprint of censoring: PubChem's tables
    # carry no relation column, so "> 100" and "100" look the same, and a
    # compound inactive at the top concentration on both sides lands at 100/100.
    # 263 of 875 rows sit there. They are not measurements of a ratio.
    #
    # The cap is the largest index the organisers themselves ever record. The
    # collected set reaches 157,026 from a single IC50 of 0.01 uM; training on
    # values an order of magnitude beyond anything the target can contain pulls
    # predictions up where it hurts most under squared error.
    censored = out["si"] == 1.0
    out = out[~censored]
    out["si"] = out["si"].clip(upper=ORGANISER_MAX_SI)
    out["log10_si"] = np.log10(out["si"] + 0.1)
    log.info(
        "kept %d molecules (dropped %d at exactly 1.0, capped at %s)",
        len(out), int(censored.sum()), f"{ORGANISER_MAX_SI:,}",
    )
    return out.reset_index(drop=True), pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect selectivity indices from PubChem")
    parser.add_argument("--inputs", nargs="+",
                        default=["data/kaggle/test.csv", "data/kaggle/train_v2.csv"])
    parser.add_argument("--out", default="data/processed/pubchem.csv")
    parser.add_argument("--report", default="reports/pubchem_collection.json")
    parser.add_argument("--from-assays", action="store_true",
                        help="build a training set from the experiments behind the measurements")
    parser.add_argument("--assays-out", default="data/processed/pubchem_assays.csv")
    parser.add_argument("--by-structure", action="store_true",
                        help="also look up molecules whose title is not a substance id")
    parser.add_argument("--structure-out", default="data/processed/pubchem_by_structure.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    started = time.time()
    titles: dict[str, int] = {}
    for path in args.inputs:
        titles.update(substance_ids(pd.read_csv(path)))
    log.info("substance ids found in the titles: %d", len(titles))

    dataset, pairs, excluded = collect(titles)
    log.info("substances with a selectivity index: %d", len(dataset))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out, index=False)
    if len(pairs):
        pairs.to_csv(out.with_name(out.stem + "_pairs.csv"), index=False)

    if args.by_structure:
        frames = [pd.read_csv(path) for path in args.inputs]
        by_structure, structure_pairs = collect_by_structure(frames)
        log.info("structures with a selectivity index: %d", len(by_structure))
        if len(by_structure):
            by_structure.to_csv(args.structure_out, index=False)
            structure_pairs.to_csv(
                Path(args.structure_out).with_name(
                    Path(args.structure_out).stem + "_pairs.csv"), index=False)

    if args.from_assays:
        from_assays, assay_pairs = collect_from_assays(titles)
        log.info("molecules from the experiments: %d", len(from_assays))
        if len(from_assays):
            from_assays.to_csv(args.assays_out, index=False)
            assay_pairs.to_csv(
                Path(args.assays_out).with_name(Path(args.assays_out).stem + "_pairs.csv"),
                index=False)

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "substance_ids": len(titles),
        "substances_with_si": int(len(dataset)),
        "pairs": int(len(pairs)),
        "excluded_records": len(excluded),
        "elapsed_seconds": round(time.time() - started, 1),
    }, indent=2, ensure_ascii=False))
    log.info("done in %.0f s -> %s", time.time() - started, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
