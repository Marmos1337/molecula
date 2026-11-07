"""Checks on the rules that decide what goes into the training labels.

These are the rules a reader has to trust: that a ratio is never built from two
incomparable measurements, that units are converted correctly, and that a bad
row is reported rather than quietly turned into a number.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from molecula import collect  # noqa: E402

ETHANOL = "CCO"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def measurement(**overrides) -> dict:
    """One activity row as the ChEMBL API hands it over."""
    row = {
        "molecule_chembl_id": "CHEMBL1",
        "canonical_smiles": CAFFEINE,
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "10",
        "standard_units": "uM",
        "assay_chembl_id": "A-VIRUS",
        "assay_description": "Antiviral activity against Vaccinia virus in HEL cells",
        "assay_type": "F",
        "document_chembl_id": "DOC1",
        "target_organism": None,
        "data_validity_comment": None,
        "potential_duplicate": 0,
    }
    row.update(overrides)
    return row


VACV_ASSAYS = pd.DataFrame(
    [
        {
            "assay_chembl_id": "A-VIRUS",
            "document_chembl_id": "DOC1",
            "assay_cell_type": "HEL",
            "assay_organism": "Vaccinia virus",
            "assay_description": "Antiviral activity against Vaccinia virus in HEL cells",
            "assay_type": "F",
        }
    ]
)


class TestUnitConversion(unittest.TestCase):
    def test_molar_units(self):
        for units, value, expected in (("nM", 1000, 1.0), ("uM", 2.5, 2.5), ("mM", 1, 1000.0)):
            with self.subTest(units=units):
                converted, _ = collect._to_micromolar(value, units, CAFFEINE)
                self.assertAlmostEqual(converted, expected, places=6)

    def test_mass_units_use_the_molecular_weight(self):
        # caffeine is 194.19 g/mol, so 19.419 ug/mL is 100 uM
        converted, note = collect._to_micromolar(19.419, "ug.mL-1", CAFFEINE)
        self.assertAlmostEqual(converted, 100.0, places=2)
        self.assertIn("converted from", note)

    def test_both_spellings_of_ug_per_ml_agree(self):
        a, _ = collect._to_micromolar(10, "ug.mL-1", CAFFEINE)
        b, _ = collect._to_micromolar(10, "ug ml-1", CAFFEINE)
        self.assertEqual(a, b)

    def test_unknown_units_are_refused_with_a_reason(self):
        converted, note = collect._to_micromolar(10, "wat", CAFFEINE)
        self.assertIsNone(converted)
        self.assertIn("unsupported units", note)


class TestMeasurementClassification(unittest.TestCase):
    def classify(self, rows: list[dict]):
        return collect.classify_measurements(pd.DataFrame(rows), VACV_ASSAYS)

    def test_censored_measurements_are_excluded_with_the_reason(self):
        kept, excluded = self.classify([measurement(standard_relation=">")])
        self.assertTrue(kept.empty)
        self.assertEqual(len(excluded), 1)
        self.assertIn("censored", excluded[0]["reason"])

    def test_empty_relation_is_accepted_but_marked(self):
        kept, _ = self.classify([measurement(standard_relation=None)])
        self.assertEqual(len(kept), 1)
        self.assertTrue(bool(kept.iloc[0]["relation_assumed_exact"]))

    def test_nan_validity_comment_is_not_a_flag(self):
        kept, _ = self.classify([measurement(data_validity_comment=float("nan"))])
        self.assertEqual(len(kept), 1)

    def test_real_validity_flag_excludes_the_row(self):
        kept, excluded = self.classify([measurement(data_validity_comment="Outside typical range")])
        self.assertTrue(kept.empty)
        self.assertIn("data validity", excluded[0]["reason"])

    def test_cowpox_is_not_vaccinia_potency(self):
        kept, _ = self.classify(
            [
                measurement(
                    assay_chembl_id="A-COWPOX",
                    assay_description="Antiviral activity against Cowpox virus (Brighton Red) in HEL cells",
                )
            ]
        )
        self.assertTrue(kept.empty, "a cowpox assay must not be counted as vaccinia potency")

    def test_antiviral_wording_wins_over_the_word_cytotoxicity(self):
        kept, _ = self.classify(
            [
                measurement(
                    standard_type="EC50",
                    assay_description=(
                        "Antiviral activity against Vaccinia virus Copenhagen assessed as "
                        "inhibition of virus-induced cytotoxicity in HEL cells"
                    ),
                )
            ]
        )
        self.assertEqual(list(kept["measurement_kind"]), ["potency"])

    def test_endpoint_families_are_recorded(self):
        kept, _ = self.classify(
            [
                measurement(standard_type="MIC"),
                measurement(
                    standard_type="MCC",
                    assay_chembl_id="A-CYTO",
                    assay_description="Cytotoxicity against uninfected HEL cells",
                ),
            ]
        )
        self.assertEqual(set(kept["endpoint_family"]), {"minimum-effect"})


class TestPairing(unittest.TestCase):
    def make(self, rows: list[dict]) -> pd.DataFrame:
        base = {
            "molecule_chembl_id": "CHEMBL1",
            "smiles": CAFFEINE,
            "document_chembl_id": "DOC1",
            "assay_description": "",
            "standard_type": "IC50",
            "raw_value": 1.0,
            "raw_units": "uM",
            "unit_note": "",
            "relation_assumed_exact": False,
        }
        return pd.DataFrame([{**base, **row} for row in rows])

    def test_a_simple_pair_becomes_si(self):
        measurements = self.make(
            [
                {"assay_chembl_id": "P", "measurement_kind": "potency", "value_um": 2.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal"},
                {"assay_chembl_id": "C", "measurement_kind": "cytotoxicity", "value_um": 100.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal", "standard_type": "CC50"},
            ]
        )
        pairs, _ = collect.pair_into_si(measurements, {"P": "HEL", "C": "HEL"})
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs.iloc[0]["si"], 50.0)

    def test_different_cell_lines_are_never_combined(self):
        measurements = self.make(
            [
                {"assay_chembl_id": "P", "measurement_kind": "potency", "value_um": 2.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal"},
                {"assay_chembl_id": "C", "measurement_kind": "cytotoxicity", "value_um": 100.0,
                 "cell_line": "Vero", "endpoint_family": "half-maximal", "standard_type": "CC50"},
            ]
        )
        pairs, excluded = collect.pair_into_si(measurements, {"P": "HEL", "C": "Vero"})
        self.assertTrue(pairs.empty)
        self.assertTrue(any("no comparable" in e["reason"] for e in excluded))

    def test_two_unknown_cell_lines_do_not_count_as_a_match(self):
        measurements = self.make(
            [
                {"assay_chembl_id": "P", "measurement_kind": "potency", "value_um": 2.0,
                 "cell_line": np.nan, "endpoint_family": "half-maximal"},
                {"assay_chembl_id": "C", "measurement_kind": "cytotoxicity", "value_um": 100.0,
                 "cell_line": np.nan, "endpoint_family": "half-maximal", "standard_type": "CC50"},
            ]
        )
        pairs, excluded = collect.pair_into_si(measurements, {"P": None, "C": None})
        self.assertTrue(pairs.empty)
        self.assertTrue(any("cell line not stated" in e["reason"] for e in excluded))

    def test_endpoint_families_are_never_mixed(self):
        measurements = self.make(
            [
                {"assay_chembl_id": "P", "measurement_kind": "potency", "value_um": 2.0,
                 "cell_line": "HEL", "endpoint_family": "minimum-effect", "standard_type": "MIC"},
                {"assay_chembl_id": "C", "measurement_kind": "cytotoxicity", "value_um": 100.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal", "standard_type": "CC50"},
            ]
        )
        pairs, _ = collect.pair_into_si(measurements, {"P": "HEL", "C": "HEL"})
        self.assertTrue(pairs.empty, "a MIC must not be divided into a CC50")

    def test_repeated_assays_are_averaged_into_one_pair(self):
        measurements = self.make(
            [
                {"assay_chembl_id": "P1", "measurement_kind": "potency", "value_um": 1.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal"},
                {"assay_chembl_id": "P2", "measurement_kind": "potency", "value_um": 4.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal"},
                {"assay_chembl_id": "C1", "measurement_kind": "cytotoxicity", "value_um": 100.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal", "standard_type": "CC50"},
                {"assay_chembl_id": "C2", "measurement_kind": "cytotoxicity", "value_um": 100.0,
                 "cell_line": "HEL", "endpoint_family": "half-maximal", "standard_type": "CC50"},
            ]
        )
        pairs, _ = collect.pair_into_si(measurements, {k: "HEL" for k in ("P1", "P2", "C1", "C2")})
        self.assertEqual(len(pairs), 1, "a cross product would give four pairs here")
        self.assertAlmostEqual(pairs.iloc[0]["ic50_um"], 2.0)  # geometric median of 1 and 4
        self.assertEqual(pairs.iloc[0]["n_potency_assays"], 2)


class TestCollectedDatasetItself(unittest.TestCase):
    """The committed dataset has to obey the rules the README states."""

    @classmethod
    def setUpClass(cls):
        path = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
        pairs = PROJECT_ROOT / "data" / "processed" / "pairs.csv"
        if not path.exists() or not pairs.exists():
            raise unittest.SkipTest("dataset not collected yet")
        cls.dataset = pd.read_csv(path)
        cls.pairs = pd.read_csv(pairs)

    def test_every_row_has_a_source(self):
        self.assertTrue(self.dataset["documents"].notna().all())

    def test_every_pair_names_one_cell_line(self):
        self.assertTrue(self.pairs["cell_line"].notna().all())

    def test_si_matches_the_two_concentrations(self):
        ratio = self.pairs["cc50_um"] / self.pairs["ic50_um"]
        self.assertTrue(np.allclose(ratio, self.pairs["si"]))

    def test_log10_si_matches_si(self):
        self.assertTrue(np.allclose(np.log10(self.dataset["si"]), self.dataset["log10_si"]))

    def test_one_row_per_structure(self):
        self.assertEqual(self.dataset["inchikey"].nunique(), len(self.dataset))

    def test_minimum_effect_pairs_use_minimum_effect_endpoints_on_both_sides(self):
        minimum = self.pairs[self.pairs["endpoint_family"] == "minimum-effect"]
        for row in minimum.to_dict("records"):
            for value in str(row["ic50_type"]).split("; ") + str(row["cc50_type"]).split("; "):
                self.assertIn(
                    collect.POTENCY_TYPES.get(value) or collect.CYTOTOX_TYPES.get(value),
                    {"minimum-effect"},
                    f"{value} is not a minimum-effect endpoint",
                )


if __name__ == "__main__":
    unittest.main()
