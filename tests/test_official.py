"""Checks on the competition data and the rules we agreed to follow with it.

The important one is :class:`TestForbiddenColumns`. The test file hands over
CC50 and IC50, and the selectivity index is their ratio, so a model that reads
them scores perfectly and breaks the task statement. These tests prove the
columns never reach a prediction: removing them from the input file does not
change a single number we produce.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from molecula import official  # noqa: E402
from molecula.experiments import _expm1_smeared, grouped_folds, tanimoto  # noqa: E402

TRAIN = PROJECT_ROOT / "data" / "kaggle" / "train_v2.csv"
TEST = PROJECT_ROOT / "data" / "kaggle" / "test.csv"
SAMPLE = PROJECT_ROOT / "data" / "kaggle" / "sample_submit.csv"
MODEL = PROJECT_ROOT / "models" / "molecula_si.joblib"


def requires(*paths: Path):
    missing = [p for p in paths if not p.exists()]
    return unittest.skipIf(missing, f"missing {[str(p) for p in missing]}")


@requires(TRAIN)
class TestOfficialTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = official.load_train(TRAIN)

    def test_every_row_is_readable_and_kept(self):
        self.assertEqual(self.data.audit["rows_kept"], self.data.audit["rows_in_file"])
        self.assertEqual(self.data.excluded, [])

    def test_zero_targets_are_kept_and_flagged(self):
        zeros = self.data.frame["si_is_zero"]
        self.assertGreater(zeros.sum(), 0, "the organisers' SI = 0 convention must survive loading")
        self.assertTrue((self.data.frame.loc[zeros, "si"] == 0).all())

    def test_no_negative_targets_survive(self):
        self.assertTrue((self.data.frame["si"] >= 0).all())

    def test_audit_records_what_the_metric_rests_on(self):
        target = self.data.audit["target"]
        self.assertGreater(target["mse_of_constant_zero"], 0)
        self.assertGreater(target["max"], target["median"] * 100)


@requires(TRAIN, TEST)
class TestForbiddenColumns(unittest.TestCase):
    def test_assert_clean_rejects_them(self):
        frame = pd.DataFrame({"SMILES": ["CCO"], "CC50_uM": [1.0]})
        with self.assertRaises(RuntimeError):
            official.assert_clean(frame)

    def test_assert_clean_accepts_a_clean_frame(self):
        official.assert_clean(pd.DataFrame({"SMILES": ["CCO"], "culture": ["vero"]}))

    def test_loaded_frames_never_carry_them(self):
        for data in (official.load_train(TRAIN), official.load_test(TEST)):
            official.assert_clean(data.frame)

    @requires(MODEL, SAMPLE)
    def test_dropping_them_from_the_file_changes_nothing(self):
        """The strongest statement available: the numbers do not depend on them."""
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            stripped = pd.read_csv(TEST).drop(columns=list(official.FORBIDDEN_COLUMNS))
            stripped_path = directory / "test_without_answers.csv"
            stripped.to_csv(stripped_path, index=False)

            with_columns = _run_submission(TEST, directory / "a.csv")
            without_columns = _run_submission(stripped_path, directory / "b.csv")
            self.assertEqual(with_columns.returncode, 0, with_columns.stderr)
            self.assertEqual(without_columns.returncode, 0, without_columns.stderr)

            a = pd.read_csv(directory / "a.csv")
            b = pd.read_csv(directory / "b.csv")
            pd.testing.assert_frame_equal(a, b)


def _run_submission(test_path: Path, out: Path):
    return subprocess.run(
        [
            sys.executable, "-m", "molecula.predict",
            "--model", str(MODEL),
            "submission",
            "--test", str(test_path),
            "--sample-submission", str(SAMPLE),
            "--target-scale", "si",
            "--out", str(out),
        ],
        cwd=PROJECT_ROOT,
        env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )


@requires(MODEL, TEST, SAMPLE)
class TestSubmissionAgainstTheRealSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        out = Path(cls.directory.name) / "submission.csv"
        completed = _run_submission(TEST, out)
        if completed.returncode != 0:
            raise AssertionError(completed.stderr)
        cls.submission = pd.read_csv(out)
        cls.sample = pd.read_csv(SAMPLE)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_columns_and_order_come_from_the_sample(self):
        self.assertEqual(list(self.submission.columns), list(self.sample.columns))
        self.assertEqual(
            self.submission[self.sample.columns[0]].tolist(),
            self.sample[self.sample.columns[0]].tolist(),
        )

    def test_one_row_per_test_molecule(self):
        self.assertEqual(len(self.submission), len(self.sample))

    def test_values_are_finite_and_in_the_si_scale(self):
        values = self.submission[self.sample.columns[1]].to_numpy(dtype=float)
        self.assertTrue(np.isfinite(values).all(), "no NaN or inf may reach the leaderboard")
        self.assertTrue((values >= 0).all(), "a selectivity index cannot be negative")


class TestFolds(unittest.TestCase):
    def test_folds_cover_every_row_exactly_once(self):
        groups = np.array([f"g{i // 3}" for i in range(60)])
        folds = grouped_folds(groups, n_folds=5, seed=42)
        covered = np.sort(np.concatenate(folds))
        np.testing.assert_array_equal(covered, np.arange(60))

    def test_a_group_never_straddles_a_fold(self):
        groups = np.array([f"g{i // 7}" for i in range(70)])
        folds = grouped_folds(groups, n_folds=5, seed=7)
        for fold in folds:
            others = np.setdiff1d(np.arange(70), fold)
            self.assertFalse(
                set(groups[fold]) & set(groups[others]),
                "the same group appeared on both sides of a fold",
            )

    def test_folds_are_reproducible(self):
        groups = np.array([f"g{i // 2}" for i in range(40)])
        first = grouped_folds(groups, 5, 42)
        second = grouped_folds(groups, 5, 42)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


@requires(TRAIN)
class TestGroupingKeys(unittest.TestCase):
    """A split scheme has to group by something coarser than the molecule."""

    @classmethod
    def setUpClass(cls):
        cls.frame = official.load_train(TRAIN).frame

    def test_murcko_is_coarser_than_connectivity(self):
        frameworks = self.frame["murcko"].nunique()
        structures = self.frame["scaffold_key"].nunique()
        self.assertLess(
            frameworks, structures * 0.8,
            "the scaffold scheme must hold out whole families, not single molecules",
        )

    def test_the_scaffold_scheme_actually_groups_by_framework(self):
        """Reverting grouping() to the connectivity key must fail here."""
        from molecula.experiments import build_features, grouping

        frame = self.frame.head(60).copy()
        groups = grouping(frame, "scaffold", build_features(frame, "smiles"))
        self.assertEqual(len(np.unique(groups)), frame["murcko"].nunique())
        self.assertLess(len(np.unique(groups)), frame["scaffold_key"].nunique())

    def test_ring_stereochemistry_does_not_split_a_framework(self):
        from molecula.chem import murcko_key, standardize

        self.assertEqual(
            murcko_key(standardize("C1C[C@H]2CC[C@@H]1C2").mol),
            murcko_key(standardize("C1CC2CCC1C2").mol),
        )

    def test_analogues_share_a_framework(self):
        from molecula.chem import murcko_key, standardize

        a = murcko_key(standardize("c1ccc(cc1)C(=O)OC").mol)
        b = murcko_key(standardize("c1ccc(cc1)C(=O)OCC").mol)
        self.assertEqual(a, b, "two esters of the same acid are one framework")

    def test_a_molecule_without_rings_is_marked_acyclic(self):
        from molecula.chem import murcko_key, standardize

        self.assertEqual(murcko_key(standardize("CCCCO").mol), "acyclic")


@requires(TRAIN)
class TestCultureMapping(unittest.TestCase):
    def test_hela_is_not_folded_into_hel(self):
        self.assertEqual(official._normalise_culture("HeLa cells"), "hela")
        self.assertEqual(official._normalise_culture("HELA cells"), "hela")
        self.assertEqual(official._normalise_culture("cells HEL"), "hel")

    def test_both_cultures_survive_in_the_real_file(self):
        cultures = set(official.load_train(TRAIN).frame["culture"])
        self.assertIn("hela", cultures)
        self.assertIn("hel", cultures)


class TestExternalDataFilter(unittest.TestCase):
    """External rows must respect whatever the scheme is holding out."""

    def setUp(self):
        from molecula.experiments import Dataset

        held_out = np.zeros((1, 2066)); held_out[0, :60] = 1
        near = held_out.copy(); near[0, 55:65] = 1       # very similar
        far = np.zeros((1, 2066)); far[0, 1000:1060] = 1  # nothing in common
        self.validation = held_out
        self.data = Dataset(
            frame=pd.DataFrame({"inchikey": ["A"], "scaffold_key": ["AAA"],
                                "murcko": ["c1ccccc1"], "publication": ["p"]}),
            y=np.array([1.0]),
            external=pd.DataFrame({"scaffold_key": ["BBB", "CCC"]}),
            external_y=np.array([2.0, 3.0]),
        )
        self.data.features = {"smiles": held_out}
        self.data.external_features = {"smiles": np.vstack([near, far])}

    def test_transfer_schemes_drop_a_similar_external_row(self):
        from molecula.experiments import external_rows_to_keep

        keep = external_rows_to_keep(self.data, "scaffold", set(), self.validation)
        self.assertFalse(bool(keep[0]), "an analogue of a held-out molecule is a leak")
        self.assertTrue(bool(keep[1]))

    def test_the_leaderboard_scheme_keeps_it_on_purpose(self):
        from molecula.experiments import external_rows_to_keep

        keep = external_rows_to_keep(self.data, "molecule", set(), self.validation)
        self.assertTrue(keep.all())

    def test_the_same_structure_is_always_dropped(self):
        from molecula.experiments import external_rows_to_keep

        keep = external_rows_to_keep(self.data, "molecule", {"BBB"}, self.validation)
        self.assertFalse(bool(keep[0]))


class TestBackTransform(unittest.TestCase):
    def test_smearing_gives_a_mean_not_a_median(self):
        """exp of the mean understates the mean of exp; smearing corrects it."""
        prediction = np.array([1.0, 2.0])
        residuals = np.array([-0.5, 0.0, 0.5])
        smeared = _expm1_smeared(prediction, residuals)
        plain = np.expm1(prediction)
        self.assertTrue((smeared >= plain).all())

    def test_without_residuals_it_is_the_plain_inverse(self):
        prediction = np.array([0.3, 1.1])
        np.testing.assert_allclose(_expm1_smeared(prediction, np.zeros(1)), np.expm1(prediction))


class TestTanimoto(unittest.TestCase):
    def test_identical_rows_score_one(self):
        X = np.zeros((2, 2100))
        X[:, 3] = 1
        X[:, 10] = 1
        similarity = tanimoto((X[:, :2048] > 0).astype(float), (X[:, :2048] > 0).astype(float))
        self.assertAlmostEqual(similarity[0, 1], 1.0)

    def test_disjoint_rows_score_zero(self):
        A = np.zeros((1, 2048)); A[0, 1] = 1
        B = np.zeros((1, 2048)); B[0, 2] = 1
        self.assertAlmostEqual(tanimoto(A, B)[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
