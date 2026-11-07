"""Checks that the parts which must agree actually agree.

Run with the project's interpreter:

    PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

The tests that need a trained model skip themselves when there is none, so this
file is also useful before the first training run.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "molecula_si.joblib"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from molecula.chem import InvalidSmilesError, featurize, standardize  # noqa: E402

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CIDOFOVIR = "NC1=NC(=O)N(C[C@@H](CO)OCP(=O)(O)O)C=C1"


class TestSmilesHandling(unittest.TestCase):
    def test_salt_is_stripped_and_canonicalised(self):
        plain = standardize("CC(=O)Oc1ccccc1C(=O)O")
        salt = standardize("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        self.assertEqual(plain.canonical_smiles, salt.canonical_smiles)
        self.assertEqual(plain.inchikey, salt.inchikey)

    def test_same_molecule_written_differently_gives_one_key(self):
        a = standardize("c1ccccc1")
        b = standardize("C1=CC=CC=C1")
        self.assertEqual(a.inchikey, b.inchikey)

    def test_rubbish_input_raises_with_a_reason(self):
        for bad in ("", "   ", "not a molecule", "C(((", None):
            with self.subTest(value=bad):
                with self.assertRaises(InvalidSmilesError) as caught:
                    standardize(bad)
                self.assertTrue(str(caught.exception))

    def test_features_are_deterministic_and_fixed_width(self):
        from molecula.chem import N_FEATURES

        first = featurize(standardize(ASPIRIN).mol)
        second = featurize(standardize(ASPIRIN).mol)
        self.assertEqual(first.shape, (N_FEATURES,))
        self.assertTrue((first == second).all())


class TestSplitHasNoLeak(unittest.TestCase):
    def setUp(self):
        self.dataset = PROJECT_ROOT / "data" / "processed" / "dataset.csv"
        if not self.dataset.exists():
            self.skipTest("dataset not collected yet")

    def test_no_molecule_on_both_sides(self):
        from molecula.dataset import load_dataset, make_split

        split = make_split(load_dataset(self.dataset))
        info = split.describe()
        self.assertEqual(info["shared_inchikeys"], 0)
        self.assertEqual(info["shared_scaffolds"], 0)
        self.assertGreater(info["n_train"], 0)
        self.assertGreater(info["n_validation"], 0)


class TestModelPaths(unittest.TestCase):
    """The API, the CLI and training must all go through the same prediction path."""

    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest("no trained model yet")
        from molecula.model import SavedModel

        cls.model = SavedModel.load(MODEL_PATH)

    def test_prediction_is_reproducible(self):
        first = self.model.predict_smiles(CIDOFOVIR)
        second = self.model.predict_smiles(CIDOFOVIR)
        self.assertEqual(first.log10_si, second.log10_si)
        self.assertEqual(first.si, second.si)

    def test_the_two_scales_describe_the_same_number(self):
        """The model predicts a raw SI; the log is a shifted view of it.

        The shift keeps log10 defined for SI = 0, which a quarter of the
        training rows carry by the organisers' convention.
        """
        import numpy as np

        prediction = self.model.predict_smiles(CIDOFOVIR)
        self.assertGreaterEqual(prediction.si, 0.0)
        self.assertAlmostEqual(prediction.log10_si, float(np.log10(prediction.si + 0.1)), places=9)

    def test_equivalent_smiles_give_the_same_number(self):
        a = self.model.predict_smiles("c1ccccc1C(=O)O")
        b = self.model.predict_smiles("OC(=O)c1ccccc1")
        self.assertEqual(a.log10_si, b.log10_si)

    def test_api_and_cli_agree(self):
        from fastapi.testclient import TestClient

        from molecula import api

        client = TestClient(api.app)
        response = client.post("/predict", json={"records": [{"id": "x", "smiles": CIDOFOVIR}]})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["model_version"], self.model.model_version)
        result = payload["results"][0]
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(
            result["prediction_si"], self.model.predict_smiles(CIDOFOVIR).si, places=3
        )

    def test_invalid_row_does_not_stop_the_batch_and_is_not_a_zero(self):
        from fastapi.testclient import TestClient

        from molecula import api

        client = TestClient(api.app)
        response = client.post(
            "/predict",
            json={
                "records": [
                    {"id": 1, "smiles": ASPIRIN},
                    {"id": 2, "smiles": "definitely not a molecule"},
                    {"id": 3, "smiles": CIDOFOVIR},
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual([r["id"] for r in results], [1, 2, 3])
        self.assertEqual(results[1]["status"], "invalid")
        self.assertIsNone(results[1]["prediction_si"])
        self.assertTrue(results[1]["error"])
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[2]["status"], "ok")

    def test_submission_takes_its_columns_from_the_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            test_csv = directory / "test.csv"
            sample_csv = directory / "sample_submission.csv"
            out_csv = directory / "submission.csv"

            with test_csv.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "smiles"])
                writer.writerow(["m2", ASPIRIN])
                writer.writerow(["m1", CIDOFOVIR])
            with sample_csv.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "SI"])  # deliberately an unusual target name
                writer.writerow(["m1", 0])
                writer.writerow(["m2", 0])

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "molecula.predict",
                    "--model",
                    str(MODEL_PATH),
                    "submission",
                    "--test",
                    str(test_csv),
                    "--sample-submission",
                    str(sample_csv),
                    "--target-scale",
                    "si",
                    "--out",
                    str(out_csv),
                ],
                cwd=PROJECT_ROOT,
                env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

            rows = list(csv.DictReader(out_csv.open()))
            self.assertEqual(list(rows[0].keys()), ["id", "SI"])
            self.assertEqual([r["id"] for r in rows], ["m1", "m2"])  # sample's order, not the test's
            self.assertTrue(all(float(r["SI"]) for r in rows))

            report = json.loads(out_csv.with_suffix(".report.json").read_text())
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["unparseable_rows"], [])

    def test_a_malformed_record_is_one_bad_row_not_a_failed_batch(self):
        from fastapi.testclient import TestClient

        from molecula import api

        client = TestClient(api.app)
        response = client.post(
            "/predict",
            json={
                "records": [
                    {"id": None, "smiles": 42},
                    {"id": 7, "smiles": None},
                    {"id": 1.5, "smiles": ""},
                    {"id": "good", "smiles": ASPIRIN},
                ]
            },
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(len(results), 4)
        self.assertEqual([r["status"] for r in results], ["invalid", "invalid", "invalid", "ok"])
        self.assertTrue(all(r["prediction_si"] is None for r in results[:3]))
        self.assertTrue(all(r["error"] for r in results[:3]))

    def test_inorganic_input_is_refused_rather_than_scored(self):
        from fastapi.testclient import TestClient

        from molecula import api

        client = TestClient(api.app)
        response = client.post(
            "/predict", json={"records": [{"id": "salt", "smiles": "[Na+].[Cl-]"}]}
        )
        result = response.json()["results"][0]
        self.assertEqual(result["status"], "invalid")
        self.assertIsNone(result["prediction_si"])

    def test_a_ragged_csv_does_not_shift_columns(self):
        from fastapi.testclient import TestClient

        from molecula import api

        client = TestClient(api.app)
        response = client.post(
            "/api/parse-csv",
            files={"file": ("m.csv", f"id,smiles\n1,{ASPIRIN},extra\n".encode(), "text/csv")},
        )
        if response.status_code == 200:
            record = response.json()["records"][0]
            self.assertEqual(record["id"], "1")
            self.assertEqual(record["smiles"], ASPIRIN)
        else:
            self.assertEqual(response.status_code, 400)

    def test_an_empty_upload_is_a_client_error_not_a_crash(self):
        from fastapi.testclient import TestClient

        from molecula import api

        client = TestClient(api.app)
        response = client.post("/api/parse-csv", files={"file": ("m.csv", b"", "text/csv")})
        self.assertEqual(response.status_code, 400)

    def test_submission_refuses_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "test.csv").write_text(f"id,smiles\nm1,{ASPIRIN}\nm1,{CIDOFOVIR}\n")
            (directory / "sample.csv").write_text("id,SI\nm1,0\n")
            completed = self._run_submission(directory, extra=[])
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicated id", completed.stderr)

    def test_submission_refuses_to_invent_a_number_for_a_bad_row(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "test.csv").write_text(f"id,smiles\nm1,{ASPIRIN}\nm2,rubbish\n")
            (directory / "sample.csv").write_text("id,SI\nm1,0\nm2,0\n")
            refused = self._run_submission(directory, extra=[])
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("Refusing to write a made-up number", refused.stderr)
            self.assertFalse((directory / "s.csv").exists())

            allowed = self._run_submission(directory, extra=["--fill-invalid", "median"])
            self.assertEqual(allowed.returncode, 0, allowed.stderr)
            report = json.loads((directory / "s.report.json").read_text())
            self.assertEqual([r["id"] for r in report["unparseable_rows"]], ["m2"])

    def _run_submission(self, directory: Path, extra: list[str]):
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "molecula.predict",
                "--model",
                str(MODEL_PATH),
                "submission",
                "--test",
                str(directory / "test.csv"),
                "--sample-submission",
                str(directory / "sample.csv"),
                "--target-scale",
                "si",
                "--out",
                str(directory / "s.csv"),
                *extra,
            ],
            cwd=PROJECT_ROOT,
            env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )

    def test_submission_refuses_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            test_csv = directory / "test.csv"
            sample_csv = directory / "sample_submission.csv"
            test_csv.write_text(f"id,smiles\nm1,{ASPIRIN}\n")
            sample_csv.write_text("id,SI\nm1,0\nGHOST,0\n")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "molecula.predict",
                    "--model",
                    str(MODEL_PATH),
                    "submission",
                    "--test",
                    str(test_csv),
                    "--sample-submission",
                    str(sample_csv),
                    "--target-scale",
                    "si",
                    "--out",
                    str(directory / "s.csv"),
                ],
                cwd=PROJECT_ROOT,
                env={"PYTHONPATH": str(PROJECT_ROOT / "src"), "PATH": "/usr/bin:/bin"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("GHOST", completed.stderr)
            self.assertFalse((directory / "s.csv").exists())


if __name__ == "__main__":
    unittest.main()
