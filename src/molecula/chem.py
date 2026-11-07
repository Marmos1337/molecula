"""Single source of truth for SMILES handling and featurisation.

Training, the HTTP API and the submission CLI all import from here, so a
molecule is parsed, standardised and vectorised exactly the same way in all
three paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_BITS = 2048

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

# Keep this list and its order frozen: it is part of the feature contract that
# a saved model was trained against. model_version changes if it changes.
DESCRIPTOR_NAMES: tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHAcceptors",
    "NumHDonors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "MolMR",
    "BertzCT",
    "NumAliphaticRings",
    "NHOHCount",
    "NOCount",
    "qed",
)

FEATURE_NAMES: tuple[str, ...] = tuple(f"fp_{i}" for i in range(FP_BITS)) + DESCRIPTOR_NAMES
N_FEATURES = len(FEATURE_NAMES)


class InvalidSmilesError(ValueError):
    """Raised when a string cannot be read as a molecule."""


@dataclass(frozen=True)
class StandardMolecule:
    input_smiles: str
    canonical_smiles: str
    inchikey: str
    mol: Chem.Mol

    @property
    def scaffold_key(self) -> str:
        """InChIKey skeleton block - groups stereoisomers and tautomer variants."""
        return self.inchikey.split("-")[0]


_UNCHARGER = rdMolStandardize.Uncharger()
_LARGEST = rdMolStandardize.LargestFragmentChooser()


def standardize(smiles: str) -> StandardMolecule:
    """Parse, strip salts/charges, canonicalise. Raises InvalidSmilesError."""
    if smiles is None or smiles != smiles:  # None or NaN
        raise InvalidSmilesError("SMILES is empty")
    text = str(smiles).strip()
    if not text:
        raise InvalidSmilesError("SMILES is empty")
    if len(text) > 2000:
        raise InvalidSmilesError("SMILES is longer than 2000 characters")

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise InvalidSmilesError("RDKit cannot parse this string as a molecule")
    if mol.GetNumHeavyAtoms() == 0:
        raise InvalidSmilesError("molecule has no heavy atoms")

    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception as exc:  # rdkit raises plain exceptions here
        raise InvalidSmilesError(f"molecule failed standardisation: {exc}") from exc

    if mol.GetNumHeavyAtoms() == 0:
        raise InvalidSmilesError("nothing left after salt stripping")
    if not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        raise InvalidSmilesError("no organic fragment left: the model is only trained on organic molecules")
    if mol.GetNumHeavyAtoms() < 3:
        raise InvalidSmilesError("too small to be a drug-like molecule (fewer than 3 heavy atoms)")

    canonical = Chem.MolToSmiles(mol)
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        raise InvalidSmilesError("cannot compute an InChIKey for this molecule")
    return StandardMolecule(text, canonical, inchikey, mol)


def _descriptors(mol: Chem.Mol) -> list[float]:
    from rdkit.Chem import QED

    values = {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtomCount": float(mol.GetNumHeavyAtoms()),
        "NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "NumHeteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "MolMR": Crippen.MolMR(mol),
        "BertzCT": Descriptors.BertzCT(mol),
        "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "NHOHCount": Descriptors.NHOHCount(mol),
        "NOCount": Descriptors.NOCount(mol),
        "qed": QED.qed(mol),
    }
    return [float(values[name]) for name in DESCRIPTOR_NAMES]


def fingerprint(mol: Chem.Mol) -> np.ndarray:
    """Morgan count fingerprint as a dense float vector."""
    fp = _MORGAN.GetCountFingerprintAsNumPy(mol)
    return fp.astype(np.float32)


def featurize(mol: Chem.Mol) -> np.ndarray:
    """One molecule -> one feature row. Order matches FEATURE_NAMES."""
    row = np.concatenate([fingerprint(mol), np.asarray(_descriptors(mol), dtype=np.float32)])
    if row.shape[0] != N_FEATURES:
        raise RuntimeError(f"feature width {row.shape[0]} != expected {N_FEATURES}")
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)


def featurize_many(mols: list[Chem.Mol]) -> np.ndarray:
    if not mols:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    return np.vstack([featurize(m) for m in mols])


def murcko_key(mol: Chem.Mol) -> str:
    """The molecule's Bemis-Murcko framework, as a grouping key.

    Used to hold whole structural families out of a fold. The InChIKey's first
    block is not a substitute: it separates stereoisomers and tautomers but
    keeps every analogue of a series apart, which makes a "scaffold" split
    indistinguishable from a random one. Molecules with no ring system share
    the key "acyclic", since they have no framework to group by.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return "unparsable"
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return "acyclic"
    # Without this, a ring stereocentre splits one framework into several keys
    # and the series it names ends up on both sides of a fold.
    return Chem.MolToSmiles(scaffold, isomericSmiles=False)


def molecular_weight(smiles: str) -> float | None:
    """MW for unit conversion during data collection; None if unparseable."""
    mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
    return float(Descriptors.MolWt(mol)) if mol is not None else None


def svg_structure(mol: Chem.Mol, width: int = 360, height: int = 260, dark: bool = False) -> str:
    """Structure drawing as an inline SVG, themed for the web UI."""
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    if dark:
        options.setAtomPalette({-1: (0.90, 0.90, 0.92)})
    else:
        options.setAtomPalette({-1: (0.10, 0.10, 0.12)})
    drawing = Chem.Mol(mol)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, drawing)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
