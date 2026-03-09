import json
import os
from pathlib import Path
from typing import ClassVar

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from dotenv import load_dotenv
from esm3.models.esm3 import ESM3
from esm3.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from huggingface_hub import login
from typing_extensions import Self

from mccop.utils.helpers import get_output_dir, logger


class StructurePredictor:
    """Singleton class to handle ESM3 structure prediction and scoring.

    Ensures the model is loaded only once and maintains a persistent cache on disk.
    """

    _instance: ClassVar["StructurePredictor | None"] = None
    _model: ClassVar[ESM3InferenceClient | None] = None

    def __new__(cls) -> Self:
        """Implements singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "cache"):
            self.cache_path = get_output_dir() / "structure_cache.json"
            self.cache: dict[str, tuple[float, float]] = self._load_cache()

    def _load_cache(self) -> dict[str, tuple[float, float]]:
        """Loads the cache from disk if it exists."""
        if self.cache_path.exists():
            try:
                with self.cache_path.open("r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load structure cache: {e}. Starting fresh.")
                return {}
        return {}

    def _save_cache(self) -> None:
        """Saves the current cache to disk."""
        try:
            with self.cache_path.open("w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save structure cache: {e}")

    def _ensure_model_loaded(self) -> None:
        """Loads the ESM3 model if it hasn't been loaded yet."""
        if self._model is not None:
            return

        logger.info("Loading ESM3 model for structure prediction...")
        load_dotenv()

        token = os.getenv("HF_TOKEN")
        if not token:
            logger.warning("HF_TOKEN not found. ESM3 loading may fail.")
        else:
            login(token=token, add_to_git_credential=True)

        try:
            self._model = ESM3.from_pretrained("esm3-open").to("cuda")
            logger.info("ESM3 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load ESM3 model: {e}")
            raise e

    def _calculate_physical_metrics(self, pdb_path: Path) -> dict[str, float]:
        """Calculates SASA and Radius of Gyration from a PDB file."""
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", str(pdb_path))

            # Compute solvent accessible surface area
            sr = ShrakeRupley()
            sr.compute(structure, level="S")
            sasa = structure.sasa

            # Compute radius of gyration (compactness)
            atoms = list(structure.get_atoms())
            coords = np.array([atom.get_coord() for atom in atoms])
            center_of_mass = np.mean(coords, axis=0)
            rg = np.sqrt(np.sum(np.sum((coords - center_of_mass) ** 2, axis=1)) / len(atoms))

            return {"sasa": float(sasa), "radius_of_gyration": float(rg)}
        except Exception as e:
            logger.warning(f"Failed to compute physical metrics for {pdb_path}: {e}")
            return {"sasa": 0.0, "radius_of_gyration": 0.0}

    def compute_scores(self, sequence: str, save_pdb_path: Path | None = None) -> dict[str, float]:
        """Computes pTM and pLDDT scores and physical metrics for a given sequence.

        Args:
            sequence: The amino acid sequence.
            save_pdb_path: Optional path to save the generated PDB structure.

        Returns:
            A dictionary with pTM, pLDDT, SASA, and Radius of Gyration.
        """
        if not sequence:
            return {"ptm": 0.0, "plddt": 0.0, "sasa": 0.0, "radius_of_gyration": 0.0}

        if sequence in self.cache:
            return self.cache[sequence]

        self._ensure_model_loaded()

        protein = ESMProtein(sequence=sequence)
        try:
            protein = self._model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        except Exception as e:
            logger.error(f"Error generating structure: {e}")
            return {"ptm": 0.0, "plddt": 0.0, "sasa": 0.0, "radius_of_gyration": 0.0}

        ptm = protein.ptm.item() if protein.ptm is not None else 0.0
        plddt = protein.plddt.mean().item() if protein.plddt is not None else 0.0

        if save_pdb_path:
            save_pdb_path.parent.mkdir(parents=True, exist_ok=True)
            protein.to_pdb(save_pdb_path)

            phys_metrics = self._calculate_physical_metrics(save_pdb_path)
        else:
            phys_metrics = {"sasa": 0.0, "radius_of_gyration": 0.0}

        results = {"ptm": ptm, "plddt": plddt, **phys_metrics}

        self.cache[sequence] = results
        self._save_cache()
        return results


def hamming_distance(seq1: str, seq2: str) -> int:
    """Calculates the Hamming distance between two sequences.

    Args:
        seq1: The first sequence.
        seq2: The second sequence.

    Returns:
        The Hamming distance (number of differing positions).

    Raises:
        ValueError: If the sequences are of different lengths.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length to compute Hamming distance.")

    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2, strict=True))


def sequence_plausibility(seq: str) -> dict[str, float]:
    """Computes basic physicochemical properties.

    Args:
        seq: Amino acid sequence.

    Returns:
        A dictionary of computed properties.
    """
    if not seq:
        return {}

    prot = ProteinAnalysis(seq)

    mw = prot.molecular_weight()
    gravy = prot.gravy()
    instability = prot.instability_index()
    iso_point = prot.isoelectric_point()

    is_soluble_proxy = (instability < 40) and (-1.5 < gravy < 0.5)

    return {
        "molecular_weight": mw,
        "gravy": gravy,
        "instability_index": instability,
        "isoelectric_point": iso_point,
        "is_soluble_proxy": int(is_soluble_proxy),
    }
