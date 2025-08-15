"""Base class for samplers
The sampling code is separate to facilitate sampling as a standalone module.
This class basically just serves as an "adaptor" of some kind because it
accepts all parameters needed for the model samplers.  Some of these parameters
are only needed by some model samplers.
FIXME: The alternative would be to remove this class and use a simple strategy
       pattern for the samplers.  This would mean that all samplers need to
       accept all parameters and all those classes need the boilerplate.  Also,
       the classes are not really needed and a simple function (with helper
       functions were needed) would suffice.
"""
from __future__ import annotations
__all__ = ["Sampler", "remove_duplicate_sequences", "validate_smiles", "SmilesState"]
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, TYPE_CHECKING
import logging
from torch import Tensor
from enum import Enum

import numpy as np
from rdkit import Chem

if TYPE_CHECKING:
    from models.model import Model

logger = logging.getLogger(__name__)

class SmilesState(Enum):
    INVALID = 0
    VALID = 1
    DUPLICATE = 2

@dataclass
class BatchRow:
    input: str
    output: str
    nll: float
    smiles: str
    state: SmilesState

@dataclass
class SampleBatch:
    """Container to hold the data returned by the adapter .sample() methods

    This is a somewhat ugly unifying implementation for all generator sample
    methods which return different data.  All return a 3-tuple with the NLL last
    but Reinvent returns one SMILES list while the others
    return two SMILES lists.
    """

    items1: List[str] | None # SMILES, None for Reinvent
    items2: List[str]  # SMILES
    nlls: Tensor  # negative log likelihoods from the model
    smilies: List[str] = None  # processed SMILES
    states: np.ndarray[SmilesState] = None  # states for items2

    def __iter__(self):
        return self

    def __next__(self):
        idx = self.idx

        try:
            if self.smilies:
                smiles = self.smilies[idx]
            else:
                smiles = None

            if self.states is not None:
                state = self.states[idx]
            else:
                state = None

            result = BatchRow(
                self.items1[idx],
                self.items2[idx],
                self.nlls[idx],
                smiles,
                state,
            )
        except IndexError:
            self.idx = 0
            raise StopIteration

        self.idx += 1

        return result

    @classmethod
    def from_list(cls, batch_rows: List[BatchRow]) -> SampleBatch:
        """Create a new dataclass from a list of BatchRow

        This factory class requires a list with a 5-tuple for the 5 fields.
        This is needed for Libinvent, Linkinvent, Mol2mol.

        FIXME: data type consistency

        :param batch_rows: list of batch rows
        :returns: a new dataclass made from the list
        """

        combined = []

        for batch_row in batch_rows:
            combined.append(
                (
                    batch_row.input,
                    batch_row.output,
                    batch_row.nll,
                    batch_row.smiles,
                    batch_row.state,
                )
            )

        transpose = list(zip(*combined))

        assert len(transpose) == 5

        sample_batch = cls(*transpose)
        sample_batch.nlls = Tensor(sample_batch.nlls)

        return sample_batch

@dataclass
class Sampler(ABC):
    """Base class for samplers"""

    modelL: Model
    # number of smiles to be generated for each input,
    # different from batch size used in dataloader which affect cuda memory
    batch_size: int
    sample_strategy: str = "multinomial"  # Transformer-based models
    isomeric: bool = False  # Transformer-based models
    randomize_smiles: bool = True
    unique_sequences: bool = False  # backwards compatibility for R3

    @abstractmethod
    def sample(self, smilies: List[str]) -> SampleBatch:
        """Use provided SMILES list for sampling"""


def validate_smiles(
        smilies, isomeric: bool = False
) -> Tuple[List, np.ndarray]:
    """Basic validation of sampled or joined SMILES

    The molecules are converted to canonical SMILES.  Each SMILES state is
    determined to be invalid, valid or duplicate.
    TODO: Can not validate whether other invalid chars like ['.'. '[N+1]',...]

    :mols: molecules
    :smilies: SMILES of molecules including invalid ones
    :returns: validated SMILES and their states
    """
    mols = [
            Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
            for smiles in smilies
        ]
    validated_smilies = []
    smilies_states = []  # valid, invalid, duplicate
    seen_before = set()

    for mol, sampled_smiles in zip(mols, smilies):
        if mol:
            failed = Chem.SanitizeMol(mol, catchErrors=True)

            if not failed:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)

                if canonical_smiles in seen_before:
                    smilies_states.append(SmilesState.DUPLICATE)
                else:
                    smilies_states.append(SmilesState.VALID)

                validated_smilies.append(canonical_smiles)
                seen_before.add(canonical_smiles)
            else:
                validated_smilies.append(sampled_smiles)
                smilies_states.append(SmilesState.INVALID)
        else:
            validated_smilies.append(sampled_smiles)
            smilies_states.append(SmilesState.INVALID)

    smilies_states = np.array(smilies_states)

    return validated_smilies, smilies_states
