import pickle
import numpy as np
import os.path as op
from abc import abstractmethod
from functools import partial
from typing import List
import time, os, math, re
from packaging import version
import pkg_resources
from tdc import Oracle
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import rdkit.Chem.QED as QED
from rdkit import rdBase
from scipy.stats.mstats import gmean

rdBase.DisableLog("rdApp.error")
from rdkit.Chem import rdMolDescriptors

def smiles_to_rdkit_mol(smiles):
    """Convert smiles into rdkit's mol (molecule) format.

    Args:
      smiles: str, SMILES string.

    Returns:
      mol: rdkit.Chem.rdchem.Mol

    """
    mol = Chem.MolFromSmiles(smiles)
    #  Sanitization check (detects invalid valence)
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
    return mol

def smiles_2_fingerprint_ECFP4(smiles):
    """Convert smiles into ECFP4 Morgan Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprint(molecule, 2)
    return fp


def smiles_2_fingerprint_FCFP4(smiles):
    """Convert smiles into FCFP4 Morgan Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprint(molecule, 2, useFeatures=True)
    return fp


def smiles_2_fingerprint_AP(smiles):
    """Convert smiles into Atom Pair Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.IntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetAtomPairFingerprint(molecule, maxLength=10)
    return fp


def smiles_2_fingerprint_ECFP6(smiles):
    """Convert smiles into ECFP6 Fingerprint.

    Args:
      smiles: str, SMILES string.

    Returns:
      fp: rdkit.DataStructs.cDataStructs.UIntSparseIntVect

    """
    molecule = smiles_to_rdkit_mol(smiles)
    fp = AllChem.GetMorganFingerprint(molecule, 3)
    return fp


fp2fpfunc = {
    "ECFP4": smiles_2_fingerprint_ECFP4,
    "FCFP4": smiles_2_fingerprint_FCFP4,
    "AP": smiles_2_fingerprint_AP,
    "ECFP6": smiles_2_fingerprint_ECFP6,
}

class ScoreModifier:
    """
    Interface for score modifiers.
    """

    @abstractmethod
    def __call__(self, x):
        """
        Apply the modifier on x.

        Args:
            x: float or np.array to modify

        Returns:
            float or np.array (depending on the type of x) after application of the distance function.
        """


class ChainedModifier(ScoreModifier):
    """
    Calls several modifiers one after the other, for instance:
        score = modifier3(modifier2(modifier1(raw_score)))
    """

    def __init__(self, modifiers: List[ScoreModifier]) -> None:
        """
        Args:
            modifiers: modifiers to call in sequence.
                The modifier applied last (and delivering the final score) is the last one in the list.
        """
        self.modifiers = modifiers

    def __call__(self, x):
        score = x
        for modifier in self.modifiers:
            score = modifier(score)
        return score


class LinearModifier(ScoreModifier):
    """
    Score modifier that multiplies the score by a scalar (default: 1, i.e. do nothing).
    """

    def __init__(self, slope=1.0):
        self.slope = slope

    def __call__(self, x):
        return self.slope * x


class SquaredModifier(ScoreModifier):
    """
    Score modifier that has a maximum at a given target value, and decreases
    quadratically with increasing distance from the target value.
    """

    def __init__(self, target_value: float, coefficient=1.0) -> None:
        self.target_value = target_value
        self.coefficient = coefficient

    def __call__(self, x):
        return 1.0 - self.coefficient * np.square(self.target_value - x)


class AbsoluteScoreModifier(ScoreModifier):
    """
    Score modifier that has a maximum at a given target value, and decreases
    linearly with increasing distance from the target value.
    """

    def __init__(self, target_value: float) -> None:
        self.target_value = target_value

    def __call__(self, x):
        return 1.0 - np.abs(self.target_value - x)


class GaussianModifier(ScoreModifier):
    """
    Score modifier that reproduces a Gaussian bell shape.
    """

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2.0))


class MinMaxGaussianModifier(ScoreModifier):
    """
    Score modifier that reproduces a half Gaussian bell shape.
    For minimize==True, the function is 1.0 for x <= mu and decreases to zero for x > mu.
    For minimize==False, the function is 1.0 for x >= mu and decreases to zero for x < mu.
    """

    def __init__(self, mu: float, sigma: float, minimize=False) -> None:
        self.mu = mu
        self.sigma = sigma
        self.minimize = minimize
        self._full_gaussian = GaussianModifier(mu=mu, sigma=sigma)

    def __call__(self, x):
        if self.minimize:
            mod_x = np.maximum(x, self.mu)
        else:
            mod_x = np.minimum(x, self.mu)
        return self._full_gaussian(mod_x)


MinGaussianModifier = partial(MinMaxGaussianModifier, minimize=True)
MaxGaussianModifier = partial(MinMaxGaussianModifier, minimize=False)


class ClippedScoreModifier(ScoreModifier):
    r"""
    Clips a score between specified low and high scores, and does a linear interpolation in between.

    This class works as follows:
    First the input is mapped onto a linear interpolation between both specified points.
    Then the generated values are clipped between low and high scores.
    """

    def __init__(self,
                 upper_x: float,
                 lower_x=0.0,
                 high_score=1.0,
                 low_score=0.0) -> None:
        """
        Args:
            upper_x: x-value from which (or until which if smaller than lower_x) the score is maximal
            lower_x: x-value until which (or from which if larger than upper_x) the score is minimal
            high_score: maximal score to clip to
            low_score: minimal score to clip to
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        self.slope = (high_score - low_score) / (upper_x - lower_x)
        self.intercept = high_score - self.slope * upper_x

    def __call__(self, x):
        y = self.slope * x + self.intercept
        return np.clip(y, self.low_score, self.high_score)


class SmoothClippedScoreModifier(ScoreModifier):
    """
    Smooth variant of ClippedScoreModifier.

    Implemented as a logistic function that has the same steepness as ClippedScoreModifier in the
    center of the logistic function.
    """

    def __init__(self,
                 upper_x: float,
                 lower_x=0.0,
                 high_score=1.0,
                 low_score=0.0) -> None:
        """
        Args:
            upper_x: x-value from which (or until which if smaller than lower_x) the score approaches high_score
            lower_x: x-value until which (or from which if larger than upper_x) the score approaches low_score
            high_score: maximal score (reached at +/- infinity)
            low_score: minimal score (reached at -/+ infinity)
        """
        assert low_score < high_score

        self.upper_x = upper_x
        self.lower_x = lower_x
        self.high_score = high_score
        self.low_score = low_score

        # Slope of a standard logistic function in the middle is 0.25 -> rescale k accordingly
        self.k = 4.0 / (upper_x - lower_x)
        self.middle_x = (upper_x + lower_x) / 2
        self.L = high_score - low_score

    def __call__(self, x):
        return self.low_score + self.L / (1 + np.exp(-self.k *
                                                     (x - self.middle_x)))


class ThresholdedLinearModifier(ScoreModifier):
    """
    Returns a value of min(input, threshold)/threshold.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def __call__(self, x):
        return np.minimum(x, self.threshold) / self.threshold

class AtomCounter:

    def __init__(self, element):
        """
        Args:
            element: element to count within a molecule
        """
        self.element = element

    def __call__(self, mol):
        """
        Count the number of atoms of a given type.

        Args:
            mol: molecule

        Returns:
            The number of atoms of the given type.
        """
        # if the molecule contains H atoms, they may be implicit, so add them
        if self.element == "H":
            mol = Chem.AddHs(mol)

        return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == self.element)

def parse_molecular_formula(formula):
    """
    Parse a molecular formulat to get the element types and counts.

    Args:
        formula: molecular formula, f.i. "C8H3F3Br"

    Returns:
        A list of tuples containing element types and number of occurrences.
    """
    import re

    matches = re.findall(r"([A-Z][a-z]*)(\d*)", formula)

    # Convert matches to the required format
    results = []
    for match in matches:
        # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
        count = 1 if not match[1] else int(match[1])
        results.append((match[0], count))

    return results


def smiles2formula(smiles):

    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    mol = Chem.MolFromSmiles(smiles)
    formula = CalcMolFormula(mol)
    return formula


def canonicalize(smiles: str, include_stereocenters=True):
    """
    Canonicalize the SMILES strings with RDKit.

    The algorithm is detailed under https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543

    Args:
        smiles: SMILES string to canonicalize
        include_stereocenters: whether to keep the stereochemical information in the canonical SMILES string

    Returns:
        Canonicalized SMILES string, None if the molecule is invalid.
    """

    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters)
    else:
        return None

class Isomer_scoring_prev:

    def __init__(self, target_smiles, means="geometric"):
        assert means in ["geometric", "arithmetic"]
        if means == "geometric":
            self.mean_func = gmean
        else:
            self.mean_func = np.mean
        atom2cnt_lst = parse_molecular_formula(target_smiles)
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        self.total_atom_modifier = GaussianModifier(mu=total_atom_num,
                                                    sigma=2.0)
        self.AtomCounter_Modifier_lst = [((AtomCounter(atom)),
                                          GaussianModifier(mu=cnt, sigma=1.0))
                                         for atom, cnt in atom2cnt_lst]

    def __call__(self, test_smiles):
        molecule = smiles_to_rdkit_mol(test_smiles)
        all_scores = []
        for atom_counter, modifier_func in self.AtomCounter_Modifier_lst:
            all_scores.append(modifier_func(atom_counter(molecule)))

        ### total atom number
        atom2cnt_lst = parse_molecular_formula(test_smiles)
        # ## todo add Hs
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        all_scores.append(self.total_atom_modifier(total_atom_num))
        return self.mean_func(all_scores)


class Isomer_scoring:

    def __init__(self, target_smiles, means="geometric"):
        assert means in ["geometric", "arithmetic"]
        if means == "geometric":
            self.mean_func = gmean
        else:
            self.mean_func = np.mean
        atom2cnt_lst = parse_molecular_formula(target_smiles)
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        self.total_atom_modifier = GaussianModifier(mu=total_atom_num,
                                                    sigma=2.0)
        self.AtomCounter_Modifier_lst = [((AtomCounter(atom)),
                                          GaussianModifier(mu=cnt, sigma=1.0))
                                         for atom, cnt in atom2cnt_lst]

    def __call__(self, test_smiles):
        #### difference 1
        #### add hydrogen atoms
        test_smiles = canonicalize(test_smiles)
        test_mol = Chem.MolFromSmiles(test_smiles)
        test_mol2 = Chem.AddHs(test_mol)
        test_smiles = Chem.MolToSmiles(test_mol2)

        molecule = smiles_to_rdkit_mol(test_smiles)
        all_scores = []
        for atom_counter, modifier_func in self.AtomCounter_Modifier_lst:
            all_scores.append(modifier_func(atom_counter(molecule)))

        #### difference 2
        ### total atom number
        test_formula = smiles2formula(test_smiles)
        atom2cnt_lst = parse_molecular_formula(test_formula)
        # atom2cnt_lst = parse_molecular_formula(test_smiles)
        # ## todo add Hs
        total_atom_num = sum([cnt for atom, cnt in atom2cnt_lst])
        all_scores.append(self.total_atom_modifier(total_atom_num))
        return self.mean_func(all_scores)

_fscores = None
def osimertinib_mpo(test_smiles):
    try:
        if "osimertinib_fp_fcfc4" not in globals().keys():
            global osimertinib_fp_fcfc4, osimertinib_fp_ecfc6
            osimertinib_smiles = (
                "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34")
            osimertinib_fp_fcfc4 = smiles_2_fingerprint_FCFP4(osimertinib_smiles)
            osimertinib_fp_ecfc6 = smiles_2_fingerprint_ECFP6(osimertinib_smiles)

        sim_v1_modifier = ClippedScoreModifier(upper_x=0.8)
        sim_v2_modifier = MinGaussianModifier(mu=0.85, sigma=0.1)
        tpsa_modifier = MaxGaussianModifier(mu=100, sigma=10)
        logp_modifier = MinGaussianModifier(mu=1, sigma=1)

        molecule = smiles_to_rdkit_mol(test_smiles)
        fp_fcfc4 = smiles_2_fingerprint_FCFP4(test_smiles)
        fp_ecfc6 = smiles_2_fingerprint_ECFP6(test_smiles)
        tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
        logp_score = logp_modifier(Descriptors.MolLogP(molecule))
        similarity_v1 = sim_v1_modifier(
            DataStructs.TanimotoSimilarity(osimertinib_fp_fcfc4, fp_fcfc4))
        similarity_v2 = sim_v2_modifier(
            DataStructs.TanimotoSimilarity(osimertinib_fp_ecfc6, fp_ecfc6))

        osimertinib_gmean = gmean(
            [tpsa_score, logp_score, similarity_v1, similarity_v2])

        return {
            "0": tpsa_score, 
            "1": logp_score,
            "2": similarity_v1,
            "3": similarity_v2, 
            "total": osimertinib_gmean
        }

    except Exception as e:
        # Log the error (you can customize this to your needs)
        print(f"Error processing {test_smiles}: {e}")
        # Return 0 for all scores if an error occurs
        return {
            "0": 0, 
            "1": 0,
            "2": 0,
            "3": 0, 
            "total": 0
        }

def fexofenadine_mpo(test_smiles):
    if "fexofenadine_fp" not in globals().keys():
        global fexofenadine_fp
        fexofenadine_smiles = (
            "CC(C)(C(=O)O)c1ccc(cc1)C(O)CCCN2CCC(CC2)C(O)(c3ccccc3)c4ccccc4")
        fexofenadine_fp = smiles_2_fingerprint_AP(fexofenadine_smiles)

    similar_modifier = ClippedScoreModifier(upper_x=0.8)
    tpsa_modifier = MaxGaussianModifier(mu=90, sigma=10)
    logp_modifier = MinGaussianModifier(mu=4, sigma=1)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ap = smiles_2_fingerprint_AP(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_value = similar_modifier(
        DataStructs.TanimotoSimilarity(fp_ap, fexofenadine_fp))
    # ========= MPO ========================
    # fexofenadine = (tpsa_score, logp_score, similarity_value)
    fexofenadine = gmean([tpsa_score, logp_score, similarity_value])
    return {
        "0": tpsa_score, 
        "1": logp_score,
        "2": similarity_value,
        "total": fexofenadine
    }
    


def ranolazine_mpo(test_smiles):
    if "ranolazine_fp" not in globals().keys():
        global ranolazine_fp, fluorine_counter
        ranolazine_smiles = "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2"
        ranolazine_fp = smiles_2_fingerprint_AP(ranolazine_smiles)
        fluorine_counter = AtomCounter("F")

    similar_modifier = ClippedScoreModifier(upper_x=0.7)
    tpsa_modifier = MaxGaussianModifier(mu=95, sigma=20)
    logp_modifier = MaxGaussianModifier(mu=7, sigma=1)
    fluorine_modifier = GaussianModifier(mu=1, sigma=1.0)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ap = smiles_2_fingerprint_AP(test_smiles)
    tpsa_score = tpsa_modifier(Descriptors.TPSA(molecule))
    logp_score = logp_modifier(Descriptors.MolLogP(molecule))
    similarity_value = similar_modifier(
        DataStructs.TanimotoSimilarity(fp_ap, ranolazine_fp))
    fluorine_value = fluorine_modifier(fluorine_counter(molecule))

    ranolazine_gmean = gmean(
        [tpsa_score, logp_score, similarity_value, fluorine_value])
        
    return {
            "0" : tpsa_score, 
            "1" : logp_score, 
            "2" : similarity_value, 
            "3":fluorine_value,
            "total":ranolazine_gmean}


def perindopril_mpo(test_smiles):
    ## no similar_modifier

    if "perindopril_fp" not in globals().keys():
        global perindopril_fp, num_aromatic_rings
        perindopril_smiles = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"
        perindopril_fp = smiles_2_fingerprint_ECFP4(perindopril_smiles)

        def num_aromatic_rings(mol):
            return rdMolDescriptors.CalcNumAromaticRings(mol)

    arom_rings_modifier = GaussianModifier(mu=2, sigma=0.5)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)

    similarity_value = DataStructs.TanimotoSimilarity(fp_ecfp4, perindopril_fp)
    num_aromatic_rings_value = arom_rings_modifier(num_aromatic_rings(molecule))

    # perindopril_gmean = gmean([similarity_value, num_aromatic_rings_value])
    # ========= MPO ========================
    perindopril = gmean([similarity_value, num_aromatic_rings_value])
    return {"0" : similarity_value,
            "1": num_aromatic_rings_value,
            "total":perindopril}


def amlodipine_mpo(test_smiles):
    ## no similar_modifier
    if "amlodipine_fp" not in globals().keys():
        global amlodipine_fp, num_rings
        amlodipine_smiles = "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC"
        amlodipine_fp = smiles_2_fingerprint_ECFP4(amlodipine_smiles)

        def num_rings(mol):
            return rdMolDescriptors.CalcNumRings(mol)

    num_rings_modifier = GaussianModifier(mu=3, sigma=0.5)

    molecule = smiles_to_rdkit_mol(test_smiles)
    fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)

    similarity_value = DataStructs.TanimotoSimilarity(fp_ecfp4, amlodipine_fp)
    num_rings_value = num_rings_modifier(num_rings(molecule))

    # amlodipine_gmean = gmean([similarity_value, num_rings_value])
    # ========= MPO ========================
    amlodipine = gmean([similarity_value, num_rings_value])
    return {
        "0": similarity_value,
        "1": num_rings_value,
        "total": amlodipine
    }



def zaleplon_mpo(test_smiles):
    if "zaleplon_fp" not in globals().keys():
        global zaleplon_fp, isomer_scoring_C19H17N3O2
        zaleplon_smiles = "O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"
        zaleplon_fp = smiles_2_fingerprint_ECFP4(zaleplon_smiles)
        isomer_scoring_C19H17N3O2 = Isomer_scoring(target_smiles="C19H17N3O2")

    fp = smiles_2_fingerprint_ECFP4(test_smiles)
    similarity_value = DataStructs.TanimotoSimilarity(fp, zaleplon_fp)
    isomer_value = isomer_scoring_C19H17N3O2(test_smiles)
    zaleplon = gmean([similarity_value, isomer_value])
    return {
        "0": similarity_value,
        "1": isomer_value,
        "total": zaleplon
    }


def sitagliptin_mpo(test_smiles):
    if "sitagliptin_fp_ecfp4" not in globals().keys():
        global sitagliptin_fp_ecfp4, sitagliptin_logp_modifier, sitagliptin_tpsa_modifier, isomers_scoring_C16H15F6N5O, sitagliptin_similar_modifier
        sitagliptin_smiles = "Fc1cc(c(F)cc1F)CC(N)CC(=O)N3Cc2nnc(n2CC3)C(F)(F)F"
        sitagliptin_fp_ecfp4 = smiles_2_fingerprint_ECFP4(sitagliptin_smiles)
        sitagliptin_mol = Chem.MolFromSmiles(sitagliptin_smiles)
        sitagliptin_logp = Descriptors.MolLogP(sitagliptin_mol)
        sitagliptin_tpsa = Descriptors.TPSA(sitagliptin_mol)
        sitagliptin_logp_modifier = GaussianModifier(mu=sitagliptin_logp,
                                                     sigma=0.2)
        sitagliptin_tpsa_modifier = GaussianModifier(mu=sitagliptin_tpsa,
                                                     sigma=5)
        isomers_scoring_C16H15F6N5O = Isomer_scoring("C16H15F6N5O")
        sitagliptin_similar_modifier = GaussianModifier(mu=0, sigma=0.1)

    molecule = Chem.MolFromSmiles(test_smiles)
    fp_ecfp4 = smiles_2_fingerprint_ECFP4(test_smiles)
    logp_score = Descriptors.MolLogP(molecule)
    logp_score = sitagliptin_logp_modifier(logp_score)
    tpsa_score = Descriptors.TPSA(molecule)
    tpsa_score = sitagliptin_tpsa_modifier(tpsa_score)
    isomer_score = isomers_scoring_C16H15F6N5O(test_smiles)
    similarity_value = sitagliptin_similar_modifier(
        DataStructs.TanimotoSimilarity(fp_ecfp4, sitagliptin_fp_ecfp4))
    sitagliptin =  gmean([similarity_value, logp_score, tpsa_score, isomer_score])
    return {
        "0": similarity_value,
        "1": logp_score,
        "2": tpsa_score,
        "3": isomer_score,
        "total": sitagliptin
    }

def reward_guacamol_mpo(mols, name='osimertinib'):
    ''' MPO task names : ['osimertinib', 'fexofenadine', 'ranolazine', 'perindopril',
                          'amlodipine', 'zaleplon', 'sitagliptin', 'task_23'] '''
    task = f'{name}_mpo'  # Construct the task name dynamically
    
    if task in globals():
        func = globals()[task]  # Retrieve the function by its name
    else:
        raise ValueError(f"Task {name} is not supported!")
    
    return [func(m) for m in mols]


