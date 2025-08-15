import os
import torch
import csv
from typing import List, Tuple, Union, Optional, Callable
import numpy as np
import random

from rdkit import Chem
import logging
import yaml
from box import Box

__all__ = ["set_random_seeds", "disable_gradients", "read_smiles_csv_file",
           "read_yaml", "log_smi_score_to_csv", "create_csv"]
logger = logging.getLogger(__name__)

def set_random_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def disable_gradients(model) -> None:
    """Disable gradient tracking for all parameters in a model

    :param model: the model for which all gradient tracking will be switched off
    """

    for param in model.parameters():
        param.requires_grad = False

def has_multiple_attachment_points_to_same_atom(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        raise RuntimeError(f"Error: Input {smiles} is not a valid molecule")

    seen = set()

    for atom in mol.GetAtoms():
        if atom.HasProp("dummyLabel"):
            neighbours = atom.GetNeighbors()

            if len(neighbours) > 1:
                raise RuntimeError("Error: dummy atom is not terminal")

            idx = neighbours[0].GetIdx()

            if idx in seen:
                return True

            seen.add(idx)

    return False


def read_smiles_csv_file(
    filename: str,
    columns: Union[int, slice],
    delimiter: str = "\t",
    header: bool = False,
    remove_duplicates: bool = False,
) -> Union[List[str], List[Tuple]]:
    """Read a SMILES column from a CSV file

    FIXME: needs to be made more robust

    :param filename: name of the CSV file
    :param columns: what number of the column to extract
    :param delimiter: column delimiter, must be a single character
    :param header: whether a header is present
    :param actions: a list of callables that act on each SMILES (only Reinvent
                    and Mol2Mol)
    :param remove_duplicates: whether to remove duplicates
    :returns: a list of SMILES or a list of a tuple of SMILES
    """

    smilies = []
    frontier = set()

    with open(filename, "r") as csvfile:
        if header:
            csvfile.readline()

        reader = csv.reader(csvfile, delimiter=delimiter)

        for row in reader:
            stripped_row = "".join(row).strip()

            if not stripped_row or stripped_row.startswith("#"):
                continue

            if isinstance(columns, int):
                smiles = row[columns].strip()

            else:
                smiles = tuple(smiles.strip() for smiles in row[columns])

                # FIXME: hard input check for libinvent / linkinvent
                #        for unsupported scaffolds containing multiple
                #        attachment points to the same atoms.
                # libinvent
                if "|" in smiles[1]:
                    if has_multiple_attachment_points_to_same_atom(smiles[0]):
                        raise ValueError(
                            f"Not supported: Smiles {smiles[0]} contains multiple attachment points for the same atom"
                        )
                # linkinvent
                if "|" in smiles[0]:
                    if has_multiple_attachment_points_to_same_atom(smiles[1]):
                        raise ValueError(
                            f"Not supported: Smiles {smiles[1]} contains multiple attachment points for the same atom"
                        )

            # SMILES transformation may fail
            # FIXME: needs sensible way to report this back to the user
            if smiles:
                if (not remove_duplicates) or (not smiles in frontier):
                    smilies.append(smiles)
                    frontier.add(smiles)

    return smilies

def read_yaml(yaml_dir):
    '''Read the configurations from yaml file and return as Box type of dict.
    Usage Example: configs.lr / configs.LLM.name
    '''
    with open(yaml_dir, "r") as file:
        configs = Box(yaml.safe_load(file))
    return configs

def create_csv(csv_file):
    '''Create a csv file to save the smiles generated.
    '''
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['SMILES', 'Score'])

def log_smi_score_to_csv(smiles_list, score_list,csv_file):
    data = list(zip(smiles_list, score_list))
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)