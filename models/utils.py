"""Helper routines

A set of common auxiliary functionality.
"""
import os
from typing import List
import logging
import torch
import models

logger = logging.getLogger(__name__)

def collate_fn(encoded_seqs: List[torch.Tensor]) -> torch.Tensor:
    """Converts a list of encoded sequences into a padded tensor

    :param encoded_seqs: encodes sequences to be padded with zeroes
    :return: padded tensor
    """
    max_length = max([seq.size(0) for seq in encoded_seqs])
    collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
    for i, seq in enumerate(encoded_seqs):
        collated_arr[i, :seq.size(0)] = seq
    return collated_arr

import pdb
import copy
def create_model(dict_filename: str, device: torch.device):
    # Avoid frequently assign tensor device
    dict_filename = os.path.abspath(dict_filename)
    save_dict = torch.load(dict_filename, map_location="cpu")
    model_class = getattr(models, f"Model", None)
    
    # model = model_class.create_from_dict(save_dict, device)
    model = copy.deepcopy(model_class.create_from_dict(save_dict, device))

    network_params = model.network.parameters()
    num_params = sum([tensor.numel() for tensor in network_params])
    logger.info(f"Number of network parameters: {num_params:,}")

    return model