import os
import time
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import argparse
import tdc
from torch.utils.tensorboard import SummaryWriter
import pdb
from utils import set_random_seeds, disable_gradients, create_csv, log_smi_score_to_csv
from sampler import SmilesState, validate_smiles
from models.utils import create_model
from oracle.MPO_scorers import reward_guacamol_mpo
from rdkit import Chem
from crossover import crossover

def scoring(smiles, states, scorer):
    '''
    Scoring function to evaluate the smiles with given states
    [Params]
        smiles: smiles
    '''
    invalid_mask = np.where(states == SmilesState.INVALID, False, True)
    duplicate_mask = np.where(states == SmilesState.DUPLICATE, False, True)
    valid_mask = np.logical_and(invalid_mask, duplicate_mask)
    valid_index = np.where(valid_mask)[0]
    valid_smiles = [smiles[i] for i in valid_index]
    
    # Check that each SMILES string can be converted to a valid molecule
    valid_smiles_final = []
    for smi in valid_smiles:
        molecule = Chem.MolFromSmiles(smi)
        if molecule is None:
            print(f"Invalid SMILES: {smi}")  
        else:
            valid_smiles_final.append(smi)
    
    # Ensure valid_smiles_final is not empty before calling scoring function
    if len(valid_smiles_final) == 0:
        print(states)
        print(valid_smiles) 
        raise ValueError("No valid SMILES found.")

    scores = reward_guacamol_mpo(mols=valid_smiles_final, name=scorer)
    assert len(valid_smiles_final) == len(scores)
    return scores, valid_smiles_final

def RL_train(args, prior, agent_idx, agents, optimizers, smiles, scores):
    '''
    Reinforcement learning for SLM
    '''
    # Remove the molecule with new token
    smiles = prior.likelihood_smiles(smiles, check=True)
    smiles = [smile for smile in smiles if smile is not None]
    scores = [score for score, smile in zip(scores, smiles) if smile is not None]
    prior_nlls = prior.likelihood_smiles(smiles)
    agent_nlls = agents[agent_idx].likelihood_smiles(smiles)
    scores = torch.tensor(scores).to(prior_nlls)
    nan_idx = torch.isnan(scores)
    scores_nonnan = scores[~nan_idx]
    agent_lls = -agent_nlls[~nan_idx] 
    prior_lls = -prior_nlls[~nan_idx]
    # ===== DAP strategy ======
    sigma = args.sigma
    augmented_lls = prior_lls + sigma * scores_nonnan
    loss = torch.pow((augmented_lls - agent_lls), 2)
    segam2 = args.segam2
    for j in range(agent_idx):
        loss += segam2 * torch.pow(agents[j].likelihood_smiles(smiles) - agent_nlls, 2) * scores_nonnan
    loss = loss.mean()
    optimizers[agent_idx].zero_grad()
    loss.backward()
    optimizers[agent_idx].step()
    return loss

def create_memory(agent_count):
    # Start with the "smiles" and "total" columns
    columns = ["smiles"]
    # Add columns for each agent based on the agent count
    for i in range(agent_count):
        columns.append(str(i)) 
    columns.append("total") 
    # Create the DataFrame with the dynamically determined columns
    memory = pd.DataFrame(columns=columns)
    
    return memory
