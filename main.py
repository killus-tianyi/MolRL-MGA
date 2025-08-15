"""
Multi-Agent Reinforcement Learning for Molecular Generation
This module implements a multi-agent reinforcement learning approach for generating
molecules using SMILES notation. The system uses multiple agents with experience
replay and crossover operations to optimize molecular properties.
"""
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
from training import scoring, RL_train, create_memory
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpo', type=str, default='MPO task', help='DPO task-related parameter')
    parser.add_argument('--agents_num', type=int, default=4, help='Number of agents')
    parser.add_argument('--replay', type=int, default=4, help='Replay value')
    parser.add_argument('--memory_size', type=int, default=1000, help='Size of the memory buffer')
    parser.add_argument('--crossoverPair', type=int, default=32, help='Number of pairs for crossover')
    parser.add_argument('--sigma', type=int, default=120, help='the impact of the score function')
    parser.add_argument('--segam2', type=float, default=0.2, help='similarity encourage')
    # Basic configs
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=600, help='Maximum number of training steps')
    parser.add_argument('--min_steps', type=int, default=20, help='Minimum number of training steps')
    parser.add_argument('--max_score', type=float, default=0.95, help='Maximum score')
    parser.add_argument('--min_score', type=float, default=0.4, help='Minimum score')
    parser.add_argument('--max_oracle_calls', type=int, default=1000000, help='Maximum number of oracle calls')
    parser.add_argument('--freq_log', type=int, default=1000, help='Frequency of logging')
    parser.add_argument('--type', type=str, default='geometric_mean', help='Type of metric to use')
    # Logger and save configs
    parser.add_argument('--slm_csv_prefix', type=str, default='slm_smiles', help='Prefix for the CSV output')
    parser.add_argument('--result_dir', type=str, default='./results/', help='Diectory to save results')
    # Small language model configs
    parser.add_argument('--prior_file', type=str, default='pretrained rnn or lstm network for prior model', help='Path to prior file')
    parser.add_argument('--agent_file', type=str, default='pretrained rnn or lstm network for agent', help='Path to agent file')
    # Strategy config
    parser.add_argument('--method', type=str, default='simple', help='RL strategy method')
    parser.add_argument('--warmup_circle', type=int, default=1000, help='Warmup circle (not used yet)')
    parser.add_argument('--optim_on', type=str, default='slm', choices=['slm', 'llm', 'both'], help='Optimization target')
    # Parsing the arguments
    return parser.parse_args()

# Ensure the directory exists
os.makedirs('output', exist_ok=True)
def main(args):
    print(f"Starting training of Reinforcement Learning")
    # Load the initial RNN model
    prior_model_filename = os.path.abspath(args.prior_file)
    agent_model_filename = os.path.abspath(args.agent_file)

    # Initiate the Prior model
    prior = create_model(prior_model_filename, device=args.device)
    disable_gradients(prior.network)

    # Initiate Multi-agents
    optimizers = []
    agents = []
    # Create multiple agents
    for i in range(args.agents_num):
        # 1. basic Architecture
        agents.append(create_model(agent_model_filename, device=args.device)) 
        # 2. Tailor 's Optimizer
        optimizers.append(torch.optim.Adam(agents[i].get_network_parameters(), lr=args.lr))

    # Create the memory storing high(average) score molecule and file to store the all score
    memory = create_memory(args.agents_num) 
    # Start running process
    for epoch in tqdm(range(args.max_steps)):
        for i in range(args.agents_num):
            # Sample a bunch of the smiles from a agent
            slm_generated_smiles, _ = agents[i].sample_smiles(num=args.batch_size, batch_size=args.batch_size)

            # Scoring smiles
            try:
                _, states = validate_smiles(slm_generated_smiles)
                scores_slm, valid_smiles = scoring(slm_generated_smiles, states, args.dpo)
            except(ValueError):
                continue

            # Update memory 
            score_df = pd.DataFrame(scores_slm)
            current_score = score_df[str(i)].tolist()
            score_df['smiles'] = valid_smiles
            score_df = score_df[['smiles'] + [str(i) for i in range(args.agents_num)] + ['total']]
            memory = pd.concat([memory, score_df], ignore_index = True)

            # Experience replay
            if args.replay > 0:
                s = min(len(memory), args.replay)
                experience = memory.head(5 * args.replay).sample(s)
                experience = experience.reset_index(drop=True)
                valid_smiles += list(experience["smiles"])
                current_score += list(experience[str(i)])
            # Update the model's parameters
            loss = RL_train(args, prior, i ,agents, optimizers, valid_smiles, current_score)
            tqdm.write(f"Agent: {i}, Epoch {epoch+1}, Loss: {loss.item():.4f}")
            parent =  memory.head(args.crossoverPair * 2)
            parent1 = parent.iloc[::2].head(args.crossoverPair)["smiles"].tolist()
            parent2 = parent.iloc[1::2].head(args.crossoverPair)["smiles"].tolist()
            childs = []
            for i in range(args.crossoverPair):
                child = crossover(parent1[i], parent2[i])  
                if (child == None):
                    pass
                else:
                    childs.append(child)
            _, states = validate_smiles(childs)
            scores_slm, valid_smiles = scoring(childs, states, args.dpo)
            # Update memory 
            score_df = pd.DataFrame(scores_slm)
            score_df['smiles'] = valid_smiles
            score_df = score_df[['smiles'] + [str(i) for i in range(args.agents_num)] + ['total']]
            memory = pd.concat([memory, score_df], ignore_index = True)
            # Update the memory
            memory = memory.drop_duplicates(subset=["smiles"])
            memory = memory.sort_values('total', ascending=False)
            memory = memory.reset_index(drop=True)

        # Check if the memory has over the limit
        if len(memory) > args.memory_size:
            memory = memory.head(args.memory_size)       
        
if __name__ == "__main__":
    # Set the parameter
    args = parse_args()
    # Global set the default device
    torch.set_default_device(args.device)
    # Set random seeds for reproducibility
    set_random_seeds()
    # Run main function
    main(args)
