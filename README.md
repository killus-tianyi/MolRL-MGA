# MolRL-MGA
This repository store the code for the essay Multi-Objective Drug Discovery via Genetic Algorithms and Reinforcement Learning in a Multi-Agent Framework


## Depdencies
```bash
pytorch
rdkit
tqdm
tensorboard
PyTDC
openbabel
numpy
pandas
argparse
```

## Dataset 

The data set: \url{https://www.ebi.ac.uk/chembl/} 

Oracle for GuacaMol Benchmarks: \url{https://www.benevolent.com/} 

Oracle for Target Protein: \url{https://tdcommons.ai/functions/oracles}

## Multi-agent Reinforcement learning
```bash
python main.py --dpo osimertinib --agents_num 4 --sigma 120 --memory_size 500 --segam2 0.2 --max_steps 600
```