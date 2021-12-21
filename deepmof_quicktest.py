#
__author__ = 'otayfuroglu'
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from schnetpack import Properties
from schnetpack.datasets import AtomsData
from schnetpack.environment import AseEnvironmentProvider

import  ase
from ase import units
from ase.io import read, write
from ase.optimize import BFGS, LBFGS

import torch

# Get device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read MOF from cif file
irmof1 = read("./IRMOFSeries/IRMOF1.cif")

# Load model
model = load_model("./NNPotentials/deepMOF_v1/best_model")

# Get calculator
calculator = SpkCalculator(
    model, device=device,
    energy=Properties.energy,
    forces=Properties.forces,
    environment_provider=AseEnvironmentProvider(cutoff=6.0)
)

# Set calculator
irmof1.set_calculator(calculator)

# Calculate energy
ei = irmof1.get_potential_energy()
print("Initial Energy: ",ei)

# Optimize MOF
print("Optimizing...")
dyn = LBFGS(irmof1)
dyn.run(fmax=0.05)
print("Optimization finished")

# Calculate energy
ef = irmof1.get_potential_energy()
print("Final Energy: ",ef)

