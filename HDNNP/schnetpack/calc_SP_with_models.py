#
import torch
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model

from ase import Atoms
from ase.io import read
from ase.db import connect
from ase.collections import g2

from schnetpack.datasets import *
import schnetpack as spk

import os, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

# path definitions
model_schnet = load_model("./from_truba/mof5_model_hdnnp_forces_1/best_model")
#model_schnet = load_model("./ethanol_model/best_model")
properties = ["energy"]#, "forces"]  # properties used for training


calc_schnet = SpkCalculator(model_schnet, device="cuda", energy="energy", forces="forces", collect_triples=True)


#path_to_db = "data/ethanol.db"
#dataset = spk.datasets.MD17("data/ethanol.db", load_only=properties, molecule="ethanol", collect_triples=True)
path_to_db = "../prepare_data/non_equ_geom_energy_forces.db"

data = connect(path_to_db)
datalist = data.select()

for i, atoms in enumerate(datalist):
    #atoms_for_schnet = atoms.toatoms()
    atoms_for_schnet = data.get_atoms(i+1)
    atoms_for_schnet.set_calculator(calc_schnet)
    if i == 0:
        print("{0:10}{1}\n".format("energy", "predict_schnet"))
    print("{:.5f}: {:.5f}".format(
        atoms.energy,
        #dataset[i]["energy"][0],
        atoms_for_schnet.get_potential_energy()[0],
    ))
    if i == 10:
        break

