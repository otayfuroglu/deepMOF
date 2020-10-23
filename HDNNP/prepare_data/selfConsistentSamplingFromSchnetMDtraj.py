#! /truba/home/otayfuroglu/miniconda3/bin/python

from calculationsWithAse import AseCalculations

from schnetpack.utils import load_model
from schnetpack.datasets import AtomsData
from schnetpack.md.utils import HDF5Loader
from schnetpack import Properties

from ase import units
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz,  write_xyz
from ase.io import read, write
from ase import Atoms

from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

import os
import numpy as np
from tqdm import trange

BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF/HDNNP/"
MODEL1_DIR = BASE_DIR + "schnetpack/models/rayTune_Behler_v3"
MODEL2_DIR = BASE_DIR + "schnetpack/rayTune_Behler_v4"
MOL_DIR = BASE_DIR + "prepare_data/"
WORKS_DIR = BASE_DIR + "prepare_data/calcWorks"
DB_DIR = BASE_DIR + "prepare_data"
SAMPLES_DIR = BASE_DIR +"prepare_data/nonEquGeometriesFromSampling"

if not os.path.exists(SAMPLES_DIR):
    os.mkdir(SAMPLES_DIR)

if not os.path.exists(WORKS_DIR):
    os.mkdir(WORKS_DIR)

model1_path= os.path.join(MODEL1_DIR, "best_model")
model1_schnet = load_model(model1_path)
model2_path= os.path.join(MODEL2_DIR, "best_model")
model2_schnet = load_model(model2_path)
properties = ["energy", "forces"]  # properties used for training

fregName = "mof5_f1"
name = fregName

molecule_path= os.path.join(MOL_DIR, "%s.xyz" %fregName)

db_path = os.path.join(DB_DIR, "nonEquGeometriesEnergyForcesWithORCAFromMD.db")
#data = AtomsData(path_to_db)
#db_atoms = data.get_atoms(0)

device = "cuda:1"
calculation1 = AseCalculations(WORKS_DIR)
calculation2 = AseCalculations(WORKS_DIR)

traj_path = os.path.join(BASE_DIR, "schnetpack/mdFiles/rpmd_simulation.hdf5")
interval = 1
traj_file = HDF5Loader(traj_path)

species = traj_file.get_property(Properties.Z, mol_idx=0)
positions = traj_file.get_property(Properties.R, mol_idx=0)
energies = traj_file.get_property(Properties.energy, mol_idx=0)


for i in range(0, len(traj_file.get_positions()), interval):
    #atoms = Atoms(traj_file.get_property(Properties.Z, mol_idx=0), positions=traj_file.get_property(Properties.R, mol_idx=0)[i])
    position = positions[i]
    atoms = Atoms(species, position)

    # positons is mean of replicas, so we must calc energy again
    calculation1.load_molecule_fromAseatoms(atoms)
    calculation1.adjust_calculator(properties=properties, ml_moldel=model1_schnet, device=device)
    Ennp1 = np.array(calculation1.get_potential_energy()).squeeze(0)

    calculation2.load_molecule_fromAseatoms(atoms)
    calculation2.adjust_calculator(properties=properties, ml_moldel=model2_schnet, device=device)
    Ennp2 = np.array(calculation2.get_potential_energy()).squeeze(0)

    if np.abs(Ennp1 - Ennp2) > 0.1:
        outFile = "%s/%s_fromSampling_"%(SAMPLES_DIR, fregName)+"{0:0>5}".format(i)+".xyz"
        print(energies[i], Ennp1, Ennp2)
        print("This configuration is added to samples")

        pose = Atoms(species, positions=position)
        write(outFile, pose, format="xyz")

    if 10 == i: 
        break
