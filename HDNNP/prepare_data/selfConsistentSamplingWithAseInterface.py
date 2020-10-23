#! /truba/home/otayfuroglu/miniconda3/bin/python

from calculationsWithAse import AseCalculations

from schnetpack.utils import load_model
from schnetpack.datasets import AtomsData

from ase import units
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz,  write_xyz
from ase.io import read, write
from ase import Atoms

from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

import os
import numpy as np

BASE_DIR = "/truba_scratch/otayfuroglu/deepMOF/HDNNP/"
MODEL1_DIR = BASE_DIR + "schnetpack/rayTune_Behler_v2"
MODEL2_DIR = BASE_DIR + "schnetpack/rayTune_Behler_v4"
MOL_DIR = BASE_DIR + "prepare_data/"
WORKS_DIR = BASE_DIR + "prepare_data/MDWorks"
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

calculation1 = AseCalculations(WORKS_DIR)
calculation1.load_molecule_fromFile(molecule_path)
calculation1.adjust_calculator(properties=properties, ml_moldel=model1_schnet, device="cuda:0")

# MD with ase-schnet model interfaces

traj_path = os.path.join(WORKS_DIR, "%s.traj" %name)
if not os.path.exists(traj_path):
    print("obtaing the non equvalent geometries")
    calculation1.init_md(
            name=name,
            time_step=0.5,
            temp_init=100,
            temp_bath=100,
            reset=False,
            interval=1,
     )
    calculation1.run_md(2000) 

calculation2 = AseCalculations(WORKS_DIR)

traj_file = Trajectory(traj_path)

for i, atoms in enumerate(traj_file):
    break
    Ennp1 = np.array(atoms.get_potential_energy()).squeeze(0)
    calculation2.load_molecule_fromAseatoms(atoms)
    calculation2.adjust_calculator(properties=properties, ml_moldel=model2_schnet, device="cuda:0")
    Ennp2 = np.array(calculation2.get_potential_energy()).squeeze(0)

    if np.abs(Ennp1 - Ennp2) > 0.1:
        outFile = "%s/%s_fromSampling_"%(SAMPLES_DIR, fregName)+"{0:0>5}".format(i)+".xyz"
        print(Ennp1, Ennp2)
        print("This configuration is added to samples")

        species = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        pose = Atoms(species, positions=positions)
        write(outFile, pose, format="xyz")

