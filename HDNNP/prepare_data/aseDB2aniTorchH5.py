#! /home/modellab/miniconda3/bin/python

from ase.io import read, write
from ase.db import connect
import pybel
import numpy as np
import h5py
import os
from tqdm import tqdm
import sys

def atoms2Smiles(atoms):
    fname = "temp.xyz"
    write(fname, atoms)
    mol = next(pybel.readfile("xyz", fname))
    smi = mol.write(format="smi")
    return smi.split()[0].strip()

def aseDb2AniTorchH5(dbDIR, dbBASE, fragName, loopCounter):
    # 
    if os.path.exists(fh5name):
        f = h5py.File(fh5name, "a")
    else:
        f = h5py.File(fh5name, "w")

    grp = f.create_group("%s/%s" % (dbBASE, fragName))

    db = connect("%s/%s_%s.db" %(dbDIR, dbBASE, fragName))
    nRows = db.count()
    db = db.select() # db to rows
    coordinates = []
    energies = []
    forces = []
    i = 0
    for row in tqdm(db, file=sys.stdout, desc="Fragment "+str(loopCounter), total=nRows):
        #if i % 100 == 0:
            #print(i)
        atoms = row.toatoms()
        coordinates.append(atoms.get_positions())
        energies.append([row["energy"]])
        forces.append(row["forces"])
        if i == 0:
            smiles = [ch.encode("utf8") for ch in atoms2Smiles(atoms)]
            species = [ch.encode("utf8") for ch in atoms.get_chemical_symbols()]
        i += 0

    coordinates = np.array(coordinates)
    energies = np.array(energies)
    forces = np.array(forces)
    smiles = np.array(smiles)
    species = np.array(species)

    grp.create_dataset("coordinates", coordinates.shape, data=coordinates)
    grp.create_dataset("energies", energies.shape, data=energies)
    grp.create_dataset("forces", forces.shape, data=forces)
    grp.create_dataset("smiles", smiles.shape, data=smiles)
    grp.create_dataset("species", species.shape, data=species)

def main():
    dbDIR = "./"
    dbBASE = "non_equ_geom_energy_coh_energy_forces_withORCA"

    global fh5name
    fh5name = "%s/%s.h5" % (dbDIR, dbBASE)

    if os.path.exists(fh5name):
        os.remove(fh5name)

    for i in range(1, 6):
        aseDb2AniTorchH5(dbDIR, dbBASE, "mof5_f"+str(i), loopCounter=i)

if __name__ == "__main__":
    main()
