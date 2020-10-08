#! /home/modellab/miniconda3/bin/python

from schnetpack.datasets import AtomsData
import os
import glob
import numpy as np
import argparse
from tqdm import trange, tqdm
from ase.db import connect
import sys


def getFragmentFromAseDb(dbDIR, dbBASE, nFragment):

    fragNames = [fragBase+str(i) for i in range(1, nFragment+1)]
    for loopCounter, fragName in enumerate(fragNames):
        db = connect("%s/%s.db" %(dbDIR, dbBASE))
        nRows = db.count()
        db = db.select() # db to rows

        new_db_path = "%s/%s_%s.db" %(dbDIR, dbBASE, fragName)
        new_db = AtomsData(new_db_path,
                           available_properties=["energy", "forces"]
                          )
        names = []
        mols = []
        property_list = []
        i = 0
        for row in tqdm(db, file=sys.stdout, desc=fragName, total=nRows):
            file_base = row["name"]
            if fragName in file_base:
                atoms = row.toatoms()
                mols.append(atoms)
                names.append(file_base)

                energy = row["energy"]
                energy = np.array([energy], dtype=np.float32)

                forces = row["forces"]
                forces = np.array(forces, dtype=np.float32)

                property_list.append({"energy": energy,
                                     "forces": forces,
                                })
        new_db.add_systems(mols, names, property_list)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-dbDIR", "--dbDIR", type=str, required=False, help="give ase db file directory")
    parser.add_argument("-dbBASE", "--dbBASE", type=str, required=True, help="give  ase db file name without extention")
    parser.add_argument("-nf", "--nFragment", type=int, required=True, help="give  number of fragment")

    args = parser.parse_args()
    if args.dbDIR is None:
        dbDIR = "./"
    else:
        dbDIR = args.dbDIR
    dbBASE = args.dbBASE
    nFragment = args.nFragment
    global fragBase
    fragBase = "mof5" + "_" + "f"
    existsFiles = glob.glob("%s/%s_%s*.db" %(dbDIR, dbBASE, fragBase))
    assert len(existsFiles) == 0, "There are db files for gived fragments, remove this files:\n %s" % existsFiles
    getFragmentFromAseDb(dbDIR, dbBASE, nFragment)
