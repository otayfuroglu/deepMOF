#! /truba/home/otayfuroglu/miniconda3/bin/python

import os, io
import numpy as np
import shutil
import argparse

from ase import Atoms
from ase.io import write

from schnetpack.md.utils import HDF5Loader
from schnetpack import Properties

from tqdm import trange

def get_trajectory(data, interval, outfilename):
    #get current working directory and make a scratch 
    #directory
    path = './.scratch'
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    #output file name
    outFileName = outfilename + ".xyz"
    #write each structure from the .traj file in .xyz format
    for i in trange(0, len(data.get_positions()), interval):
        #print(i)
        atoms = Atoms(data.get_property(Properties.Z, mol_idx=0), positions=data.get_property(Properties.R, mol_idx=0)[i])
        string = 'structure%03d' % (i,) +'.xyz'
        outStruct = os.path.join(path, string)
        write(outStruct, atoms)
    #combines all optimization structures in one trajectory 
    #file
        inFile = open(os.path.join(path, 'structure%03d' %
                      (i,)  +'.xyz'), 'r')
        fileStr = inFile.read()
        outFile = open(outFileName, 'a')
        outFile.write(fileStr)
    shutil.rmtree(path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Give samefile.hdf5 file name")
    parser.add_argument("-fbase", "--fbase", type=str, required=True, help="give hdf5 file base")
    parser.add_argument("-n", "--interval", type=int, required=True, help="give interval collection of file")

    args = parser.parse_args()
    log_file = os.path.join(args.fbase+".hdf5")
    data = HDF5Loader(log_file)
    get_trajectory(data, interval=args.interval, outfilename=args.fbase)
