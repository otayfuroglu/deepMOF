#   
from ase.io import read, write
from ase.visualize import view
from ase.build import bulk
import numpy as np

from multiprocessing import Pool
from itertools import product
import os

"""ase Atoms object; rattle
Randomly displace atoms.
This method adds random displacements to the atomic positions,
taking a possible constraint into account. The random numbers are drawn
from a normal distribution of standard deviation stdev.
For a parallel calculation, it is important to use the same seed on all processors!
"""

file_path = "./mof_fragments"
non_equ_geom_xyz_path = "non_equ_geom_xyz_files_non_scaled"
if not os.path.exists(non_equ_geom_xyz_path):
    os.mkdir(non_equ_geom_xyz_path)

file_bases = ["mof5_f1", "mof5_f2", "mof5_f3", "mof5_f4", "mof5_f5"]
#file_bases = ["mof5_new_f2", "mof5_new_f3", "mof5_new_f4", "mof5_new_f5", "mof5_new_f6"]
#file_bases = ["mof5_new_single", ]
scaled=False

def scale_atoms_distence(atoms, scale_factor):
    atoms = atoms.copy()
    atoms.center(vacuum=0.0)
    atoms.set_cell(scale_factor * atoms.cell, scale_atoms=True)
    #print(atoms.cell)
    #write("test.xyz", atoms)
    #view(atoms)
    return atoms

#atoms = read("./fragments/mof5_f1.xyz")
#scale_atoms_distence(atoms, 2)


def random_scale_direction(direction):
    return np.random.uniform(0.98 * direction, 1.05 * direction)

def calc_displacement_atom(directions_distance):
    return np.sqrt(sum(direction**2 for direction in directions_distance))


def displaced_atomic_positions(atom_positions):

    while True:
        n_atom_positions = np.array([random_scale_direction(direction)
                                    for direction in atom_positions])
        if calc_displacement_atom(atom_positions - n_atom_positions) <= 0.16:
            return n_atom_positions

def get_non_equ_geo(i, file_name):
    if scaled:
        atoms = read("%s/%s" %(non_equ_geom_xyz_path, file_name))
    else:
        atoms = read("%s/%s" %(file_path, file_name))

    for atom in atoms:
        atom.position = displaced_atomic_positions(atom.position)
    write("{}/{}_{}.xyz".format(non_equ_geom_xyz_path, file_name.replace(".xyz",""), i), atoms)

def main(n_proc):
    #file_bases = ["test0"]

    if scaled:
        for file_base in file_bases:
            atoms = read("%s/%s.xyz" %(file_path, file_base))

            scale_range = (0.96, 1.10)
            scale_step = 0.01
            for i, scale_factor in enumerate(np.arange(scale_range[0], scale_range[1], scale_step)):
                scaled_atoms = scale_atoms_distence(atoms, scale_factor)
                write("{}/{}_{}.xyz".format(non_equ_geom_xyz_path, file_base, i), scaled_atoms)

        file_names = os.listdir(non_equ_geom_xyz_path)
    else:
        print("Don't apply scale procedure")
        file_names = os.listdir(file_path)

    with Pool(n_proc) as pool:
        pool.starmap(get_non_equ_geo, product(range(1000), file_names))

main(8)

