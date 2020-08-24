
from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton
from ase.calculators.orca import ORCA

from ase import Atoms
from ase.io import read
from ase.db import connect

import os


def calc_opt(atoms, calc_tor):
    atoms.set_calculator(calc_tor)
    opt = QuasiNewton(atoms)
    opt.run(fmax=0.1, steps=1000)
    return atoms.get_potential_energy()

def calc_SP(atoms, calc_tor):
    atoms.set_calculator(calc_tor)
    return atoms.get_potential_energy()


def orca_calculator(multiplicity, label, n_proc):
    return ORCA(label="ase_orca_%s"%label,
              maxiter=2000,
              charge=0, mult=multiplicity,
              orcasimpleinput='RPW86PBE def2-TZVP PAL%d'%n_proc,
              orcablocks='%scf Convergence normal \n maxiter 300 \n maxcore 1000 end'
              )


fragment_path = "../prepare_data/fragments"
file_names = os.listdir("../prepare_data/fragments")

def calc_free_atom_energy():
    multiplicity_atoms = {"H": 2, "C": 3, "O": 3, "Zn": 1}
    free_atoms_energies = {}
    for chemical_symbol, multiplicity in multiplicity_atoms.items():
        free_atoms_energies[chemical_symbol] = calc_SP(Atoms(chemical_symbol), orca_calculator(multiplicity_atoms[chemical_symbol], chemical_symbol, 4))
    return free_atoms_energies

def calc_cohesive_E(atoms):
    free_atoms_energies = calc_free_atom_energy()
    chemical_symbols = atoms.get_chemical_symbols()
    chemical_symbols_numbers = {i:chemical_symbols.count(i) for i in chemical_symbols}
    free_energies_all_atoms = 0.0
    for chemical_symbol, number_of_atoms in chemical_symbols_numbers.items():
        free_energies_all_atoms += number_of_atoms * free_atoms_energies[chemical_symbol]
    total_E = calc_SP(atoms, orca_calculator(1, atoms.get_chemical_formula(), 4))
    os.system("rm ase_orca_*" %chemical_symbols )
    return (total_E - free_energies_all_atoms) / len(atoms)

atoms = read("%s/%s" %(fragment_path, "mof5_f3.xyz"))
#chemical_symbols = atoms.get_chemical_symbols()
#chemical_symbols_numbers = {i:chemical_symbols.count(i) for i in chemical_symbols}
#print(chemical_symbols_numbers)
print(calc_cohesive_E(atoms))
