#
"""
!!! This module was created by quoting from Schnetpack!!!

This module provides a ASE calculator class [#ase1]_ for SchNetPack models, as
well as a general Interface to all ASE calculation methods, such as geometry
optimisation, normal mode computation and molecular dynamics simulations.
References
----------
.. [#ase1] Larsen, Mortensen, Blomqvist, Castelli, Christensen, Dułak, Friis,
    Groves, Hammer, Hargus: The atomic simulation environment -- a Python
        library for working with atoms.
            Journal of Physics: Condensed Matter, 9, 27. 2017.

"""

from schnetpack.data.atoms import AtomsConverter
from schnetpack.utils.spk_utils import DeprecationHelper
from schnetpack import Properties
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from schnetpack.datasets import AtomsData

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz,  write_xyz
from ase.io import read

from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

import os

class AseCalculations(object):
    """
    x
    """

    def __init__(self,
                 working_dir,
                 molecule_path=None,
                 db_atoms=None,
                ):

        #setup dir
        self.working_dir=working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        #load molecule
        self.molecule = None
        self._load_molecule(molecule_path, db_atoms)
        #self.molecule = molecule

        # unless initialized, set dynamics to False
        self.dynamics=False

    def adjust_calculator(self,
                          device="cuda",
                          properties=None,
                          calculator=None,
                          ml_moldel=None,
                          energy_units="eV",
                          forces_units="eV/Angstrom",
                         ):

        # setup calculator
        if calculator is None:
            if properties is None:
                raise ValueError("Properties not given !!!")
            calculator = SpkCalculator(ml_moldel,
                                       device=device,
                                       energy=properties[0],
                                       forces=properties[1],
                                       energy_units=energy_units,
                                       forces_units=forces_units,
                                       collect_triples=True
                                      )
        self.molecule.set_calculator(calculator)

    def _load_molecule(self, molecule_path, db_atoms):
        if db_atoms is None:
            self.molecule = read(molecule_path)
        else:
            self.molecule = db_atoms

    def save_molecule(self, name, file_format="xyz", append=False):
        molecule_path = os.path.join(self.working_dir, "%s.%s" % (name, file_format))
        write_xyz(molecule_path, self.molecule, plain=True)

    def calculate_single_point(self):
        self.molecule.energy = self.molecule.get_potential_energy()
        self.molecule.forces = self.molecule.get_forces()
        self.save_molecule("single_point", file_format="extxyz")

    def optimize(self, fmax=1.0e-1, steps=1000):
        name = "Optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimezer = QuasiNewton(self.molecule,
                                trajectory="%s.traj" % optimize_file,
                                restart="%s.pkl" % optimize_file,
                               )
        optimezer.run(fmax, steps)
        self.save_molecule(name)
    def print_calc(self):
        print(self.molecule.get_potential_energy())

model_schnet = load_model("./models/from_truba/mof5_model_hdnnp_forces/best_model")
#properties = ["cohesive_E_perAtom", "energy"]#, "forces"]  # properties used for training
properties = ["energy", "forces"]  # properties used for training

molecule_path="../prepare_data/mof_fragments/mof5_f1.xyz"
working_dir="./test"

#path_to_db = "../prepare_data/non_equ_geom_energy_coh_energy_forces_withORCA_v4.db"
#data = AtomsData(path_to_db)
#db_atoms = data.get_atoms(0)

calculation = AseCalculations(working_dir,
                              molecule_path=molecule_path
                              #db_atoms=db_atoms,
                             )
calculation.adjust_calculator(properties=properties, ml_moldel=model_schnet)
calculation.optimize()
calculation.print_calc()

def test():
    calc_schnet = SpkCalculator(model_schnet, device="cuda",
                                energy=properties[1],
                               # forces="forces",
                                collect_triples=True)
    db_atoms.set_calculator(calc_schnet)
    print(db_atoms.get_potential_energy())
#test()
