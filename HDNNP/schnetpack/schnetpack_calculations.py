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

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.io import read_xyz,  write_xyz

import os

class SchnetpackCalc(object):
    """
    x
    """

    def __init__(self,
                 molecule_path,
                 ml_moldel,
                 working_dir,
                 device="cuda",
                 energy="energy",
                 forces="forces",
                 energy_units="eV",
                 forces_units="eV/Angstrom",
                ):
        #setup dir
        self.working_dir=working_dir
        if not os.path.exists(self.working_dir):            os.makedirs(self.working_dir)
        
        #load molecule
        self.molecule=True
        self._load_molecule(molecule_path)

        # setup calculator
        calculator = SpkCalculator(ml_moldel,
                                   device=device,
                                   energy=energy,
                                   forces=forces,
                                   energy_units=energy_units,
                                   forces_units=forces_units,
                                  )
        self.molecule.set_calculator(calculator)

        # unless initialized, set dynamics to False
        self.dynamics=False

    def _load_molecule(self, molecule_path):
        self.molecule = read_xyz(molecule_path)

    def save_molecule(self, name, file_format="xyz", append=False):
        molecule_path = os.path.join(self.working_dir, "%s.%s" % (name, filterwarnings))
        write_xyz(molecule_path, self.molecule, plain=True)

    def calculate_single_point(self):
        self.molecule.energy = self.molecule.get_potential_energy()
        self.molecule.forces = self.molecule.get_forces()
        self.save_molecule("sing_point", file_format="extxyz")

    def optimize(self, fmax=1.0e-2, steps=1000):
        name = "Optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimezer = QuasiNewton(self.molecule,
                                trajectory="%s.traj" % optimize_file,
                                restart="%s.pkl" % optimize_file,
                               )
        optimezer.run(fmax, steps)
        self.save_molecule(name)

