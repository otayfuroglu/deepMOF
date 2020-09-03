#
from calculations_withAse import AseCalculations

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

from ase.calculators.orca import ORCA

from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton

import os

def orca_calculator(label, n_cpu, initial_gbw=['','']):
    return ORCA(label=label,
                   maxiter=2000,
                   charge=0, mult=1,
                   orcasimpleinput='SP PBE D4 DEF2-SVP DEF2/J RIJDX MINIPRINT NOPRINTMOS NOPOP' + ' ' + initial_gbw[0],
                   orcablocks='%scf Convergence sloppy \n maxiter 300 end \n %pal nprocs ' + str(n_cpu) + ' end' + initial_gbw[1]
                   )

model_schnet = load_model("./mof5_model_hdnnp_forces_v4_mof5_f1/best_model")
properties = ["total_E", "forces"]  # properties used for training
#properties = ["energy", "forces"]  # properties used for training

file_base = "mof5_f1"
molecule_path="../prepare_data/mof_fragments/%s.xyz" %file_base
working_dir="./test"

#path_to_db = "../prepare_data/non_equ_geom_energy_coh_energy_forces_withORCA_v4.db"
#data = AtomsData(path_to_db)
#db_atoms = data.get_atoms(0)

calculation = AseCalculations(working_dir,
                              molecule_path=molecule_path
                              #db_atoms=db_atoms,
                             )
calculation.adjust_calculator(properties=properties, ml_moldel=model_schnet)
#calculation.optimize()
calculation.print_calc()
calculation.adjust_calculator(properties=properties, calculator=orca_calculator(file_base, 4))
calculation.print_calc()

def test():
    calc_schnet = SpkCalculator(model_schnet, device="cuda",
                                energy=properties[1],
                               # forces="forces",
                                collect_triples=True)
    db_atoms.set_calculator(calc_schnet)
    print(db_atoms.get_potential_energy())
#test()
