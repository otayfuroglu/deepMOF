#
import torch
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model
from dftd4 import D4_model
from ase.optimize import BFGS, LBFGS, GPMin, QuasiNewton
from ase.calculators.orca import ORCA

from ase import Atoms
from ase.io import read
from ase.db import connect
from ase.collections import g2

from schnetpack.datasets import *
import schnetpack as spk

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import os, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

# path definitions
model_schnet = load_model("./non_equ_geom_energy_forces_withORCA_v4.db/best_model")
#model_schnet = load_model("./ethanol_model/best_model")
properties = ["cohesive_E_perAtom"]#, "forces"]  # properties used for training
calc_schnet = SpkCalculator(model_schnet, device="cuda",
                            energy=properties[0],
                             # forces="forces",
                            collect_triples=True)
calc_dft4 = D4_model(xc='pbe0')

def calc_opt(atoms, calc_tor):
    atoms.set_calculator(calc_tor)
    dyn = QuasiNewton(atoms)
    dyn.run(fmax=0.1, steps=1000)
    return atoms.get_potential_energy()


#fragment_path = "../prepare_data/fragments"
#file_names = os.listdir("../prepare_data/fragments")

path_to_db = "../prepare_data/non_equ_geom_energy_coh_energy_forces_withORCA_v4.db"
data = connect(path_to_db)
datalist = data.select()

def main():
    qm_opt_energies = []
    sch_opt_energies = []

    #for i, file_name in enumerate(file_names):
    #    atoms = read("%s/%s" %(fragment_path, file_name))
    #    opt_e = calc_opt(atoms)
    for i, atoms in enumerate(datalist):
        mol = atoms.toatoms()

        opt_e = calc_opt(mol, calc_dft4)
        sch_opt_e = calc_opt(mol, calc_schnet)

        qm_opt_energies.append(opt_e)
        sch_opt_energies.append(sch_opt_e[0])

        #if i == 0:
        #    print("{0:10}{1}\n".format("energy", "predict_schnet"))
        #print("{:.5f}: {:.5f}".format(
        #    opt_e,
        #    #dataset[i]["energy"][0],
        #    sch_opt_e,
        #))
        if i == 100:
            break
    df_data = pd.DataFrame()
    df_data["QM_opt_E"] = qm_opt_energies
    df_data["Scnetpcak_opt_E"] = sch_opt_energies
    df_data.to_csv("qm_sch_opt_energies.csv")

def plot_linear_reg():
    df_data = pd.read_csv("./qm_sch_opt_energies.csv")
    edges = [-4.0, -1.5]
    linreg = stats.linregress(df_data["QM_opt_E"], df_data["Scnetpcak_opt_E"])
    ax = df_data.plot.scatter(x="QM_opt_E",
                         xlim=(edges[0], edges[1]),
                         ylim=(edges[0], edges[1]),
                         y="Scnetpcak_opt_E",
                         #colormap='viridis',
                        )
    ax.plot(edges, edges, ls="-", color="b")
    plt.text(-2, -3.5, "R^2=%.2f"%linreg.rvalue)
    plt.show()
    #plt.savefig("%s_QM_E.png"%keyword)

main()
plot_linear_reg()
