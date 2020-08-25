#
import torch
from schnetpack.interfaces import SpkCalculator
from schnetpack.utils import load_model

from ase import Atoms
from ase.io import read
from ase.db import connect
from ase.collections import g2

from schnetpack.datasets import *
import schnetpack as spk

import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt

import os, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

# path definitions
model_schnet = load_model("./mof5_model_hdnnp_forces_v4/best_model")
#model_schnet = load_model("./ethanol_model/best_model")
properties = ["cohesive_E_perAtom"]#, "forces"]  # properties used for training


calc_schnet = SpkCalculator(model_schnet, device="cuda",
                            energy=properties[0],
                           # forces="forces",
                            collect_triples=True)


#path_to_db = "data/ethanol.db"
#dataset = spk.datasets.MD17("data/ethanol.db", load_only=properties, molecule="ethanol", collect_triples=True)
path_to_db = "../prepare_data/non_equ_geom_energy_coh_energy_forces_withORCA_v4.db"
data = AtomsData(path_to_db)

column_names = ["qm_SP_energies", "schnet_SP_energies"]
csv_file_name = "qm_sch_SP_energies.csv"

def main():
    file_names = []
    SP_energies = []
    schnet_SP_energies = []
    for i in range(len(data)):
        #row_for_schnet = atoms.toatoms()
        file_names += [data.get_name(i)]
        mol = data.get_atoms(i)
        mol.set_calculator(calc_schnet)

        coh_E = float(data[i][properties[0]][0])
        coh_schnet_E = float(mol.get_potential_energy()[0])

        SP_energies += [coh_E]
        schnet_SP_energies += [coh_schnet_E]
        print(i)
        #if i == 0:
        #    print("{0:10}{1}\n".format("energy", "predict_schnet"))
        #print("{:.5f}: {:.5f}".format(coh_E, coh_schnet_E))
        #if i == 10:
        #    break

    df_data = pd.DataFrame()
    df_data["FileNames"] = file_names
    df_data[column_names[0]] = SP_energies
    df_data[column_names[1]] = schnet_SP_energies
    df_data["Error"] = SP_energies - schnet_SP_energies
    df_data.to_csv(csv_file_name)

#main()
df_data = pd.read_csv(csv_file_name)
df_data["Error"] = np.array(df_data[column_names[0]]) - np.array(df_data[column_names[1]])
df_data.to_csv(csv_file_name)

def plot_linear_reg():
    df_data = pd.read_csv(csv_file_name)
    edges = [-5.5, -4.0]
    linreg = stats.linregress(df_data[column_names[0]], df_data[column_names[0]])
    ax = df_data.plot.scatter(x=column_names[0],
                        # xlim=(edges[0], edges[1]),
                        # ylim=(edges[0], edges[1]),
                         y=column_names[1],
                         #colormap='viridis',
                        )
    ax.plot(edges, edges, ls="-", color="b")
    plt.text(-2, -3.5, "R^2=%.2f"%linreg.rvalue)
    plt.show()
    #plt.savefig("%s_QM_E.png"%keyword)
#plot_linear_reg()
