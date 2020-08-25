#
#from ase import Atoms
#from ase.io import read
#from ase.db import connect
#
#from schnetpack.utils import load_model
#from schnetpack.datasets import *

import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import os, warnings
index_warning = 'Converting sparse IndexedSlices'
warnings.filterwarnings('ignore', index_warning)

sns.set()
sns.set_style("darkgrid")
sns.set_context("paper")
# path definitions
#model_schnet = load_model("./mof5_model_hdnnp_forces_v4/best_model")
##model_schnet = load_model("./ethanol_model/best_model")
#properties = ["cohesive_E_perAtom"]#, "forces"]  # properties used for training
#

csv_file_name = "qm_sch_SP_energies.csv"
column_names = ["FileNames", "qm_SP_energies", "schnet_SP_energies", "Error"]
df_data = pd.read_csv(csv_file_name)[column_names]
#df = df_data.astype(str).groupby("FileNames").agg(";".join)
mof5_f1 = df_data.loc[df_data["FileNames"].str.contains("mof5_f1")]
print(mof5_f1.head())

def plot_linear_reg(df_data, column_names):
    edges = [-5.5, -4.0]
    linreg = stats.linregress(df_data[column_names[1]], df_data[column_names[2]])
    ax = df_data.plot.scatter(x=column_names[1],
                        # xlim=(edges[0], edges[1]),
                        # ylim=(edges[0], edges[1]),
                         y=column_names[2],
                         #colormap='viridis',
                        )
    ax.plot(edges, edges, ls="-", color="b")
    plt.text(-2, -3.5, "R^2=%.2f"%linreg.rvalue)
    plt.show()
    #plt.savefig("%s_QM_E.png"%keyword)
#plot_linear_reg()

def plot_error(df_data, column_names):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_ylim(0, 1)
    maker_types = ["v", "8", "s", "o", "x"]
    for i in range(5):
        df_frag = df_data.loc[df_data["FileNames"].str.contains("mof5_f%s"%i)]
        ax.scatter(x=df_frag[column_names[1]], y=df_frag[column_names[3]], marker=maker_types[i])
    #line2, =  ax.plot(df["Validation loss"], label= "Validation loss")

    #palette = sns.color_palette("bright", 2)
    #sns.scatterplot(ax=ax, x=df_data[column_names[1]], y="", markers=True,
    #             palette=palette, data=pd.melt(df_data, "FileNames"), linewidth=0, alpha = 0.7)
    #ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()
plot_error(df_data, column_names)
