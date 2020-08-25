#
from ase.db import connect
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()
sns.set_style("darkgrid")
sns.set_context("paper")

df = pd.read_csv("./mof5_model_hdnnp_forces_v4_200/log.csv")#[["Time", "Train loss", "Validation loss"]]
n_epoch = df.shape[0]
print(n_epoch)

#df2 = data=pd.melt(df, "Time")
#print(df2.head())

#Validation loss
def plot_losses():
    x = range(n_epoch)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 1.1)
    ax.plot(x, df["Train loss"], label="Train loss")
    ax.plot(x, df["Validation loss"], label= "Validation loss")
    ax.plot(x, df["MAE_cohesive_E_perAtom"], label= "Error")

    # same result with seaborn
    #palette = sns.color_palette("bright", 2)
    #sns.lineplot(ax=ax, x="Time", y="value",  hue='variable', markers=True,
    #             palette=palette, data=pd.melt(df, "Time"))
    #ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()
plot_losses()
