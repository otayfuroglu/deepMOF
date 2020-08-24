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

df = pd.read_csv("./from_truba/mof5_model_hdnnp_forces_1/log.csv")[["Time", "Train loss", "Validation loss"]]
#df2 = data=pd.melt(df, "Time")
#print(df2.head())

#Validation loss
def plot_losses():
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 10)
    #line1, =  ax.plot(df["Train loss"], label="Train loss")
    #line2, =  ax.plot(df["Validation loss"], label= "Validation loss")

    # same result with seaborn
    palette = sns.color_palette("bright", 2)
    sns.lineplot(ax=ax, x="Time", y="value",  hue='variable', markers=True,
                 palette=palette, data=pd.melt(df, "Time"))
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()
plot_losses()
