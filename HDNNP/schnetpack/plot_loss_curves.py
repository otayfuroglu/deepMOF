#
from ase.db import connect
import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
sns.set()
sns.set_style("darkgrid")
sns.set_context("paper")

def plot():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 20)
    df = pd.read_csv("./mof5_model_hdnnp_forces_v4_mof5_f1/log.csv")#[["Time", "Train loss", "Validation loss"]]
    column_names = df.columns.values
    n_epoch = df.shape[0]
    x = range(n_epoch)
    for column_name in column_names[2:]:
        ax.plot(x, df[column_name], label=column_name)
        ax.legend()

    #ax.plot(x, df["Train loss"], label="Train loss")
    #ax.plot(x, df["Validation loss"], label= "Validation loss")
    #ax.plot(x, df["MAE_cohesive_E_perAtom"], label= "Error")
    # same result with seaborn
    #palette = sns.color_palette("bright", 2)
    #sns.lineplot(ax=ax, x="Time", y="value",  hue='variable', markers=True,
    #             palette=palette, data=pd.melt(df, "Time"))
    #ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()

def dynamic_plot():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ##ax.set_ylim(0, 1.2)
    while True:
        df = pd.read_csv("./mof5_model_hdnnp_forces_v4_v2/log.csv")#[["Time", "Train loss", "Validation loss"]]
        n_epoch = df.shape[0]
        x = range(n_epoch)
        ax.plot(x, df["Train loss"], label="Train loss")
        ax.plot(x, df["Validation loss"], label= "Validation loss")
        ax.plot(x, df["MAE_cohesive_E_perAtom"], label= "Error")
        # same result with seaborn
        #palette = sns.color_palette("bright", 2)
        #sns.lineplot(ax=ax, x="Time", y="value",  hue='variable', markers=True,
        #             palette=palette, data=pd.melt(df, "Time"))
        #ax.xaxis.set_major_formatter(ticker.EngFormatter())
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(60)

plot()
