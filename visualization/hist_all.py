import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("moncattle/data/lomba.csv", float_precision='high')
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.canvas.set_window_title('histogram all')
    fig.suptitle('Todos Colares', fontsize=20)
    fig.text(0.5, 0.04, '', ha='center', fontsize=15)
    fig.text(0.07, 0.5, 'Frequência Absoluta',
             va='center', rotation='vertical', fontsize=15)
    [ax.tick_params(labelsize=12) for ax in axs]
    axs[0].set_title('Acelerômetro', fontsize=15)
    axs[0].hist(df.iloc[:, 1:4],
                edgecolor='black', alpha=0.7)
    axs[1].set_title('Magnetômetro', fontsize=15)
    axs[1].hist(df.iloc[:, 4:7],
                edgecolor='black', alpha=0.7)
    axs[2].set_title('Giroscópio', fontsize=15)
    axs[2].hist(df.iloc[:, 7:10],
                edgecolor='black', alpha=0.7, label=["x", "y", "z"])
    plt.legend()
    plt.show()
