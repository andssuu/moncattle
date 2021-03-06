import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("moncattle/data/lomba.csv", float_precision='high')
    fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
    fig.canvas.set_window_title('histogram gyroscope all')
    fig.suptitle('Giroscópio: Todos Colares', fontsize=20)
    fig.text(0.5, 0.04, '°/s', ha='center', fontsize=15)
    fig.text(0.07, 0.5, 'Frequência Absoluta',
             va='center', rotation='vertical', fontsize=15)
    [ax.tick_params(labelsize=12) for _axs in axs for ax in _axs]
    bases = ["A2", "A3", "B2", "B3", "C3", "C4", "D1", "D2", "D3", "D4"]
    for i, base in enumerate(bases):
        data = df[df.id_colar == base].iloc[:, 7:10]
        axs[int(i >= 5)][i % 5].hist(data.iloc[:, 0],
                                     edgecolor='black', alpha=0.7, label='giro_x')
        axs[int(i >= 5)][i % 5].hist(data.iloc[:, 1],
                                     edgecolor='black', alpha=0.7, label='giro_y')
        axs[int(i >= 5)][i % 5].hist(data.iloc[:, 2],
                                     edgecolor='black', alpha=0.7, label='giro_z')
        axs[int(i >= 5)][i % 5].set_title(base, fontsize=15)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()
