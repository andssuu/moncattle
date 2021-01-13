import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    bases = ["A2", "A3", "B2", "B3", "C3", "C4", "D1", "D2", "D3", "D4"]
    base = bases[0]
    df = pd.read_csv("moncattle/data/lomba.csv", float_precision='high')
    data = df[df.id_colar == base].iloc[:, 7:10]
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.canvas.set_window_title(
        'histograma giroscopio {}'.format(base.lower()))
    fig.text(0.5, 0.04, '°/s', ha='center', fontsize=15)
    fig.text(0.07, 0.5, 'Frequência Absoluta',
             va='center', rotation='vertical', fontsize=15)
    [ax.hist(data.iloc[:, i], edgecolor='black', alpha=0.7)
     for i, ax in enumerate(axs)]
    [ax.set_title(t, fontsize=15) for t, ax in zip(data.columns, axs)]
    fig.suptitle('Colar {}'.format(base), fontsize=20)
    [ax.tick_params(labelsize=12) for ax in axs]
    plt.show()
