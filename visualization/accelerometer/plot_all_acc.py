import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig, axs = plt.subplots(2, 5, sharex=True, sharey=True)
    fig.canvas.set_window_title('plot accelerometer all')
    fig.suptitle('Todos Colares', fontsize=20)
    fig.text(0.07, 0.5, 'g (m/s²)',
             va='center', rotation='vertical', fontsize=15)
    [ax.tick_params(labelsize=12) for _axs in axs for ax in _axs]
    bases = ["A2", "A3", "B2", "B3", "C3", "C4", "D1", "D2", "D3", "D4"]
    df = pd.read_csv("moncattle/data/lomba.csv", float_precision='high')
    # data = df[df.id_colar == base].iloc[:, 1:4]
    # labels = df[df.id_colar == base].iloc[:, -1]
    # hours = df[df.id_colar == base].loc[:, "horario"]
    # dates = df[df.id_colar == base].loc[:, "data"]
    # _dates = [datetime.strptime('{}:{}'.format(_d, _h), '%d%m%y:%H%M%S.%f')
    #           for _d, _h in zip(dates, hours)]
    # LE = LabelEncoder()
    # encode_labels = LE.fit_transform(labels)
    for i, base in enumerate(bases):
        data = df[df.id_colar == base].iloc[:, 1:4]
        plt.title('Colar {}'.format(base), fontsize=20)
        axs[int(i >= 5)][i % 5].plot(
            range(data.shape[0]), data.iloc[:, 0], label='acc_x')
        axs[int(i >= 5)][i % 5].plot(
            range(data.shape[0]), data.iloc[:, 1], label='acc_y')
        axs[int(i >= 5)][i % 5].plot(
            range(data.shape[0]), data.iloc[:, 2], label='acc_z')
        axs[int(i >= 5)][i % 5].set_title(base, fontsize=15)
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()
