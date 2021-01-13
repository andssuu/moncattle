import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(
        zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))


if __name__ == "__main__":
    bases = ["A2", "A3", "B2", "B3", "C3", "C4", "D1", "D2", "D3", "D4"]
    base = bases[1]
    df = pd.read_csv("moncattle/data/lomba.csv", float_precision='high')
    data = df[df.id_colar == base].iloc[:, :4]
    labels = df[df.id_colar == base].iloc[:, -1]
    hours = df[df.id_colar == base].loc[:, "horario"]
    dates = df[df.id_colar == base].loc[:, "data"]
    _dates = [datetime.strptime('{}:{}'.format(_d, _h), '%d%m%y:%H%M%S.%f')
              for _d, _h in zip(dates, hours)]
    LE = LabelEncoder()
    encode_labels = LE.fit_transform(labels)
    fig, ax = plt.subplots()
    fig.suptitle('Colar {}'.format(base), fontsize=20)
    colors = ['red', 'green', 'blue', 'purple']
    for name, _t, _acc in zip(encode_labels, _dates, data.values):
        ax.plot(_t, _acc[1],
                marker=".",  c=colors[name], label=LE.inverse_transform([name])[0])
        ax.plot(_t, _acc[2],
                marker=".",  c=colors[name], label=LE.inverse_transform([name])[0])
        ax.plot(_t, _acc[3],
                marker=".",  c=colors[name], label=LE.inverse_transform([name])[0])
    plt.legend()
    #plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    legend_without_duplicate_labels(ax)
    plt.show()
