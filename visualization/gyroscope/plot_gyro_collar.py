import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    bases = ["A2", "A3", "B2", "B3", "C3", "C4", "D1", "D2", "D3", "D4"]
    base = bases[1]
    df = pd.read_csv("moncattle/data/lomba.csv", float_precision='high')
    data = df[df.id_colar == base].iloc[:, 7:10]
    ax = data.plot()
    ax.set_ylabel("°/s", fontsize=15)
    plt.title('Colar {}'.format(base), fontsize=20)
    ax.figure.canvas.set_window_title(
        'plot gyroscope {}'.format(base.lower()))
    plt.legend(loc="best")
    plt.show()
