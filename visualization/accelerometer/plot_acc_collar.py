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
    data = df[df.id_colar == base].iloc[:, 1:4]
    labels = df[df.id_colar == base].iloc[:, -1]
    hours = df[df.id_colar == base].loc[:, "horario"]
    dates = df[df.id_colar == base].loc[:, "data"]
    _dates = [datetime.strptime('{}:{}'.format(_d, _h), '%d%m%y:%H%M%S.%f')
              for _d, _h in zip(dates, hours)]
    LE = LabelEncoder()
    encode_labels = LE.fit_transform(labels)
    axs = data.plot()
    plt.title('Colar {}'.format(base), fontsize=20)
    axs.figure.canvas.set_window_title(
        'plot accelerometer {}'.format(base.lower()))
    plt.legend(loc="best")
    plt.show()
