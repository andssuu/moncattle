import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("moncattle/data/lomba.csv")
    data = df.iloc[:, :4]
    labels = df.iloc[:, -1]
    hours = df.loc[:, "horario"]
    dates = df.loc[:, "data"]
    #datetime.strptime('250515:131911.203', '%d%m%y:%H%M%S.%f')
    LE = LabelEncoder()
    encode_labels = LE.fit_transform(labels)
    fig = plt.figure()
    colors = ['red', 'green', 'blue', 'purple']
    plt.plot(range(data.acc_x.size), data.acc_x, label=labels.values)
    plt.plot(range(data.acc_y.size), data.acc_y, label=labels.values)
    plt.plot(range(data.acc_z.size), data.acc_z, label=labels.values)
    plt.legend()
    plt.show()

    # fig = plt.figure()
    # i = 0
    # colors = ['red', 'green', 'blue', 'purple']
    # for name, group in zip(encode_labels, data.values):
    #     plt.plot(i, float(group[1]),
    #              marker="o", linestyle="", c=colors[name], label=LE.inverse_transform([name])[0])
    #     plt.plot(i, float(group[2]),
    #              marker="o", linestyle="", c=colors[name], label=LE.inverse_transform([name])[0])
    #     plt.plot(i, float(group[3]),
    #              marker="o", linestyle="", c=colors[name], label=LE.inverse_transform([name])[0])
    #     i += 1
    #     # LE.inverse_transform(0)
    # plt.legend()
    # plt.show()
