import math
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fig, axs = plt.subplots(3, sharex=True, sharey=True)
    fig.canvas.set_window_title('series all')
    fig.suptitle('Todos Colares', fontsize=20)
    fig.text(0.07, 0.5, '',
             va='center', rotation='vertical', fontsize=15)
    [ax.tick_params(labelsize=12) for ax in axs]
    df = pd.read_csv("moncattle/data/lomba.csv", float_precision='high')
    axs[0].set_title('Acelerômetro', fontsize=15)
    axs[0].plot(df.iloc[:, 1:4], alpha=0.7)
    axs[1].set_title('Magnetômetro', fontsize=15)
    axs[1].plot(df.iloc[:, 4:7], alpha=0.7)
    axs[2].set_title('Giroscópio', fontsize=15)
    _ = axs[2].plot(df.iloc[:, 7:10], alpha=0.7)
    plt.legend(iter(_), ('x', 'y', 'z'))
    # axs[2].legend()
    plt.show()
