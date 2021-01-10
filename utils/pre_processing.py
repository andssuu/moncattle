import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv("moncattle/data/cows.csv")

#data = df.loc[:, :"velocidadeDeslocamento"]
#data = df.iloc[:, :-1]
#data = df.iloc[:, :4]
data_a_2 = df[df.id_colar == "A2"].iloc[:, :4]
label_a_2 = df[df.id_colar == "A2"].iloc[:, -1]

# a_3 = df[df.id_colar == "A3"]

# data[data.id_colar == "B2"]
# data[data.id_colar == "B3"]

# data[data.id_colar == "C3"]
# data[data.id_colar == "C4"]

# data[data.id_colar == "D1"]
# data[data.id_colar == "D2"]
# data[data.id_colar == "D3"]
# data[data.id_colar == "D4"]

#labels = df.loc[:, "comportamento"]
labels = df.iloc[:, -1]


_labels = label_a_2
_data = data_a_2
LE = LabelEncoder()
_labels = LE.fit_transform(_labels)
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(_data.acc_x, _data.acc_y,
             _data.acc_z, c=_labels)
_labels = label_a_2
_data = data_a_2
LE = LabelEncoder()
_labels = LE.fit_transform(_labels)

fig2 = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter3D(_data.acc_x, _data.acc_y,
             _data.acc_z)
plt.legend()
plt.show()
