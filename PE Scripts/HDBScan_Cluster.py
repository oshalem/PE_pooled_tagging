import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import hdbscan
import os, sys


def Cycle_Colors(_n, color_num=None):
    # color_num: Number of colors through
    # n: Length of data

    if color_num == None:
        color_num = _n

    _colors = plt.cm.hsv(np.linspace(0, 1, color_num))
    _colors_rep = _colors.copy()
    for i in range(int(np.ceil(_n / color_num)) - 1):
        _colors_rep = np.concatenate((_colors_rep, _colors), axis=0)

    return _colors_rep


df = pd.read_csv('labels_1500cpg_ED-64_EN-512_embedded.csv')

df['cluster'] = hdbscan.HDBSCAN(min_cluster_size=min_clus_size, cluster_selection_epsilon=float(eps)).fit_predict(df[['X', 'Y']])

unq = np.unique(df['cluster'])
colors = Cycle_Colors(len(unq) - 1)
plots = np.empty([len(unq)], dtype=object)
plt.figure(figsize=(20, 10))
for p, c in (enumerate(unq)):
    if c < 0:
        color = [0.65, 0.65, 0.65, 1]
    else:
        color = colors[c]

    idx = np.where(df['cluster'] == c)[0]
    X = df['X'].iloc[idx]
    Y = df['Y'].iloc[idx]
    plots[p] = plt.scatter(X, Y, s=0.5, color=color, edgecolors='none')
    if c >= 0:
        plt.text(np.mean(X), np.mean(Y), s=c, fontsize=12)

title_name = 'min-cluster-size-' + str(min_clus_size) + '_epsilon-' + str(eps)
plt.title(title_name)
plt.legend(plots, unq, fontsize=8, markerscale=8)

"""
# Save Image
plt.savefig(os.path.join('HDBScan_Clusters', title_name + '.png'))
plt.savefig(os.path.join(title_name + '.png'))
plt.clf()
plt.close()
"""

plt.show()
