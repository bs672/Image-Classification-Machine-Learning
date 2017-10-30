import sys
import os
"""
    Hack allow modules DataSets and Modules to work
    Also allows you to read .csv from either inside DataSets or in the main folder
"""
base = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base,'DataSets'))
sys.path.append(os.path.join(base,'Modules'))
# print(sys.path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import rcca
# Custom implementation of CCA, as scikit support for CCA is not optimal.
# Source: https://github.com/gallantlab/pyrcca
#(Docs: https://www.frontiersin.org/articles/10.3389/fninf.2016.00049/full)

from matplotlib import offsetbox
from time import time
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, scale
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.cross_decomposition import CCA
from csv_read import load_extracted_features, load_graph, load_seed, load_csv

X = load_extracted_features(subset = range(6000))
assert X.shape == (6000, 1084)
matrix = load_graph(shape_match=True)
seeds = load_seed()

# This will make a matrix containing to all points from eachother via simularity
def shortest_path(i, j, seen):
    # computes shortest path from element i to j
    point = (i,j)
    if seen.get(str(point)):
        return seen[str(point)]
    elif (i == j):
        seen[str((i,j))] = 0
        return 0
    elif matrix[i,j]:
        return 1 # same point
    else:
        queue = []
        # Initialize queue
        for neighbor in matrix[i,:]:
            if neighbor:
                seen[str((i,neighbor))] = 1
                queue.append((neighbor,j))

        while queue and seen.get(str(point)) is None:
            (n, j) = queue.pop()
            if seen.get(str((n, j))):
                seen[str(point)] = seen[str((i,n))] + seen[str((n,j))]
            else:
                for neighbor in matrix[n,:]:
                    if neighbor and seen.get(str((n,neighbor))) is None:
                        seen[str((n,neighbor))] = 1
                        queue.append((neighbor,j))
                        if seen.get(str((i,neighbor))) is None:
                            seen[str((i,neighbor))] = seen[str((i,n))] + seen[str((n,neighbor))]

        if seen.get(str(point)):
            return seen[str(point)]
        else:
            return -1

output = np.zeros(matrix.shape, dtype='int')
table = {}
for i in range(output.shape[0]):
    for j in range(i, output.shape[1]):
        p = shortest_path(i, j, table)
        output[i,j] = output[j,i] = p
        print('Got length for {}: {}'.format((i,j), p))
        print('Current dictionary size: {}'.format(len(table)))

np.savetxt('HopeThisWorks.csv', np.asarray(output), delimiter=",", fmt='%d')







# Spec_comp = 2
# correctness = np.zeros(8)
# for n_comp in range(2,10):
#
#     X_pca = load_csv('X_pca_{}.csv'.format(n_comp))
#     if X_pca is None:
#         X_pca = PCA(n_components=n_comp).fit_transform(X)
#         np.savetxt('DataSets/X_pca_{}.csv'.format(n_comp),np.asarray(X_pca), delimiter=",")
#     M_spec = load_csv('MSpec_{}.csv'.format(Spec_comp))
#     if M_spec is None:
#         M_spec = SpectralEmbedding(n_comp=Spec_comp).fit_transform(matrix)
#         np.savetxt('DataSets/MSpec_{}.csv'.format(Spec_comp),np.asarray(M_spec), delimiter=",")
#     cca = CCA(n_components=n_comp)
#     X_cca, M_cca = cca.fit_transform(X_pca[:6000], M_spec)
#     # np.savetxt('DataSets/X_cca_{}.csv'.format(n_comp),np.asarray(X_cca), delimiter=",")
#     # np.savetxt('DataSets/M_cca_{}.csv'.format(Spec_comp),np.asarray(M_cca), delimiter=",")
#     X_pred_cca = cca.predict(X_pca[6000:])
#     # np.savetxt('DataSets/X_pred_cca_{}.csv'.format(n_comp),np.asarray(X_pred_cca), delimiter=",")
#
#     centroids = np.zeros((10,n_comp))
#     for i, label in seeds:
#         centroids[label] += X_cca[i]
#     centroids /= 10
#
#     kmeans = KMeans(n_clusters = 10, init=centroids).fit(X_cca)
#
#     print('For N_Comp {} and Spec_comp {}'.format(n_comp, Spec_comp))
#     count = 0.0
#     for i, label in seeds:
#         count += int(label == kmeans.labels_[i])
#         correctness[n_comp-2] += int(label == kmeans.labels_[i])
#         if not(label == kmeans.labels_[i]):
#             print('Seed {} is mislabeled'.format(i))
#             print('{} should be {}'.format(kmeans.labels_[i], label))
#     print('Correctness : {} '.format( count/60. ))
#
#     # scores = np.zeros(10)
#     # for i, label in seeds:
#     #     scores[label] += np.linalg.norm(centroids[label] - X_cca[i])
#     #
#     # print('For N_Comp {}'.format(n_comp))
#     # print('Range {} - {}'.format(np.min(scores),np.max(scores)))
#     # for i, s in enumerate(scores):
#     #     print('Score for label {}, {}'.format(i,s))
# print(correctness/60.)
