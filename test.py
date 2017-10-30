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
from scipy.sparse import dia_matrix, csr_matrix
from scipy.sparse.csgraph import dijkstra
# https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.sparse.csgraph.dijkstra.html#scipy.sparse.csgraph.dijkstra
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, scale
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.cross_decomposition import CCA
from csv_read import load_extracted_features, load_graph, load_seed, load_csv


preload = load_csv('HopeThisWorks.csv')
if preload is not None:
    np.savetxt('Graph_Matrix_Edge_Distance.csv', np.asarray(preload), delimiter=",", fmt='%d')
else:
    X = load_extracted_features(subset = range(6000))
    assert X.shape == (6000, 1084)
    matrix = load_graph(shape_match=True)
    seeds = load_seed()
    output = dijkstra(csr_matrix(matrix), directed=False)
    out_max = np.nanmax(output) + 1
    print('Going to change NaN to {}'.format(out_max))
    output[np.isnan(output)] = out_max
    print('Done: output shape {}'.format(output.shape))
    np.savetxt('Graph_Matrix_Edge_Distance.csv', np.asarray(output), delimiter=",", fmt='%d')







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
