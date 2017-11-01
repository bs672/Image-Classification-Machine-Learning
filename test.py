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
from csv_read import load_extracted_features_PCA, load_graph, load_seed, load_csv, load_extracted_features
from Comp import CCACluster
from models import SiameseModel, Classifier


# Sanity
best = load_csv('AllLabelsCCAGraphPred-8-500-500viaMax.csv') # 58.5

features = load_csv('Graph_Matrix_Path_Distance.csv')
features = SpectralEmbedding(n_components=100, affinity='precomputed', n_jobs=-1).fit_transform(features)
# assert features is not None
test = SiameseModel(features, load_seed(), "PathDistance", batch_size = 256)
test.create_model()
test.fit()
test.evaluate()
labels = test.predict()

classifier_features = load_extracted_features_PCA(k=10)
if labels.shape == (10000,2):
    print('Reducing labels')
    labels = labels[:6000]
test2 = Classifier(classifier_features, labels, "PathDistanceClassifier", batch_size = 256)
test2.fit()
output = test2.predict()

sim = 0
for i in range(output.shape[0]):
    assert (best[i][0] == output[i][0])
    sim += int(best[i][1] == output[i][1])

print(sim/output.shape[0]))


# e_Dist = load_csv('Graph_Matrix_Edge_Distance.csv')
# f_Dist = load_graph(shape_match=True, g_type='dist')
#
# # Running with no preembedding
# # Fails bad
# # ccacluster = CCACluster(f_Dist[:,:6000], e_Dist, k_PCA = 6000, k_SE=6000)
# # cca = CCA(n_components=8)
# # cca.fit(f_Dist[:6000], e_Dist[:6000])
# # ccacluster.cca_predictions = cca.transform(f_Dist[6000:])
# # ccacluster.save_predictions()
#
# e_Dist_spec = SpectralEmbedding(n_components=100, affinity='precomputed', n_jobs=-1).fit_transform(e_Dist)
# f_Dist_spec = SpectralEmbedding(n_components=100, affinity='precomputed', n_jobs=-1).fit_transform(f_Dist)
# # print(f_Dist_spec.shape)
# ccacluster = CCACluster(f_Dist_spec, e_Dist_spec, k_PCA = 100, k_SE=100)
# # ccacluster.save_predictions()
#
# print('Now with 10 components')
# ccacluster.k_CCA = 10
# ccacluster.cca_predictions = None
# ccacluster.save_predictions()
