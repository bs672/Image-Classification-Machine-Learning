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
from csv_read import load_extracted_features_PCA, load_graph, load_seed, load_csv, load_extracted_features, load_spectral_embedding
from Comp import CCACluster
from models import SiameseModel, Classifier, FatSiameseModel


# Sanity
oldbest = load_csv('OldBest.csv') # 58.5
newbest = load_csv('NewBEST.csv')
# features = load_csv('NewBESTFeatures.csv')

modelname = "CCAExtractedFeaturesEXP"
fname = "ExtractedFeaturesCCA-EXP"
cca = CCACluster(load_extracted_features_PCA(k=500),
        load_spectral_embedding(k=500, g_type='exp'), k_PCA = 500, k_SE=500, k_CCA=8, fname=fname)
# cca.determine_CCA_components(CCrange=np.arange(1,11))
cca.save_predictions()
features = cca.cca_predictions
# assert features.shape[0] > 5
print('Norm difference with best features, {}'.format(np.linalg.norm(features-load_csv('NewBESTFeatures.csv'))))
# test = SiameseModel(features, load_csv('newseeds.csv'), modelname, batch_size = 256)
test = SiameseModel(features, load_seed(), modelname, batch_size = 256)
test.fit(epochs=10)
test.evaluate()
labels, check = test.predict(split=True)

sim, simold = 0, 0
for i in range(labels.shape[0]):
    assert (newbest[i][0] == labels[i][0])
    sim += int(newbest[i][1] == labels[i][1])
    simold += int(oldbest[i][1] == labels[i][1])

print('Similarity to full labels (71.5): {}'.format(sim/labels.shape[0]))
print('Similarity to full labels (58.5): {}'.format(simold/labels.shape[0]))

labels, check = test.predict(split=True, via_max=False)

sim, simold = 0, 0
for i in range(labels.shape[0]):
    assert (newbest[i][0] == labels[i][0])
    sim += int(newbest[i][1] == labels[i][1])
    simold += int(oldbest[i][1] == labels[i][1])

print('Similarity to full labels (71.5): {}'.format(sim/labels.shape[0]))
print('Similarity to full labels (58.5): {}'.format(simold/labels.shape[0]))
# fname = os.path.join('DataSets', 'SpectralEmbeddingPathDistance.csv')
# trying CCA on SpectralEmbeddingEXP and SpectralEmbeddingDist
# features2 = SpectralEmbedding(n_components=1000, affinity='precomputed').fit_transform(load_csv('Graph_Matrix_Path_Distance.csv'))
# np.savetxt(fname, np.asarray(features2), delimiter=",", fmt='%.4f')
# similarity_2_best = np.zeros(20)
# similarity_2_old = np.zeros(20)
# results_2_best = np.zeros(20)
# results_2_old = np.zeros(20)
# count = 0
# for g_type in ['exp']:
#     for k_PCA in range(30, 10, -1):
#         fname = "FeaturesCCA-GraphDist{}-Graph{}".format(k_PCA, g_type)
#         cca = CCACluster(load_spectral_embedding(k=500, g_type='dist', k_PCA=k_PCA),
#                 load_spectral_embedding(k=500, g_type=g_type), k_PCA = 500, k_SE=500, fname=fname)
#         cca.determine_CCA_components(CCrange=np.arange(1,11))
#         cca.save_predictions()
#         features = cca.cca_predictions
#         assert features is not None and features.shape[0] == 10000
#         # Fat is not better than Regular
#         modelname = "SiameseCCA{}-GraphDist{}-Graph{}".format(cca.k_CCA, k_PCA, g_type)
#         test = SiameseModel(features, load_seed(), modelname, batch_size = 256)
#         if test.model is None: test.fit(epochs=5)
#         test.evaluate()
#         labels, check = test.predict(split=True)
#
#         sim, simold = 0, 0
#         for i in range(labels.shape[0]):
#             assert (newbest[i][0] == labels[i][0])
#             sim += int(newbest[i][1] == labels[i][1])
#             simold += int(oldbest[i][1] == labels[i][1])
#
#         print('Similarity to full labels (71.5): {}'.format(sim/labels.shape[0]))
#         print('Similarity to full labels (58.5): {}'.format(simold/labels.shape[0]))
#         similarity_2_best[count]=(sim/labels.shape[0])
#         similarity_2_old[count]=(simold/labels.shape[0])
#         # check = np.array(load_csv('ResultsPathDistance.csv')[1:], dtype='int')
#
#         sim, simold = 0, 0
#         for i in range(check.shape[0]):
#             assert (newbest[i+6000][0] == check[i][0])
#             sim += int(newbest[i+6000][1] == check[i][1])
#             simold += int(oldbest[i+6000][1] == check[i][1])
#
#         print('Check Results to labels (71.5): {}'.format(sim/labels.shape[0]))
#         print('Check Results to labels (58.5): {}'.format(simold/labels.shape[0]))
#         results_2_best[count]=(sim/labels.shape[0])
#         results_2_old[count]=(simold/labels.shape[0])
#
#         count += 1
# print('Similarity to full labels (71.5) Path: {}'.format(similarity_2_best[:21]))
# print('Similarity to full labels (58.5) Path: {}'.format(similarity_2_old[:21]))
# print('Similarity to full labels (71.5) EXP: {}'.format(similarity_2_best[21:]))
# print('Similarity to full labels (58.5) EXP: {}'.format(similarity_2_old[21:]))
# print('Check Results to labels (71.5) Path: {}'.format(results_2_best[:21]))
# print('Check Results to labels (58.5) Path: {}'.format(results_2_old[:21]))
# print('Check Results to labels (71.5) EXP : {}'.format(results_2_best[21:]))
# print('Check Results to labels (58.5) EXP: {}'.format(results_2_old[21:]))
# assert features.shape[0] == (10000)
# classifier_features = features
# if labels.shape == (10000,2):
#     print('Reducing labels')
#     labels = labels[:6000]
# test2 = Classifier(classifier_features, labels, "DeepLayerEXPClassifier", batch_size = 256)
# test2.fit()
# output = test2.predict()

# output = np.array(load_csv('DeepLayerEXPClassifier.csv')[1:], dtype='int')
#
# sim, simold = 0, 0
# for i in range(output.shape[0]):
#     assert (newbest[i+6000][0] == output[i][0])
#     sim += int(newbest[i+6000][1] == output[i][1])
#     simold += int(oldbest[i+6000][1] == output[i][1])
#
# print('Similarity to labels (71.5): {}'.format(sim/labels.shape[0]))
# print('Similarity to labels (58.5): {}'.format(simold/labels.shape[0]))
#
# sim = 0
# assert output.shape == (4000,2)
# for i in range(output.shape[0]):
#     assert (check[i][0] == output[i][0])
#     sim += int(check[i][1] == output[i][1])
#
# print('Similarity to Siamese labels: {}'.format(sim/output.shape[0]))

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
