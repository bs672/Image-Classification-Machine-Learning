"""
    Method of trying to produce more 'seeds' to improve the siamese cluster
"""

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

from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, scale
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.cross_decomposition import CCA
from csv_read import load_extracted_features_PCA, load_graph, load_seed, load_csv, load_extracted_features, load_spectral_embedding
from Comp import CCACluster
from models import SiameseModel, Classifier, FatSiameseModel

"""
    Idea is to do a siamese clustering for 5 different views of the data, and
    add any points that are labeled the same between all of them
    v1 : Current best submissions
    v2 : Previous best submissions
    v3 : Spectral Siamese Cluster of GraphEXP (k=10) NOT USED
    v4 : Spectral Siamese Cluster of GraphPathDistance (k=10) NOT USED
    v5 : CCA of v4, v5

    The mix will be a 6000x5 matrix
    dim1[6000] will be the indexs
    dim2[5] will be the labels from each view
"""

# Labels is a list of label sets
def concatenate_labels(labels):
    output = np.zeros((6000, len(labels)), dtype=int)
    for i, l in enumerate(labels):
        assert l.shape == (6000,2), 'Shape should be 6000x2, {}'.format(l.shape)
        output[:,i] = l[:,1]
    return output

# filters labels that are not all the same
def new_seeds(merged_labels):
    (indices, sets) = merged_labels.shape
    assert indices == 6000
    output = []
    label_counts = np.zeros(10)
    for i in range(indices):
        labels, counts = np.unique(merged_labels[i,:], return_counts=True)
        if len(labels) == 1 and counts == sets:
            output.append([i+1, labels[0]])
            label_counts[labels[0]] += 1
            # print(merged_labels[i,:])
    output = np.array(output, dtype='int')
    print(output.shape)
    print(label_counts)
    # # sanity check
    # for index, label in load_seed():
    #     assert output[index-1,:] = label # forces the keeping of old seed
    return output, np.min(label_counts)

v2 = load_csv('OldBest.csv')[:6000] # 58.5
v1 = load_csv('NewBEST.csv')[:6000]
# v3 = load_csv('V3.csv')
# v4 = load_csv('V4.csv')
v5 = load_csv('V5.csv')

new, min_labels = new_seeds(concatenate_labels([v1,v2,v5]))
# print(new)

assert min_labels == 21

# now I merge old seeds with new seeds, getting 21 seeds for each label

output = np.zeros((210,2), dtype='int')
output_counter = 0
label_counts = np.zeros(10)
max_counts = np.array([min_labels]*10)
for index, label in new:
    if label_counts[label] < min_labels:
        output[output_counter] = index, label
        output_counter += 1
        label_counts[label] += 1
# print(output)
print(label_counts)
# print(output.shape)

np.savetxt('newseeds.csv', np.asarray(output), delimiter=',', fmt='%d')

# v3
# test = SiameseModel(load_spectral_embedding(k=10, g_type='exp'), load_seed(), 'V3', batch_size = 256)
# # test.fit(epochs=10)
# # test.evaluate()
# v3 = test.predict()
#
# test = SiameseModel(load_spectral_embedding(k=10, g_type='path'), load_seed(), 'V4', batch_size = 256)
# # test.fit(epochs=10)
# # test.evaluate()
# v4 = test.predict()
#
# #v5
# # fname = "CCA-GraphPath-GraphEXP"
# # cca = CCACluster(load_spectral_embedding(k=500, g_type='path'),
# #         load_spectral_embedding(k=500, g_type='exp'), k_PCA = 500, k_SE=500, fname=fname)
# # cca.determine_CCA_components(CCrange=np.arange(1,11))
# # cca.save_predictions()
# # features = cca.cca_predictions
# # Fat is not better than Regular
# test = SiameseModel(load_csv("CCA-GraphPath-GraphEXP.csv"), load_seed(), 'V5', batch_size = 256)
# # test.fit(epochs=10)
# test.evaluate()
# v5 = test.predict()
