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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler, scale
from sklearn.manifold import SpectralEmbedding, TSNE
from sklearn.cross_decomposition import CCA
from csv_read import load_extracted_features_PCA, load_spectral_embedding, load_seed, load_csv

class CCACluster():

    # Only define k_CCA if you don't want to automatically determine the best value
    def __init__(self, k_PCA, k_SE, k_CCA=None, best_reg=None, verbose=True, type='adj'):
        self.features = load_extracted_features_PCA(k=k_PCA)
        self.graph = load_spectral_embedding(k=k_SE,type=type)
        self.seeds = load_seed()
        self.cca_predictions = None
        self.fname = 'CCACluster_PCA{}_Spec{}.hdf5'.format(k_PCA, k_SE)
        if verbose: print('Loaded all data')
        self.ccaCV = None
        self.k_PCA, self.k_SE, self.k_CCA = k_PCA, k_SE, k_CCA
        self.best_reg = best_reg
        if os.path.isfile(self.fname):
            if verbose: print('Found saved ccaCV')
            self.ccaCV = rcca.CCACrossValidate()
            self.ccaCV.load(self.fname)
            # print(self.ccaCV.best_numCC)
            self.k_CCA = self.ccaCV.best_numCC
            self.best_reg = self.ccaCV.best_reg
            # self.cca = rcca.CCA(kernelcca = False,
            #                     numCC = self.k_CCA,
            #                     reg = self.best_reg,
            #                     verbose=self.verbose)
        self.verbose = verbose
        assert self.features.shape == (10000, k_PCA)
        assert self.graph.shape == (6000, k_SE)
        assert self.seeds.shape == (60, 2)
        if verbose:
            print('Using {} features from PCA reduced data'.format(k_PCA))
            print('Using {} features from Spectral Embedding of Graph.csv'.format(k_PCA))

    def determine_CCA_components(self, CCrange = None):
        if self.ccaCV is not None:
            print('Cross Validation already computed')
            print('You should delete old saved .hdf5 if you wish to recompute')
            return

        if CCrange:
            ccaCV = rcca.CCACrossValidate(kernelcca = False,
                                numCCs = CCrange,
                                regs = [0, 1e2, 1e4, 1e6],
                                verbose=self.verbose)
        else:
            ccaCV = rcca.CCACrossValidate(kernelcca = False,
                                numCCs = np.arange(1, min(self.k_PCA, self.k_SE)),
                                regs = [0, 1e2, 1e4, 1e6],
                                verbose=self.verbose)

        if self.features is None or self.graph is None:
            print('What the fuck is this?')
        ccaCV.train([self.features[:6000], self.graph[:6000]])
        self.k_CCA, self.best_reg = ccaCV.best_numCC, ccaCV.best_reg
        if self.verbose:
            print('Best CCA components: {}'.format(self.k_CCA))
            print('Best Regularization: {}'.format(self.best_reg))
        testcorrsCV = ccaCV.validate([self.features[5000:6000], self.graph[5000:]])
        if self.verbose:
            print('Test correlations')
            print(testcorrsCV)
        ccaCV.compute_ev([self.features[5000:6000], self.graph[5000:]])
        if self.verbose:
            print('Expected Variance has been computed')
            # print(ccaCV.ev)
        if not os.path.isfile(self.fname):
            ccaCV.save(self.fname)
        self.ccaCV = ccaCV

    def predict(self):
        if self.ccaCV is None:
            if verbose: print('Going to compute best components first')
            self.determine_CCA_components()

        # self.cca_predictions, _ = self.ccaCV.predict(self.features, self.ccaCV.ws)
        cca = CCA(n_components=self.k_CCA)
        cca.fit(self.features[:6000], self.graph[:6000])
        self.cca_predictions = cca.transform(self.features)
        if self.verbose:
            print('Produced predictions')

    def cluster(self, from_save=None, type='kmeans'):
        if self.cca_predictions is None:
            if from_save:
                self.cca_predictions = load_csv(from_save)

            if self.cca_predictions is None: #still none
                self.predict()

        kmeans = KMeans(n_clusters = 10).fit(self.cca_predictions)
        count = 0.0
        for i, label in self.seeds:
            count += int(label == kmeans.labels_[i])
            # correctness[n_comp-2] += int(label == kmeans.labels_[i])
            if not(label == kmeans.labels_[i]):
                print('Seed {} is mislabeled'.format(i))
                print('{} should be {}'.format(kmeans.labels_[i], label))
        print('Correctness : {} '.format( count/60. ))


    def save_predictions(self):
        if self.cca_predictions is None:
            self.predict()

        np.savetxt('CCAPred-{}-{}-{}.csv'.format(self.k_CCA, self.k_PCA, self.k_SE), np.asarray(self.cca_predictions), delimiter=",", fmt='%.5f')
        if self.verbose:
            print('Saved predictions')

if __name__ == '__main__':
    test = CCACluster(k_PCA = 100, k_SE = 100)
    test.determine_CCA_components()
    print('How many recommended components : {}'.format(test.k_CCA))
    test.predict()
    test.save_predictions()
    # test.cluster(from_save='CCAPred8.csv')
