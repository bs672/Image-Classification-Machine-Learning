import sys
import os

base = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base, 'DataSets'))
sys.path.append(os.path.join(base, 'Modules'))


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets
from sklearn.semi_supervised import label_propagation
from sklearn.metrics import confusion_matrix, classification_report
from csv_read import load_seed, load_data_features


k = 8
FNAME = 'CCAGraphPred-{}-500-500'.format(k)
print('Reading from {}.csv'.format(FNAME))
print('Saving to {}.hdf5'.format(FNAME))

seeds = load_seed()
seed_features = load_data_features(fname=FNAME, onlyseeds=True)
X_features = load_data_features(fname=FNAME)
# Give an input target file (Labels of all 10,000 points)
target = np.zeros(10000)

rng = np.random.RandomState(0)
indices = np.arange(len(X_features))
rng.shuffle(indices)

X = X_features[indices[:10000]]
y = target[:, 1][indices[:10000]]
for i in range(len(seeds)):
    y[seeds[i, 0]] = seeds[i, 1]

n_total_samples = len(y)
n_labeled_points = 60

indices = np.arange(n_total_samples)

unlabeled_set = np.array([i for i in indices if i not in seeds[:, 0]])

# #############################################################################
# Shuffle everything around
y_train = np.copy(y)
y_train[unlabeled_set] = -1

# #############################################################################
# Learn with LabelSpreading
lp_model = label_propagation.LabelSpreading(gamma=0.25)
lp_model.fit(X, y_train)
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = y[unlabeled_set]

cm = confusion_matrix(true_labels, predicted_labels, labels=lp_model.classes_)

print("Label Spreading model: %d labeled & %d unlabeled points (%d total)" %
      (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples))

print(classification_report(true_labels, predicted_labels))

print("Confusion matrix")
print(cm)

# #############################################################################
# Calculate uncertainty values for each transduced distribution
pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

# #############################################################################
# Pick the top 10 most uncertain labels
uncertainty_index = np.argsort(pred_entropies)[-10:]

labels = np.copy(target)
labels[:, 1][unlabeled_indices] = predicted_labels
assert not np.array_equal(labels[6000:], target[6000:])

#
# def purity_score(clusters, classes):
#     """
#     Calculate the purity score for the given cluster assignments and ground truth classes
#
#     :param clusters: the cluster assignments array
#     :type clusters: numpy.array
#
#     :param classes: the ground truth classes
#     :type classes: numpy.array
#
#     :returns: the purity score
#     :rtype: float
#     """
#
#     A = np.c_[(np.array([clusters[index-1] for index, label in classes]), classes)]
#
#     n_accurate = 0.
#
#     for j in np.unique(A[:,0]):
#         z = A[A[:,0] == j, 1]
#         x = np.argmax(np.bincount(z))
#         n_accurate += len(z[z == x])
#
#     return n_accurate / A.shape[0]
#
# print(purity_score(labels_max, seeds))
# print('Confidence in each prediction via max similarity')
# for index, c in confidence:
#     print('Element {} has confidence {:1.3f}'.format(int(index),c))
np.savetxt('results{}viaLabelSpread.csv'.format(FNAME), np.asarray(
    labels[6000:]), delimiter=',', fmt='%d', header='Id,Label')
