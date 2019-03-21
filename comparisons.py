# coding: utf-8


# get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set()
import numpy as np

import numpy as np
import pandas as pd
import torch
from main import *

from sklearn.datasets.samples_generator import make_blobs
from sklearn.mixture import GMM
from sklearn.cluster import KMeans

from model import *
from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
# to supress depreciation warnings
import warnings;

warnings.simplefilter('ignore')

X, y_true = make_blobs(n_samples=60, n_features=118, centers=2,
                       cluster_std=0.60, random_state=0)
# X = X[:, ::-1] # flip axes for better plotting

plt.scatter(X[:, 0], X[:, 1])

# gmm = GMM(n_components=2).fit(X)
# labels = gmm.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis');
# print(X.shape)
# print(labels.shape)
#
#
# # # PR and F-Score for GMM
#
#
#
#
# precision, recall, f_score, support = prf(y_true, labels, average='binary')
# print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision,recall, f_score))


# # Experimentation with DAGMM 


lambda_energy = 0.1
lambda_cov_diag = 0.005
num_epochs = 100
lr = 1e-4

model = DaGMM()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
gamma_ls = []
for i in range(num_epochs):
    #     X, y_true = make_blobs(n_samples=60, n_features = 118, centers=2,
    #                        cluster_std=0.60, random_state=0)
    X, y_true = make_blobs(n_samples=60, n_features=118, centers=2,
                           cluster_std=0.60)
    X = to_var(torch.tensor(X, dtype=torch.float))

    enc, dec, z, gamma = model.forward(X)
    total_loss, sample_energy, recon_error, cov_diag = model.loss_function(X, dec, z, gamma, lambda_energy,
                                                                           lambda_cov_diag)
    gamma_ls.append(gamma.detach().numpy())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
#     print(model.mu.numpy())
#     print(model.cov.numpy())


# # PR and F-Score for DAGMM       


# Testing the precision recall score on test data
X_test, y_test = make_blobs(n_samples=60, n_features=118, centers=2,
                            cluster_std=0.60)
X_test = to_var(torch.tensor(X_test, dtype=torch.float))
enc, dec, z, gamma = model.forward(X_test)

labels = []
new_gamma = gamma.detach().numpy()

for i in range(new_gamma.shape[0]):
    cluster_one_id = new_gamma[i][0]
    cluster_two_id = new_gamma[i][1]
    if (cluster_one_id <= cluster_two_id):
        labels.append(0)
    else:
        labels.append(1)

assert (len(labels) == new_gamma.shape[0])

accuracy = accuracy_score(y_test, labels)
precision, recall, f_score, support = prf(y_test, labels, average='binary')
# print("Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(precision,recall, f_score))
print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(accuracy, precision, recall,
                                                                                            f_score))

print(labels)
print(y_test)
