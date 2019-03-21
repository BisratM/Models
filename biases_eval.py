import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import numpy as np
import pandas as pd
import torch
from main import *
from biase_DAGMM import *

#to supress depreciation warnings
import warnings; warnings.simplefilter('ignore')


# LOAD BIASES DATA SET
DATASET = 'biase'  # sys.argv[1]
PREFIX = 'biase'  # sys.argv[2]

filename = DATASET + '.txt'
data = open(filename)
head = data.readline().rstrip().split()

label_file = open(DATASET + '_label.txt')
label_dict = {}

for line in label_file:
    temp = line.rstrip().split()
    label_dict[temp[0]] = temp[1]
label_file.close()

label = []
for c in head:
    if c in label_dict.keys():
        label.append(label_dict[c])
    else:
        print(c)

label_set = []
for c in label:
    if c not in label_set:
        label_set.append(c)
name_map = {value: idx for idx, value in enumerate(label_set)}
id_map = {idx: value for idx, value in enumerate(label_set)}
label = np.asarray([name_map[name] for name in label])

expr = []
for line in data:
    temp = line.rstrip().split()[1:]
    temp = [float(x) for x in temp]
    expr.append(temp)

expr = np.asarray(expr).T
n_cell, _ = expr.shape
if n_cell > 150:
    batch_size = config['batch_size']
else:
    batch_size = 32



# Parameters of DAGMM

lambda_energy = 0.1
lambda_cov_diag = 0.005
num_epochs = 3
lr = 1e-4


#make DAGMM model

model = DaGMM()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)
expr = to_var(torch.tensor(expr, dtype=torch.float))
for i in range(num_epochs):
    enc, dec, z, gamma = model.forward(expr)
    total_loss, sample_energy, recon_error, cov_diag = model.loss_function(expr, dec, z, gamma, lambda_energy,lambda_cov_diag)
    # gamma_ls.append(gamma.detach().numpy())
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print("COMPLETE")