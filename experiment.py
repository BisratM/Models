import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from model import *
import matplotlib.pyplot as plt
from utils import *
from data_loader import *
import IPython
from tqdm import tqdm

def sample_data(mu, var, num_samples = 100):
	""" sample from a normal distribution and return number of samples 
  	Parameters
  	----------
  	mu:torch.FloatTensor
  		Mean for distribution of size MxN
  	var: torch.FloatTensor
  		Variance for distribution of size MxN
  	num_samples: int
  		Number of sample wanted from distribution 

  	Returns 
  	Drawn samples of size MxNxnum_samples 
   	-------

	"""
	return Normal(mu, var).sample((num_samples,))


def visualize_cluster(clump):
	group = clump.numpy()
	x = group[:,0]
	y = group[:,1]
	plt.scatter(x,y)


# means and variances for our three gaussian clusters 
mu1, var1 = torch.Tensor([2.5, 2.5]), torch.Tensor([1.5, 0.8])
mu2, var2 = torch.Tensor([7.5, 7.5]), torch.Tensor([0.75, 0.5])
mu3, var3 = torch.Tensor([8, 1.5]), torch.Tensor([0.6, 0.8])


num_epochs = 3
#establish two hyper-params 
lambda_energy = 10.0
lambda_cov_diag = 14.0

model = DaGMM(n_gmm = 3, latent_dim = 3)

#This is probably overkill 
# optimizer = torch.optim.Adam(model.parameters(), lr=1)
optimizer = torch.optim.SGD(model.parameters(), lr = 1)
for i in range(num_epochs):

  #Input is composed of sampels from the three distributions 
  cluster1, cluster2, cluster3 = sample_data(mu1, var1), sample_data(mu2, var2), sample_data(mu3, var3)
  sample = torch.cat([cluster1, cluster2, cluster3])
  x = sample.transpose(0, 1)
  x = to_var(x)
  enc, dec, z, gamma = model.forward(x)

  total_loss, sample_energy, recon_error, cov_diag = model.loss_function(x, dec, z, gamma,lambda_energy,lambda_cov_diag)

  #Ref to https://pytorch.org/tutorials/beginner/pytorch_with_examples.html for review 
  model.zero_grad()
  total_loss.backward()
  optimizer.step()

  print("enc: ", enc)
  print("dec: ", dec)
  print("z: ", z)
  print("gamma", gamma)

  print ("reconstruction error: ", recon_error)
  print("covariance diagonal: ", cov_diag)
  print("sample energy: ", sample_energy)
  print("LOSS: ", total_loss)



print("mean: ",  model.mu.numpy())
print("cov: ", model.cov.numpy())


# if __name__ == '__main__':
# 	dagmm = DaGMM(n_gmm = 2)







