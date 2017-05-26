from scipy.stats import multivariate_normal, uniform
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from nf import NormalizingFlow
#from iaf import AutoregressiveFlow
import tensorflow as tf
from IPython import embed
import sys

OUTPUT_DIR = "normalizing_flow_figures/"

def test_func1(z):
    """This is the first test pdf from the paper in table 1"""
    temp1 = 0.5*((np.linalg.norm(z,axis=1)-2.0)/0.4)**2
    exp1 = np.exp(-0.5*((z[:,0]-2)/0.6)**2)
    exp2 = np.exp(-0.5*((z[:,0]+2)/0.6)**2)

    return np.exp(-(temp1 - np.log(exp1 + exp2)))

mu = 0.0
rv = multivariate_normal([mu,mu], [[1,0],[0,1]]) # Assumed form of original q
#rv.pdf = lambda x: np.full(x.shape[:2], 1.0/x.size) # Uniform Distribution

shape = 2
grid_size = 3
layers = 2
x, y = np.mgrid[mu-grid_size:mu+grid_size:0.1, mu-grid_size:mu+grid_size:0.1]
pos = np.empty(x.shape + (2,))
pos[:,:,0] = x; pos[:,:,1] = y
q0_grid = rv.pdf(pos)
grid_shape = q0_grid.shape
iters = 10000
batch_size = 100
# Below are tranformation parameters for each layer [[w1,u1,b1],...,[wk,uk,bk]]
tp = [[[0.,1.],[0.,-.8],[0.]], [[50.,0.],[0.35,0.],[0.]]]
tp = None
nf = NormalizingFlow(layers=layers, input_dim=shape, transform_params=tp)
#nf = NormalizingFlow(layers=layers, input_dim=shape)
z0 = rv.rvs(batch_size)
q0 = rv.pdf(z0).reshape(-1,1) # Starting distribution
print("params before training = ", nf.getParams(z0,q0))
cost = []
for i in range(iters):
    #z0 = np.random.multivariate_normal([0,0],[[1,0],[0,1]], size=batch_size)
    z0 = rv.rvs(batch_size)
    q0 = rv.pdf(z0).reshape(-1) # Starting distribution
    exact_pdf = test_func1(z0).reshape(-1) # Trying to fit this pdf
    #cost.append(nf.partial_fit(z0, q0, exact_pdf))
    cost.append(nf.partial_fit(z0, q0, test_func1))
    #print "cost = ", cost[-1]

print("params after training = ", nf.getParams(z0,q0))

plt.figure(0)
plt.title("Cost")
plt.plot(cost)
plt.savefig(OUTPUT_DIR+"cost.png"); plt.close()

plt.figure(1)
plt.title("Target PDF")
plt.contourf(x,y,test_func1(pos.reshape(-1,2)).reshape(grid_shape))
plt.savefig(OUTPUT_DIR+"target.png"); plt.close()

plt.figure(2)
plt.title("Original Gaussian")
plt.contourf(x,y,q0_grid)
plt.savefig(OUTPUT_DIR+"orig_gaussian.png"); plt.close()

zk, qk = nf.flow(pos.reshape(-1,2), q0_grid.reshape(-1,1))

for k in range(layers):
    plt.figure(k+3)
    plt.title("After %d Transformation" % (k+1))
    plt.contourf(zk[k+1][:,0].reshape(grid_shape), zk[k+1][:,1].reshape(grid_shape), qk[k+1].reshape(grid_shape))
    plt.savefig(OUTPUT_DIR+"transf%d_%diters.png" % (k,iters)); plt.close()
#plt.show()
