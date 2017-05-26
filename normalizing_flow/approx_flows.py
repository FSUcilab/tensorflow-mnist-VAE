from scipy.stats import multivariate_normal, uniform
import matplotlib.pyplot as plt
import numpy as np
from normalizing_flow import NormalizingFlow
from nf_linear import LinearTransform
from nf_radial import RadialTransform
import tensorflow as tf
from IPython import embed
from kingma_tests import test_bean, test_displaced_gaussian, test_sine, test_kingma3, test_kingma4
from plots import makeScatterPlots, makeScatterPlot
import sys

#target_pdf = test_sine
target_pdf = test_bean
target_pdf = test_kingma4
mu = 0.0
rv = multivariate_normal([mu,mu], [[1,0],[0,1]]) # Assumed form of original q
#rv.pdf = lambda x: np.full(x.shape[:2], 1.0/x.size) # Uniform Distribution

iz0 = rv.rvs(30000)
iq0 = rv.pdf(iz0).reshape(-1,1) # Starting distribution
x = iz0[:,0]
y = iz0[:,1]
makeScatterPlot(iz0[:,0], iz0[:,1], iq0, title="Original Distribution", xlab="x", ylab="y", file_name="orig_dist.png")

shape = 2
grid_size = 3
transform_names = ['radial']*4
transform_names = ['linear']*4
transform_names = ['radial', 'linear', 'radial', 'linear']
transform_names = ['linear'] * 8
transform_names = ['radial'] * 8
transform_names = ['radial', 'linear'] * 4
transform_names = ['linear', 'radial'] * 4
transform_names = ['linear', 'radial'] * 8
transform_names = ['linear'] * 16
transform_names = ['radial'] * 16
transform_names = ['radial'] * 8
transform_names = ['radial'] * 6
transform_names = ['linear'] * 4
#transform_names = ['radial', 'linear'] * 3
transform_names = ['radial', 'linear'] * 16

layers = len(transform_names)
x, y = np.mgrid[mu-grid_size:mu+grid_size:0.1, mu-grid_size:mu+grid_size:0.1]
pos = np.empty(x.shape + (2,))
pos[:,:,0] = x; pos[:,:,1] = y
q0_grid = rv.pdf(pos)
grid_shape = q0_grid.shape
iters = 100000
batch_size = 100
nf = NormalizingFlow(transform_names=transform_names, input_dim=shape, exact=target_pdf)
z0 = rv.rvs(batch_size)
q0 = rv.pdf(z0).reshape(-1,1) # Starting distribution

print("params before training = ", nf.getParams(z0,q0))
#zk, qk = nf.flow(z0, q0)
#print("qk= ", qk)
#quit()

cost = []
tot_cost = 0.0
batch_cost = []

params = nf.getParams(z0, q0)
for k in range(layers):
    print("layer %d: " % k, params[k])
print("\n")

for i in range(iters):
    z0 = rv.rvs(batch_size)
    q0 = rv.pdf(z0).reshape(-1,1) # Starting distribution
    # The weight is updated twice: once for each nf.partial_fit
    #print(nf.partial_fit(z0, q0))
    #print(nf.partial_fit(z0, q0))
    partial_cost = nf.partial_fit(z0, q0)
    print("(it%05d) partial_cost= " % i, partial_cost)
    tot_cost += partial_cost
    cost.append(tot_cost / (iters))
    batch_cost.append(partial_cost)

    #if i % 10 == 0:
        #print(i, " batch_ Cost: ", partial_cost)
        #partial_cost = 0.0
        #print("pos shape: ", pos.reshape(-1,2).shape)
        #print("q0_grid shape: ", q0_grid.reshape(-1,1).shape)
        #print("det= ", det)

    if i % 10 == 0:
        #zk, qk = nf.flow(z0, q0)
        #print("qk= ", qk)
        #quit()

        det = nf.getDeterminant(z0, q0)
        """
        params = nf.getParams(z0, q0)
        print("params= ", params)
        print("det= ", det[0][0])
        print("coef= ", det[0][1])
        print("hhprime= ", det[0][2])
        print("fac= ", det[0][3])
        print("hh= ", det[0][4])
        print("r= ", det[0][5])
        print("z= ", det[0][6])
        print("q= ", det[0][7])
        print("z00= ", det[0][8])
        print("r= ", det[0][9])
        quit()
        """
        if np.isnan(det[0][0]).any():
            print("iteration %d, nan")
            quit()

    from matplotlib.collections import LineCollection

    def plotGrid(zk,k, title, save):
        xy = zk[k+1]
        xk = xy[:,0].reshape(grid_shape)
        yk = xy[:,1].reshape(grid_shape)
        segsx = np.zeros(grid_shape+(2,)) # [grid_shape[0],grid_shape[1],2]
        segsy = np.zeros(grid_shape+(2,)) # [grid_shape[0],grid_shape[1],2]
        for i in range(grid_shape[1]):
            segsx[:,i,0] = xk[:,i]
            segsx[:,i,1] = yk[:,i]
        linesx = LineCollection(segsx, linewidths=0.3, color='red')

        for i in range(grid_shape[0]):
            segsy[:,i,0] = xk[i,:]
            segsy[:,i,1] = yk[i,:]
        linesy = LineCollection(segsy, linewidths=0.3, color='red')

        fig, ax = plt.subplots(1,1)
        ax.set_xlim(-4.,4.)
        ax.set_ylim(-4.,4.)
        ax.add_collection(linesx)
        ax.add_collection(linesy)
        ax.set_xlabel("x0")
        ax.set_ylabel("y0")
        ax.set_title(title)
        plt.savefig(save)
        plt.close()

    if i % 1000 == 0:
        zk, qk = nf.flow(pos.reshape(-1,2), q0_grid.reshape(-1,1))
        for k in range(0,layers):
            plotGrid(zk, k, "zk[%02d] grid, it %05d" % (k, i), "zk[%02d]_it%05d.pdf" % (k, i))
        #quit()

    #if i % 1000 == 0:
    if i % 1000 == 0:
        #print("q0_grid shape:", q0_grid.shape); quit()
        zk, qk = nf.flow(pos.reshape(-1,2), q0_grid.reshape(-1,1))
        #print("zk= ", zk)
        #print("qk= ", qk)
        k = layers-1
        #quit()
        #det = nf.getDeterminant(pos.reshape(-1,2), q0_grid.reshape(-1,1))
        #print("pos shape: ", pos.reshape(-1,2).shape)
        #print("q0_grid shape: ", q0_grid.reshape(-1,1).shape)
        #print("det= ", det)
        #print("determinant shape: ", nf.getDeterminant(pos.reshape(-1,2), q0_grid.reshape(-1,1))[0][0].shape)
        #print("qk[0] shape:", qk[0].reshape(-1,1).shape); 
        #print("qk[1] shape:", qk[1].reshape(-1,1).shape); quit()
        #print("qk[k+1] shape:", qk[k+1].reshape(-1,1)); quit()
        plt.figure(k+3)
        if (k == 0): plt.title("it %05d, after %02d Transformation" % (i, k+1))
        elif (k > 0): plt.title("it %05d, after %02d Transformations" % (i, k+1))
        #print("k= ", k)
        #print("zk len = ", len(zk))
        plt.contourf(zk[k+1][:,0].reshape(grid_shape), zk[k+1][:,1].reshape(grid_shape), qk[k+1].reshape(grid_shape))
        plt.savefig("nf_it%05d_layer%02d.pdf" % (i, layers-1))
        plt.savefig("nf_it%05d_layer%02d.png" % (i, layers-1))
        plt.close()

    # plot cost function
    if i % 1000 == 0:
    #if i % 1 == 0:
        plt.plot(batch_cost)
        plt.title("cost per batch")
        plt.xlabel("iter")
        plt.ylabel("batch cost")
        plt.savefig("batch cost.pdf")
        plt.savefig("batch cost.png")
        plt.close()

print("params after training = ", nf.getParams(z0,q0))

zk, qk = nf.flow(pos.reshape(-1,2), q0_grid.reshape(-1,1))
plt.figure(0)
plt.title("Cost")
plt.plot(cost)

plt.figure(1)
plt.title("Target PDF")
plt.contourf(x,y,nf.evalTargetPDF(pos.reshape(-1,2)).reshape(grid_shape))

plt.figure(2)
plt.title("Original Gaussian")
plt.contourf(x,y,q0_grid)

# Below are tranformation parameters for each layer [[w1,u1,b1],...,[wk,uk,bk]]
#tp = [[[0.,1.],[0.,-.8],[0.]], [[1000.,0.],[0.35,0.],[0.]]]
#nf = NormalizingFlow(layers=layers, input_dim=shape, transform_params=tp)
#zk, qk = nf.flow(pos.reshape(-1,2), q0_grid.reshape(-1,1))

for k in range(layers):
    plt.figure(k+3)
    plt.title("After %02d Transformation" % (k+1))
    plt.contourf(zk[k+1][:,0].reshape(grid_shape), zk[k+1][:,1].reshape(grid_shape), qk[k+1].reshape(grid_shape))
plt.show()
