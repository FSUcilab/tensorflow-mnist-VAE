import numpy as np
import tensorflow as tf
from IPython import embed
import sys
import kingma_tests as km
from plots import makeScatterPlots, makeScatterPlot


class RadialTransform():
    def __init__(self, input_dim):
        #print(transform_params); quit()
        self.input_dim = input_dim
        # only valid for 2D latency spaces
        #self.mu_init  = tf.constant([0.,0.])
        #self.sig_init = tf.constant([1.,1.])

        # Initial values for random variables (which will evolve)
        print("Random variables for parameters")
        self.z00    = tf.Variable(tf.random_uniform([self.input_dim], -1., 1.), name="z00")
        alpha0 = np.random.uniform(-.5, .5, 1).astype(np.float32)
        self.alpha0 = tf.Variable(alpha0, name="alpha0")
        #self.alpha  = tf.log(1. + tf.exp(self.alpha0)) / np.log(2.)  # make sure alpha does not go negative. Alpha approx 1
        self.alpha  = tf.log(1. + tf.exp(self.alpha0))  # make sure alpha does not go negative. Alpha approx 1
        #self.alpha  = self.alpha0
        #self.gamma = tf.Variable(tf.random_uniform([1,],  .01, .02), name="gamma")
        #self.gamma = tf.constant([1.0,])
        #self.gamma = tf.log(1.+tf.exp(self.gamma))
        
        # see Appendix of Rezende et al (Normalizing flows)
        # prevent non-invertibility of tranformation (beta + alpha > 0)
        # beta + alpha = tf.log(1 + tf.exp(beta+alpha))
        #self.beta0  = tf.Variable(tf.random_uniform([1,],  -.47, -.46), name="beta0")
        #self.beta0  = tf.Variable(tf.log(tf.exp(self.alpha-1.)), name="beta0")
        #self.beta0  = tf.Variable(tf.log(tf.exp(alpha0-1.)), name="beta0")
        self.beta0  = tf.Variable(0., name="beta0")
        #self.beta0  = tf.Variable(tf.random_uniform([1], -1., 1.), name="beta0")
        self.beta = -self.alpha + tf.log(1. + tf.exp(self.beta0)) #+self.alpha))
		# beta=0 ==> exp(alpha) = 1 + exp(beta0)
		# exp(beta0) = (exp(alpha)-1)
		# beta0 = log(exp(alpha)-1.)

    def _h(self, r):
        return 1. / (self.alpha + r)
        #return self.gamma / (self.alpha + r)

    def hprime(self, r):
        return -1. / tf.square((self.alpha + r))
        #return -self.gamma / tf.square((self.alpha + r))

    def _f(s, z):   # s == self
        s.r = tf.norm(z - s.z00, axis=1)
        hh = tf.tile(tf.reshape(s._h(s.r), [-1,1]), [1,2])
        return z + s.beta * hh * (z-s.z00)

    def __call__(s, z, q):  # s == self
        z_out = s._f(z)
        # fac = 0 if beta = 0
        fac = s.beta * s._h(s.r)

        s.z = z
        s.q = q
        s.fac = fac
        s._hh = s._h(s.r)
        s.rr = tf.norm(z - s.z00, axis=1)

        # coef = 1, if fac = 0
        coef = tf.pow((1.+fac), s.input_dim-1.)

        s.coef = coef
        s.hhprime = s.hprime(s.r)

        s.determinant = tf.abs(coef * (1.+fac+s.beta*s.hprime(s.r)*s.r))
        s.determinant = tf.reshape(s.determinant, (-1,1))  # necessary to prevent explosion of size for q_out. Do not understand. 
        q_out = tf.exp(tf.log(q) - tf.log(s.determinant))
        # tranform from inital pdf(z) to transformed pdf(z_out)
        #q_out = q / tf.abs(s.determinant)
        return (z_out, q_out)

    def getParams(self):
        return [self.z00, self.alpha, self.beta]

    def getDeterminant(self):
        return [self.determinant, self.coef, self.hhprime, self.fac, self._hh, self.r, self.z, self.q, self.z00, self.rr]

    def getR(self):
        return [self.r]

    def getVar(self, var):
        return self.var

#----------------------------------------------------------------------

def targetPdf(z):
    """This is the first test pdf from the paper in table 1"""
    temp1 = 0.5*((np.linalg.norm(z,axis=1)-2.0)/0.4)**2
    exp1 = np.exp(-0.5*((z[:,0]-2)/0.6)**2)
    exp2 = np.exp(-0.5*((z[:,0]+2)/0.6)**2)
    return np.exp(-(temp1 - np.log(exp1 + exp2)))



if __name__ == '__main__':
    from scipy.stats import multivariate_normal, uniform
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    mu = 0.0
    rv = multivariate_normal([mu,mu], [[1,0],[0,1]]) # Assumed form of original q
    #rv.pdf = lambda x: np.full(x.shape[:2], 1.0/x.size) # Uniform Distribution

    shape = 2
    grid_size = 3
    layers = 12
    #x, y = np.mgrid[mu-grid_size:mu+grid_size:0.1, mu-grid_size:mu+grid_size:0.1]
    #pos = np.empty(x.shape + (2,))
    #pos[:,:,0] = x; pos[:,:,1] = y
    iz0 = rv.rvs(30000)
    iq0 = rv.pdf(iz0).reshape(-1,1) # Starting distribution
    x = iz0[:,0]
    y = iz0[:,1]
    #z0 = pos.reshape(-1,2)
    #q0_grid = rv.pdf(pos) # Gaussian Distribution
    #q0 = q0_grid.reshape(-1)
    #grid_shape = q0_grid.shape
    #embed()
    #sys.exit()

    makeScatterPlot(x, y, iq0, title="Original Distribution", 
      xlab="x", ylab="y", file_name="orig_dist.png")
    """
    plt.figure(0)
    plt.title("Original Distribution")
    plt.scatter(x,y,iq0)
    plt.savefig("orig_dist.png")
    plt.close()
    """

    #z0,alpha,beta for each layer
    #nf = NormalizingFlow(layers=layers, input_dim=shape, transform_params=tp)
    nf = NormalizingFlow(layers=layers, input_dim=shape)
    #zk, qk = nf.flow(iz0, iq0)

    #plotAllDeterminants(iz0, iq0)
    #quit()


    # Get Determinant
    #det = nf.getDeterminant(z0, q0)
    #print(det); quit()

    """
    for k in range(layers):
        plt.figure(k+1)
        plt.title("After %d Transformation" % (k+1))
        plt.contourf(zk[k+1][:,0].reshape(grid_shape), zk[k+1][:,1].reshape(grid_shape), qk[k+1].reshape(grid_shape))
    plt.savefig("transf.png")
    """

    max_iter = 100000
    batch_size = 256
    batch_size = 100
    #batch_size = 10

    for i in range(max_iter):
        z0 = rv.rvs(batch_size)
        q0 = rv.pdf(z0).reshape(-1,1) # Starting distribution
        #lst = nf.getParams(z0, q0)
        #print("z0= ", lst)
        #quit()
        lst = nf.getParams(z0, q0)
        #print("\ninitial params: ", lst)
        cost = nf.partial_fit(z0, q0)
        #print("\niter %06d, cost= %14.7e" % (i, cost))
        #lst = nf.getParams(z0, q0)
        #print("\nparams after cost: ", lst)

        if i % 100 == 0:
            lst = nf.getDeterminant(z0, q0)
            #print("\nlst= ", lst)
            lst = nf.getParams(z0, q0)
            #print("\nparams= ", lst)
            #quit()

        #"""
        if i % 1000 == 0:
            print("make plots: iter ", i)
            #plotDeterminant(iz0, iq0, nf)
            zk, qk = nf.flow(iz0, iq0)
            makeScatterPlots(i, zk, qk) # pos.reshape(-1,2), q0_grid.reshape(-1))
        #"""

