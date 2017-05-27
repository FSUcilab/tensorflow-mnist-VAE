import numpy as np
import tensorflow as tf
from IPython import embed
import sys

# How to get this recognized when running from parent directory? 
#from plots import makeScatterPlots, makeScatterPlot


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

