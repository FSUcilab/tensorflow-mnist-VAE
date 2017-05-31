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

        # Initial values for random variables (which will evolve)
        print("Random variables for parameters")
        self.z00    = tf.Variable(tf.random_uniform([self.input_dim], -1., 1.), name="z00")
        alpha0 = np.random.uniform(-.5, .5, 1).astype(np.float32)
        self.alpha0 = tf.Variable(alpha0, name="alpha0")
        self.alpha  = tf.log(1. + tf.exp(self.alpha0))  # make sure alpha does not go negative. Alpha approx 1
        
        # see Appendix of Rezende et al (Normalizing flows)
        # prevent non-invertibility of tranformation (beta + alpha > 0)
        # beta + alpha = tf.log(1 + tf.exp(beta+alpha))
        #self.beta0  = tf.Variable(tf.random_uniform([1,],  -.47, -.46), name="beta0")
        #self.beta0  = tf.Variable(tf.log(tf.exp(self.alpha-1.)), name="beta0")
        #self.beta0  = tf.Variable(tf.log(tf.exp(alpha0-1.)), name="beta0")
        self.beta0  = tf.Variable(0., name="beta0")
        #self.beta0  = tf.Variable(tf.random_uniform([1], -1., 1.), name="beta0")
        self.beta = -self.alpha + tf.log(1. + tf.exp(self.beta0)) #+self.alpha))

    def _h(self, r):
        return 1. / (self.alpha + r)

    def hprime(self, r):
        return -1. / tf.square((self.alpha + r))

    def _f(s, z):   # s == self
        s.r = tf.norm(z - s.z00, axis=1)
        hh = tf.tile(tf.reshape(s._h(s.r), [-1,1]), [1,s.input_dim])
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
        s.determinant = tf.reshape(s.determinant, (-1,1)) 
        # Equivalent to: q_out = q / tf.abs(s.determinant)
        q_out = tf.exp(tf.log(q) - tf.log(s.determinant))
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

