import numpy as np
import tensorflow as tf
#from IPython import embed
import sys

class LinearTransform():
    def __init__(self, input_dim): 
        self.input_dim = input_dim
        #self.w = tf.Variable([.1,.1], name="w")  # orig
        # I want planes in all directions
        self.w = tf.Variable(tf.random_uniform([input_dim], -.1, .1), name="w")
        u = tf.Variable(np.zeros(input_dim, 'float32'))

        #upar0 = [np.log(np.e-1.)] * self.w
        #u = tf.Variable(upar0 / tf.norm(self.w)**2, name="u")
        #print("shape u: ", tf.shape(u))

        self.b = tf.Variable(tf.zeros([1]), name="b")

        # ensure invertibility (see Rezende paper on Normalizing Flows, Appendix)
        # impose invertibility of the transformation
        wtu = tf.reduce_sum(self.w * u, 0)
        m = -1. + tf.log(1 + tf.exp(wtu))
        self.u = u + (m - wtu) * self.w / tf.norm(self.w)**2  # invertibility condition

        # no invertibility condition
        #self.u = u 

        # I would like self.u = 0. Therefore, what should be the value of u? 
        # 0 = wtu + m - wtu ==> m = 0 ==> e = 1 + exp(wtu) ==> wtu = log(e-1)
        #  u = upar + uperp  with upar = a*w ==> a = log(e-1)/||w||^2
        #  uperp can be random (we will set to zero)


    def _f(self, z, x):
        return z + self.u * tf.reshape(tf.tanh(x), (-1,1))

    def _psi(self, x):
        return self.w * tf.reshape((1-tf.tanh(x)**2), (-1,1))


    def __call__(self, z, q):
        wT_dot_z_plus_b = tf.reduce_sum(self.w * z, 1) + self.b
        z_out = self._f(z, wT_dot_z_plus_b)
        self.determinant = tf.abs(1 + tf.matmul(self._psi(wT_dot_z_plus_b), tf.reshape(self.u, (-1,1))))
        # equivalent to q_out = q / self.determinant (more numerically stable?)
        q_out = tf.exp(tf.log(q) - tf.log(self.determinant))
        #q_out = tf.exp(tf.log(tf.reshape(q, [-1,1])) - tf.log(self.determinant))
        #print("determ linear shape: ", self.determinant.shape)
        #print("q shape: ", q.shape)
        #print("q shape: ", q.get_shape())
        #return (z_out, tf.log(tf.reshape(q, [-1,1])))
        return (z_out, q_out)  # returns (300,300) 

    def getParams(self):
        return [self.w, self.u, self.b]

    def getVar(self, var):
        return self.var

    def getDeterminant(self):
        return [self.determinant]

