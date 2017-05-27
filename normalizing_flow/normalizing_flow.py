import numpy as np
import tensorflow as tf
from IPython import embed
import sys
from .nf_radial import RadialTransform
from .nf_linear import LinearTransform
# DEBUGGING
#import tensorflow.python.debug as tf_debug  


class NormalizingFlow():
    def __init__(self, input_dim=2, transform_names=None, exact=None):

        self.transform_names = transform_names

        if transform_names == None:
            print("Must specify transform_names: ('linear'/'radial')")
            quit()

        self.layers = len(transform_names)

        self.input_dim = input_dim
        self.target_pdf = exact

        self.transform_dict = {}
        self.transform_dict['radial'] = RadialTransform
        self.transform_dict['linear'] = LinearTransform

        print("execute transforms")
        self.transforms = [self.transform_dict[name](self.input_dim) for name in self.transform_names] 

    def flow(self, z0, q0):
        return self.sess.run((self.zk, self.qk), feed_dict={self.zk[0]: z0, self.qk[0]: q0})

    def partial_fit(self, z0, q0):
        cost, _ = self.sess.run((self.cost, self.optimizer), feed_dict={self.zk[0]: z0, self.qk[0]: q0})
        return cost

    def getParams(self, z0, q0):
        params = [self.sess.run(self.transforms[i].getParams(), feed_dict={self.zk[0]: z0, self.qk[0]: q0}) for i in range(self.layers)]
        return params

    def evalTargetPDF(self, z0):
        return self.target_pdf(z0).eval(session=self.sess)

    def getVar(self, var, layer, z0, q0):
        return self.sess.run(self.transforms[layer].getVar(var), feed_dict={self.zk[0]: z0, self.qk[0]: q0})

    def getDeterminant(self, z0, q0):
        params = [self.sess.run(self.transforms[i].getDeterminant(), feed_dict={self.zk[0]: z0, self.qk[0]: q0}) for i in range(self.layers)]
        return params

    def getLogDeterminant(self):
        logdet = tf.log(self.transforms[0].determinant)
        return logdet

