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

        if 0:
            self._build_graph()

            # Launch the session
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            self.sess = tf.InteractiveSession(config=config)
            #self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)   ### DEBUGGING
            #self.sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            self.sess.run(tf.global_variables_initializer())  # Tensor flow before 1.0
            #self.sess.run(tf.initialize_all_variables())  # TF 1.x


    def _build_graph(self):
        self.zk = [tf.placeholder(tf.float32, [None, self.input_dim], name='zk')]
        self.qk = [tf.placeholder(tf.float32, [None, 1], name='qk')]
        #self.transforms = [self.transform_dict[name](self.input_dim) for name in self.transform_names] 

        #print("dir= ", dir(self.transforms[0]))
        #xx = self.getVar(self.transforms[0].determinant, 0, self.zk[0], self.qk[0])

        for k in range(self.layers):
            z_next, q_next = self.transforms[k](self.zk[k], self.qk[k])
            self.zk.append(z_next)
            self.qk.append(q_next)

        # Cost should be called in the higher level program computing the cost
        #self._cost()
        #xx = self.transforms[0].getDeterminant()
        #det = self.sess.run((xx), feed_dict={self.zk[0]: z0, self.qk[0]: q0})
        #print("determinant shape= ", xx); quit()

    def flow(self, z0, q0):
        return self.sess.run((self.zk, self.qk), feed_dict={self.zk[0]: z0, self.qk[0]: q0})

    '''
    def _cost(self):
        """Solving for the sum in equation (13), we can substitute
        it into equation (15). This is the simplified form of eqn (15)"""

        # This is not how they do it in the paper, but I only want to fit a pdf, not do the whole VAE
        # This is the KL divergence between the final transformed pdf and the target pdf
        if self.target_pdf is not None:
            self.exact = self.target_pdf(self.zk[-1])
            self.cost = tf.reduce_mean(tf.log(self.qk[-1]+1e-8) - tf.log(self.exact+1e-8)) # orig
        else:
            print("self.target_pdf should not be None!")
            #self.cost = tf.reduce_mean(tf.log(self.qk[-1]+1e-8) - tf.log(self.qk[0]+1e-8))
            quit()

        # GradientDescent ok if lr=0, but not if lr=1.e-22 (at first iteration). How can that be? 
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.cost)
            #tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(self.cost)
            #tf.train.GradientDescentOptimizer(learning_rate=1.e-22).minimize(self.cost)  # debugging tool
            #tf.train.AdagradOptimizer(learning_rate=0.00001).minimize(self.cost)
    '''

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
        for k in range(1,len(self.transforms)):
            logdet += tf.log(self.transforms[k].determinant)
        return logdet

