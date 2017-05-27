import tensorflow as tf
import numpy as np

def test_bean(z):
    "This is the first test pdf from the paper in table 1"
    # Bean pdf (formula in paper by Rezende on Normalizing flows
    temp1 = 0.5*((tf.norm(z,axis=1)-2.0)/0.4)**2
    exp1 = tf.exp(-0.5*((z[:,0]-2)/0.6)**2)
    exp2 = tf.exp(-0.5*((z[:,0]+2)/0.6)**2)
    return tf.exp(-(temp1 - tf.log(exp1 + exp2)))

def test_displaced_gaussian(z):
    # return pdf evaluated at points z
    exp1 = (1./(2.*np.math.pi))**2 * tf.exp(-0.5*((z[:,0]-1.)**2 + (z[:,1]-1.)**2))
    #self.mu_exact = tf.constant([1.,1.])
    #self.sig_exact = tf.constant([1.,1.])
    return exp1

def test_bean_left(z):
    "This is the first test pdf from the paper in table 1"
    # Bean pdf (formula in paper by Rezende on Normalizing flows
    temp1 = 0.5*((tf.norm(z,axis=1)-2.0)/0.4)**2
    exp1 = tf.exp(-0.5*((z[:,0]+2)/0.6)**2)
    return tf.exp(-(temp1 - tf.log(exp1)))

def test_bean_right(z):
    "This is the first test pdf from the paper in table 1"
    # Bean pdf (formula in paper by Rezende on Normalizing flows
    temp1 = 0.5*((tf.norm(z,axis=1)-2.0)/0.4)**2
    exp2 = tf.exp(-0.5*((z[:,0]-2)/0.6)**2)
    return tf.exp(-(temp1 - tf.log(exp2)))

def w1(z):
    return tf.sin(2.*np.math.pi*z[:,0]*0.25)

def w2(z):
    return 3.*tf.exp(-.5*((z[:,0]-1)/0.6)**2)

def w3(z):
    arg = (z[:,0]-1.)/0.3
    sig = 1./(1.+tf.exp(-arg))
    return sig

def test_sine(z):
    "Second test function in Kingma et al paper, table 1"
    temp = 0.5*((z[:,1]-w1(z))/0.4)**2
    return tf.exp(-temp)

def test_kingma3(z):
    temp = -tf.log(tf.exp(-0.5*((z[:,1]-w1(z))/0.35)**2)
            + tf.exp(-0.5*((z[:,1]-w1(z)+w2(z))/0.35)**2))
    return tf.exp(-temp)

def test_kingma4(z):
    temp = -tf.log(tf.exp(-0.5*((z[:,1]-w1(z))/0.40)**2)
            + tf.exp(-0.5*((z[:,1]-w1(z)+w3(z))/0.35)**2))
    return tf.exp(-temp)

