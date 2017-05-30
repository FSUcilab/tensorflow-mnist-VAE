import tensorflow as tf
from normalizing_flow.normalizing_flow import NormalizingFlow
import numpy as np

# Prior
class Prior():
    def __init__(self, input_dim, transform_names):
        # Not clear how this initialization works
        self.nf = NormalizingFlow(transform_names=transform_names, input_dim=input_dim) #, exact=target_pdf)

    def getLogDeterminant(self):
        return self.nf.getLogDeterminant()

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(x, w0) + b0
        h0 = tf.nn.elu(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.tanh(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer
        # borrowed from https: // github.com / altosaar / vae / blob / master / vae.py
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output * 2], initializer=w_init)
        bo = tf.get_variable('bo', [n_output * 2], initializer=b_init)
        gaussian_params = tf.matmul(h1, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :n_output]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, n_output:])

    return mean, stddev

# Bernoulli MLP as decoder
def bernoulli_MLP_decoder(z, n_hidden, n_output, keep_prob, reuse=False):
    with tf.variable_scope("bernoulli_MLP_decoder", reuse=reuse):
        # initializers
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        w0 = tf.get_variable('w0', [z.get_shape()[1], n_hidden], initializer=w_init)
        b0 = tf.get_variable('b0', [n_hidden], initializer=b_init)
        h0 = tf.matmul(z, w0) + b0
        h0 = tf.nn.tanh(h0)
        h0 = tf.nn.dropout(h0, keep_prob)

        # 2nd hidden layer
        w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden], initializer=w_init)
        b1 = tf.get_variable('b1', [n_hidden], initializer=b_init)
        h1 = tf.matmul(h0, w1) + b1
        h1 = tf.nn.elu(h1)
        h1 = tf.nn.dropout(h1, keep_prob)

        # output layer-mean
        wo = tf.get_variable('wo', [h1.get_shape()[1], n_output], initializer=w_init)
        bo = tf.get_variable('bo', [n_output], initializer=b_init)
        y = tf.sigmoid(tf.matmul(h1, wo) + bo)

    return y

# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob):
    # encoding
    mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)

    # prior via normalizing flow
    transform_names = ['radial', 'linear'] * 2
    prior = Prior(dim_z, transform_names)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # Feed this z to the prior calculation
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z0 = z  # to connect to Normalizing flow code

    # Normal distribution evaluated at z0
    q0 = tf.exp(-0.5*(z-mu)**2 / sigma**2) / (2.*np.pi*sigma**2)**0.5
    q0 = tf.reduce_prod(q0, axis=1)

    # Feed (z0,q0) to the normalizing flow code (Must make this easier)
    layers = len(transform_names)
    zk = [z0]
    qk = [q0]
    for k in range(layers):
        z_next, q_next = prior.nf.transforms[k](zk[k], qk[k])
        zk.append(z_next)
        qk.append(q_next)

    log_det = prior.getLogDeterminant() # orig

    # decoding 
    # The input to the decoder is still a sample from the Gaussian q(z|x)
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)

    # this would correspond to the posterior being modified via IAF or NF
    #y = bernoulli_MLP_decoder(zk[-1], n_hidden, dim_img, keep_prob)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    # loss
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)
    # KL(q(z|x) || p(z)) = E_q0 [ log(q(z|x) - p(z)) ]
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
    
	# with normalizing flow
    #KL_divergence += tf.reduce_sum(log_det, 1)

    #print("shape log_det: ", log_det.shape)
    #print("shape mu: ", mu.shape)

    # KL += log_det  leads to determinants going to zero. WHY? 
    # KL -= log_det  leads to determinants going to infinity, but much slower. Why? 
    #
    #

    # implement normal flow for the prior (inital mean=0, sigma=1)
    # E_q0 \log p(z) = E_q0 \log p(z0) - E_q0 \sum_i \log(det_i )
    # I need to be able to sample from p(z)
    # z0 ~ N(0,1) ==> zk = f(z0)

    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)

    ELBO = marginal_likelihood - KL_divergence

    loss = -ELBO

    return y, z, loss, -marginal_likelihood, KL_divergence, log_det

def decoder(z, dim_img, n_hidden):

    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)

    return y
