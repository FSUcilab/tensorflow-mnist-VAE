import tensorflow as tf
from normalizing_flow.normalizing_flow import NormalizingFlow
import numpy as np

# Prior
class Prior():
    def __init__(self, input_dim, transform_names):
        # Not clear how this initialization works
        self.input_dim = input_dim
        self.transform_names = transform_names
        self.nb_layers = len(transform_names)
        self.nf = NormalizingFlow(transform_names=transform_names, input_dim=input_dim) #, exact=target_pdf)
    # Feed (z0,q0) to the normalizing low code (Must make this easier)

    def flow(self, z0, q0):
        self.z0 = z0
        self.q0 = q0
        self.zk = [self.z0]
        self.qk = [self.q0]
        for k in range(self.nb_layers):
            z_next, q_next = self.nf.transforms[k](self.zk[k], self.qk[k])
            self.zk.append(z_next)
            self.qk.append(q_next)

        return self.zk, self.qk

    def getLogDeterminant(self):
        return self.nf.getLogDeterminant()

    def sample(self):
        # N(0,1)
        # return actual real numbers
        n = 10000
        z0 = np.random.normal(np.zeros([n, self.input_dim]), np.ones([n, self.input_dim]))
        q0 = np.exp(-0.5*z0**2) / (2.*np.pi)**0.5
        q0 = np.prod(q0, axis=1, keepdims=True)  # shape of (n,1)
        return z0, q0

# Gaussian MLP as encoder
def gaussian_MLP_encoder(x, n_hidden, n_output, keep_prob):
    with tf.variable_scope("gaussian_MLP_encoder"):
        # initializers
        # get initialization of " Delving Deep into Rectifiers" paper (2015)
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

def KLAnalytic(mu, sigma):
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
    return KL_divergence

def KLStochastic(mu, sigma):
    # KL(q(z|x) || p(z))
    norm = tf.contrib.distributions.Normal(tf.zeros_like(mu), tf.ones_like(sigma))
    zq = mu + norm.sample() * sigma;
    zp = norm.sample()
    log_qz_x = norm.log_prob((zq-mu)/sigma)
    log_pz = norm.log_prob(zp)
    KL_divergence = log_qz_x - log_pz
    #KL_divergence = tf.log(qz_x) - tf.log(pz)
    return KL_divergence

def KLNormalizingFlowPrior(prior):
    log_det = prior.getLogDeterminant()
    #KL_divergence = tf.reduce_sum(log_det, 1)
    KL_divergence = log_det
    return KL_divergence, log_det

# Gateway
def autoencoder(x_hat, x, dim_img, dim_z, n_hidden, keep_prob, KL_anal, NF, NF_pairs):
    # encoding
    mu, sigma = gaussian_MLP_encoder(x_hat, n_hidden, dim_z, keep_prob)

    # prior via normalizing flow
    transform_names = ['radial', 'linear'] * NF_pairs
    #transform_names = ['linear']
    prior = Prior(dim_z, transform_names)

    # sampling by re-parameterization technique
    z = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # Feed this z in N(0,1) to the prior calculation
    z0 = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)

    # Normal distribution evaluated at z0
    q0 = tf.exp(-0.5*(z0-mu)**2 / sigma**2) / (2.*np.pi*sigma**2)**0.5
    q0 = tf.reduce_prod(q0, axis=1, keep_dims=True)  # shape [npts, 1]

    # Feed (z0,q0) to the normalizing flow code (Must make this easier)
    if NF:
        zk, qk = prior.flow(z0, q0)
    else:
        prior.z0 = z0
        prior.q0 = q0
        zk, qk = None, None

    # decoding 
    # The input to the decoder is still a sample from the Gaussian q(z|x)
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, keep_prob)

    # this would correspond to the posterior being modified via IAF or NF
    #y = bernoulli_MLP_decoder(zk[-1], n_hidden, dim_img, keep_prob)
    y = tf.clip_by_value(y, 1e-8, 1 - 1e-8)

    # loss
    # summation over dimensions of x
    marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)

    # KL(q(z|x) || p(z)) = E_q0 [ log(q(z|x) - p(z)) ]
    #KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)

    analytic_KL = KL_anal
    normalizing_flow = NF

    if analytic_KL:
        KL_divergence = KLAnalytic(mu, sigma)
    else:
        KL_divergence = KLStochastic(mu, sigma)
    
    # set to True for normalizing flow
    if normalizing_flow:
        nf_divergence, log_det = KLNormalizingFlowPrior(prior)
        KL_divergence += nf_divergence
    else:
        log_det = tf.zeros_like([1])  # only so that calling routine works. This is a Kludge.


    # implement normal flow for the prior (inital mean=0, sigma=1)
    # E_q0 \log p(z) = E_q0 \log p(z0) - E_q0 \sum_i \log(det_i )
    # I need to be able to sample from p(z)
    # z0 ~ N(0,1) ==> zk = f(z0)

    # mean over batch dimension
    marginal_likelihood = tf.reduce_mean(marginal_likelihood)
    KL_divergence = tf.reduce_mean(KL_divergence)
    ELBO = marginal_likelihood - KL_divergence
    loss = -ELBO
    return y, z, loss, -marginal_likelihood, KL_divergence, log_det, prior, zk, qk

def decoder(z, dim_img, n_hidden):
    y = bernoulli_MLP_decoder(z, n_hidden, dim_img, 1.0, reuse=True)
    return y
