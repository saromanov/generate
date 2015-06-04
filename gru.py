import numpy as np
import theano.tensor as T
import theano
from base import Recurrent, shared
from activations import sigmoid, tanh

#Implementation of Gated Recurrent Unit
#Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
#http://arxiv.org/abs/1412.3555

class GRU(Recurrent):
	def __init__(self, ninp, nhid, nout):
		super(GRU, self).__init__(ninp, nhid, nout)
		self.rng = np.random.RandomState(seed=1234)
		#Init hidden states
		self.h0 = T.zeros((nhid, ))
		low = -0.001
		high = 0.001
		self.Wh = shared(self.rng, low, high, ((ninp, nhid,)), 'Wh')
		self.Uh = shared(self.rng, low, high, (nhid, nhid), 'Uh')
		self.Wx = shared(self.rng, low, high, ((ninp, nhid,)), 'Wx')
		self.Ux = shared(self.rng, low, high, (nhid, nhid), 'Ux')
		self.Wz = shared(self.rng, low, high, ((ninp, nhid,)), 'Wz')
		self.Uz = shared(self.rng, low, high, (nhid, nhid), 'Uz')
		self.Wo = shared(self.rng, low, high, (nhid, nout), 'Wo')
		self.bh = theano.shared(np.asarray(np.zeros(nhid, ), dtype=theano.config.floatX), name='bh')
		self.bx = theano.shared(np.asarray(np.zeros(nhid), dtype=theano.config.floatX), name='bv')
		self.bz = theano.shared(np.asarray(np.zeros(nhid), dtype=theano.config.floatX), name='bv')
		self.bo = theano.shared(np.asarray(np.zeros(nout), dtype=theano.config.floatX), name='bo')
		self.params = [self.Wh, self.Uh, self.Wo, self.Wx, self.Ux, self.Uz, self.Wz, self.bx, self.bo, self.bz, self.bh]
    
    def check_activations(self, activations):
    	pass
		
	def _forward(self, x_t, h_prev, activations=None):
		""" activations - list of activations for wach step( all is three steps) """
		z = sigmoid(T.dot(x_t, self.Wh) + self.bh + T.dot(h_prev, self.Uh))
		r = sigmoid(T.dot(x_t, self.Wx) + self.bx + T.dot(h_prev, self.Ux))
		h = T.tanh(T.dot(x_t, self.Wz) + self.bz + T.dot(r * h_prev, self.Uz))
		return (1 - z) * h_prev + z * h