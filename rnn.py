from base import Recurrent, shared
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
#Implementation of simple RNN


#Need to append norm clipping (paper "on the difficulty training RNN")



class RNN(Recurrent):
	def __init__(self, ninp, nhid, nout,low_weight=-0.001, high_weight=0.001, steps=10, epoch=10, batch_size=None):
		super(RNN, self).__init__(ninp, nhid, nout, steps=steps, epoch=epoch, batch_size=batch_size)
		self.rng = np.random.RandomState(seed=1234)
		#Init hidden states
		self.h0 = T.zeros((nhid, ))
		low = low_weight
		high = high_weight
		self.Wh = shared(self.rng, low, high, ((ninp, nhid,)), 'Wh')
		self.Uh = shared(self.rng, low, high, (nhid, nhid), 'Uh')
		self.Wo = shared(self.rng, low, high, (nhid, nout), 'Wo')
		self.bh = theano.shared(np.asarray(np.zeros(nhid, ), dtype=theano.config.floatX), name='bh')
		self.bo = theano.shared(np.asarray(np.zeros(nout), dtype=theano.config.floatX), name='bo')
		self.params = [self.Wh, self.bh, self.Uh, self.Wo, self.bo]

	def _forward(self, x_t, h_prev, activation='tanh'):
		""" Note: If activation is sigmoid """
		func = T.nnet.sigmoid
		if activation == 'tanh':
			func = T.tanh
		return func(T.dot(x_t, self.Wh) + self.bh + T.dot(h_prev, self.Uh))
