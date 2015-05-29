import numpy as np
import theano.tensor as T
import theano
from sklearn import datasets


class ParametersInit:
	def __init__(self, rng, low, high):
		"""
			rng - theano or numpy
			low - low value for init
			high - high value for init
		"""
		self.rng = rng
		self.low = low
		self.high = high

	def get_weights(self, size, name='W'):
		"""
			size in tuple format
			name - current name for weights
		"""
		return self._initW(size, name)

	def _initW(self, size, name):
		return theano.shared(value = \
			np.asarray(
				self.rng.uniform(low=self.low, high=self.high, size=size), dtype=theano.config.floatX
				), name='W')

	def _initW2(self, size, nin, nout, name):
		return theano.shared(value = \
			np.asarray(
				self.rng.uniform(low=-np.sqrt(6)/np.sqrt(nin + nout), high=np.sqrt(6)/np.sqrt(nin + nout), \
					size=size), dtype=theano.config.floatX
				), name='W')


#Generate to generate sequences of words or images (DRAW)

class LSTM:
	def __init__(self, inp, targets, num_inp, num_hid, num_out, num_seq):
		self.inp = inp
		self.targets = targets
		self.x = T.matrix('x')
		self.num_inp = num_inp
		self.num_hid = num_hid
		self.num_out = num_out
		self.num_seq = num_seq
		par = ParametersInit(np.random.RandomState(seed=1234), -0.001, 0.001)
		self.Wih = par.get_weights(((num_inp, num_hid)))
		self.Whh = par.get_weights(((num_hid, num_hid)))
		self.Who = par.get_weights(((num_hid, num_out)))
		self.Wf = par.get_weights(((num_inp, num_hid)))
		self.Uf = par.get_weights(((num_hid, num_hid)))
		self.Wic = par.get_weights(((num_inp, num_hid)))
		self.Uic = par.get_weights(((num_inp, num_hid)))
		self.Wo =par.get_weights(((num_inp, num_hid)))
		self.Uo = par.get_weights(((num_inp, num_hid)))
		self.Vo = par.get_weights(((num_inp, num_hid)))
		self.bi = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
		self.bf = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
		#self.bv = theano.shared(np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')


	def step(self, x_t, h_t_prev, c_g_prev):
		""" One step of LSTM """
		#Input gate
		i_g = T.nnet.sigmoid(T.dot(x_t, self.Wih) + T.dot(h_t_prev.T, self.Whh) + self.bi)

		#Forgot gate
		f_g = T.nnet.sigmoid(T.dot(x_t, self.Wf) + T.dot(h_t_prev.T, self.Uf) + self.bf)

		#c_g = T.dot(f_g, c_g_prev) + T.dot(i_g, T.tanh(T.dot(x_t, self.Wic) + T.dot(h_t_prev, self.Uic) + self.bc))

		#Output gate
		#o_g = T.nnet.sigmoid(T.dot(x_t, self.Wo) + T.dot(h_t_prev, self.Uo) + T.dot(c_g, self.Vo) + self.bo)

		#new hidden state
		#h = T.dot(o_g, T.tanh(c_g))

		#return c_g, h
		return i_g, f_g

	def train(self, iters=100, decay=0.9):
		hidden = np.random.random((self.num_hid, self.num_out, ))
		hid = T.matrix('h')
		func = theano.function([self.x, hid], self.step(self.x, hid, self.x))
		print(func(self.inp, hidden))
