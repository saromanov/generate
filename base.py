import numpy as np
import theano
import theano.tensor as T
from training import SGD, RMSProp
import os
import pickle

def shared(rng, low, high, size, name):
	return theano.shared(value = \
			np.asarray(rng.uniform(low=low, high=high, size=size),dtype=theano.config.floatX), name=name)

class Recurrent:
	""" Base class from training networks with reccurent architecture """
	def __init__(self, ninp, nhid, nout, steps=10, epoch=10, batch_size=None):
		self.ninp = ninp
		self.nhid = nhid
		self.nout = nout
		self.steps = steps
		self.epoch = epoch
		self.batch_size = batch_size
		self.params = []
		self.x = T.tensor3('x')
		self.y = T.matrix('y')

	def _forward(self, x_t, h_prev):
		""" Note: If activation is sigmoid """
		pass

	def _cost(self, value, target, type_cost='lse'):
		#cost = ((t - y)**2).mean(axis=0).sum()
		if type_cost == 'lse':
			return ((value - target)**2).mean(axis=0).sum()
		if type_cost == 'bce':
			return self._cost_binary_ce(value, target)

	def _cost_binary_ce(self, value, target):
		return T.nnet.binary_crossentropy(value, target).mean(axis=0)

	def compute_step(self, learning='sgd'):
		hidden = T.zeros((self.steps, self.nhid), )
		h, updates = theano.scan(fn=self._forward, sequences=self.x, outputs_info=hidden)

		output = T.nnet.softmax(T.dot(h[-1], self.Wo) + self.bo)
		#cost_value = self._cost(output, self.y)
		error = self._get_error(output, self.y) + self._regularization()
		grads = T.grad(error, self.params)
		if learning == 'rmsprop':
			results = [(oldparam, RMSProp().run(oldparam, newparam)) for (oldparam, newparam) in zip(self.params, grads)]
		else:
			results = [(oldparam, SGD().run(oldparam, newparam)) for (oldparam, newparam) in zip(self.params, grads)]
		return error, results

	def _regularization(self):
		''' Append L1 and L2 regularization to cost function '''
		result = theano.shared(0.0)
		for param in self.params:
			result += T.sum(param**2)
		return result

	def _get_error(self, result, target):
		""" Compute errors of network on each iteration """
		return T.mean(T.sqr((result - target)**2).sum(axis=0))

	def fit(self, inp, targets, epoch=10, learning='sgd'):
		cost_value, updates = self.compute_step()
		func = theano.function([], cost_value, updates=updates, givens={self.x: inp, self.y: targets})
		#classify = theano.function(inputs=idxs, outputs=pred)
		for i in range(self.epoch):
			msg = "Epoch: {0}, cost {1} ".format(i, func())
			print(msg)
		

	def generate(self, X):
		""" Generate samples after training"""
		func = theano.function([self.x], self.y)
		return func(X)

	def load_model(path):
		if not os.path.exists(path):
			raise Exception("Path to model not found")
		f = open(path, 'rb')
		model = pickle.load(f)