import theano.tensor as T
import theano

def sigmoid(X):
	return T.nnet.sigmoid(X)

def tanh(X):
	return T.tanh(X)