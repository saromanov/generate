from rnn import RNN
from gru import GRU
import numpy as np
from sklearn.datasets import load_iris

''' Generating examples with prediction with softmax every character at time
	softmax(output). Maximize total log probability of training sequence
	which implies that the RNN learns a probability distribution over sequences.
	We can sample from this conditional distribution to get the next character in
	a generated string  and  provide  it  as  the  next  input  to  the  RNN
'''


'''e = 1
for i in range(10):
	X = np.random.rand(100,2)
	y = np.dot(X[:,0], X[:,1])

	net = RNN(2, 30, 1)
	c = net.fit(X, y)
	e = 0.1*np.sqrt(c)+0.9*e
	print(e)'''

nout = 2
net = RNN(2, 30, nout)
np.random.seed(123)
X = np.random.rand(10, 10, 2)
y = np.random.rand(10,2)
'''for i in range(nout):
	y[:,i+1:,i] = X[:,:-i-1,i]'''
tresh=0.5
'''y[0:,][X[1:-1, :,1] > X[:-2,:,0] + tresh] = 1
y[1][X[1:-1, :,1] > X[:-2,:,0] + tresh] = 2'''
net.fit(X, y)
