#Isoteric records

#Every optimization methods, compute only one step

class RMSProp:
	""" Implementation of RMSProp for training """
	def __init__(self, learning_rate=0.001, eps=1e-5, delta=0.99):
		self.learning_rate = learning_rate
		self.eps = eps
		self.delta = delta
		self.oldparams = None

	def run(self, params, gparams, decay=0.005):
		value= self.delta * params + (1 - self.delta) * gparams**2
		result = self.learning_rate * gparams/T.sqrt(result + self.eps)
		self.oldparams = oldparams
		return result


class SGD:
	def __init__(self):
		self.velocity = 0.01

	def run(self, params, gparams, learning_rate=0.001, momentum=0.99):
		velocity_next = self.velocity * momentum - learning_rate * gparams
		self.velocity = velocity_next
		return params - params * velocity_next