#Isoteric records

#Every optimization methods, compute only one step

class RMSProp:
	""" Implementation of RMSProp for training """
	def __init__(self):
		pass

	def run(self, params, gparams, decay=0.005):
		return params - gparams * decay


class SGD:
	def __init__(self):
		self.velocity = 0.01

	def run(self, params, gparams, learning_rate=0.001, momentum=0.99):
		velocity_next = self.velocity * momentum - learning_rate * gparams
		self.velocity = velocity_next
		return params - params * velocity_next