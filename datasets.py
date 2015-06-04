import imdb

def get_imdb(n_words=100000, varid_portion=0.1, maxlen=None):
	""" Get of Large Movie Review datasets """
	return imdb.load_data(path="./imdb.pkl", n_words = n_words, valid_portion=varid_portion, maxlen=maxlen)

def get_wikipedia_data(path):
	pass