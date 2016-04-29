import numpy as np
import numpy.random
from math import exp

def logisticSigmoid(x):
	return 1.0/(1.0 + exp(-x))


class Gate():
	outBias = -1
	def __init__(self, type):
		self.W = None
		if type == 'in':
			self.b = 0
		elif type == 'out':
			self.b = outBias
			outBias -= 1
		elif type == 'forget':
			self.b = 1
		self.f = logisticSigmoid

	def initWeights(self, hiddenN):
		self.W = np.random.rand(hiddenN,1)*0.4 - 0.2

class Cell():
	def __init__(self):
		self.W = None
		self.s = None
		self.h = lambda x: 2.0*logisticSigmoid(x) - 1
		self.g = lambda x: 4.0*logisticSigmoid(x) - 2

	def initWeights(self, hiddenN):
		self.W = np.random.rand(hiddenN,1)*0.4 - 0.2
		self.s = np.random.rand()*0.4 - 0.2

class CellBlock():
	def __init__(self, n):
		self.cells = [Cell() for i in range(n)]
		self.inGate = Gate('in')
		self.outGate = Gate('out')
		self.forgetGate = Gate('forget')

	def initWeights(self, inDim, hiddenN):
		for c in self.cells:
			c.initWeights(hiddenN)
		self.inGate.initWeights(hiddenN)
		self.outGate.initWeights(hiddenN)
		self.forgetGate.initWeights(hiddenN)

class OutLayer():
	def __init__(self, n):
		self.W = None
		self.n = n

	def initWeights(self, hiddenN):
		self.W = np.random.rand(hiddenN,n)*0.4 - 0.2

class Network():
	def __init__(self, inDim, blockSizes, outDim):
		self.blocks = []
		for n in blockSizes:
			self.blocks.append(CellBlock(n))
		self.out = OutLayer(outDim)
		hiddenN = sum([n+2 for n in blockSizes])

		for b in self.blocks:
			b.initWeights(inDim, hiddenN)










def main():

	topo = [2,2]

	n = Network(1, topo, 1)




if __name__ == '__main__':
	main()
