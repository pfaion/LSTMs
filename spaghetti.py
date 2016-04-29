import numpy as np
import numpy.random
from matplotlib import pyplot as plt


def logisticSigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def h(x):
	return 2.0/(1.0 + np.exp(-x)) - 1

def g(x):
	return 4.0/(1.0 + np.exp(-x)) - 2


class LSTMNetwork:
    def __init__(self, inDim, blockSizes, nHiddenUnits, outDim):
        self.inDim = inDim
        self.blockSizes = blockSizes
        self.blocks = range(len(blockSizes))
        self.nHiddenUnits = nHiddenUnits
        self.outDim = outDim
        self.nUnits = self.inDim + self.nHiddenUnits + sum([n+2 for n in self.blockSizes])

        self.hiddenW = np.random.rand(self.nHiddenUnits, self.nUnits) * 0.4 - 0.2
        self.inGatesW = [np.random.rand(1, self.nUnits) * 0.4 - 0.2 for n in self.blocks]
        self.outGatesW = [np.random.rand(1, self.nUnits) * 0.4 - 0.2 for n in self.blocks]
        self.cellW = [np.random.rand(n, self.nUnits) * 0.4 - 0.2 for n in self.blockSizes]
        self.outW = np.random.rand(self.outDim, self.nUnits) * 0.4 - 0.2

        self.hiddenB = np.zeros((self.nHiddenUnits, 1))
        self.inGatesB = [np.zeros((1,1))]*len(self.blocks)
        self.outGatesB = [np.array([[-1*(i+1)]]) for i in self.blocks]
        self.cellB = [np.zeros((n,1)) for n in self.blockSizes]
        self.outB = np.zeros((self.outDim,1))

        self.networkState = np.random.rand(self.nUnits, 1)
        self.cellStates = [np.random.rand(n,1) for n in self.blockSizes]

    def update(self, inp):

        netHidden = self.hiddenW @ self.networkState
        yHidden = logisticSigmoid(netHidden + self.hiddenB)

        netsInGates = [self.inGatesW[i] @ self.networkState for i in self.blocks]
        yInGates = [logisticSigmoid(netsInGates[i] + self.inGatesB[i]) for i in self.blocks]

        netsOutGates = [self.outGatesW[i] @ self.networkState for i in self.blocks]
        yOutGates = [logisticSigmoid(netsOutGates[i] + self.outGatesB[i]) for i in self.blocks]

        netsCells = [self.cellW[i] @ self.networkState for i in self.blocks]
        newCellStates = [self.cellStates[i] + yInGates[i]*g(netsCells[i] + self.cellB[i]) for i in self.blocks]
        yCells = [yOutGates[i]*h(newCellStates[i]) for i in self.blocks]

        netOut = self.outW @ self.networkState
        yOut = logisticSigmoid(netOut + self.outB)

        newNetworkState = np.concatenate((
                inp,
                yHidden,
                np.concatenate(yInGates),
                np.concatenate(yOutGates),
                np.concatenate(yCells)
        ))

        self.networkState = newNetworkState
        return self.networkState

n = LSTMNetwork(2, [2,4,6], 3, 1)

time = 100
coll = np.zeros((n.nUnits, time))
inp = lambda t: np.random.rand(2, 1)
for t in range(time):
    coll[:,t] = n.update(inp(t))[:,0]

plt.matshow(coll)
plt.show()


#
# inDim = 2
# blockSizes = [2,4,6]
# blocks = range(len(blockSizes))
# nHiddenUnits = 4
# outDim = 1
#
# nUnits = inDim + nHiddenUnits + sum([n+2 for n in blockSizes])
# unitActivation = np.random.rand(nUnits, 1)
#
# hiddenWeights = np.random.rand(nHiddenUnits, nUnits) * 0.4 - 0.2
# hiddenBiases = np.zeros((nHiddenUnits, 1))
# def yHidden():
#     net = hiddenWeights @ unitActivation
#     return logisticSigmoid(net + hiddenBiases)
#
# inGatesWeights = [np.random.rand(1, nUnits) * 0.4 - 0.2 for n in blocks]
# inGatesBiases = [np.zeros((1,1))]*len(blocks)
# def yInGates():
#     nets = [inGatesWeights[i] @ unitActivation for i in blocks]
#     return [logisticSigmoid(nets[i] + inGatesBiases[i]) for i in blocks]
#
# outGatesWeights = [np.random.rand(1, nUnits) * 0.4 - 0.2 for n in blocks]
# outGatesBiases = [np.array([[-1*(i+1)]]) for i in blocks]
# def yOutGates():
#     nets = [outGatesWeights[i] @ unitActivation for i in blocks]
#     return [logisticSigmoid(nets[i] + outGatesBiases[i]) for i in blocks]
#
# cellWeights = [np.random.rand(n, nUnits) * 0.4 - 0.2 for n in blockSizes]
# states = [np.random.rand(n,1) for n in blockSizes]
# cellBiases = [np.zeros((n,1)) for n in blockSizes]
# def yCells():
#     nets = [cellWeights[i] @ unitActivation for i in blocks]
#     newStates = [states[i] + yInGates()[i]*g(nets[i] + cellBiases[i]) for i in blocks]
#     return [yOutGates()[i]*h(newStates[i]) for i in blocks]
#
# outWeights = np.random.rand(outDim, nUnits) * 0.4 - 0.2
# outBiases = np.zeros((outDim,1))
# def yOut():
#     net = outWeights @ unitActivation
#     return logisticSigmoid(net + outBiases)
#
# def yInp():
#     return np.random.rand(inDim, 1)
#
# steps = 100
# acColl = np.zeros((nUnits, steps))
# outColl = np.zeros((outDim, steps))
# for t in range(steps):
#     inp = yInp()
#     hid = yHidden()
#     inG = yInGates()
#     inGCat = np.concatenate(inG)
#     outG = yOutGates()
#     outGCat = np.concatenate(outG)
#     c = yCells()
#     cCat = np.concatenate(c)
#     out = yOut()
#
#     newAc = np.concatenate((inp, hid, inGCat, outGCat, cCat))
#     acColl[:,t] = newAc[:,0]
#     unitActivation = newAc
#
#     outColl[:,t] = out[:,0]
#
#
#
# plt.matshow(acColl, cmap=plt.cm.gray)
# plt.matshow(outColl, cmap=plt.cm.gray)
#
# plt.show()
