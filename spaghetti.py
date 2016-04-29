import numpy as np
import numpy.random
from matplotlib import pyplot as plt


def logisticSigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def derivSig(x):
    return np.multiply(logisticSigmoid(x), (1 - logisticSigmoid(x)))

def h(x):
	return 2.0/(1.0 + np.exp(-x)) - 1

def derivH(x):
    return 2*derivSig(x)

def g(x):
	return 4.0/(1.0 + np.exp(-x)) - 2

def derivG(x):
    return 4*derivSig(x)

class LSTMNetwork:
    def __init__(self, inDim, blockSizes, nHiddenUnits, outDim):
        self.inDim = inDim
        self.blockSizes = blockSizes
        self.blocks = range(len(blockSizes))
        self.nHiddenUnits = nHiddenUnits
        self.outDim = outDim
        self.nUnits = self.inDim + self.nHiddenUnits + sum([n+2 for n in self.blockSizes])
        self.nBlocks = len(self.blocks)

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

    def run(self, inp, target):

        # forward pass
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

        inpIdxStart = 0
        inpIdxStop = hiddenIdxStart = self.inDim
        hiddenIdxStop = inGatesIdxStart = self.inDim + self.nHiddenUnits
        inGatesIdxStop = outGatesIdxStart = self.inDim + self.nHiddenUnits + self.nBlocks
        outGatesIdxStop = cellsIdxStart = self.inDim + self.nHiddenUnits + 2*self.nBlocks
        cellsIdxStop = self.nUnits






        learningRate = 0.2

        err = np.sum(target - yOut)

        outErr = derivSig(netOut) * err
        deltaOutW = learningRate * (outErr @ self.networkState.T)

        hiddenErr = derivSig(netHidden) * (self.outW[:, hiddenIdxStart:hiddenIdxStop].T @ outErr)
        deltaHiddenW = learningRate * (hiddenErr @ self.networkState.T)

        blockIdx = 0
        outGatesErr = []
        for b, n in enumerate(self.blockSizes):
            idx = cellsIdxStart + blockIdx
            cellsErr = (h(self.cellStates[b])).T @ (self.outW[:, idx:idx + n].T @ outErr)
            outGatesErr.append(derivSig(netsOutGates[b]) * cellsErr)
            blockIdx += n
        deltaOutGatesW = [learningRate * (outGatesErr[i] @ self.networkState.T) for i in self.blocks]

        blockIdx = 0
        statesErr = []
        for b, n in enumerate(self.blockSizes):
            idx = cellsIdxStart + blockIdx
            statesErr.append(derivSig(netsOutGates[b]) * derivH(self.cellStates[b]) * (self.outW[:, idx:idx + n].T @ outErr))
            blockIdx += n

        derivStateInGatesW = [0 + (g(netsCells[i]) * derivSig(netsInGates[i])) @ self.networkState.T for i in self.blocks]
        deltaInGatesW = [learningRate * (statesErr[i].T @ derivStateInGatesW[i]) for i in self.blocks]

        derivStateCellW = [0 + (derivG(netsCells[i]) * logisticSigmoid(netsInGates[i])) @ self.networkState.T for i in self.blocks]
        deltaCellW = [learningRate * statesErr[i] * derivStateCellW[i] for i in self.blocks]



        self.outW += deltaOutW
        self.hiddenW += deltaHiddenW
        self.outGatesW = [self.outGatesW[i] + deltaOutGatesW[i] for i in self.blocks]
        self.inGatesW = [self.inGatesW[i] + deltaInGatesW[i] for i in self.blocks]
        self.cellW = [self.cellW[i] + deltaCellW[i] for i in self.blocks]



        self.networkState = newNetworkState
        self.cellStates = newCellStates
        return (self.networkState, yOut)






def toCat(i, base):
    return [1 if b == i else 0 for b in range(base)]

base = 10
sequence = [0] + [np.random.randint(1,base) for i in range(10)] + [0]
print(sequence)
sequence = [toCat(s,base) for s in sequence]
sequence = np.array(sequence).T

n = LSTMNetwork(base, [2,2], 3, base)
time = 10
for t in range(time):
    coll = np.zeros((n.nUnits, len(sequence)-1))
    for i in range(len(sequence)-1):
        # the None is a dirty hack to keep the array shape (x, 1) while slicing
        state, out = n.run(sequence[:,i,None],sequence[:,i+1,None])
        coll[:,i] = state[:,0]
    plt.matshow(coll)
    plt.show()
