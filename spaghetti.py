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
        self.inGatesW = [np.random.rand(1, self.nUnits) * 0.4 - 0.2 for _ in self.blocks]
		self.inGatesPHW = [np.random.rand(1, n) * 0.4 - 0.2 for n in self.blockSizes]
		self.forgetGatesW = [np.random.rand(1, self.nUnits) * 0.4 - 0.2 for _ in self.blocks]
		self.forgetGatesPHW = [np.random.rand(1, n) * 0.4 - 0.2 for n in self.blockSizes]
		self.cellW = [np.random.rand(n, self.nUnits) * 0.4 - 0.2 for n in self.blockSizes]
        self.outGatesW = [np.random.rand(1, self.nUnits) * 0.4 - 0.2 for _ in self.blocks]
		self.outGatesPHW = [np.random.rand(1, n) * 0.4 - 0.2 for n in self.blockSizes]
        self.outW = np.random.rand(self.outDim, self.nUnits) * 0.4 - 0.2

        self.hiddenB = np.zeros((self.nHiddenUnits, 1))
        self.inGatesB = [np.zeros((1,1))]*self.nBlocks
		self.forgetGatesB = [np.zeros((1,1))]*self.nBlocks
        self.cellB = [np.zeros((n,1)) for n in self.blockSizes]
        self.outGatesB = [np.array([[-1*(i+1)]]) for i in self.blocks]
        self.outB = np.zeros((self.outDim,1))

        self.networkState = np.random.rand(self.nUnits, 1)
        self.cellStates = [np.zeros((n,1)) for n in self.blockSizes]

    def train(self, inp_seq, target_seq):
        # initialize states
        self.networkState = np.random.rand(self.nUnits, 1)
        self.cellStates = [np.zeros((n,1)) for n in self.blockSizes]

        # initialize derivatives
        derivStateCellW = [np.zeros((s, self.nUnits)) for s in self.blockSizes]
        derivStateInGatesW = [np.zeros((s, self.nUnits)) for s in self.blockSizes]
        derivStateInGatesPHW = [np.zeros((s, s)) for s in self.blockSizes]
		derivStateForgetGatesW = [np.zeros((s, self.nUnits)) for s in self.blockSizes]
        derivStateForgetGatesPHW = [np.zeros((s, s)) for s in self.blockSizes]

        # loop over the sequence
        for seq_idx, inp in enumerate(inp_seq):
            target = target_seq[seq_idx]

            # forward pass
            netHidden = self.hiddenW @ self.networkState
            yHidden = logisticSigmoid(netHidden + self.hiddenB)

            netsInGates = [(self.inGatesW[i] @ self.networkState) + (self.inGatesPHW[i] @ self.cellStates[i]) for i in self.blocks]
            yInGates = [logisticSigmoid(netsInGates[i] + self.inGatesB[i]) for i in self.blocks]

			netsForgetGates =[(self.forgetGatesW[i] @ self.networkState) + (self.forgetGatesPHW @ self.cellStates[i]) for i in self.blocks]
			yForgetGates = [logisticSigmoid(netsForgetGates[i] + self.forgetGatesB[i]) for i in self.blocks]

			netsCells = [self.cellW[i] @ self.networkState for i in self.blocks]
            self.cellStates = [yForgetGates[i] * self.cellStates[i] + yInGates[i]*g(netsCells[i] + self.cellB[i]) for i in self.blocks]

            netsOutGates = [(self.outGatesW[i] @ self.networkState) + (self.outGatesPHW[i] @ self.cellStates[i]) for i in self.blocks]
            yOutGates = [logisticSigmoid(netsOutGates[i] + self.outGatesB[i]) for i in self.blocks]

			# yCells = [yOutGates[i]*h(self.cellStates[i]) for i in self.blocks]
			yCells = [yOutGates[i] * self.cellStates[i] for i in self.blocks]

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


            if seq_idx == len(inp_seq)-1:
                sqerr = 1.0/self.inDim * np.sum((target - yOut)**2)
                print("\ntarget", target.T)
                print("output", yOut.T)
                print("error", sqerr)



            # backward pass
            learningRate = 0.2

            err = np.sum(target - yOut)


            outErr = derivSig(netOut) * err

            hiddenErr = derivSig(netHidden) * (self.outW[:, hiddenIdxStart:hiddenIdxStop].T @ outErr)

            blockIdx = 0
            outGatesErr = []
            statesErr = []
            for b, n in enumerate(self.blockSizes):
                idx = cellsIdxStart + blockIdx
				blockIdx += n

                cellsErr = self.cellStates[b].T @ (self.outW[:, idx:idx + n].T @ outErr)
				outGatesErr.append(derivSig(netsOutGates[b]) * cellsErr)

				statesErr.append(derivSig(netsOutGates[b]) * derivH(self.cellStates[b]) * (self.outW[:, idx:idx + n].T @ outErr))



            derivStateInGatesW = [derivStateInGatesW[i] + (g(netsCells[i]) * derivSig(netsInGates[i])) @ self.networkState.T for i in self.blocks]
            deltaInGatesW = [learningRate * (statesErr[i].T @ derivStateInGatesW[i]) for i in self.blocks]

            derivStateCellW = [derivStateCellW[i] + (derivG(netsCells[i]) * logisticSigmoid(netsInGates[i])) @ self.networkState.T for i in self.blocks]
            deltaCellW = [learningRate * statesErr[i] * derivStateCellW[i] for i in self.blocks]


            deltaOutW = learningRate * (outErr @ self.networkState.T)
            deltaHiddenW = learningRate * (hiddenErr @ self.networkState.T)
			deltaOutGatesW = [learningRate * (outGatesErr[i] @ self.networkState.T) for i in self.blocks]


            self.outW += deltaOutW
            self.hiddenW += deltaHiddenW
            self.outGatesW = [self.outGatesW[i] + deltaOutGatesW[i] for i in self.blocks]
            self.inGatesW = [self.inGatesW[i] + deltaInGatesW[i] for i in self.blocks]
            self.cellW = [self.cellW[i] + deltaCellW[i] for i in self.blocks]



            self.networkState = newNetworkState
            for w in self.cellW:
                plt.matshow(w)
                plt.show()


def generateSequence(seqLen, n_numbers):
    indicator = 0
    numbers = range(1,n_numbers)

    memNum = np.random.choice(numbers)
    restNums = [n for n in numbers if n != memNum]
    length = np.random.randint(seqLen[0], seqLen[1])

    randomPart = [np.random.choice(restNums) for _ in range(length)]
    seq = [memNum] + randomPart + [indicator] + [memNum]

    return seq


def generateSequences(n_sequences, seqLen, n_numbers):
    return [generateSequence(seqLen, n_numbers) for _ in range(n_sequences)]


def seqsToCategoricalAndNumpy(seqs, n_numbers):
    newSeqs = []
    for seq in seqs:
        newSeq = []
        for n in seq:
            catVec = np.array([[1 if i == n else 0 for i in range(n_numbers)]]).T
            newSeq.append(catVec)
        # newSeq = np.array(newSeq)
        newSeqs.append(newSeq)
    return newSeqs

def catToNum(cat):
    for i, c in enumerate(cat):
        if c == 1:
            seq.append(i)


def makeInputTargetPairs(seqs):
    inps = []
    tars = []
    for seq in seqs:
        inp = []
        tar = []
        for i in range(len(seq)-1):
            inp.append(seq[i])
            tar.append(seq[i+1])
        inps.append(inp)
        tars.append(tar)
    return inps, tars


n_nums = 4
sequences = generateSequences(100, [5,10], n_nums)
sequences = seqsToCategoricalAndNumpy(sequences, n_nums)
train_inps, train_tars = makeInputTargetPairs(sequences)



n = LSTMNetwork(n_nums, [1,1], 0, n_nums)
for i in range(len(train_inps)):
    n.train(train_inps[i], train_tars[i])

# the None is a hack to keep the numpy ndarrays confined to 2D, although its only a row vector in this context
# n.train(train_seqs[0][0,:,None], train_seqs[0][1,:,None])
