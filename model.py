import numpy as np
import function as F
from module import Module, FulllyConnect, Dropout
from loss import CrossEntropyWithSoftMax


class Model(Module):
    def __init__(self, inDim, outDim, hiddenDim=15):
        super(Model, self).__init__()
        self.fc1 = FulllyConnect(inDim, hiddenDim)
        self.relu1 = F.Relu()
        self.fc2 = FulllyConnect(hiddenDim, outDim)
        self.softmax = F.SoftMax()
        self.dropout = Dropout(prob=0.5)
        self.loss = CrossEntropyWithSoftMax()

    def forward(self, x, y=None):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        if y is not None:
            x = self.dropout(x)
            l = self.loss(x, y)
            return self.softmax(x), l
        else:
            return self.softmax(x)

    def optimize(self, lr=0.01):
        # BP1
        deltaL = self.dropout.backward() * self.loss.backward()
        
        # BP2
        deltaRelu = self.relu1.backward()
        deltaRelu = np.hstack((deltaRelu, np.ones((deltaRelu.shape[0], 1))))
        deltaFc2 = deltaRelu * (deltaL @ self.fc2.W)

        # BP3-BP4
        self.fc2.backward(deltaL, lr)
        self.fc1.backward(deltaFc2[:, :-1], lr)

        return
