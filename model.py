import numpy as np
import function as F
from module import Module, FulllyConnect, Dropout, Conv2D
from loss import CrossEntropyWithSoftMax


class Model(Module):
    def __init__(self, inDim, outDim):
        super(Model, self).__init__()
        self.conv1 = Conv2D(1, 32, stride=2)
        self.relu1 = F.Relu()
        self.conv2 = Conv2D(32, 16, stride=2)
        self.relu2 = F.Relu()
        self.pool = F.GlobalMeanPooling()
        self.fc = FulllyConnect(16, outDim)
        self.softmax = F.SoftMax()
        self.dropout = Dropout(prob=0.5)
        self.loss = CrossEntropyWithSoftMax()

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.fc(x)
        if y is not None:
            x = self.dropout(x)
            l = self.loss(x, y)
            return self.softmax(x), l
        else:
            return self.softmax(x)

    def optimize(self, lr=0.01):
        # BP1
        deltaFc = self.dropout.backward() * self.loss.backward()

        # BP2
        deltaConv2 = self.relu2.backward() * self.pool.backward(self.fc.res(deltaFc))
        deltaConv1 = self.relu1.backward() * self.conv2.res(deltaConv2)

        # BP3-BP4
        self.fc.backward(deltaFc, lr)
        self.conv2.backward(deltaConv2, lr)
        self.conv1.backward(deltaConv1, lr)

        return
