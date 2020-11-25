import numpy as np


class Module:
    def __init__(self):
        pass

    def forward(self, *wargs, **kwargs):
        pass

    def __call__(self, *wargs, **kwargs):
        return self.forward(*wargs, **kwargs)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class FulllyConnect(Module):
    def __init__(self, inputDim, outputDim):
        super(FulllyConnect, self).__init__()
        self.W = np.random.normal(
            0, 1./np.sqrt(inputDim+1), size=(outputDim, inputDim+1))
        self.grad = None

    def forward(self, x):
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.x = x
        z = x @ self.W.T
        return z

    def backward(self, delta, lr):
        N = delta.shape[0]
        self.grad = delta.T @ self.x / N
        self.W -= lr * self.grad
        return self.grad


class Dropout(Module):
    def __init__(self, prob=0.5, training=True):
        super(Dropout, self).__init__()
        self.prob = prob
        self.training = training
        self.grad = None

    def forward(self, x):
        if self.training:
            self.flag = np.random.random(x.shape)
            a = x * (self.flag > self.prob)
        else:
            a = x
        return a

    def backward(self):
        self.grad = (self.flag > self.prob) * 1.0
        return self.grad


class Conv2D(Module):
    """
    The tensor are arranged in (N, C, H, W) order.
    """
    def __init__(self, inputDim, outputDim, kernelSize=(3, 3), stride=1):
        super(Conv2D, self).__init__()
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.kernelSize = kernelSize
        self.stride = stride

        rowW = kernelSize[0] * kernelSize[1]*inputDim
        colW = outputDim
        self.weight = np.random.normal(
            0, 1./np.sqrt(inputDim), size=(rowW, colW))
        self.grad = None

    def outShape(self, x):
        N, C, H, W = x.shape
        kw, kh = self.kernelSize
        Oh = ((H - kh + 1) + self.stride - 1) / self.stride
        Ow = ((W - kw + 1) + self.stride - 1) / self.stride
        return int(Oh), int(Ow)

    def forward(self, x):
        N, C, H, W = x.shape
        Oh, Ow = self.outShape(x)
        self.x_ = self.unrolling(x)
        x = self.x_ @ self.weight
        x = x.reshape((N, Oh, Ow, self.outputDim))
        x = x.transpose((0, 3, 1, 2))
        return x

    def unrolling(self, x):
        kw, kh = self.kernelSize
        N, C, H, W = x.shape
        Oh, Ow = self.outShape(x)
        
        MX = np.zeros((N, Oh*Ow, kw*kh*C))
        index = 0
        for h in range(0, H-kh+1, self.stride):
            for w in range(0, W-kw+1, self.stride):
                subX = x[:, :, h:h+kh, w:w+kw].reshape(N, -1)
                MX[:, index, :] = subX
        return MX

    def backward(self, delta, lr):
        N = delta.shape[0]
        self.grad = delta.T @ self.x_ / N
        self.weight -= lr * self.grad
        return self.grad


if __name__ == '__main__':
    conv = Conv2D(64, 128)
    x = np.random.random((3, 64, 28, 56))
    y = conv(x)
    pass