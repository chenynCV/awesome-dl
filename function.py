import numpy as np


class Function:
    def __init__(self):
        pass

    def forward(self, *wargs, **kwargs):
        pass

    def __call__(self, *wargs, **kwargs):
        return self.forward(*wargs, **kwargs)


class Relu(Function):
    def __init__(self):
        super(Relu, self).__init__()
        self.grad = None

    def forward(self, x):
        self.x = x
        a = x * (x > 0)
        return a

    def backward(self):
        self.grad = (self.x > 0) * 1.0
        return self.grad


class Sigmoid(Function):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.grad = None

    def forward(self, x):
        self.ex = np.exp(-x)
        a = 1.0 / (1.0 + self.ex)
        return a

    def backward(self):
        self.grad = self.ex / ((1 + self.ex)**2)
        return self.grad


class SoftMax(Function):
    def __init__(self):
        super(SoftMax, self).__init__()
        self.grad = None

    def forward(self, x):
        ex = np.exp(x - x.max())
        a = ex / np.sum(ex, axis=1, keepdims=True)
        return a


class GlobalMeanPooling(Function):
    def __init__(self):
        super(GlobalMeanPooling, self).__init__()
        self.grad = None

    def forward(self, x):
        self.shape = (N, C, H, W) = x.shape
        a = x.reshape((N, C, -1)).mean(axis=-1)
        return a

    def backward(self, delta=None):
        N, C, H, W = self.shape
        self.grad = np.ones((N, C, H, W)) / (H*W)
        if delta is not None:
            self.grad *= delta.reshape((N, C, 1, 1))
        return self.grad


class GlobalMaxPooling(Function):
    def __init__(self):
        super(GlobalMaxPooling, self).__init__()
        self.grad = None

    def forward(self, x):
        self.shape = (N, C, H, W) = x.shape
        x_ = x.reshape((N, C, -1))
        self.index = x_.argmax(axis=-1)
        a = x_.max(axis=-1)
        return a

    def backward(self, delta=None):
        N, C, H, W = self.shape
        self.grad = np.zeros((N, C, H*W))
        self.grad[self.index] = 1.0
        self.grad = self.grad.reshape((N, C, H, W))
        if delta is not None:
            self.grad *= delta.reshape((N, C, 1, 1))
        return self.grad
