import numpy as np
import function as F


class Loss:
    def __init__(self):
        pass

    def forward(self, *wargs, **kwargs):
        pass

    def __call__(self, *wargs, **kwargs):
        return self.forward(*wargs, **kwargs)


class CrossEntropy(Loss):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, y_, y):
        self.y_ = y_
        self.y = y
        loss = -np.sum(y*np.log(y_), axis=1)
        return loss

    def backward(self):
        self.grad = -self.y/self.y_
        return self.grad


class CrossEntropyWithSoftMax(Loss):
    def __init__(self):
        super(CrossEntropyWithSoftMax, self).__init__()
        self.softmax = F.SoftMax()

    def forward(self, x, y):
        a = self.softmax(x)
        self.a = a
        self.y = y
        loss = -np.sum(y*np.log(a), axis=1)
        return loss

    def backward(self):
        self.grad = self.a - self.y
        return self.grad
