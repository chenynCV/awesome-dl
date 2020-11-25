import numpy as np
from model import Model
from dataset import Dataset, DataLoader


if __name__ == '__main__':
    trainDs = Dataset('./mnist.pkl.gz', classNum=10, split='train')
    trainDataLoader = DataLoader(trainDs, batch=128)

    valDs= Dataset('./mnist.pkl.gz', classNum=10, split='val')
    valDataLoader = DataLoader(valDs, batch=128)

    model = Model(inDim=(28, 28), outDim=10)
    for epoch in range(10000):
        model.train()
        for step, (X, Y) in enumerate(trainDataLoader):
            Y_, loss = model(X, Y) 
            model.optimize(lr=0.01)
            if step % 100 == 0:
                print('epoch={}, step={}, loss={:.4f}'.format(epoch, step, np.mean(loss)))
            pass

        model.eval()
        TP = 0
        for (X, Y) in valDataLoader:
            Y_ = np.argmax(model(X), axis=1)
            TP += sum(Y == Y_) 
        print('epoch={}, acc={:.4f}'.format(epoch, TP/len(valDs)))