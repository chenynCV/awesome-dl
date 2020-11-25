import numpy as np
import pickle as pkl
import gzip


class Dataset:
    def __init__(self, dataPath, classNum, split='train'):
        self.classNum = classNum
        self.split = split
        self.data = self.load(dataPath)

    def load(self, dataPath):
        with gzip.open(dataPath, 'rb') as f:
            trainData, valData, testData = pkl.load(f, encoding='bytes')
        if self.split == 'train':
            X = [np.reshape(x, (1, 28, 28)) for x in trainData[0]]
            Y = [self.oneHot(y) for y in trainData[1]]
        elif self.split == 'val':
            X = [np.reshape(x, (1, 28, 28)) for x in valData[0]]
            Y = valData[1]
        elif self.split == 'test':
            X = [np.reshape(x, (1, 28, 28)) for x in testData[0]]
            Y = testData[1]
        else:
            raise NotImplementedError
        return np.array(X), np.array(Y)

    def shuffle(self):
        index = np.random.permutation(len(self))
        X = self.data[0][index]
        Y = self.data[1][index]
        return X, Y

    def __getitem__(self, index):
        X = self.data[0][index]
        Y = self.data[1][index]
        return X, Y

    def __len__(self):
        return len(self.data[0])

    def __iter__(self):
        for i in range(0, len(self)):
            yield self.data[0][i], self.data[1][i]

    def oneHot(self, y):
        e = np.zeros((self.classNum, 1)).reshape(-1)
        e[y] = 1.0
        return e


class DataLoader:
    def __init__(self, dataset, batch, shuffle=False):
        self.dataset = dataset
        self.batch = batch
        self.shuffle = shuffle
        self.numSample = len(dataset)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()
        for i in range(0, len(self)):
            if i + self.batch >= self.numSample:
                i = self.numSample - self.batch
            yield self.dataset[i:i+self.batch]

    def __getitem__(self, index):
        return self.dataset[index:index+self.batch]

    def __len__(self):
        return self.numSample // self.batch


if __name__ == '__main__':
    dataset = Dataset('./mnist.pkl.gz', 10)
    dataloader = DataLoader(dataset, batch=12)
    batchData = dataloader[0]
    print('Done!')
