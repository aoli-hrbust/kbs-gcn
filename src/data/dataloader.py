from torch.utils.data import DataLoader, Dataset, TensorDataset
from src.data.dataset import *


class PyTorchSingleViewDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray = None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.sampleNum

    @property
    def sampleNum(self):
        return self.X[0].shape[0]

    @property
    def viewNum(self):
        return len(self.X)

    def __getitem__(self, item):
        X = self.X[item]
        if self.Y is None:
            return X
        return [X, self.Y[item]]


class PyTorchMultiviewDataset(Dataset):
    def __init__(self, X: List[np.ndarray], Y: np.ndarray = None):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.sampleNum

    @property
    def sampleNum(self):
        return self.X[0].shape[0]

    @property
    def viewNum(self):
        return len(self.X)

    def __getitem__(self, item):
        X = [self.X[v][item, :] for v in range(self.viewNum)]
        if self.Y is None:
            return X
        return X + [self.Y[item]]

    @classmethod
    def from_multiview_dataset(cls, data: MultiviewDataset):
        return cls(data.X)


class PyTorchPartialMultiviewDataset(PyTorchMultiviewDataset):
    def __init__(self, X: List[np.ndarray], M: np.ndarray, Y: np.ndarray = None):
        super().__init__(X, Y)
        self.M = M

    def __getitem__(self, item):
        X_batch = [self.X[v][item, :] for v in range(self.viewNum)]
        M_batch = self.M[item, :]
        if self.Y is None:
            return X_batch + [M_batch]
        return X_batch + [M_batch, self.Y[item]]

    @classmethod
    def from_partial_multiview_dataset(cls, data: PartialMultiviewDataset):
        return cls(data.X, data.mask)
