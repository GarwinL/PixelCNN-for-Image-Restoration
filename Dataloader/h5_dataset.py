# adapted from https://github.com/twtygqyy/pytorch-vdsr/blob/master/dataset.py

import torch.utils.data as data
import torch
import h5py

class HDF5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(HDF5Dataset, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]