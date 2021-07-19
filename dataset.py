"""
模型的输入数据类，继承自torch.utils.data.Dataset
"""

import torch
from torch.utils.data import Dataset


class LEDataset(Dataset):
    def __init__(self, tokenized_list, label_list):
        self.dataset = tokenized_list
        self.labels = label_list

    def __getitem__(self, item):
        return self.dataset[item], torch.tensor(int(self.labels[item]))

    def __len__(self):
        return len(self.dataset)