import torch
from torch.utils.data import Dataset


class LEDataset(Dataset):
    def __init__(self, tokenized_list, label_list):
        self.dataset = tokenized_list
        self.labels = label_list

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]

    def __len__(self):
        return len(self.dataset)