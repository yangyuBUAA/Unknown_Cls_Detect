"""
使用bert+label embedding+权重共享+lmcl进行未知意图识别（异常检测）
"""

import pickle
import torch
import torch.nn.functional as F
import yaml
from numpy import sqrt
from model import Model
from transformers_model.models.bert.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import LEDataset


def filter(config):
    model = Model(config)
    # model.load_state_dict(torch.load("checkpoint/checkpoint-epoch1-batch3000.bin"))
    # model.load_state_dict(torch.load("checkpointv2/checkpoint-epoch0-batch2000.bin"))
    model.load_state_dict(torch.load("checkpointv3/checkpoint-epoch0-batch2000.bin"))
    model.eval()
    if config["use_cuda"] and torch.cuda.is_available():
        # model = torch.nn.DataParallel(model)
        model = model.cuda()
        # model = torch.nn.parallel.DistributedDataParallel(model)
    dataset = load_dataset()

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    confuse_matrix = torch.zeros((4, 4))

    for index, batch in tqdm(enumerate(dataloader)):
        input_ids, attention_mask, token_type_ids = \
            batch[0]["input_ids"].squeeze(), batch[0]["attention_mask"].squeeze(), batch[0]["token_type_ids"].squeeze()
        label = batch[1]
        if config["use_cuda"] and torch.cuda.is_available():
            input_ids, attention_mask, token_type_ids = \
                input_ids.cuda(), attention_mask.cuda(), token_type_ids.cuda()
            label = label.cuda()
        model_output = model(input_ids, attention_mask, token_type_ids)
        model_output = model_output.logits
        # print(model_output)
        # argmax = torch.argmax(model_output, dim=1).squeeze().detach().cpu().numpy()
        max = torch.max(model_output, dim=1)
        max_values = max.values.cpu().detach().numpy()
        max_indices = max.indices.cpu().detach().numpy()
        label = label.squeeze().cpu().numpy()

        # print(max_values)
        # print(max_indices)
        # print(label)
        try:
            for i in range(64):
                if max_values[i] < 1:
                    max_indices[i] = 3
            # print(max_indices)
            for i in range(64):
                confuse_matrix[label[i]][max_indices[i]] = confuse_matrix[label[i]][max_indices[i]] + 1
        except:
            pass
    print(confuse_matrix)


def load_dataset():
    return_data = list()
    return_label = list()
    with open("data/test/test_with_abnormal_shuffled.txt", "r", encoding="utf-8") as dataset:
        data = dataset.readlines()

    for line in data:
        a = line.strip()
        return_data.append(a[:-2])
        return_label.append(a[-1])

    tokenizer = BertTokenizer.from_pretrained("huggingface_pretrained_model/bert-base-chinese")
    tokenized_list, label_list = list(), list()
    for index in tqdm(range(len(return_data))):
        tokenized = tokenizer(return_data[index], return_tensors="pt", max_length=20, padding="max_length",
                              truncation=True)
        tokenized_list.append(tokenized)
        label_list.append(return_label[index])

    return LEDataset(tokenized_list, label_list)


if __name__ == "__main__":
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    filter(config)