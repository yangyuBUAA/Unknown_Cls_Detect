import yaml
import torch
import os

from tqdm import tqdm
from model import Model
from dataset import LEDataset
from transformers_model.models.bert.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader


def train(config):

    CURRENT_DIR = config["CURRENT_DIR"]

    train_set, eval_set, test_set = load_dataset(config)

    model = Model(config)
    if config["use_cuda"] and torch.cuda.is_available():
        model = model.cuda()

    train_dataloader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    eval_dataloader = DataLoader(dataset=eval_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=True)





def evaluate(config):
    pass


def load_dataset(config):
    CURRENTDIR = config["CURRENT_DIR"]
    # TRAIN_CACHED_PATH = os.path.join(CURRENTDIR, "data/train/train_cached.bin")
    # EVAL_CACHED_PATH = os.path.join(CURRENTDIR, "data/eval/eval_cached.bin")
    # TEST_CACHED_PATH = os.path.join(CURRENTDIR, "data/test/test_cached.bin")
    #
    # if os.path.exists(TEST_CACHED_PATH):
    #     torch.load()

    TRAIN_SOURCE_PATH = os.path.join(CURRENTDIR, config["TRAIN_DIR"])
    EVAL_SOURCE_PATH = os.path.join(CURRENTDIR, config["EVAL_DIR"])
    TEST_SOURCE_PATH = os.path.join(CURRENTDIR, config["TEST_DIR"])

    tokenizer = BertTokenizer.from_pretrained(os.path.join(CURRENTDIR, config["bert_model_path"]))
    print("构建数据集...")
    train_set = construct_dataset(tokenizer, TRAIN_SOURCE_PATH)
    eval_set = construct_dataset(tokenizer, EVAL_SOURCE_PATH)
    test_set = construct_dataset(tokenizer, TEST_SOURCE_PATH)
    print("构建完成...训练集{}条数据，验证集{}条数据，测试集{}条数据...".format(len(train_set), len(eval_set), len(test_set)))
    return train_set, eval_set, test_set


def construct_dataset(tokenizer, SOURCE_PATH):
    tokenized_list = list()
    label_list = list()
    with open(SOURCE_PATH, "r", encoding="utf-8") as train_set:
        data = train_set.readlines()
        for line in tqdm(data):
            line_list = line.strip().split()
            sequence, label = line_list[0], line_list[1]
            # print(sequence, label)
            # break
            sequence_tokenized = tokenizer(sequence, return_tensors="pt", max_length=20, padding="max_length",
                                           truncation=True)
            tokenized_list.append(sequence_tokenized)
            label_list.append(label)

    return LEDataset(tokenized_list, label_list)


if __name__ == '__main__':
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    train(config)
