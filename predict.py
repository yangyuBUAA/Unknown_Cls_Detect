# 模型预测接口

import torch
import yaml
from model import Model
from transformers_model.models.bert.tokenization_bert import BertTokenizer


def inference(config):
    model = Model(config)
    model.load_state_dict(torch.load("checkpoint/checkpoint-epoch0-batch1000.bin"))
    tokenizer = BertTokenizer.from_pretrained("huggingface_pretrained_model/bert-base-chinese")

    sentence = ["旅游", "旅游哪里好玩"]
    tokenized = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=20, padding="max_length")
    input_ids = tokenized["input_ids"]
    print(input_ids.shape)
    attention_mask = tokenized["attention_mask"]
    token_type_ids = tokenized["token_type_ids"]

    print(model(input_ids, attention_mask, token_type_ids))

if __name__=="__main__":
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    inference(config)