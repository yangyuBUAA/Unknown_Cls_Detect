'''
读取训练好的模型，输出：
1.每条数据的高维特征
2.每个类别的label embedding特征
3.每条数据中的每个token针对每个label的的attention score
'''

import pickle
import torch
import torch.nn.functional as F
import yaml
from numpy import sqrt
from model import Model
from transformers_model.models.bert.tokenization_bert import BertTokenizer


def analysis(config):
    model = Model(config)
    model.load_state_dict(torch.load("checkpoint/checkpoint-epoch1-batch3000.bin"))
    model.load_state_dict(torch.load("checkpointv2/checkpoint-epoch0-batch2000.bin"))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained("huggingface_pretrained_model/bert-base-chinese")

    sentence = ["淮安创业开公司流程", "征信不好买车以租代购可靠吗", "今天天气好买车吗", "天气", "盐城晋安怎么注册营业执照"]
    tokenized = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=20, padding="max_length")
    
    input_ids = tokenized["input_ids"]
    print(input_ids)
    attention_mask = tokenized["attention_mask"]
    token_type_ids = tokenized["token_type_ids"]

    output = model(input_ids, attention_mask, token_type_ids)
    print(output.logits)  # 模型输出的logits [batch_size, label_nums]
    print(output.attention_raw)  # 模型输出的每个token对每个label的attention，[batch_size, seq_len, label_nums]
    print(output.attention_normalized)  # [batch_size, 1, seq_len]
    print(cosine_distance(output.features_after_attention, output.label_features))
    print(euclidean_distance(output.features_after_attention, output.label_features))
    
    store(sentence, config["categories"], output.attention_raw, output.attention_normalized)

def euclidean_distance(model_output_features, label_embedding_features):
    """计算向量之间的欧几里得距离

    用于欧氏距离计算，如果分类的loss函数未使用余弦距离，则适用此函数查看分类效果

    Args:
        model_output_features: 模型的输出特征，直接将ModelOutput.features_after_attention传入，[batch_size, bert_hidden_dim]

        label_embedding_features: 模型的类别特征，直接将ModelOutput.label_features传入，[bert_embedding_dim, label_nums]

    Returns:
        将batch_size条模型输出特征分别与每个类别的label embedding计算欧几里得距离，返回tensor

        tensor([[18.7260, 18.8430, 18.9314],
                [19.0084, 19.2090, 19.1634]])

        此时，batch_size为2，label_nums为3

    Raises:
        None
    """
    label_nums = label_embedding_features.shape[1]
    features = model_output_features.unsqueeze(-1)
    features = features.repeat(1, 1, label_nums)
    # print(features.shape) # [batch_size, bert_hidden_dim, label_nums]
    # print(F.pairwise_distance(features, label_embedding_features, p=2).detach())
    return F.pairwise_distance(features, label_embedding_features, p=2).detach()

    # t1 = features[1, :, 0].unsqueeze(0)
    # t2 = label_embedding_features[:, 1].unsqueeze(0)
    # print(sqrt((t1-t2)*(t1-t2)))
    # print(F.pairwise_distance(t1, t2, p=2))


def cosine_distance(model_output_features, label_embedding_features):
    """计算向量之间的余弦距离

    用于余弦距离计算

    Args:
        model_output_features: 模型的输出特征，直接将ModelOutput.features_after_attention传入，[batch_size, bert_hidden_dim]

        label_embedding_features: 模型的类别特征，直接将ModelOutput.label_features传入，[bert_embedding_dim, label_nums]

    Returns:
        将batch_size条模型输出特征分别与每个类别的label embedding计算余弦距离，返回tensor

        tensor([[-0.0120, -0.0419, -0.1388],
        [-0.0487, -0.0937, -0.1616]])

        此时，batch_size为2，label_nums为3

    Raises:
        None
    """
    batch_size = model_output_features.shape[0]
    label_nums = label_embedding_features.shape[1]
    return_tensor = torch.zeros(batch_size, label_nums)

    for i in range(batch_size):
        for j in range(label_nums):
            return_tensor[i, j] = torch.cosine_similarity(model_output_features[i, :].unsqueeze(0), label_embedding_features[:, j].unsqueeze(0)).detach()
    return return_tensor
    

def store(sentences, labels, attention_raw, attention_cls):
    """存储模型输出信息用于可视化分析

    将训练好的模型输出数据进行存储

    """
    torch.save([sentences, labels, attention_raw, attention_cls], "analysis_data/data.bin")

if __name__=="__main__":
    with open(r'config.yaml', 'r', encoding='utf-8') as f:
        result = f.read()
        config = yaml.load(result)
    analysis(config)