import torch
import torch.nn.functional as F
import yaml
# BertModelModified是使用了BertEmbeddingsModified进行embedding的bert，同时将返回数据置为embedding，用于读取预训练embedding层的输出
from transformers_model.models.bert.modeling_bert import BertModel, BertModelModified
from transformers_model.models.bert.tokenization_bert import BertTokenizer


# 使用bert抽取特征，返回最后一层的hidden state作为对每个token的encoder
class FeatureExtractLayer(torch.nn.Module):
    def __init__(self, config):
        super(FeatureExtractLayer, self).__init__()
        self.config = config
        self.bert_model = BertModel.from_pretrained("huggingface_pretrained_model/bert-base-chinese")

    def forward(self, tokenized):
        bert_output = self.bert_model(tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"])
        return bert_output.last_hidden_state


# 使用bert的embedding层对每个label进行初始的embedding
class LabelEmbedding(torch.nn.Module):
    def __init__(self, config):
        super(LabelEmbedding, self).__init__()
        self.config = config
        self.bert_model_modified = BertModelModified.from_pretrained("huggingface_pretrained_model/bert-base-chinese")

    def forward(self, tokenized):
        embedding_layer = self.bert_model_modified(tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"])
        return embedding_layer


# 获取类别的embedding并构成labelembedding层
class LabelEmbeddingLayer(torch.nn.Module):
    def __init__(self, config):
        super(LabelEmbeddingLayer, self).__init__()
        self.config = config
        self.label_embedding = LabelEmbedding(self.config)
        self.categories = self.config["categories"]
        self.tokenizer = BertTokenizer.from_pretrained("huggingface_pretrained_model/bert-base-chinese")
        self.label_embedded = list()
        for cate in self.categories:
            tokenized = self.tokenizer(cate, return_tensors="pt", truncation=True, max_length=10, padding="max_length")
            embedded = self.label_embedding(tokenized).squeeze()
            attention_mask_list_sum = sum(tokenized["attention_mask"].squeeze().numpy().tolist())
            # print(attention_mask_list_sum)
            # print(embedded.shape, tokenized["attention_mask"])
            embedded_meaningful = embedded[1:attention_mask_list_sum-1, :]
            # print(embedded_meaningful.shape)
            one_label_embedding = torch.sum(embedded_meaningful, dim=0).unsqueeze(0)
            # print(embedding.shape)
            self.label_embedded.append(one_label_embedding)

        # (bert_embedding_dim, label_nums)
        self.label_embedding_layer_params = torch.cat(self.label_embedded, dim=0).permute(1, 0)



class LabelEmbeddingLayerWithParametersShare(torch.nn.Module):
    def __init__(self, config):
        super(LabelEmbeddingLayerWithParametersShare, self).__init__()
        self.config = config


class AttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        self.config = config
        self.max_pool1d = torch.nn.MaxPool1d(kernel_size=3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, deep_features, label_embeddings):
        # deep_features (bsz, max_seq_len, hidden_dim)
        # label_embeddings (bert_embedding_dim, label_nums)
        deep_features_normalized = F.normalize(deep_features, dim=-1)
        label_embeddings_normalized = F.normalize(label_embeddings, dim=0)
        atten = torch.matmul(deep_features_normalized, label_embeddings_normalized)
        # print(atten)
        pooled = self.max_pool1d(atten).squeeze()
        # (bsz, seq_len)
        normalized = self.softmax(pooled).unsqueeze(1)
        # print(normalized.shape)
        feature_attention = torch.matmul(normalized, deep_features).squeeze()
        # (bsz, label_embedding_dim)
        return feature_attention


class AjustiveAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super(AjustiveAttentionLayer, self).__init__()
        self.config = config


# 分类层，输出为logits，后面可以接单标签分类或者多标签分类的loss function
class ClassificationLayer(torch.nn.Module):
    def __init__(self, config, weight_matrix_params):
        super(ClassificationLayer, self).__init__()
        self.config = config
        self.weight_matrix_params = weight_matrix_params

    def forward(self, features):
        return torch.matmul(features, self.weight_matrix_params)


# 使用LMCLoss进行参数优化，将分类层和loss层整合在一起，返回logits和loss，logits用于评估，loss用于优化
class LMCLossLayer(torch.nn.Module):
    def __init__(self, config, weight_matrix_params):
        super(LMCLossLayer, self).__init__()
        self.config = config
        self.weight_matrix_params = weight_matrix_params

    def forward(self, features):
        pass


class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.sequence_feature_extract_layer = FeatureExtractLayer(config)

        # 将label embedding设置为可学习参数，维度为（bert_hidden_dim, label_nums）
        self.label_embedding_layer = torch.nn.Parameter(LabelEmbeddingLayer(config).label_embedding_layer_params, requires_grad=True)

        self.attention_layer = AjustiveAttentionLayer(config) if config["use_ajustive_attention"] else AttentionLayer(config)

        self.classification_layer = ClassificationLayer(config, self.label_embedding_layer)

    def forward(self, tokenized):
        deep_features = self.sequence_feature_extract_layer(tokenized)
        features = self.attention_layer(deep_features, self.label_embedding_layer)
        # shape of features(bsz, label_embedding_dim) eg:(32, 768)

        return self.classification_layer(features)


if __name__ == '__main__':
    f = open(r'config.yaml', 'r', encoding='utf-8')
    result = f.read()
    config = yaml.load(result)
    model = Model(config)
    tokenizer = BertTokenizer.from_pretrained("huggingface_pretrained_model/bert-base-chinese")
    tokenized = tokenizer(["今天天气怎么样", "明天天气怎么样"], return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    print(model(tokenized))
