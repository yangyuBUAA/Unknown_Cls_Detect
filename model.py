import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import yaml
# BertModelModified是使用了BertEmbeddingsModified进行embedding的bert，同时将返回数据置为embedding，用于读取预训练embedding层的输出
from transformers_model.models.bert.modeling_bert import BertModel, BertModelModified
from transformers_model.models.bert.tokenization_bert import BertTokenizer


# 使用bert抽取特征，返回最后一层的hidden state作为对每个token的encoder
class FeatureExtractLayer(torch.nn.Module):
    """加载预训练bert，使用bert抽取sequence特征，将last_hidden_state进行输出

    Attributes:
        config: 配置
        bert_model: 读取的预训练模型
    
    """
    def __init__(self, config):
        super(FeatureExtractLayer, self).__init__()
        self.config = config
        self.bert_model = BertModel.from_pretrained(self.config["bert_model_path"])

    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert_model(input_ids, attention_mask, token_type_ids)
        return bert_output.last_hidden_state


# 使用bert的embedding层对每个label进行初始的embedding
class LabelEmbedding(torch.nn.Module):
    """使用bert预训练的embedding对每个label进行初始的embedding

    Attributes:
        config: 配置
        bert_model_modified: 修改后的bert模型，此模型返回embedding层的编码，源码位置与BertModel相同
    """
    def __init__(self, config):
        super(LabelEmbedding, self).__init__()
        self.config = config
        self.bert_model_modified = BertModelModified.from_pretrained(self.config["bert_model_path"])

    def forward(self, tokenized):
        embedding_layer = self.bert_model_modified(tokenized["input_ids"], tokenized["attention_mask"], tokenized["token_type_ids"])
        return embedding_layer


class LabelEmbeddingLayer(torch.nn.Module):
    """将LabelEmbedding类提取的每个类别的embedding特征进行整合

    整合为(label nums, bert_embedding_dim)的tensor

    Attributes:
        config: 配置
        label_embedding: LabelEmbedding类，用于提取每个类别对应的embedding特征
        categories: 存放分类的类别标签
        tokenizer: bert的tokenizer
        label_embedded: 存放每个类别标签对应的初始化label embedding
        label_embedding_layer_params: 将每个类别标签对应的初始化的label embedding tensor拼接
    """
    def __init__(self, config):
        super(LabelEmbeddingLayer, self).__init__()
        self.config = config
        self.label_embedding = LabelEmbedding(self.config)
        self.categories = self.config["categories"]
        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_model_path"])
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
    """计算bert输出特征和label embedding的attention

    Longer class information....

    Attributes:
        config: 配置
        max_pool1d: 使用torch的MaxPool1d用来过滤特征
        softmax: 使用torch提供的softmax进行特征归一化
    """

    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        self.config = config
        self.max_pool1d = torch.nn.MaxPool1d(kernel_size=3)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, deep_features, label_embeddings, attention_mask):
        # deep_features (bsz, max_seq_len, hidden_dim)
        # label_embeddings (bert_embedding_dim, label_nums)
        deep_features_normalized = F.normalize(deep_features, dim=-1)
        label_embeddings_normalized = F.normalize(label_embeddings, dim=0)
        atten = torch.matmul(deep_features_normalized, label_embeddings_normalized)

        # 修复attention为遮挡padding的问题
        ones = torch.ones_like(attention_mask)
        reverse = 10000 * (attention_mask - ones)
        reverse = reverse.unsqueeze(-1).repeat(1, 1, label_embeddings.shape[1]) # (batch_size, max_seq_len, label_nums)
        atten = atten + reverse
        
        # print(atten)
        pooled = self.max_pool1d(atten).squeeze()
        # (bsz, seq_len)
        normalized = self.softmax(pooled).unsqueeze(1)
        # print(normalized.shape)
        feature_attention = torch.matmul(normalized, deep_features).squeeze()
        # (bsz, label_embedding_dim)
        return feature_attention, atten, normalized


class AjustiveAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super(AjustiveAttentionLayer, self).__init__()
        self.config = config


# 分类层，输出为logits，后面可以接单标签分类或者多标签分类的loss function
class ClassificationLayer(torch.nn.Module):
    """分类层，将attention后的特征通过全连接层进行分类

    全连接层的参数与label embedidng矩阵参数共享，bias为0

    Attributes:
        config: 配置
        weight_matrix_params: 实例化时传入label embedding矩阵

    """
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
    """bert+由bert embedding初始化的可学习的label embedding+与label embedding权重共享的分类层模型

    模型需要根据配置进行初始化，配置中需要写入类别标签或者类别相关进行进行label embedidng的初始化
    模型返回为logits，可以使用CrossEntropy进行优化，或者softmax + nlloss

    Attributes:
        config: 配置
        sequence_feature_extract_layer: sequence特征抽取层，bert，但只返回bert的last_hidden_state
        label_embedding_layer: 初始化label embedidng，将其封装为可学习参数
        attention_layer: 计算bert输出last_hidden_state与label embedding的attention值，无可学习参数
        classification_layer: 分类层，与label embedding参数共享，为可学习参数

    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.sequence_feature_extract_layer = FeatureExtractLayer(config)

        # 将label embedding设置为可学习参数，维度为（bert_hidden_dim, label_nums）
        self.label_embedding_layer = torch.nn.Parameter(LabelEmbeddingLayer(config).label_embedding_layer_params, requires_grad=True)

        self.attention_layer = AjustiveAttentionLayer(config) if config["use_ajustive_attention"] else AttentionLayer(config)

        self.classification_layer = ClassificationLayer(config, self.label_embedding_layer)

    def forward(self, input_ids, attention_mask, token_type_ids):
        deep_features = self.sequence_feature_extract_layer(input_ids, attention_mask, token_type_ids)
        features, atten, normalized = self.attention_layer(deep_features, self.label_embedding_layer, attention_mask)
        # shape of features(bsz, label_embedding_dim) eg:(32, 768)
        logits = self.classification_layer(features)
        label_features = self.label_embedding_layer.clone().detach()
        return ModelOutput(logits, features, atten, normalized, label_features)


class ModelOutput:
    def __init__(self, logits, features, atten, normalized, label_features):
        self.logits = logits
        self.features_after_attention = features
        self.attention_raw = atten
        self.attention_normalized = normalized
        self.label_features = label_features


class LargeMarginCosineLoss(torch.nn.Module):
    """
    Softmax and sigmoid focal loss
    """

    def __init__(self, num_labels=3,  scale=30, margin=0.35,activation_type='softmax'):

        super(LargeMarginCosineLoss, self).__init__()
        self.num_labels = num_labels
        self.scale = scale
        self.margin = margin
        self.activation_type = activation_type

    def forward(self, input, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            
        """
        if self.activation_type == 'softmax':
            logits = F.softmax(input, dim=-1)
        elif self.activation_type == 'sigmoid':
            logits = F.sigmoid(input)
        y_true = F.one_hot(target, self.num_labels)
        y_pred = y_true * (logits - self.margin) + (1 - y_true) * logits
        y_pred *= self.scale
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(y_pred.view(-1, self.num_labels), target.view(-1))

        return loss