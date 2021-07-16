"""
根据label embedding的值绘制热力图
"""
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def heat_map(bin_data_path):
    data = torch.load(bin_data_path)
    sentences, labels, attention_raw, attention_cls = data[0], data[1], data[2], data[3]
    print(sentences)
    print(labels)
    print(attention_raw)
    print(attention_cls)
    for i in range(len(sentences)):
        one_sentence = sentences[i]
        one_attention_raw = attention_raw[i, :, :]
        one_attention_cls = attention_cls[i, :, :]
        print(one_sentence)
        print(one_attention_raw.shape)
        print(one_attention_cls.shape)


if __name__=="__main__":
    # sns.set()
    # np.random.seed(0)
    # uniform_data = np.random.rand(10, 12)
    # ax = sns.heatmap(uniform_data)
    # plt.show()
    heat_map("analysis_data/data.bin")