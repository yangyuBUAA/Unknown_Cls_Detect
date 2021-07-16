"""
根据label embedding的值绘制热力图
"""
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['Songti SC']
plt.rcParams['axes.unicode_minus'] = False


def heat_map_raw(bin_data_path):
    data = torch.load(bin_data_path)
    sentences, labels, attention_raw, attention_cls = \
        data[0], data[1], data[2], data[3].detach().numpy()
    # print(sentences)
    # print(labels)
    # print(attention_raw)
    # print(attention_cls)
    for i in range(len(sentences)):
        one_sentence = sentences[i]
        one_attention_raw = attention_raw[i, :, :]
        one_attention_cls = attention_cls[i, :, :]
        one_sentence_split = list(one_sentence)
        one_sentence_split.insert(0, "s")
        one_sentence_split.append("e")
        for i in range(20 - len(one_sentence_split)):
            one_sentence_split.append("p")
        print(one_sentence_split)
        one_attention_raw = torch.softmax(one_attention_raw, dim=0).detach().numpy()

        df1 = pd.DataFrame(one_attention_raw, columns=labels, index=one_sentence_split)
        p = plt.figure(figsize=(6, 20), dpi=80)
        ax = sns.heatmap(df1, annot=True)
        plt.savefig("heatmap/raw" + one_sentence + ".jpg")
        plt.show()
        # print(sentences[i])
        # break


def heat_map_cls(bin_data_path):
    data = torch.load(bin_data_path)
    sentences, labels, attention_raw, attention_cls = \
        data[0], data[1], data[2], data[3].detach().numpy()
    for i in range(len(sentences)):
        one_sentence = sentences[i]
        one_attention_raw = attention_raw[i, :, :]
        one_attention_cls = attention_cls[i, :, :]
        one_sentence_split = list(one_sentence)
        one_sentence_split.insert(0, "s")
        one_sentence_split.append("e")
        for i in range(20 - len(one_sentence_split)):
            one_sentence_split.append("p")

        print(one_attention_cls[0])
        df1 = pd.DataFrame(one_attention_cls, columns=one_sentence_split)
        print(df1)
        p = plt.figure(figsize=(20, 5), dpi=80)
        ax = sns.heatmap(df1, annot=True)
        plt.savefig("heatmap/cls" + one_sentence + ".jpg")
        plt.show()


if __name__=="__main__":
    # sns.set()
    # np.random.seed(0)
    # uniform_data = np.random.rand(10, 12)
    # ax = sns.heatmap(uniform_data)
    # plt.show()
    heat_map_raw("analysis_data/data3.bin")
    heat_map_cls("analysis_data/data3.bin")