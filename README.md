## 使用预训练模型+label embedding+改进loss+异常检测算法进行未知类别识别；新词检测
1. 将label embedding嵌入到bert的分类框架中，将label与sentence在相同的语义空间中进行编码，模型训练完成后，在欧几里得空间中，能够起到同类的文本聚到该类别的label embedding周围。
2. 将改进的loss函数引入，改进的loss函数能够在角空间中增加类间距离，减小内类距离，实现了LMCL以及Center Loss，使用Center Loss能够更容易使用基于密度的异常检测算法继续增强。
3. 实现异常检测算法，发现异常样本，异常样本即未知类别的样本，分类器不应该将其分类到任何一个类别，所以希望能够通过异常检测算法检测出。
---
### Label Embedding
### Loss Function优化
### 异常检测算法
