"""
根据label embedding的值绘制热力图
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__=="__main__":
    sns.set()
    np.random.seed(0)
    uniform_data = np.random.rand(10, 12)
    ax = sns.heatmap(uniform_data)
    plt.show()