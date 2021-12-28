# -*- coding: utf-8 -*-
"""
社会网络分析期末大作业
Created on Wed Dec 22 17:20:26 2021

@author: 10428
"""
#0-导入库和模块
import pandas as pd
import numpy as np
import random
import networkx as nx
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#【了解数据】
#1加载 Facebook 页面作为节点，页面之间的相互喜欢作为边缘
# load nodes details
with open("fb-pages-food.nodes",encoding='utf-8') as f:
    fb_nodes = f.read().splitlines() 

# load edges (or links)
with open("fb-pages-food.edges") as f:
    fb_links = f.read().splitlines() 

len(fb_nodes), len(fb_links)	#会输出有 620 个节点和 2,102 个链接


#2创建所有节点的数据框。此数据帧的每一行代表分别由“node_1”和“node_2”列中的节点形成的链接
# captture nodes in 2 separate lists
node_list_1 = []
node_list_2 = []

for i in tqdm(fb_links):
  node_list_1.append(i.split(',')[0])
  node_list_2.append(i.split(',')[1])

fb_df = pd.DataFrame({'node_1': node_list_1, 'node_2': node_list_2})	#测试fb_df.head()
#fb_df.head()																			#节点“276”、“58”、“132”、“603”和“398”与节点“0”形成链接。

#3画图
# create graph
G = nx.from_pandas_edgelist(fb_df, "node_1", "node_2", create_using=nx.Graph())

# plot graph
plt.figure(figsize=(10,10))

pos = nx.random_layout(G,seed=23)
nx.draw(G, with_labels=False,  pos = pos, node_size = 40, alpha = 0.6, width = 0.7)

plt.show()	#描绘图表

#【模型构建的数据集准备】
#（1）检索未连接的节点对——负样本
#1使用邻接矩阵的这个属性从原始图G 中找到所有未连接的节点对：
# combine all nodes in a list
node_list = node_list_1 + node_list_2

# remove duplicate items from the list
node_list = list(dict.fromkeys(node_list))

# build adjacency matrix
adj_G = nx.to_numpy_matrix(G, nodelist = node_list)

#adj_G.shape
#测试邻接矩阵形状是(620, 620)的方阵

#2搜索对角线上方的值（绿色部分）或下方的值（红色部分）。让我们搜索零的对角线值：
# get unconnected node-pairs
all_unconnected_pairs = []


# traverse adjacency matrix
offset = 0
for i in tqdm(range(adj_G.shape[0])):
  for j in range(offset,adj_G.shape[1]):
    if i != j:
      if nx.shortest_path_length(G, str(i), str(j)) <=2:
        if adj_G[i,j] == 0:
          all_unconnected_pairs.append([node_list[i],node_list[j]])

  offset = offset + 1

#len(all_unconnected_pa​​irs)
#测试数据集中有多少个未连接的节点对：19018个未连接的对

#3这些节点对将在链路预测模型的训练过程中充当负样本。让我们将这些对保存在一个数据框中：
node_1_unlinked = [i[0] for i in all_unconnected_pairs]
node_2_unlinked = [i[1] for i in all_unconnected_pairs]

data = pd.DataFrame({'node_1':node_1_unlinked, 
                     'node_2':node_2_unlinked})

# add target variable 'link'
data['link'] = 0

#（2）从连接的节点对中删除链接 - 正样本
initial_node_count = len(G.nodes)

fb_df_temp = fb_df.copy()

# empty list to store removable links
omissible_links_index = []

for i in tqdm(fb_df.index.values):
  
  # remove a node pair and build a new graph
  G_temp = nx.from_pandas_edgelist(fb_df_temp.drop(index = i), "node_1", "node_2", create_using=nx.Graph())
  
  # check there is no spliting of graph and number of nodes is same
  if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
    omissible_links_index.append(i)
    fb_df_temp = fb_df_temp.drop(index = i)

#len(omissible_links_index)
#有超过 1400 个链接可以从图中删除。这些丢弃的边缘将在链路预测模型训练期间充当正训练示例。
    
    
#（3）模型训练数据
# create dataframe of removable edges
fb_df_ghost = fb_df.loc[omissible_links_index]

# add the target variable 'link'
fb_df_ghost['link'] = 1

data = data.append(fb_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)
#data['link'].value_counts()
#说明这是高度不平衡的数据。链接与无链接的比率接近 8%

#【特征提取】
# drop removable edges
fb_df_partial = fb_df.drop(index=fb_df_ghost.index.values)

# build graph
G_data = nx.from_pandas_edgelist(fb_df_partial, "node_1", "node_2", create_using=nx.Graph())

#在我们的图（G_data）上训练node2vec模型
from node2vec import Node2Vec

# Generate walks
node2vec = Node2Vec(G_data, dimensions=100, walk_length=16, num_walks=50)

# train node2vec model
n2w_model = node2vec.fit(window=7, min_count=1)

#要计算一对或边的特征，我们将把该对中节点的特征相加
#x = [(n2w_model[str(i)]+n2w_model[str(j)]) for i,j in zip(data['node_1'], data['node_2'])]
x = [(n2w_model.wv[str(i)]+n2w_model.wv[str(j)]) for i,j in zip(data['node_1'], data['node_2'])]

#【建立链接预测模型】
#将数据分成两部分——一部分用于训练模型，另一部分用于测试模型的性能：
xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'], 
                                                test_size = 0.3, 
                                                random_state = 35)

#先拟合一个逻辑回归模型： 
lr = LogisticRegression(class_weight="balanced")

lr.fit(xtrain, ytrain)

#对测试集进行预测
#predictions = lr.predict_proba(xtest)
#使用 AUC-ROC 分数来检查我们模型的性能
#roc_auc_score(ytest, predictions[:,1])

#使用更复杂的模型LightGBM
import lightgbm as lgbm

train_data = lgbm.Dataset(xtrain, ytrain)
test_data = lgbm.Dataset(xtest, ytest)

# define parameters
parameters = {
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'num_threads' : 2,
    'seed' : 76
}

# train lightGBM model
model = lgbm.train(parameters,
                   train_data,
                   valid_sets=test_data,
                   num_boost_round=1000,
                   early_stopping_rounds=20)

