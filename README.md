# Facebook-Link-Prediction
这篇报告会将链接预测的所有步骤应用到有趣的真实场景中。
报告将使用一个图形数据集，其中的节点是来自全球各地的流行餐厅和知名厨师的 Facebook页面。如果任何两个页面（节点）相互喜欢，则它们之间存在边（链接）。
数据集中包含620个节点，超过2100条边，整个网络的密度大约是0.0108969，网络的平均聚类系数大概是0.330897。数据集的特征均符合真实网络。

## 算法
### （1）随机删除链接

按照基本原则，应该避免删除任何可能产生孤立节点（没有任何边的节点）或孤立网络的边。我们必须确保在丢弃边缘的过程中，图的所有节点都应该保持连接。
在下面的代码中，我们将首先检查删除节点对是否会导致图分裂（number_connected_components>1）或节点数量减少。如果这两种情况都没有发生，那么我们删除那个节点对，并对下一个节点对重复相同的过程。这样就能得到过去时间的网络连接图。
  
```python
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
    
```
### （2）node2vec
  node2vec的思想同DeepWalk一样：生成随机游走，对随机游走采样得到（节点，上下文）的组合，然后用处理词向量的方法对这样的组合建模得到网络节点的表示。不过在生成随机游走过程中做了一些创新，可以说node2vec是一种有偏见的随机游走模型。
	node2vec既能保持节点邻居信息而且又容易训练。它满足同一个社区内的节点表示相似，拥有类似结构特征的节点表示相似两个基本要求。
	随机游走的采样主要有两种方式，深度优先游走（Depth-first Sampling，DFS）和广度优先游走（Breadth-first Sampling，BFS）。BFS倾向于在初始节点的周围游走，可以反映出一个节点的邻居的微观特性；而DFS一般会跑的离初始节点越来越远，可以反映出一个节点邻居的宏观特性。所以node2vec改进了DeepWalk中随机游走的方式，使它综合DFS和BFS的特性，引入了两个参数p和q来控制随机游走的方式。
![image](https://user-images.githubusercontent.com/44738680/147557916-079dd99f-7b30-4519-80dc-6378ed976a52.png)

### （3）逻辑回归模型
逻辑回归是一种分类算法，是机器学习中最简单也常用的一种训练模型。它用于在给定一组自变量的情况下预测二元结果（1/0、是/否、真/假）。为了表示二元/分类结果，我们使用虚拟变量，当结果变量是分类变量时，您也可以将逻辑回归视为线性回归的一个特例，即逻辑回归模型就是一个被logistic方程归一化后的线性分类模型。简单地说，它通过将数据拟合到 logit 函数来预测事件发生的概率。
典型的逻辑模型图如下所示，可以看到概率永远不会低于0和高于1。

![image2](https://s2.loli.net/2021/12/28/a7dyfhZDzBSArsG.png)

### （4）LightGBM
在逻辑回归模型之外，我们还采用了一个更加复杂的训练模型LightGBM。
Light GBM (Light Gradient Boosting Machines)是一种基于决策树算法的快速、分布式、高性能梯度提升框架，用于排序、分类和许多其他机器学习任务。
由于它基于决策树算法，因此它以最佳拟合方式分割树叶，而其他提升算法则是按深度或水平而不是按叶分割树。因此，当在 Light GBM中的同一片叶子上生长时，leaf-wise 算法可以比level-wise算法减少更多的损失，从而导致更好的精度，这是任何现有的 boosting 算法都很难实现的。此外，它的速度非常快，因此有“光”这个词。
在下图中可以看到XGBOOST的 逐级树增长和Light GBM中的逐叶拆分树，可以看到后者明显是一种更为高效的算法。逐叶拆分会导致复杂性增加并可能导致过度拟合，可以通过指定另一个参数max-depth来解决这一个问题，该参数指定将发生拆分的深度。
![image](https://s2.loli.net/2021/12/28/4i3zZHUKrwFPOWd.png)
![image](https://s2.loli.net/2021/12/28/RK8wu9PMLlhbUOx.png)



