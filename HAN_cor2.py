import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# 1. 数据准备

# 读取数据
data = pd.read_csv('Meituan_bj_impl.csv')

# 提取 'userID', 'gender', 'age', 'kid', 'martial' 列并去重
unique_user_data = data[['userID', 'gender', 'age', 'kid', 'martial']].drop_duplicates()

# 提取 'gender', 'age', 'kid', 'martial' 四列作为用户特征
user_features = unique_user_data[['gender', 'age', 'kid', 'martial']].values

# 将 numpy 数组转换为 PyTorch 张量
user_features = torch.tensor(user_features, dtype=torch.float)

# 方法一：正确读取数据，指定列名
data.columns = [col.split(':')[0] for col in data.columns]

# 创建 time-location 联合节点
data['time_location'] = data['timeID'].astype(str) + '-' + data['locationID'].astype(str)

# 编码节点和标签
time_location_encoder = LabelEncoder()
user_encoder = LabelEncoder()
cate_encoder = LabelEncoder()  # 新增加cate节点

data['time_location'] = time_location_encoder.fit_transform(data['time_location'])
data['userID'] = user_encoder.fit_transform(data['userID'])
data['cateID'] = cate_encoder.fit_transform(data['cateID'])  # 对cateID进行编码

intent_encoder = LabelEncoder()
data['intentID'] = intent_encoder.fit_transform(data['intentID'])

# 2. 构造异质图

# 创建空的异质图数据对象
hetero_data = HeteroData()

# 设置节点数量
hetero_data['user'].num_nodes = data['userID'].nunique()
hetero_data['time_location'].num_nodes = data['time_location'].nunique()
hetero_data['cate'].num_nodes = data['cate_ID'].nunique()

# 添加边（用户与时间-空间节点的交互）
edge_index_u_tl = torch.tensor([
    data['userID'].values,
    data['time_location'].values
], dtype=torch.long)
hetero_data['user', 'interacts', 'time_location'].edge_index = edge_index_u_tl

# 添加反向边
edge_index_tl_u = torch.tensor([
    data['time_location'].values,
    data['userID'].values
], dtype=torch.long)
hetero_data['time_location', 'interacts', 'user'].edge_index = edge_index_tl_u

# 添加边（用户与类别节点的交互）
edge_index_u_c = torch.tensor([
    data['userID'].values,
    data['cateID'].values
], dtype=torch.long)
hetero_data['user', 'interacts', 'cate'].edge_index = edge_index_u_c

# 添加反向边
edge_index_c_u = torch.tensor([
    data['cateID'].values,
    data['userID'].values
], dtype=torch.long)
hetero_data['cate', 'interacts', 'user'].edge_index = edge_index_c_u

# 添加边（类别与时间-空间节点的交互）
edge_index_c_tl = torch.tensor([
    data['cateID'].values,
    data['time_location'].values
], dtype=torch.long)
hetero_data['cate', 'interacts', 'time_location'].edge_index = edge_index_c_tl

# 添加反向边
edge_index_tl_c = torch.tensor([
    data['time_location'].values,
    data['cateID'].values
], dtype=torch.long)
hetero_data['time_location', 'interacts', 'cate'].edge_index = edge_index_tl_c

# 添加节点特征
embedding_dim = 128
num_nodes_dict = {
    'user': hetero_data['user'].num_nodes,
    'time_location': hetero_data['time_location'].num_nodes,
    'cate': hetero_data['cate'].num_nodes,
}

for node_type in num_nodes_dict.keys():
    hetero_data[node_type].x = torch.arange(num_nodes_dict[node_type])

# 3. 构建模型
class HAN(torch.nn.Module):
    def __init__(self, num_nodes_dict, embedding_dim, num_classes, metadata, feature_dict=None):
        super(HAN, self).__init__()
        self.num_nodes_dict = num_nodes_dict
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Trainable node embeddings
        self.embeddings = torch.nn.ModuleDict()
        for node_type in num_nodes_dict.keys():
            self.embeddings[node_type] = torch.nn.Embedding(num_nodes_dict[node_type], embedding_dim)

        # Initial feature vectors
        self.features = {}
        if feature_dict is not None and 'user' in feature_dict:
            user_features = feature_dict['user']
            feature_dim = user_features.shape[1]
            if feature_dim < embedding_dim:
                # Create an embedding layer to expand the feature vectors
                self.user_embedding = torch.nn.Embedding(num_nodes_dict['user'], embedding_dim)
                # Initialize the first 'feature_dim' dimensions of the embedding layer with 'user_features'
                self.user_embedding.weight.data[:feature_dim] = user_features
            else:
                self.features['user'] = user_features

        # HAN layer
        self.han = HANConv(
            embedding_dim,
            num_classes,
            num_layers=2,
            heads=2,
            concat=True,
            dropout=0.6,
        )

        # MLP Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x_dict, edge_index_dict):
        # Get embeddings for all node types
        x_dict = {node_type: self.embeddings[node_type](x_dict[node_type]) for node_type in x_dict.keys()}

        # If user features are available, use them
        if 'user' in self.features:
            x_dict['user'] = self.features['user']

        # Apply HAN layer
        x = self.han(x_dict, edge_index_dict)

        # Apply MLP classifier
        x = self.classifier(x)

        return x.log_softmax(dim=-1)   

# 获取所有的边
all_edges = []
for rel in hetero_data.edge_index_dict.keys():
    edges = hetero_data.edge_index_dict[rel].t().numpy()
    all_edges.append(edges)
all_edges = np.concatenate(all_edges, axis=0)

# 划分边为训练集、验证集和测试集
train_val_edges, test_edges = train_test_split(all_edges, test_size=0.3, random_state=42)
train_edges, val_edges = train_test_split(train_val_edges, test_size=0.15, random_state=42)

# 确保所有的节点在训练集中可见
for node_type in ['user', 'time_location', 'cate']:
    num_nodes = hetero_data[node_type].num_nodes
    train_nodes = np.unique(train_edges)
    missing_nodes = set(range(num_nodes)) - set(train_nodes)
    for node in missing_nodes:
        # 找到与缺失节点相关的边
        related_edges = all_edges[(all_edges == node).any(axis=1)]
        # 将这些边添加到训练集中
        train_edges = np.concatenate([train_edges, related_edges], axis=0)

# 获取所有的节点
all_nodes = np.unique(all_edges)

# # 创建一个字典来存储每种类型的节点的数量
# num_nodes_dict = {node_type: hetero_data[node_type].num_nodes for node_type in ['user', 'time_location', 'cate']}

# # Assume that 'user_behaviors' is a dictionary where the keys are user IDs and the values are their behaviors
# labels = {node_type: torch.zeros(num_nodes_dict[node_type], dtype=torch.long) for node_type in num_nodes_dict.keys()}
# for user_id, behavior in user_behaviors.items():
#     labels['user'][user_id] = behavior
    
# # 创建训练和验证的掩码
# train_mask = {node_type: np.isin(range(num_nodes_dict[node_type]), np.unique(train_edges)) for node_type in ['user', 'time_location', 'cate']}
# val_mask = {node_type: np.isin(range(num_nodes_dict[node_type]), np.unique(val_edges)) for node_type in ['user', 'time_location', 'cate']}

# # 将掩码转换为Tensor
# for node_type in ['user', 'time_location', 'cate']:
#     train_mask[node_type] = torch.from_numpy(train_mask[node_type])
#     val_mask[node_type] = torch.from_numpy(val_mask[node_type])

# device = torch.device('cpu')
# hetero_data = hetero_data.to(device)

# metadata = [('user', 'time_location'), ('time_location', 'user'), ('user', 'cate'), ('cate', 'user'), ('cate', 'time_location'), ('time_location', 'cate')]
# num_classes = data['intentID'].nunique()
# feature_dict = {'user': user_features}

# # 初始化模型
# model = HAN(num_nodes_dict=num_nodes_dict, embedding_dim=embedding_dim, num_classes=num_classes, metadata=metadata,feature_dict=feature_dict)
# criterion = torch.nn.CrossEntropyLoss()  # Loss function
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01,weight_decay=0.005)  # Optimizer

# # Early stopping parameters
# patience = 6
# best_val_loss = float('inf')
# counter = 0
# num_epochs=120

# for epoch in range(num_epochs):
#     # Training
#     model.train()
#     optimizer.zero_grad()  # Clear gradients
#     out = model(hetero_data.x_dict, hetero_data.edge_index_dict)  # Forward pass
#     loss = sum(criterion(out[node_type][train_mask[node_type]], labels[node_type][train_mask[node_type]]) for node_type in out.keys())  # Compute loss
#     loss.backward()  # Backward pass
#     optimizer.step()  # Update weights

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         val_out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
#         val_loss = sum(criterion(val_out[node_type][val_mask[node_type]], labels[node_type][val_mask[node_type]]) for node_type in val_out.keys())

#     # Print progress
#     print(f'Epoch: {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}')

#     # Check if we need to early stop
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         counter = 0
#     else:
#         counter += 1
#         if counter >= patience:
#             print('Early stopping...')
#             break