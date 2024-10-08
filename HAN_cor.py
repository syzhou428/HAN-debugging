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

# 如果您使用方法二，请使用以下代码
# data = pd.read_csv('Meituan_bj_no_cate.csv', sep=' ', header=None, skiprows=1, names=['userID', 'locationID', 'timeID', 'cateID', 'intentID'])

# 编码节点和标签
user_encoder = LabelEncoder()
location_encoder = LabelEncoder()
time_encoder = LabelEncoder()
#cate_encoder = LabelEncoder()

data['userID'] = user_encoder.fit_transform(data['userID'])
data['locationID'] = location_encoder.fit_transform(data['locationID'])
data['timeID'] = time_encoder.fit_transform(data['timeID'])
#data['cateID'] = cate_encoder.fit_transform(data['cateID'])

intent_encoder = LabelEncoder()
data['intentID'] = intent_encoder.fit_transform(data['intentID'])

# 2. 构建异质图

hetero_data = HeteroData()

hetero_data['user'].num_nodes = data['userID'].nunique()
hetero_data['location'].num_nodes = data['locationID'].nunique()
hetero_data['time'].num_nodes = data['timeID'].nunique()
#hetero_data['category'].num_nodes = data['cateID'].nunique()

# 添加边
edge_index_u_l = torch.tensor([
    data['userID'].values,
    data['locationID'].values
], dtype=torch.long)
hetero_data['user', 'interacts', 'location'].edge_index = edge_index_u_l

edge_index_u_t = torch.tensor([
    data['userID'].values,
    data['timeID'].values
], dtype=torch.long)
hetero_data['user', 'interacts', 'time'].edge_index = edge_index_u_t

# edge_index_u_c = torch.tensor([
#     data['userID'].values,
#     data['cateID'].values
# ], dtype=torch.long)
# hetero_data['user', 'interacts', 'category'].edge_index = edge_index_u_c

# 添加反向边
edge_index_l_u = torch.tensor([
    data['locationID'].values,
    data['userID'].values
], dtype=torch.long)
hetero_data['location', 'interacts', 'user'].edge_index = edge_index_l_u

edge_index_t_u = torch.tensor([
    data['timeID'].values,
    data['userID'].values
], dtype=torch.long)
hetero_data['time', 'interacts', 'user'].edge_index = edge_index_t_u

# edge_index_c_u = torch.tensor([
#     data['cateID'].values,
#     data['userID'].values
# ], dtype=torch.long)
# hetero_data['category', 'interacts', 'user'].edge_index = edge_index_c_u

# 添加节点特征
embedding_dim = 128
num_nodes_dict = {
    'user': hetero_data['user'].num_nodes,
    'location': hetero_data['location'].num_nodes,
    'time': hetero_data['time'].num_nodes,
    #'category': hetero_data['category'].num_nodes,
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
                self.user_embedding.weight.data[:, :feature_dim] = user_features
                self.features['user'] = self.user_embedding.weight.data
            else:
                self.features['user'] = user_features

        # Define HANConv layer
        self.conv = HANConv(in_channels=embedding_dim, out_channels=embedding_dim, dropout=0.6,metadata=metadata)

        # Classifier
        self.classifier = torch.nn.ModuleDict()
        for node_type in ['user', 'time', 'location']:
            self.classifier[node_type] = torch.nn.Linear(embedding_dim, num_classes)

    def forward(self, x_dict, edge_index_dict):
        # Get node embeddings
        x_dict = {node_type: self.embeddings[node_type](x_dict[node_type]) if node_type not in self.features else self.features[node_type] for node_type in x_dict.keys()}

    # HANConv
        x_dict = self.conv(x_dict, edge_index_dict)

    # Classification
        out_dict = {node_type: self.classifier[node_type](x_dict[node_type]) for node_type in ['user', 'time', 'location']}

        return out_dict
    
# Assign 'intent' labels to 'time' and 'location' nodes
for node_type in ['user', 'time', 'location']:
    labels = torch.zeros(hetero_data[node_type].num_nodes, dtype=torch.long)
    for idx, row in data.iterrows():
        node_idx = row[node_type+'ID']
        label = row['intentID']
        labels[node_idx] = label
    hetero_data[node_type].y = labels

    # Split the dataset
    num_nodes = hetero_data[node_type].num_nodes
    all_indices = np.arange(num_nodes)
    train_val_indices, test_indices = train_test_split(all_indices, test_size=0.3, random_state=42)
    train_indices, valid_indices = train_test_split(train_val_indices, test_size=0.15, random_state=42)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    valid_mask[valid_indices] = True
    test_mask[test_indices] = True

    hetero_data[node_type].train_mask = train_mask
    hetero_data[node_type].valid_mask = valid_mask
    hetero_data[node_type].test_mask = test_mask

# 训练设置
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
hetero_data = hetero_data.to(device)
for node_type, features in hetero_data.x_dict.items():
    print(f"Shape of {node_type} features: {features.shape}")
num_classes = data['intentID'].nunique()
#model = HAN(num_nodes_dict, embedding_dim, num_classes, metadata=hetero_data.metadata()).to(device)
# Initialize HAN with the given user_features for 'user' nodes
feature_dict = {'user': user_features}
model = HAN(num_nodes_dict, embedding_dim, num_classes, hetero_data.metadata(), feature_dict).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.006)
criterion = torch.nn.CrossEntropyLoss()


# 早停机制
patience = 6
best_valid_loss = float('inf')
epochs_no_improve = 0
best_model_path = 'han_best_model.pth'
num_epochs = 160  # 最大训练轮数
loss_dict={'user':0.4,'time':0.5,'location':0.1}

# Training and validation
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    out_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    loss = 0
    for node_type in ['user', 'time', 'location']:
        loss += loss_dict[node_type]*criterion(out_dict[node_type][hetero_data[node_type].train_mask], hetero_data[node_type].y[hetero_data[node_type].train_mask])
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        out_dict = model(hetero_data.x_dict, hetero_data.edge_index_dict)
        val_loss = 0
        for node_type in ['user', 'time', 'location']:
            val_loss += loss_dict[node_type]*criterion(out_dict[node_type][hetero_data[node_type].valid_mask], hetero_data[node_type].y[hetero_data[node_type].valid_mask])
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}')

    if val_loss.item() < best_valid_loss:
        best_valid_loss = val_loss.item()
        epochs_no_improve = 0
        # 保存最佳模型
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation loss decreased, saving model to {best_model_path}")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epoch(s)")

    # 早停判断
    if epochs_no_improve >= patience and epoch > 32: 
        print("Early stopping!")
        break

# 5. 测试与评估

# 加载最佳模型
# model.load_state_dict(torch.load(best_model_path))

# model.eval()
# with torch.no_grad():
#     out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
#     preds_dict = {}
#     labels_dict = {}
#     for node_type in ['user', 'time', 'location']:
#         logits = out[node_type][hetero_data[node_type].test_mask]
#         preds_dict[node_type] = logits.argmax(dim=1).cpu().numpy()
#         labels_dict[node_type] = hetero_data[node_type].y[hetero_data[node_type].test_mask].cpu().numpy()
# # 测试模型
# # model.eval()
# # with torch.no_grad():
# #     out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
# #     logits = out['user'][hetero_data['user'].test_mask]
# #     preds = logits.argmax(dim=1).cpu().numpy()
# #     labels = hetero_data['user'].y[hetero_data['user'].test_mask].cpu().numpy()
# # with torch.no_grad():
# #     out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
# #     logits = out[hetero_data['user'].test_mask]
# #     preds = logits.argmax(dim=1).cpu().numpy()
# #     labels = hetero_data['user'].y[hetero_data['user'].test_mask].cpu().numpy()

# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# for node_type in ['user', 'time', 'location']:
#     precision = precision_score(labels_dict[node_type], preds_dict[node_type], average='macro', zero_division=0)
#     recall = recall_score(labels_dict[node_type], preds_dict[node_type], average='macro', zero_division=0)
#     f1 = f1_score(labels_dict[node_type], preds_dict[node_type], average='macro', zero_division=0)
#     accuracy = accuracy_score(labels_dict[node_type], preds_dict[node_type])

#     print(f"{node_type} Test Precision: {precision:.4f}")
#     print(f"{node_type} Test Recall:    {recall:.4f}")
#     print(f"{node_type} Test F1 Score:  {f1:.4f}")
#     print(f"{node_type} Test Accuracy:  {accuracy:.4f}")
# 计算指标
# precision = precision_score(labels, preds, average='macro', zero_division=0)
# recall = recall_score(labels, preds, average='macro', zero_division=0)
# f1 = f1_score(labels, preds, average='macro', zero_division=0)
# accuracy = accuracy_score(labels, preds)

# print(f"Test Precision: {precision:.4f}")
# print(f"Test Recall:    {recall:.4f}")
# print(f"Test F1 Score:  {f1:.4f}")
# print(f"Test Accuracy:  {accuracy:.4f}")

# 删除非最优模型（此处仅保存最佳模型，无需删除其他模型）
    
from scipy import stats

# Test and evaluate
model.load_state_dict(torch.load(best_model_path))
model.eval()
with torch.no_grad():
    out = model(hetero_data.x_dict, hetero_data.edge_index_dict)
    preds_dict = {}
    labels_dict = {}
    for node_type in ['user', 'time', 'location']:
        logits = out[node_type][hetero_data[node_type].test_mask]
        preds_dict[node_type] = logits.argmax(dim=1).cpu().numpy()
        labels_dict[node_type] = hetero_data[node_type].y[hetero_data[node_type].test_mask].cpu().numpy()

    # Combine the predictions and labels of 'user', 'time' and 'location' nodes
    combined_preds = [stats.mode(preds)[0] for preds in zip(*preds_dict.values())]
    combined_labels = [stats.mode(labels)[0] for labels in zip(*labels_dict.values())]

    precision = precision_score(combined_labels, combined_preds, average='macro', zero_division=0)
    recall = recall_score(combined_labels, combined_preds, average='macro', zero_division=0)
    f1 = f1_score(combined_labels, combined_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(combined_labels, combined_preds)

    print(f"Combined Test Precision: {precision:.4f}")
    print(f"Combined Test Recall:    {recall:.4f}")
    print(f"Combined Test F1 Score:  {f1:.4f}")
    print(f"Combined Test Accuracy:  {accuracy:.4f}")