import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import argparse

# PyTorch Geometric Imports
import torch_geometric.transforms as T
from torch_geometric.data import Data
# 关键: 我们使用基线项目中的 NeighborSampler
from torch_geometric.loader import NeighborSampler
from torch_geometric.nn import SAGEConv

# -----------------------------------------------------------------
# 1. 从 baseline 复制 SAGE_NeighSampler 模型
# (来源: models/sage_neighsampler.py)
#
# **重要修改**: 我已经移除了模型末尾的 log_softmax()，
# 因为 FocalLoss (和 CrossEntropyLoss) 期望原始的 logits。
# -----------------------------------------------------------------
class SAGE_NeighSampler(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(SAGE_NeighSampler, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        
    def forward(self, x, adjs):
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]] # 目标节点是批次中的中心节点
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        # **修改**: 返回 logits, 而非 log_softmax
        return x
    
    def inference(self, x_all, layer_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm: 
                        x = self.bns[i](x)
                xs.append(x)
                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)
        pbar.close()
        
        # **修改**: 返回 logits, 而非 log_softmax
        return x_all

# -----------------------------------------------------------------
# 2. 我们自己的 Focal Loss (保持不变)
# -----------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt)**self.gamma * ce_loss
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------------------------------------------
# 3. 数据加载 (适配 baseline 和 我们的数据)
# -----------------------------------------------------------------
def load_data(filepath='phase1_gdata.npz'):
    print("开始加载数据...")
    try:
        data_np = np.load(filepath, allow_pickle=True)
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {filepath}")
        return None, None
        
    # 特征归一化 (来自 gnn_mini_batch.py)
    x_raw = data_np['x']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    x = torch.tensor(x_scaled, dtype=torch.float32)

    # 修复 y (squeeze)
    y = torch.tensor(data_np['y'], dtype=torch.long).squeeze()

    # 修复 edge_index (transpose)
    edge_index_raw = torch.tensor(data_np['edge_index'], dtype=torch.long)
    edge_index = edge_index_raw.t().contiguous() 

    # 修复 masks (load as long)
    train_mask = torch.tensor(data_np['train_mask'], dtype=torch.long)
    test_mask = torch.tensor(data_np['test_mask'], dtype=torch.long)

    # 创建 Data 对象 (NeighborLoader/Sampler 需要)
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # **关键**: 基线项目需要 'valid_mask'
    # 我们从 'train_mask' 中分割出一个验证集 (例如 90% 训练, 10% 验证)
    print("创建训练集/验证集分割...")
    indices = torch.randperm(train_mask.size(0))
    split_point = int(train_mask.size(0) * 0.9)
    data.train_mask = train_mask[indices[:split_point]]
    data.valid_mask = train_mask[indices[split_point:]]
    data.test_mask = test_mask # 原始测试集
    
    print("将图转换为 SparseTensor (adj_t)...")
    # 使用 baseline 的 ToSparseTensor 转换 (来自 gnn_mini_batch.py)
    # 这会创建 data.adj_t, NeighborSampler 需要它
    data = T.ToSparseTensor()(data)
    
    # 存储分割索引 (用于 test 函数)
    split_idx = {'train': data.train_mask, 
                 'valid': data.valid_mask, 
                 'test': data.test_mask}

    print("数据加载完成。")
    return data, split_idx

# -----------------------------------------------------------------
# 4. 训练函数 (来自 baseline gnn_mini_batch.py)
# -----------------------------------------------------------------
def train(model, train_loader, optimizer, loss_fn, device, data_x, data_y, train_idx_len):
    model.train()
    
    # data_x 和 data_y 已经在GPU上
    pbar = tqdm(total=train_idx_len, ncols=80)
    pbar.set_description('Training')

    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        
        # 手动获取子图的特征
        x_batch = data_x[n_id].to(device)
        
        # 模型前向传播 (只获取中心节点的输出)
        out = model(x_batch, adjs)
        
        # 获取中心节点的标签
        y_batch = data_y[n_id[:batch_size]]
        
        # 使用我们的 FocalLoss
        loss = loss_fn(out, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    return total_loss / len(train_loader)

# -----------------------------------------------------------------
# 5. 评测函数 (改编自 baseline)
# -----------------------------------------------------------------
@torch.no_grad()
def evaluate_auc(y_true, probs_np):
    """
    我们自定义的多分类AUC评测函数 (处理-100标签)
    (已更新: 使用 baseline 的 one-hot 编码策略来修复 sklearn bug)
    """
    
    # 1. 过滤掉 -100 标签
    valid_indices = (y_true != -100)
    if valid_indices.sum() == 0:
        return 0.5 # 默认
        
    y_true_filtered = y_true[valid_indices]
    probs_np_filtered = probs_np[valid_indices]
    
    # 2. 检查剩余的有效标签是否多于一类
    if len(np.unique(y_true_filtered)) < 2:
        return 0.5 # 无法计算
        
    # 3. 采用 baseline 的 one-hot 编码策略
    #    (修复 roc_auc_score 在 y_true 只有两类时忽略 'multi_class' 的bug)
    
    num_classes = probs_np_filtered.shape[1] # 应该是 4
    
    # 我们 one-hot 编码 y_true。
    # 即使 y_true_filtered 只有 0 和 1, 
    # onehot_code[y_true_filtered] 也会创建
    # 一个 (N_filtered, 4) 的数组，最后两列为0，这是正确的。
    onehot_code = np.eye(num_classes)
    y_true_onehot = onehot_code[y_true_filtered]
    
    # 现在我们比较 (N, 4) vs (N, 4), roc_auc_score 会正确处理
    # 我们使用 'weighted' 来处理不平衡
    return roc_auc_score(y_true_onehot, probs_np_filtered, average='weighted')

@torch.no_grad()
def test(model, data, layer_loader, device, split_idx, loss_fn):
    model.eval()
    
    # 使用 baseline 的高效 inference 方法
    out = model.inference(data.x, layer_loader, device)
    y_pred = F.softmax(out, dim=1)
    
    losses, eval_results = dict(), dict()
    
    for key in ['train', 'valid', 'test']:
        mask = split_idx[key]
        
        # --- 计算 Loss ---
        if key in ['train', 'valid']:
            # 这些掩码只指向有效的标签 (0 或 1)
            losses[key] = loss_fn(out[mask], data.y[mask]).item()
        else:
            losses[key] = 0.0 # 不计算测试集的损失 (因为有-100)
            
        # --- 计算 Eval (AUC) ---
        y_true = data.y[mask].cpu().numpy()
        y_pred_np = y_pred[mask].cpu().numpy()
        eval_results[key] = evaluate_auc(y_true, y_pred_np)
            
    return eval_results, losses

# -----------------------------------------------------------------
# 6. 主执行逻辑 (改编自 baseline)
# -----------------------------------------------------------------
def main():
    # --- 超参数 (来自 gnn_mini_batch.py SAGE) ---
    LR = 0.0005
    NUM_LAYERS = 2
    HIDDEN_CHANNELS = 128
    DROPOUT = 0.0
    BATCHNORM = False
    L2 = 5e-7
    
    EPOCHS = 50
    GNN_BATCH_SIZE = 1024
    NUM_WORKERS = 4
    GAMMA = 2.0 # Focal Loss
    
    # 1. 加载数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, split_idx = load_data('phase1_gdata.npz')
    if data is None:
        return
        
    # **关键**: 将整个数据 (特征、标签、图结构) 一次性移到GPU
    # NeighborSampler 将从这里拉取数据
    data = data.to(device)
    
    # 获取训练集索引 (用于 Sampler)
    train_idx = split_idx['train'].to(device)
    NUM_CLASSES = 2 # 我们的数据是4分类

    # 2. 定义数据加载器 (来自 baseline)
    print("创建 NeighborSampler...")
    train_loader = NeighborSampler(data.adj_t, 
                                   node_idx=train_idx, 
                                   sizes=[10, 5], 
                                   batch_size=GNN_BATCH_SIZE, 
                                   shuffle=True, 
                                   num_workers=NUM_WORKERS)
    
    # layer_loader 用于 'inference'，采样所有邻居 (-1)
    layer_loader = NeighborSampler(data.adj_t, 
                                   node_idx=None, 
                                   sizes=[-1], 
                                   batch_size=4096, 
                                   shuffle=False, 
                                   num_workers=NUM_WORKERS)
    
    # 3. 实例化模型 (来自 baseline)
    model = SAGE_NeighSampler(
        in_channels = data.x.size(-1), 
        hidden_channels = HIDDEN_CHANNELS,
        out_channels = NUM_CLASSES, # 4 分类
        num_layers = NUM_LAYERS,
        dropout = DROPOUT,
        batchnorm = BATCHNORM
    ).to(device)
    
    # 4. 定义损失函数 (我们自己的)
    train_y_valid = data.y[train_idx] # train_idx 已经只包含 0 和 1
    class_counts = torch.bincount(train_y_valid, minlength=NUM_CLASSES)
    total_samples = train_y_valid.size(0)
    class_weights = []
    for count in class_counts:
        count_item = count.item()
        if count_item == 0: class_weights.append(0.0)
        else: class_weights.append(total_samples / (NUM_CLASSES * count_item))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"训练集中各类别数量: {class_counts.cpu().numpy()}")
    print(f"使用 FocalLoss (Gamma={GAMMA}, Alpha={class_weights.cpu().numpy()})")
    loss_fn = FocalLoss(alpha=class_weights, gamma=GAMMA)
    
    # 5. 定义优化器 (来自 baseline)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    
    # 6. 训练循环
    print(f"开始在 {device} 上训练 (GraphSAGE Mini-Batch)...")
    
    # 将 x 和 y 显式传递给训练/测试函数
    # (这是 baseline 的做法，可以避免在循环中重复索引 data 对象)
    data_x = data.x
    data_y = data.y
    
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_loader, optimizer, loss_fn, 
                     device, data_x, data_y, train_idx.size(0))
        
        eval_results, losses = test(model, data, layer_loader, device, 
                                      split_idx, loss_fn)
        
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f} | "
              f"Train AUC: {eval_results['train']:.4f} (Loss: {losses['train']:.4f}) | "
              f"Valid AUC: {eval_results['valid']:.4f} (Loss: {losses['valid']:.4f}) | "
              f"Test AUC: {eval_results['test']:.4f}")

    print("训练完成。")
    #保存模型
    torch.save(model.state_dict(), 'sage_neighsampler_focalloss.pth')
    #生成测试集预测结果
    eval_results, losses = test(model, data, layer_loader, device, split_idx, loss_fn)
    y_pred = model.inference(data.x, layer_loader, device)
    y_pred_probs = F.softmax(y_pred, dim=1)
    test_mask = split_idx['test']
    test_preds = y_pred_probs[test_mask].cpu().numpy()
    np.savez_compressed('test_predictions.npz', predictions=test_preds)
    print("测试集预测结果已保存到 'test_predictions.npz'。")

if __name__ == "__main__":
    main()