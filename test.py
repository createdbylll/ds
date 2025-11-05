import numpy as np
import os
# 设置cuda gpt id
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TGNMemory, GATConv
from torch_geometric.nn.models.tgn import TimeEncoder, LastAggregator, IdentityMessage
# 导入评估指标
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
# 导入特征归一化工具
from sklearn.preprocessing import StandardScaler

# --- 1. Focal Loss 实现 ---
class FocalLoss(nn.Module):
    """
    一个健壮且数值稳定的Focal Loss实现。
    
    用法：
      loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)
      loss = loss_fn(logits, labels)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor, optional): 形状为 (C,) 的类别权重张量。 
                                           例如 [weight_for_class_0, weight_for_class_1]
            gamma (float, optional): 聚焦参数 gamma. 默认为 2.0.
            reduction (str, optional): 规约方式: 'mean', 'sum', 'none'. 
                                     默认为 'mean'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): 模型的原始输出 (logits), 
                                   形状 (N, C)
            targets (torch.Tensor): 真实标签, 形状 (N,)
        """
        
        # --- 关键步骤 1: 计算标准的交叉熵损失 (但不安抚) ---
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # --- 关键步骤 2: 计算 p_t (模型对正确类别的预测概率) ---
        pt = torch.exp(-ce_loss)
        
        # --- 关键步骤 3: 计算Focal Loss项 ---
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        # --- 关键步骤 4: 应用 alpha 权重 ---
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
            
        # --- 关键步骤 5: 应用规约 (Reduction) ---
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 2. 数据加载与预处理 (已更新) ---
def load_and_preprocess_data(filepath='phase1_gdata.npz'):
    """
    加载.npz文件并将其转换为TGN所需的格式。
    (已修复: train_mask/test_mask 作为索引 (long) 加载, 而不是布尔值 (bool))
    """

    data = np.load(filepath, allow_pickle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 优化点 1: 特征归一化 ---
    x_raw = data['x']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw)
    x = torch.tensor(x_scaled, dtype=torch.float32)

    y = torch.tensor(data['y'], dtype=torch.long).squeeze()
    edge_type = torch.tensor(data['edge_type'], dtype=torch.float32)
    edge_timestamp = torch.tensor(data['edge_timestamp'], dtype=torch.float32).squeeze()

    # --- 关键修复: Edge Index (保持不变) ---
    edge_index_raw = torch.tensor(data['edge_index'], dtype=torch.long)
    edge_index = edge_index_raw.t().contiguous() 
    
    num_nodes = x.size(0)

    # --- 关键修复: 加载 Mask (从 bool -> long) ---
    if 'train_mask' not in data or 'test_mask' not in data:
         print("错误：数据中未找到 'train_mask' 或 'test_mask'。")
         print("将使用随机索引创建占位符...")
         num_nodes_for_mask = x.size(0)
         
         # 创建随机索引，而不是布尔掩码
         num_train = int(0.6 * num_nodes_for_mask)
         indices = torch.randperm(num_nodes_for_mask)
         train_mask = indices[:num_train] # (dtype=long by default)
         test_mask = indices[num_train:]  # (dtype=long by default)
    else:
        # 将 mask 作为 long 索引加载
        train_mask = torch.tensor(data['train_mask'], dtype=torch.long)
        test_mask = torch.tensor(data['test_mask'], dtype=torch.long)
    # --- 修复结束 ---

    # --- TGN 核心预处理 (保持不变) ---
    perm = edge_timestamp.argsort()
    src = edge_index[0][perm].to(device)
    dst = edge_index[1][perm].to(device)
    t = edge_timestamp[perm].to(device)
    msg = edge_type[perm].view(-1, 1).to(device)
    static_edge_index = edge_index.to(device) 

    # 移动其他数据到设备
    x = x.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device) # 现在是 LongTensor
    test_mask = test_mask.to(device)   # 现在是 LongTensor
    
    print("数据加载与预处理完成 (已归一化, 已转置, mask为索引)。") # 更新了打印信息
    print(f"节点数: {num_nodes}")
    print(f"边数 (事件数): {src.size(0)}")
    print(f"节点特征维度: {x.size(1)}")
    
    return (num_nodes, x, y, src, dst, t, msg, 
            static_edge_index, train_mask, test_mask, device)


# --- 3. TGN 模型定义 (无变化) ---
class TGNNodeClassifier(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, msg_dim, 
                 time_dim, embed_dim, gnn_out_dim, num_classes):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        
        self.time_enc = TimeEncoder(time_dim)
        
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=msg_dim,
            memory_dim=node_feat_dim, 
            time_dim=time_dim,
            # FIX 1: Use PyG's 'IdentityMessage' module, not 'nn.Identity'
            message_module=IdentityMessage(msg_dim, node_feat_dim, time_dim), 
            # FIX 2: Remove the 'nn.' prefix
            aggregator_module=LastAggregator()
        )
        
        self.gnn = GATConv(node_feat_dim, gnn_out_dim)
        self.classifier = nn.Linear(gnn_out_dim, num_classes)
        self.register_buffer('initial_node_features', torch.zeros(num_nodes, node_feat_dim))

    def forward(self, src, dst, t, msg, static_edge_index):
        self.memory.reset_state()
        self.memory.update_state(src, dst, t, msg)
        z_final_memory = self.memory.memory.clone()
        z_gnn = self.gnn(z_final_memory, static_edge_index)
        z_gnn = F.relu(z_gnn)
        out = self.classifier(z_gnn)
        return out
        
    def detach_memory(self):
        self.memory.detach()

# --- 4. 训练与评估 (已更新) ---

def train(model, optimizer, loss_fn, src, dst, t, msg, static_edge_index, y, train_mask):
    model.train()
    optimizer.zero_grad()
    
    out = model(src, dst, t, msg, static_edge_index)
    
    loss = loss_fn(out[train_mask], y[train_mask])
    
    loss.backward()
    optimizer.step()
    
    model.detach_memory()
    
    return loss.item()

@torch.no_grad()
def test(model, src, dst, t, msg, static_edge_index, y, mask):
    """
    评估函数 (已更新为多类别)
    """
    model.eval()
    
    out = model(src, dst, t, msg, static_edge_index)
    
    # 获取预测
    logits = out[mask]
    # Probs shape is (N_test, 4)
    probs = F.softmax(logits, dim=1) 
    preds = logits.argmax(dim=1)            

    y_true = y[mask].cpu().numpy()
    preds_np = preds.cpu().numpy()
    probs_np = probs.cpu().numpy()
    
    # 1. 使用 'weighted' F1
    f1 = f1_score(y_true, preds_np, average='weighted', zero_division=0)
    
    # 2. 过滤掉测试集中未标记的样本 (label == -100)
    #    (根据赛题描述，测试样本label为-100)
    valid_indices = (y_true != -100)
    if valid_indices.sum() == 0:
        print("警告: 测试集中没有有效标签 (所有标签都是-100?)")
        return 0.0, 0.5 # 返回默认值
        
    y_true_filtered = y_true[valid_indices]
    probs_np_filtered = probs_np[valid_indices]
    
    # 3. 检查剩余的有效标签是否多于一类
    if len(np.unique(y_true_filtered)) < 2:
        roc_auc = 0.5 # 无法计算
    else:
        # 4. 使用 multi_class='ovr' (One-vs-Rest) 计算ROC-AUC
        roc_auc = roc_auc_score(y_true_filtered, 
                                probs_np_filtered, 
                                multi_class='ovr', 
                                average='weighted')
    
    # PR-AUC在多类别下定义不明确，我们统一报告 ROC-AUC
    return f1, roc_auc, roc_auc

# --- 5. 主执行逻辑 (已更新) ---
if __name__ == "__main__":
    
    # --- 超参数 (已更新) ---
    LR = 1e-4  # (0.0001) - 已调整的学习率
    EPOCHS = 50
    TIME_DIM = 32
    EMBED_DIM = 64
    GAMMA = 2.0 # Focal Loss 的 gamma 参数
    
    # 1. 加载数据
    (num_nodes, x, y, src, dst, t, msg, 
     static_edge_index, train_mask, test_mask, device) = load_and_preprocess_data()
    
    NODE_FEAT_DIM = x.size(1)
    MSG_DIM = msg.size(1)
    NUM_CLASSES = y.max().item() + 1
    
    # 2. 实例化模型
    model = TGNNodeClassifier(
        num_nodes=num_nodes,
        node_feat_dim=NODE_FEAT_DIM,
        msg_dim=MSG_DIM,
        time_dim=TIME_DIM,
        embed_dim=EMBED_DIM,
        gnn_out_dim=EMBED_DIM,
        num_classes=NUM_CLASSES
    ).to(device)
    
    model.initial_node_features = x.clone() 

# 3. 定义损失函数和优化器 (已更新为多类别)
    
    # --- 关键: 仅使用训练集(train_mask)中的标签来计算权重 ---
    train_y = y[train_mask]
    
    # 过滤掉训练集中可能存在的-100标签
    train_y_valid = train_y[train_y != -100]
    if train_y_valid.numel() == 0:
        raise ValueError("训练集中没有有效的标签!")

    # 根据赛题描述，标签为 0, 1, 2, 3
    NUM_CLASSES = 4 # 硬编码为4，或使用 y.max() + 1
    
    print(f"检测到 {NUM_CLASSES} 个类别。")

    # 计算所有4个类别的数量
    class_counts = torch.bincount(train_y_valid, minlength=NUM_CLASSES)
    total_samples = train_y_valid.size(0)
    
    class_weights = []
    for count in class_counts:
        count_item = count.item()
        if count_item == 0:
            # 如果训练集中没有这个类，给一个0权重，防止除以零
            class_weights.append(0.0) 
        else:
            # 标准权重: 总数 / (类别数 * 该类别数量)
            weight = total_samples / (NUM_CLASSES * count_item)
            class_weights.append(weight)
    
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"训练集中各类别数量: {class_counts.cpu().numpy()}")
    
    # --- 优化点 3: 切换到 Focal Loss ---
    print(f"使用 FocalLoss (Gamma={GAMMA}, Alpha={class_weights.cpu().numpy()})")
    loss_fn = FocalLoss(alpha=class_weights, gamma=GAMMA)
    
    # --- 优化点 4: 调整学习率 ---
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 4. 训练循环 (已更新打印信息)
    print("开始训练...")
    for epoch in range(1, EPOCHS + 1):
        loss = train(model, optimizer, loss_fn, 
                     src, dst, t, msg, static_edge_index, y, train_mask)
        
        # --- 优化点 5: 报告所有指标 ---
        f1, pr_auc, roc_auc = test(model, src, dst, t, msg, 
                                     static_edge_index, y, test_mask)
        
        print(f"Epoch: {epoch:02d}, Loss: {loss:.4f} | "
            f"Test F1 (weighted): {f1:.4f}, Test ROC-AUC (ovr): {roc_auc:.4f}")
    print("训练完成。")