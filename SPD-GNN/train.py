import dgl
import dgl.data
from dgl.nn import GraphConv, HeteroGraphConv, SAGEConv
from dgl.dataloading import GraphDataLoader
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import random_split
from torchmetrics import F1Score, Accuracy, Precision, Recall

from dgl.data import DGLDataset
from torch.utils.data import DataLoader
from model import *

import os
import numpy as np
import random
import itertools
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore", UserWarning)

import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.model_selection import KFold
import argparse
import copy

ModelName = "GCN"
featNum = 1

current_time = datetime.now().strftime('%Y-%m-%d')
# log_filename = f'SavedModels/logs/log_{current_time}.log'
# result_paths = ['Visualization/Address_Label_Pool/', 'Visualization/PU_Learning/','SavedModels/']
result_paths = ['SavedModels/', 'Visualization/AUC/', 'Visualization/Feature_Importance/']

for rp in result_paths:
    if not os.path.exists(rp+current_time):
        os.makedirs(rp+current_time, exist_ok=True)

# 获取当前时间并格式化为字符串

# # 设置日志配置
# logging.basicConfig(
#     filename=log_filename,  # 使用时间作为日志文件名
#     filemode='a',           # 追加模式
#     format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
#     level=logging.INFO       # 设置日志级别
# )

# 在导入部分之后，设置设备部分之前添加以下代码
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 只使用 GPU 3

# 修改设备选择代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")

def plot_auc(y_true, y_pred_prob, current_time, model_name):
    """
    绘制并保存 AUC 曲线及其原始数据
    Args:
        y_true: 真实标签
        y_pred_prob: 预测概率（需要正类的概率）
        current_time: 当前时间戳
        model_name: 模型名称
    """
    # 创建保存目录
    auc_dir = f'Visualization/AUC/{current_time}'
    os.makedirs(auc_dir, exist_ok=True)
    
    # 计算 ROC 曲线的值
    fpr, tpr, thresholds = roc_curve(y_true.cpu(), y_pred_prob.cpu())
    roc_auc = auc(fpr, tpr)
    
    # 保存原始数据
    data_dict = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': roc_auc,
        'y_true': y_true.cpu().numpy(),
        'y_pred_prob': y_pred_prob.cpu().numpy()
    }
    np.savez(f'{auc_dir}/{model_name}_ROC_data.npz', **data_dict)
    
    # 绘制 ROC 曲线
    plt.style.use('seaborn')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    
    # 保存图形
    plt.savefig(f'{auc_dir}/{model_name}_AUC_{roc_auc:.3f}.png', dpi=300, bbox_inches='tight')
    plt.close()

def pred(model, average, dataloader, debug=False):
    f1_score = F1Score(num_classes=2, task='multiclass', average=average)
    precision_score = Precision(num_classes=2, task='multiclass', average=average)
    recall_score = Recall(num_classes=2, task='multiclass', average=average)
    accuracy_score = Accuracy(num_classes=2, task='multiclass', average=average)
    y_pred = torch.tensor([], dtype=torch.float32).to(device)
    y_label = torch.tensor([], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['Feature'][:,:-featNum].float()
            #feat = batched_graph.ndata['Feature'].float()
            pred = model(batched_graph, feat)

            # 展平预测和标签
            pred = pred.view(-1, pred.shape[-1])  # (batch_size * num_nodes, num_classes)
            labels = labels.view(-1)  # (batch_size * num_nodes,)

            y_pred = torch.cat((y_pred, pred), 0)
            y_label = torch.cat((y_label, labels.to(device)), 0)

    # 获取预测概率
    probabilities = torch.softmax(y_pred, dim=1)
    
    acc = accuracy_score(y_pred.cpu().argmax(1), y_label.cpu())
    f1 = f1_score(y_pred.cpu().argmax(1), y_label.cpu())
    pr = precision_score(y_pred.cpu().argmax(1), y_label.cpu())
    re = recall_score(y_pred.cpu().argmax(1), y_label.cpu())
    
    if debug:
        print('Test accuracy:', acc)
        print('Precision: {}, Recall: {}, F1-score: {}'.format(pr, re, f1))
        
    return y_label, probabilities, acc, pr, re, f1


def train_original(model, n_epoch, optimizer, scheduler, dataloader, val_dataloader=None, class_weights=None):
    """原始版本的训练函数（没有 Recall 限制）"""
    best_f1 = 0
    best_model = None
    best_predictions = None
    best_labels = None
    best_metrics = None
    
    for epoch in tqdm(range(n_epoch)):
        for i, (batched_graph, labels) in enumerate(dataloader):
            batched_graph = batched_graph.to(device)
            model.train()
            feat = batched_graph.ndata['Feature'][:,:-featNum].float()
            optimizer.zero_grad()
            predd = model(batched_graph, feat)

            # 展平预测和标签
            predd = predd.view(-1, predd.shape[-1])
            labels = labels.view(-1)

            # 只使用标准交叉熵损失
            loss = F.cross_entropy(predd, labels.to(device), weight=class_weights)
            loss.backward()
            optimizer.step()

        # 验证阶段
        labels, probs, acc, pr, re, f1 = pred(model, 'weighted', val_dataloader)
        scheduler.step(f1)
        
        if(f1 > best_f1):
            best_f1 = f1
            best_model = model
            best_predictions = probs
            best_labels = labels
            best_metrics = {'acc': acc, 'f1': f1, 'pr': pr, 're': re}

    # 保存最佳模型
    model_name = f'SavedModels/{current_time}/Original_{ModelName}_{best_metrics["acc"]:.4f}_{best_metrics["f1"]:.4f}_{best_metrics["pr"]:.4f}_{best_metrics["re"]:.4f}.pt'
    torch.save(best_model, model_name)
    
    # 绘制 AUC 曲线
    plot_auc(best_labels, best_predictions[:, 1], current_time, f'Original_{ModelName}')
    
    # 分析特征重要性
    save_feature_importance(best_model, val_dataloader, current_time, f'Original_{ModelName}')

    return best_model, best_predictions, acc, pr, re, f1

def focal_loss(predd, labels, gamma=2.0, alpha=0.75):
    """
    Focal Loss 实现
    alpha: 正样本权重 (0.75 意味着正样本权重更大)
    gamma: 聚焦参数，使模型更关注难分类的样本
    """
    ce_loss = F.cross_entropy(predd, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1-pt)**gamma * ce_loss
    return focal_loss.mean()

def train_with_recall(model, n_epoch, optimizer, scheduler, dataloader, val_dataloader=None, class_weights=None):
    best_f1 = 0
    best_model = None
    best_predictions = None
    best_labels = None
    
    # 动态阈值，初始为0.5
    threshold = 0.5
    target_recall = 0.85
    min_precision = 0.75  # 添加最小精确率限制
    
    for epoch in range(n_epoch):
        model.train()
        total_loss = 0
        
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            feat = batched_graph.ndata['Feature'][:,:-featNum].float()
            optimizer.zero_grad()
            predd = model(batched_graph, feat)
            
            # 展平预测和标签
            predd = predd.view(-1, predd.shape[-1])
            labels = labels.view(-1).to(device)
            
            # 标准交叉熵损失
            loss = F.cross_entropy(predd, labels, weight=class_weights)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证阶段
        if val_dataloader:
            # 获取原始预测概率
            labels, predictions, _, _, _, _ = pred(model, 'weighted', val_dataloader)
            
            # 尝试不同的阈值
            best_threshold = threshold
            best_metrics = None
            
            # 在当前阈值附近搜索最佳阈值
            for t in np.arange(max(0.1, threshold-0.1), min(0.9, threshold+0.1), 0.02):
                pred_labels = (predictions[:, 1] > t).float()
                acc = accuracy_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
                pr = precision_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
                re = recall_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
                f1 = f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
                
                # 如果达到目标召回率且精确率可接受，更新最佳阈值
                if re >= target_recall and pr >= min_precision:
                    if best_metrics is None or f1 > best_metrics['f1']:
                        best_threshold = t
                        best_metrics = {'acc': acc, 'pr': pr, 're': re, 'f1': f1}
                
                # 如果找不到满足条件的阈值，选择精确率和召回率平衡的阈值
                elif best_metrics is None and re >= target_recall * 0.9:
                    if best_metrics is None or (pr + re) > (best_metrics['pr'] + best_metrics['re']):
                        best_threshold = t
                        best_metrics = {'acc': acc, 'pr': pr, 're': re, 'f1': f1}
            
            # 更新阈值和指标
            if best_metrics is not None:
                threshold = best_threshold
                acc, pr, re, f1 = best_metrics['acc'], best_metrics['pr'], best_metrics['re'], best_metrics['f1']
            else:
                # 使用当前阈值的指标
                pred_labels = (predictions[:, 1] > threshold).float()
                acc = accuracy_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
                pr = precision_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
                re = recall_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
                f1 = f1_score(labels.cpu().numpy(), pred_labels.cpu().numpy())
            
            if scheduler:
                scheduler.step(f1)  # 使用 F1 分数来调整学习率
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model)
                best_predictions = predictions
                best_labels = labels
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {total_loss:.4f}, F1: {f1:.4f}, Recall: {re:.4f}, '
                  f'Precision: {pr:.4f}, Threshold: {threshold:.2f}')
    
    return best_model, best_predictions, acc, pr, re, f1

def collate_fn(batch):
    # 使用 dgl.batch 将多个图合并成一个图
        batched_graph = dgl.batch([item[0] for item in batch])  # 获取每个样本的图
        labels = torch.cat([item[1] for item in batch])  # 获取每个样本的标签
        return batched_graph, labels

class MultiAddressDataset(DGLDataset):
    def __init__(self, graphs, feature_key='Feature', label_keys=['MLAddress', 'SPAddress']):
        self.graphs = graphs
        self.graphs = []
        self.node_labels = []

        for g in graphs:
            # 获取节点特征
            node_features = g.ndata[feature_key]
            
            # 获取标签
            ml_labels = g.ndata.get(label_keys[0], None)
            sp_labels = g.ndata.get(label_keys[1], None)
            
            if ml_labels is None or sp_labels is None:
                raise ValueError(f"Label keys {label_keys} are missing in the graph.")

            # 删除节点特征全为零的节点
            non_zero_nodes = (node_features.sum(dim=1) != 0)  # 判断每个节点的特征是否全为零
            g = dgl.node_subgraph(g, non_zero_nodes)  # 仅保留非零节点的子图

            # 获取更新后的节点特征和标签
            node_features = g.ndata[feature_key]
            ml_labels = g.ndata[label_keys[0]]
            sp_labels = g.ndata[label_keys[1]]

            # 合并 MLAddress 和 SPAddress 为一个标签：MLAddress为负样本（0），SPAddress为正样本（1）
            labels = torch.zeros_like(ml_labels)  # 默认值为0
            labels[sp_labels == 1] = 1  # 标记 SPAddress 为 1

            self.graphs.append(g)
            self.node_labels.append(labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.node_labels[idx]


def load_graphs(directory, directory2):
    all_graphs = []
    # 首先加载directory中的图以确定标准格式
    standard_graphs, _ = dgl.load_graphs(os.path.join(directory, os.listdir(directory)[0]))
    standard_graph = standard_graphs[0]
    standard_node_keys = set(standard_graph.ndata.keys())
    standard_edge_keys = set(standard_graph.edata.keys())

    def convert_graph_format(graph, standard_node_keys, standard_edge_keys):
        # 移除节点多余的标签
        current_node_keys = set(graph.ndata.keys())
        for key in current_node_keys - standard_node_keys:
            del graph.ndata[key]
        
        # 移除边多余的特征
        current_edge_keys = set(graph.edata.keys())
        for key in current_edge_keys - standard_edge_keys:
            del graph.edata[key]
        
        # 转换节点数据类型以匹配标准格式
        for key in graph.ndata:
            if key in standard_graph.ndata:
                target_dtype = standard_graph.ndata[key].dtype
                graph.ndata[key] = graph.ndata[key].to(target_dtype)
        
        # 转换边数据类型以匹配标准格式
        for key in graph.edata:
            if key in standard_graph.edata:
                target_dtype = standard_graph.edata[key].dtype
                graph.edata[key] = graph.edata[key].to(target_dtype)
        
        return graph

    # 加载directory中的图
    for filename in os.listdir(directory):
        if filename.endswith('.dgl'):
            file_path = os.path.join(directory, filename)
            graphs, _ = dgl.load_graphs(file_path)
            all_graphs.extend(graphs)

    # 加载并转换directory2中的图
    for filename in os.listdir(directory2):
        if filename.endswith('.dgl'):
            file_path = os.path.join(directory2, filename)
            graphs, _ = dgl.load_graphs(file_path)
            # 转换每个图的格式
            converted_graphs = [convert_graph_format(g, standard_node_keys, standard_edge_keys) for g in graphs]
            all_graphs.extend(converted_graphs)

    return all_graphs

def analyze_feature_importance(model, dataloader, feature_names=None):
    """
    分析特征重要性
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        feature_names: 特征名称列表（可选）
    Returns:
        weight_importance: 基于权重的特征重要性
        grad_importance: 基于梯度的特征重要性
    """
    model.eval()
    n_features = 32 - featNum  # 特征数量
    
    # 如果没有提供特征名称，使用索引作为名称
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # 基于权重的重要性 - 使用第一层的 SAGEConv 权重
    weight_importance = torch.abs(model.gcnlayer[0].fc_self.weight).mean(dim=0)
    
    # 基于梯度的重要性
    grad_importance = torch.zeros(n_features).to(device)
    n_samples = 0
    
    for batched_graph, labels in dataloader:
        batched_graph = batched_graph.to(device)
        feat = batched_graph.ndata['Feature'][:,:-featNum].float()
        feat.requires_grad = True
        
        # 前向传播
        output = model(batched_graph, feat)
        output = output.view(-1, output.shape[-1])
        labels = labels.view(-1).to(device)
        
        # 计算损失
        loss = F.cross_entropy(output, labels)
        
        # 反向传播
        loss.backward()
        
        # 累积梯度的绝对值
        grad_importance += torch.abs(feat.grad).mean(dim=0)
        n_samples += 1
    
    grad_importance /= n_samples
    
    return weight_importance.cpu(), grad_importance.cpu(), feature_names

def plot_feature_importance(importance, feature_names, title, current_time, model_name, method):
    """
    绘制并保存特征重要性图
    """
    plt.figure(figsize=(12, 6))
    importance_scores = importance.detach().numpy()
    
    # 创建特征重要性的排序索引
    sorted_idx = np.argsort(importance_scores)
    pos = np.arange(len(sorted_idx)) + .5
    
    # 绘制条形图
    plt.barh(pos, importance_scores[sorted_idx])
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.xlabel('Importance Score')
    plt.title(f'{title} - {model_name}')
    
    # 保存图像
    save_dir = f'Visualization/Feature_Importance/{current_time}'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{model_name}_{method}_importance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存原始数据
    data_dict = {
        'feature_names': np.array(feature_names),
        'importance_scores': importance_scores,
        'sorted_indices': sorted_idx
    }
    np.savez(f'{save_dir}/{model_name}_{method}_importance_data.npz', **data_dict)

def save_feature_importance(model, dataloader, current_time, model_name):
    """
    分析并保存特征重要性
    """
    # 添加 Visualization/Feature_Importance 到 result_paths
    feature_dir = f'Visualization/Feature_Importance/{current_time}'
    os.makedirs(feature_dir, exist_ok=True)
    
    # 特征名称（根据实际特征调整）
    feature_names = [f'Feature_{i}' for i in range(32 - featNum)]  # 31个特征
    
    # 计算特征重要性
    weight_importance, grad_importance, feature_names = analyze_feature_importance(
        model, dataloader, feature_names)
    
    # 绘制并保存基于权重的特征重要性
    plot_feature_importance(
        weight_importance,
        feature_names,
        'Weight-based Feature Importance',
        current_time,
        model_name,
        'weight'
    )
    
    # 绘制并保存基于梯度的特征重要性
    plot_feature_importance(
        grad_importance,
        feature_names,
        'Gradient-based Feature Importance',
        current_time,
        model_name,
        'gradient'
    )

def plot_comparison_roc(current_time):
    """
    绘制原始模型和带 Recall 约束模型的 ROC 曲线对比图
    """
    # 加载数据
    auc_dir = f'Visualization/AUC/{current_time}'
    original_data = np.load(f'{auc_dir}/Original_{ModelName}_ROC_data.npz')
    recall_data = np.load(f'{auc_dir}/Recall_{ModelName}_ROC_data.npz')
    
    # 设置绘图样式
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 8))
    
    # 绘制两条 ROC 曲线
    plt.plot(original_data['fpr'], original_data['tpr'], 
            color='royalblue', lw=2, linestyle='-',
            label=f'Original Model (AUC = {original_data["auc"]:.3f})')
    
    plt.plot(recall_data['fpr'], recall_data['tpr'], 
            color='darkorange', lw=2, linestyle='-',
            label=f'Recall-Enhanced Model (AUC = {recall_data["auc"]:.3f})')
    
    # 添加对角线
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    
    # 设置图形属性
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存对比图
    plt.savefig(f'{auc_dir}/ROC_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存对比数据
    comparison_data = {
        'original_fpr': original_data['fpr'],
        'original_tpr': original_data['tpr'],
        'original_auc': original_data['auc'],
        'recall_fpr': recall_data['fpr'],
        'recall_tpr': recall_data['tpr'],
        'recall_auc': recall_data['auc']
    }
    np.savez(f'{auc_dir}/ROC_Comparison_data.npz', **comparison_data)

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='Train GCN models with different loss functions')
    parser.add_argument('--mode', type=str, default='both',
                      choices=['original', 'recall', 'both'],
                      help='Training mode: original (standard loss), recall (with recall constraint), or both')
    args = parser.parse_args()
    
    directory = "/home/wch/nfs/wch/Research5/Dataset/SPDDataset/"
    directory2 = "/home/wch/nfs/wch/Research5/Dataset/SPDDatasetML"

    
    
    graphs = load_graphs(directory, directory2)
    dataset = MultiAddressDataset(graphs=graphs)

    # 5折交叉验证
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    indices = list(range(len(dataset)))
    
    # 存储结果
    original_metrics = []
    recall_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n=== Training Fold {fold+1}/{n_splits} ===")
        
        # 创建数据加载器
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_dataloader = DataLoader(dataset, batch_size=32, sampler=train_subsampler, collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset, batch_size=32, sampler=val_subsampler, collate_fn=collate_fn)

        # 计算类别权重
        train_labels = []
        for idx in train_idx:
            _, labels = dataset[idx]
            train_labels.append(labels)
        train_labels = torch.cat(train_labels)
        class_counts = torch.bincount(train_labels.long())
        total_samples = len(train_labels)
        class_weights = total_samples / (len(class_counts) * class_counts)
        class_weights = class_weights.to(device)

        # 模型参数
        in_feats, hid_feats, out_feats = 32-featNum, 64, 32
        n_classes, n_layer = 2, 3

        # 根据选择的模式训练模型
        if args.mode in ['original', 'both']:
            print("\nTraining Original Model...")
            model_original = GCN(in_feats, hid_feats, out_feats, n_classes, n_layer).to(device)
            optimizer = optim.Adam(model_original.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
            _, _, acc, pr, re, f1 = train_original(model_original, n_epoch=60, optimizer=optimizer, 
                                                scheduler=scheduler, dataloader=train_dataloader, 
                                                val_dataloader=val_dataloader, class_weights=class_weights)
            print(f"Fold {fold+1} Results - Acc: {acc:.4f}, Prec: {pr:.4f}, Rec: {re:.4f}, F1: {f1:.4f}")
            original_metrics.append({'fold': fold + 1, 'accuracy': acc, 'precision': pr, 'recall': re, 'f1': f1})

        if args.mode in ['recall', 'both']:
            print("\nTraining Model with Recall Constraint...")
            model_recall = GCN(in_feats, hid_feats, out_feats, n_classes, n_layer).to(device)
            optimizer = optim.Adam(model_recall.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, min_lr=1e-6)
            _, _, acc, pr, re, f1 = train_with_recall(model_recall, n_epoch=60, optimizer=optimizer, 
                                                    scheduler=scheduler, dataloader=train_dataloader, 
                                                    val_dataloader=val_dataloader, class_weights=class_weights)
            recall_metrics.append({'fold': fold + 1, 'accuracy': acc, 'precision': pr, 'recall': re, 'f1': f1})
        
        torch.cuda.empty_cache()

    # 打印结果
    print("\n=== Cross Validation Results ===")
    if args.mode in ['original', 'both']:
        print("\nOriginal Model:")
        print_metrics_summary(original_metrics)
    
    if args.mode in ['recall', 'both']:
        print("\nModel with Recall Constraint:")
        print_metrics_summary(recall_metrics)
    
    # 如果两种模型都训练了，生成 ROC 对比图
    if args.mode == 'both':
        plot_comparison_roc(current_time)

def print_metrics_summary(metrics):
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in metrics]),
        'precision': np.mean([m['precision'] for m in metrics]),
        'recall': np.mean([m['recall'] for m in metrics]),
        'f1': np.mean([m['f1'] for m in metrics])
    }
    
    std_metrics = {
        'accuracy': np.std([m['accuracy'] for m in metrics]),
        'precision': np.std([m['precision'] for m in metrics]),
        'recall': np.std([m['recall'] for m in metrics]),
        'f1': np.std([m['f1'] for m in metrics])
    }
    
    print(f"Average Accuracy: {avg_metrics['accuracy']:.4f} ± {std_metrics['accuracy']:.4f}")
    print(f"Average Precision: {avg_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
    print(f"Average Recall: {avg_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
    print(f"Average F1-score: {avg_metrics['f1']:.4f} ± {std_metrics['f1']:.4f}")

if __name__ == "__main__":
    
    main()
