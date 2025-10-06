#!/usr/bin/env python3
"""
专门用于绘制ROC-AUC曲线的脚本
按流域分别绘制HyperGAT(SANN-WA)和HyperGCN(SANN-WoA)的各类别ROC-AUC曲线
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn

from train_eval import TrainConfig, build_model
from data_processing import prepare_temporal_data_from_csv, build_knn_edge_index
from models import HyperGCN, HyperGAT, HyperTemporalModel


def plot_roc_auc_for_basin(
    model: nn.Module, 
    data: dict, 
    basin: str, 
    model_name: str,
    class_names: list = None,
    output_dir: str = "./roc_auc_plots"
):
    """为单个流域绘制ROC-AUC曲线"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 移动数据到设备
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    
    # 获取预测结果
    with torch.no_grad():
        if "x_seq" in data:
            logits = model(data["x"], data["hyperedge_index"], data["x_seq"])
        else:
            logits = model(data["x"], data["hyperedge_index"])
        
        y_prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_true = data["y"].cpu().numpy()
    
    # 使用测试集数据
    test_mask = data["test_mask"].cpu().numpy()
    y_true_test = y_true[test_mask]
    y_prob_test = y_prob[test_mask]
    
    # 获取类别数量
    num_classes = len(np.unique(y_true))
    
    # 如果没有提供类别名称，使用默认名称
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # 计算每个类别的ROC-AUC
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_test, y_prob_test[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算宏平均
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    colors = cycle(['navy', 'darkorange', 'cornflowerblue', 'seagreen', 'purple', 'brown', 'pink', 'gray'])
    
    # 绘制每个类别的ROC曲线
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    # 绘制宏平均ROC曲线
    plt.plot(fpr["macro"], tpr["macro"], color='red', linestyle='--', lw=3,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})')
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    
    # 设置图形属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {model_name} - Basin: {basin}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    output_path = os.path.join(output_dir, f"{model_name}_{basin}_roc_auc.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"ROC-AUC曲线已保存到: {output_path}")
    
    return roc_auc


def plot_all_basins_roc_auc(
    config: TrainConfig,
    data: dict,
    basins: np.ndarray,
    model_name: str,
    class_names: list = None,
    output_dir: str = "./roc_auc_plots"
):
    """为所有流域绘制ROC-AUC曲线"""
    
    print(f"\n开始为 {model_name} 绘制所有流域的ROC-AUC曲线...")
    
    # 构建模型
    in_channels = data["x"].size(1)
    out_channels = int(data["y"].max().item()) + 1
    model = build_model(config, in_channels, out_channels)
    
    # 获取唯一流域
    unique_basins = np.unique(basins)
    all_roc_auc = {}
    
    for basin in unique_basins:
        print(f"绘制流域 {basin} 的ROC-AUC曲线...")
        
        # 获取该流域的数据
        basin_indices = np.where(basins == basin)[0]
        if len(basin_indices) == 0:
            continue
        
        # 创建流域子数据
        idx_tensor = torch.tensor(basin_indices, dtype=torch.long)
        sub_data = {
            "x": data["x"][idx_tensor],
            "y": data["y"][idx_tensor],
            "hyperedge_index": data["hyperedge_index"],  # 需要重新构建
            "train_mask": data["train_mask"][idx_tensor],
            "val_mask": data["val_mask"][idx_tensor],
            "test_mask": data["test_mask"][idx_tensor],
        }
        
        if "x_seq" in data:
            sub_data["x_seq"] = data["x_seq"][idx_tensor]
        
        # 重新构建该流域的超图
        sub_x_np = sub_data["x"].cpu().numpy()
        sub_edge_index = build_knn_edge_index(
            sub_x_np, 
            k_neighbors=config.k_neighbors, 
            include_self=True, 
            make_undirected=True
        )
        sub_data["hyperedge_index"] = sub_edge_index
        
        # 绘制ROC-AUC曲线
        try:
            roc_auc = plot_roc_auc_for_basin(
                model, sub_data, basin, model_name, class_names, output_dir
            )
            all_roc_auc[basin] = roc_auc
        except Exception as e:
            print(f"绘制流域 {basin} 的ROC-AUC曲线时出错: {e}")
            continue
    
    # 保存所有ROC-AUC结果
    roc_auc_file = os.path.join(output_dir, f"{model_name}_all_basins_roc_auc.json")
    with open(roc_auc_file, "w", encoding="utf-8") as f:
        # 转换numpy类型为Python原生类型以便JSON序列化
        serializable_roc_auc = {}
        for basin, metrics in all_roc_auc.items():
            serializable_roc_auc[str(basin)] = {str(k): float(v) for k, v in metrics.items()}
        json.dump(serializable_roc_auc, f, ensure_ascii=False, indent=2)
    
    print(f"所有流域的ROC-AUC结果已保存到: {roc_auc_file}")
    
    return all_roc_auc


def main():
    """主函数"""
    
    print("="*60)
    print("开始绘制ROC-AUC曲线")
    print("="*60)
    
    # 配置参数
    config = TrainConfig(
        csv_path="./包含藻密度的所有数据.csv",
        enable_temporal=True,
        epochs=50,  # 减少epochs用于快速测试
        early_stop_patience=10
    )
    
    # 加载数据
    print("加载数据...")
    full_data, _, basins = prepare_temporal_data_from_csv(
        csv_path=config.csv_path,
        label_col=config.label_col,
        id_col=config.id_col,
        time_col=config.time_col,
        features=list(config.features),
        interval_days=int(config.interval_days),
        steps=int(config.steps),
        k_neighbors=config.k_neighbors,
        val_size=config.val_size,
        test_size=config.test_size,
        random_state=config.random_state,
        group_col=config.group_col,
    )
    
    # 设置水质类别名称
    class_names = ['Class I', 'Class II', 'Class III', 'Class IV', 'Class V']
    
    # 为每个模型绘制ROC-AUC曲线
    models = ["HyperGCN", "HyperGAT"]
    all_results = {}
    
    for model_name in models:
        print(f"\n处理模型: {model_name} ({'SANN-WoA' if 'GCN' in model_name else 'SANN-WA'})")
        config.model_name = model_name
        
        try:
            roc_auc_results = plot_all_basins_roc_auc(
                config=config,
                data=full_data,
                basins=basins,
                model_name=model_name,
                class_names=class_names,
                output_dir="./roc_auc_plots"
            )
            all_results[model_name] = roc_auc_results
            print(f"{model_name} 的ROC-AUC曲线绘制完成")
        except Exception as e:
            print(f"绘制 {model_name} 的ROC-AUC曲线时出错: {e}")
    
    # 生成汇总报告
    generate_roc_auc_summary(all_results)
    
    print("\n所有ROC-AUC曲线绘制完成！")
    print("结果保存在: ./roc_auc_plots/")


def generate_roc_auc_summary(all_results):
    """生成ROC-AUC汇总报告"""
    
    print("\n" + "="*60)
    print("ROC-AUC汇总报告")
    print("="*60)
    
    for model_name, results in all_results.items():
        sann_type = "SANN-WA" if "GAT" in model_name else "SANN-WoA"
        print(f"\n{model_name} ({sann_type}):")
        
        # 计算平均AUC
        macro_aucs = []
        for basin, metrics in results.items():
            if "macro" in metrics:
                macro_aucs.append(metrics["macro"])
                print(f"  {basin}: Macro-AUC = {metrics['macro']:.3f}")
        
        if macro_aucs:
            avg_macro_auc = np.mean(macro_aucs)
            print(f"  平均Macro-AUC: {avg_macro_auc:.3f}")
    
    # 保存汇总结果
    summary_file = "./roc_auc_plots/roc_auc_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n汇总结果已保存到: {summary_file}")


if __name__ == "__main__":
    main()

