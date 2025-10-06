"""
计算效率评估模块
用于对比HyperGAT(SANN-WA)和HyperGCN(SANN-WoA)的训练时间、推理速度和内存使用
"""

import os
import time
import psutil
import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from itertools import cycle
from sklearn.metrics import roc_curve, auc

from train_eval import TrainConfig, build_model, evaluate
from data_processing import prepare_temporal_data_from_csv, prepare_data_from_csv


@dataclass
class EfficiencyMetrics:
    """计算效率指标"""
    # 训练相关
    total_train_time: float = 0.0  # 总训练时间(秒)
    avg_epoch_time: float = 0.0     # 平均每轮训练时间(秒)
    train_memory_peak: float = 0.0  # 训练峰值内存(MB)
    train_memory_avg: float = 0.0   # 训练平均内存(MB)
    
    # 推理相关
    inference_time: float = 0.0     # 单次推理时间(秒)
    inference_memory: float = 0.0   # 推理内存使用(MB)
    throughput: float = 0.0         # 吞吐量(样本/秒)
    
    # 模型复杂度
    num_parameters: int = 0         # 模型参数数量
    model_size_mb: float = 0.0      # 模型大小(MB)
    flops: int = 0                 # 浮点运算次数(近似)
    
    # 收敛性
    convergence_epoch: int = 0      # 收敛轮数
    best_val_loss: float = 0.0      # 最佳验证损失


class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self):
        self.peak_memory = 0
        self.memory_samples = []
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """开始监控"""
        self.peak_memory = 0
        self.memory_samples = []
    
    def sample_memory(self):
        """采样当前内存使用"""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        self.memory_samples.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
    
    def get_metrics(self) -> Tuple[float, float]:
        """获取内存指标"""
        avg_memory = np.mean(self.memory_samples) if self.memory_samples else 0
        return self.peak_memory, avg_memory


@contextmanager
def timer_and_memory():
    """计时和内存监控上下文管理器"""
    monitor = MemoryMonitor()
    monitor.start_monitoring()
    start_time = time.time()
    
    try:
        yield monitor
    finally:
        end_time = time.time()
        monitor.sample_memory()
        elapsed_time = end_time - start_time
        peak_memory, avg_memory = monitor.get_metrics()
        
        # 存储结果
        timer_and_memory.last_time = elapsed_time
        timer_and_memory.peak_memory = peak_memory
        timer_and_memory.avg_memory = avg_memory


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_size(model: nn.Module) -> float:
    """估算模型大小(MB)"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024 / 1024
    return size_all_mb


def estimate_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """估算FLOPs（简化版本）"""
    # 这是一个简化的FLOPs估算，实际应用中可能需要更精确的工具
    total_flops = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Linear层: input_size * output_size
            total_flops += module.in_features * module.out_features
        elif isinstance(module, nn.MultiheadAttention):
            # 注意力机制: 4 * d_model^2 * seq_len (简化)
            d_model = module.embed_dim
            seq_len = input_shape[1] if len(input_shape) > 1 else 1
            total_flops += 4 * d_model * d_model * seq_len
    
    return total_flops


class EfficiencyEvaluator:
    """计算效率评估器"""
    
    def __init__(self, output_dir: str = "./efficiency_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_model_efficiency(
        self, 
        config: TrainConfig, 
        model_name: str,
        data: Dict[str, torch.Tensor],
        num_runs: int = 3
    ) -> EfficiencyMetrics:
        """评估单个模型的效率"""
        
        print(f"\n========== 评估 {model_name} 计算效率 ==========")
        
        # 构建模型
        in_channels = data["x"].size(1)
        out_channels = int(data["y"].max().item()) + 1
        model = build_model(config, in_channels, out_channels)
        
        # 基础指标
        metrics = EfficiencyMetrics()
        metrics.num_parameters = count_parameters(model)
        metrics.model_size_mb = estimate_model_size(model)
        
        # 估算FLOPs
        input_shape = data["x"].shape
        metrics.flops = estimate_flops(model, input_shape)
        
        print(f"模型参数数量: {metrics.num_parameters:,}")
        print(f"模型大小: {metrics.model_size_mb:.2f} MB")
        print(f"估算FLOPs: {metrics.flops:,}")
        
        # 训练效率评估
        train_metrics = self._evaluate_training_efficiency(model, data, config, num_runs)
        metrics.__dict__.update(train_metrics)
        
        # 推理效率评估
        inference_metrics = self._evaluate_inference_efficiency(model, data, num_runs)
        metrics.__dict__.update(inference_metrics)
        
        return metrics
    
    def _evaluate_training_efficiency(
        self, 
        model: nn.Module, 
        data: Dict[str, torch.Tensor], 
        config: TrainConfig,
        num_runs: int
    ) -> Dict[str, Any]:
        """评估训练效率"""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # 移动数据到设备
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        train_times = []
        memory_peaks = []
        memory_avgs = []
        
        for run in range(num_runs):
            print(f"训练效率测试 - 运行 {run + 1}/{num_runs}")
            
            # 重置模型
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # 训练几个epoch来测试效率
            test_epochs = min(10, config.epochs)
            epoch_times = []
            
            with timer_and_memory() as monitor:
                for epoch in range(test_epochs):
                    epoch_start = time.time()
                    
                    model.train()
                    optimizer.zero_grad()
                    
                    if "x_seq" in data:
                        logits = model(data["x"], data["hyperedge_index"], data["x_seq"])
                    else:
                        logits = model(data["x"], data["hyperedge_index"])
                    
                    loss = criterion(logits[data["train_mask"]], data["y"][data["train_mask"]])
                    loss.backward()
                    optimizer.step()
                    
                    epoch_time = time.time() - epoch_start
                    epoch_times.append(epoch_time)
                    monitor.sample_memory()
            
            train_times.append(timer_and_memory.last_time)
            memory_peaks.append(timer_and_memory.peak_memory)
            memory_avgs.append(timer_and_memory.avg_memory)
        
        return {
            "total_train_time": np.mean(train_times),
            "avg_epoch_time": np.mean([t / test_epochs for t in train_times]),
            "train_memory_peak": np.mean(memory_peaks),
            "train_memory_avg": np.mean(memory_avgs)
        }
    
    def _evaluate_inference_efficiency(
        self, 
        model: nn.Module, 
        data: Dict[str, torch.Tensor], 
        num_runs: int
    ) -> Dict[str, Any]:
        """评估推理效率"""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # 移动数据到设备
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)
        
        inference_times = []
        memory_peaks = []
        
        for run in range(num_runs):
            print(f"推理效率测试 - 运行 {run + 1}/{num_runs}")
            
            with torch.no_grad():
                with timer_and_memory() as monitor:
                    # 多次推理取平均
                    num_inference = 100
                    for _ in range(num_inference):
                        if "x_seq" in data:
                            _ = model(data["x"], data["hyperedge_index"], data["x_seq"])
                        else:
                            _ = model(data["x"], data["hyperedge_index"])
                        monitor.sample_memory()
            
            avg_inference_time = timer_and_memory.last_time / num_inference
            inference_times.append(avg_inference_time)
            memory_peaks.append(timer_and_memory.peak_memory)
        
        # 计算吞吐量
        batch_size = data["x"].size(0)
        throughput = batch_size / np.mean(inference_times)
        
        return {
            "inference_time": np.mean(inference_times),
            "inference_memory": np.mean(memory_peaks),
            "throughput": throughput
        }
    
    def compare_models(
        self, 
        config: TrainConfig, 
        data: Dict[str, torch.Tensor],
        models: List[str] = ["HyperGCN", "HyperGAT"]
    ) -> Dict[str, EfficiencyMetrics]:
        """对比多个模型的效率"""
        
        results = {}
        
        for model_name in models:
            config.model_name = model_name
            metrics = self.evaluate_model_efficiency(config, model_name, data)
            results[model_name] = metrics
        
        # 保存结果
        self._save_comparison_results(results)
        
        # 生成对比图表
        self._generate_comparison_plots(results)
        
        return results
    
    def _save_comparison_results(self, results: Dict[str, EfficiencyMetrics]):
        """保存对比结果"""
        
        # 转换为可序列化的字典
        serializable_results = {}
        for model_name, metrics in results.items():
            serializable_results[model_name] = asdict(metrics)
        
        # 保存JSON
        with open(os.path.join(self.output_dir, "efficiency_comparison.json"), "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式的汇总表
        self._save_summary_table(serializable_results)
    
    def _save_summary_table(self, results: Dict[str, Dict]):
        """保存汇总表"""
        
        import pandas as pd
        
        summary_data = []
        for model_name, metrics in results.items():
            summary_data.append({
                "Model": model_name,
                "Parameters": f"{metrics['num_parameters']:,}",
                "Model Size (MB)": f"{metrics['model_size_mb']:.2f}",
                "FLOPs": f"{metrics['flops']:,}",
                "Avg Epoch Time (s)": f"{metrics['avg_epoch_time']:.3f}",
                "Inference Time (ms)": f"{metrics['inference_time']*1000:.2f}",
                "Throughput (samples/s)": f"{metrics['throughput']:.1f}",
                "Peak Memory (MB)": f"{metrics['train_memory_peak']:.1f}",
                "Inference Memory (MB)": f"{metrics['inference_memory']:.1f}"
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.output_dir, "efficiency_summary.csv"), index=False)
        print("\n效率对比汇总表:")
        print(df.to_string(index=False))
    
    def _generate_comparison_plots(self, results: Dict[str, EfficiencyMetrics]):
        """生成对比图表"""
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('模型计算效率对比分析', fontsize=16, fontweight='bold')
        
        model_names = list(results.keys())
        
        # 1. 训练时间对比
        epoch_times = [results[name].avg_epoch_time for name in model_names]
        axes[0, 0].bar(model_names, epoch_times, color=['#1f77b4', '#ff7f0e'])
        axes[0, 0].set_title('平均每轮训练时间')
        axes[0, 0].set_ylabel('时间 (秒)')
        
        # 2. 推理时间对比
        inference_times = [results[name].inference_time * 1000 for name in model_names]  # 转换为毫秒
        axes[0, 1].bar(model_names, inference_times, color=['#1f77b4', '#ff7f0e'])
        axes[0, 1].set_title('单次推理时间')
        axes[0, 1].set_ylabel('时间 (毫秒)')
        
        # 3. 内存使用对比
        memory_usage = [results[name].train_memory_peak for name in model_names]
        axes[0, 2].bar(model_names, memory_usage, color=['#1f77b4', '#ff7f0e'])
        axes[0, 2].set_title('训练峰值内存使用')
        axes[0, 2].set_ylabel('内存 (MB)')
        
        # 4. 模型大小对比
        model_sizes = [results[name].model_size_mb for name in model_names]
        axes[1, 0].bar(model_names, model_sizes, color=['#1f77b4', '#ff7f0e'])
        axes[1, 0].set_title('模型大小')
        axes[1, 0].set_ylabel('大小 (MB)')
        
        # 5. 参数数量对比
        param_counts = [results[name].num_parameters for name in model_names]
        axes[1, 1].bar(model_names, param_counts, color=['#1f77b4', '#ff7f0e'])
        axes[1, 1].set_title('模型参数数量')
        axes[1, 1].set_ylabel('参数数量')
        
        # 6. 吞吐量对比
        throughputs = [results[name].throughput for name in model_names]
        axes[1, 2].bar(model_names, throughputs, color=['#1f77b4', '#ff7f0e'])
        axes[1, 2].set_title('推理吞吐量')
        axes[1, 2].set_ylabel('样本/秒')
        
        # 调整布局
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n对比图表已保存到: {os.path.join(self.output_dir, 'efficiency_comparison.png')}")
    
    def plot_roc_auc_curves(
        self, 
        model: nn.Module, 
        data: Dict[str, torch.Tensor], 
        basin: str,
        model_name: str,
        class_names: List[str] = None
    ):
        """绘制ROC-AUC曲线，按流域分别绘制各类别"""
        
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
        output_path = os.path.join(self.output_dir, f"{model_name}_{basin}_roc_auc.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"ROC-AUC曲线已保存到: {output_path}")
        
        return roc_auc
    
    def plot_all_basins_roc_auc(
        self, 
        config: TrainConfig, 
        data: Dict[str, torch.Tensor],
        basins: np.ndarray,
        model_name: str,
        class_names: List[str] = None
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
            from data_processing import build_knn_edge_index
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
                roc_auc = self.plot_roc_auc_curves(
                    model, sub_data, basin, model_name, class_names
                )
                all_roc_auc[basin] = roc_auc
            except Exception as e:
                print(f"绘制流域 {basin} 的ROC-AUC曲线时出错: {e}")
                continue
        
        # 保存所有ROC-AUC结果
        roc_auc_file = os.path.join(self.output_dir, f"{model_name}_all_basins_roc_auc.json")
        with open(roc_auc_file, "w", encoding="utf-8") as f:
            # 转换numpy类型为Python原生类型以便JSON序列化
            serializable_roc_auc = {}
            for basin, metrics in all_roc_auc.items():
                serializable_roc_auc[str(basin)] = {str(k): float(v) for k, v in metrics.items()}
            json.dump(serializable_roc_auc, f, ensure_ascii=False, indent=2)
        
        print(f"所有流域的ROC-AUC结果已保存到: {roc_auc_file}")
        
        return all_roc_auc


def run_efficiency_evaluation(include_roc_auc=True, class_names=None):
    """运行完整的效率评估"""
    
    # 配置
    config = TrainConfig(
        csv_path="./包含藻密度的所有数据.csv",
        model_name="HyperGAT",  # 会被覆盖
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
    
    # 创建评估器
    evaluator = EfficiencyEvaluator()
    
    # 对比模型
    results = evaluator.compare_models(
        config=config,
        data=full_data,
        models=["HyperGCN", "HyperGAT"]  # SANN-WoA vs SANN-WA
    )
    
    # 打印详细对比结果
    print("\n" + "="*60)
    print("详细效率对比结果")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name} ({'SANN-WA' if 'GAT' in model_name else 'SANN-WoA'}):")
        print(f"  参数数量: {metrics.num_parameters:,}")
        print(f"  模型大小: {metrics.model_size_mb:.2f} MB")
        print(f"  平均每轮训练时间: {metrics.avg_epoch_time:.3f} 秒")
        print(f"  单次推理时间: {metrics.inference_time*1000:.2f} 毫秒")
        print(f"  推理吞吐量: {metrics.throughput:.1f} 样本/秒")
        print(f"  训练峰值内存: {metrics.train_memory_peak:.1f} MB")
        print(f"  推理内存使用: {metrics.inference_memory:.1f} MB")
    
    # 绘制ROC-AUC曲线
    if include_roc_auc:
        print("\n" + "="*60)
        print("开始绘制ROC-AUC曲线")
        print("="*60)
        
        # 设置类别名称（如果未提供）
        if class_names is None:
            # 根据水质类别设置名称
            class_names = ['Class I', 'Class II', 'Class III', 'Class IV', 'Class V']
        
        # 为每个模型绘制ROC-AUC曲线
        for model_name in ["HyperGCN", "HyperGAT"]:
            print(f"\n为 {model_name} 绘制ROC-AUC曲线...")
            config.model_name = model_name
            
            try:
                roc_auc_results = evaluator.plot_all_basins_roc_auc(
                    config=config,
                    data=full_data,
                    basins=basins,
                    model_name=model_name,
                    class_names=class_names
                )
                print(f"{model_name} 的ROC-AUC曲线绘制完成")
            except Exception as e:
                print(f"绘制 {model_name} 的ROC-AUC曲线时出错: {e}")
    
    return results


if __name__ == "__main__":
    results = run_efficiency_evaluation()
