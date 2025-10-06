#!/usr/bin/env python3
"""
运行效率评估实验的脚本
用于对比HyperGAT(SANN-WA)和HyperGCN(SANN-WoA)的计算效率
"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from train_eval import run_efficiency_comparison, TrainConfig

def create_efficiency_report(include_roc_auc=True):
    """创建详细的效率评估报告"""
    
    print("开始运行效率评估实验...")
    
    # 运行效率对比
    results = run_efficiency_comparison()
    
    if not results:
        print("未获取到效率评估结果")
        return
    
    # 创建报告目录
    report_dir = "./efficiency_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # 生成详细报告
    generate_detailed_report(results, report_dir)
    
    # 生成可视化图表
    generate_efficiency_plots(results, report_dir)
    
    # 绘制ROC-AUC曲线
    if include_roc_auc:
        print("\n开始绘制ROC-AUC曲线...")
        from efficiency_evaluation import run_efficiency_evaluation
        
        # 设置水质类别名称
        class_names = ['Class I', 'Class II', 'Class III', 'Class IV', 'Class V']
        
        # 运行包含ROC-AUC的评估
        roc_auc_results = run_efficiency_evaluation(
            include_roc_auc=True, 
            class_names=class_names
        )
        
        print("ROC-AUC曲线绘制完成")
    
    print(f"\n效率评估报告已生成到: {report_dir}")


def generate_detailed_report(results, report_dir):
    """生成详细的文本报告"""
    
    report_content = []
    report_content.append("# SANN-WA vs SANN-WoA 计算效率评估报告")
    report_content.append("=" * 60)
    report_content.append("")
    
    # 模型对比
    report_content.append("## 模型对比")
    report_content.append("")
    report_content.append("- **SANN-WA (HyperGAT)**: 带注意力机制的空间注意力神经网络")
    report_content.append("- **SANN-WoA (HyperGCN)**: 无注意力机制的空间注意力神经网络")
    report_content.append("")
    
    # 详细指标
    report_content.append("## 详细效率指标")
    report_content.append("")
    
    for model_name, metrics in results.items():
        sann_type = "SANN-WA" if "GAT" in model_name else "SANN-WoA"
        report_content.append(f"### {model_name} ({sann_type})")
        report_content.append("")
        report_content.append(f"- **参数数量**: {metrics['num_parameters']:,}")
        report_content.append(f"- **模型大小**: {metrics['model_size_mb']:.2f} MB")
        report_content.append(f"- **平均推理时间**: {metrics['avg_inference_time']*1000:.2f} ms")
        report_content.append(f"- **推理吞吐量**: {metrics['throughput']:.1f} 样本/秒")
        report_content.append(f"- **推理内存使用**: {metrics['inference_memory_mb']:.1f} MB")
        report_content.append("")
    
    # 相对性能分析
    if len(results) == 2:
        models = list(results.keys())
        gcn_metrics = results[models[0]] if "GCN" in models[0] else results[models[1]]
        gat_metrics = results[models[1]] if "GAT" in models[1] else results[models[0]]
        
        report_content.append("## 相对性能分析")
        report_content.append("")
        report_content.append("### 注意力机制的计算开销")
        report_content.append("")
        
        param_ratio = gat_metrics['num_parameters'] / gcn_metrics['num_parameters']
        time_ratio = gat_metrics['avg_inference_time'] / gcn_metrics['avg_inference_time']
        memory_ratio = gat_metrics['inference_memory_mb'] / gcn_metrics['inference_memory_mb']
        throughput_ratio = gat_metrics['throughput'] / gcn_metrics['throughput']
        
        report_content.append(f"- **参数数量增加**: {param_ratio:.2f}x")
        report_content.append(f"- **推理时间增加**: {time_ratio:.2f}x")
        report_content.append(f"- **内存使用增加**: {memory_ratio:.2f}x")
        report_content.append(f"- **吞吐量变化**: {throughput_ratio:.2f}x")
        report_content.append("")
        
        # 分析结论
        report_content.append("### 分析结论")
        report_content.append("")
        
        if time_ratio > 1.5:
            report_content.append("- 注意力机制显著增加了推理时间")
        elif time_ratio > 1.2:
            report_content.append("- 注意力机制适度增加了推理时间")
        else:
            report_content.append("- 注意力机制对推理时间影响较小")
        
        if memory_ratio > 1.5:
            report_content.append("- 注意力机制显著增加了内存使用")
        elif memory_ratio > 1.2:
            report_content.append("- 注意力机制适度增加了内存使用")
        else:
            report_content.append("- 注意力机制对内存使用影响较小")
        
        if throughput_ratio < 0.8:
            report_content.append("- 注意力机制显著降低了吞吐量")
        elif throughput_ratio < 0.9:
            report_content.append("- 注意力机制适度降低了吞吐量")
        else:
            report_content.append("- 注意力机制对吞吐量影响较小")
        
        report_content.append("")
        
        # 实用性建议
        report_content.append("### 实用性建议")
        report_content.append("")
        report_content.append("1. **计算资源充足时**: 推荐使用SANN-WA，注意力机制能提供更好的性能")
        report_content.append("2. **计算资源受限时**: 推荐使用SANN-WoA，在保持合理性能的同时降低计算成本")
        report_content.append("3. **实时应用场景**: 根据具体的延迟要求选择模型")
        report_content.append("4. **内存受限环境**: 优先考虑SANN-WoA")
        report_content.append("")
    
    # 保存报告
    with open(os.path.join(report_dir, "efficiency_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))
    
    # 保存JSON数据
    with open(os.path.join(report_dir, "efficiency_data.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def generate_efficiency_plots(results, report_dir):
    """生成效率对比图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备数据
    model_names = []
    sann_types = []
    param_counts = []
    model_sizes = []
    inference_times = []
    throughputs = []
    memory_usage = []
    
    for model_name, metrics in results.items():
        model_names.append(model_name)
        sann_types.append("SANN-WA" if "GAT" in model_name else "SANN-WoA")
        param_counts.append(metrics['num_parameters'])
        model_sizes.append(metrics['model_size_mb'])
        inference_times.append(metrics['avg_inference_time'] * 1000)  # 转换为毫秒
        throughputs.append(metrics['throughput'])
        memory_usage.append(metrics['inference_memory_mb'])
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SANN-WA vs SANN-WoA 计算效率对比', fontsize=16, fontweight='bold')
    
    # 1. 参数数量对比
    bars1 = axes[0, 0].bar(sann_types, param_counts, color=['#1f77b4', '#ff7f0e'])
    axes[0, 0].set_title('模型参数数量')
    axes[0, 0].set_ylabel('参数数量')
    for i, v in enumerate(param_counts):
        axes[0, 0].text(i, v + max(param_counts)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # 2. 模型大小对比
    bars2 = axes[0, 1].bar(sann_types, model_sizes, color=['#1f77b4', '#ff7f0e'])
    axes[0, 1].set_title('模型大小')
    axes[0, 1].set_ylabel('大小 (MB)')
    for i, v in enumerate(model_sizes):
        axes[0, 1].text(i, v + max(model_sizes)*0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # 3. 推理时间对比
    bars3 = axes[0, 2].bar(sann_types, inference_times, color=['#1f77b4', '#ff7f0e'])
    axes[0, 2].set_title('单次推理时间')
    axes[0, 2].set_ylabel('时间 (毫秒)')
    for i, v in enumerate(inference_times):
        axes[0, 2].text(i, v + max(inference_times)*0.01, f'{v:.2f}', ha='center', va='bottom')
    
    # 4. 吞吐量对比
    bars4 = axes[1, 0].bar(sann_types, throughputs, color=['#1f77b4', '#ff7f0e'])
    axes[1, 0].set_title('推理吞吐量')
    axes[1, 0].set_ylabel('样本/秒')
    for i, v in enumerate(throughputs):
        axes[1, 0].text(i, v + max(throughputs)*0.01, f'{v:.1f}', ha='center', va='bottom')
    
    # 5. 内存使用对比
    bars5 = axes[1, 1].bar(sann_types, memory_usage, color=['#1f77b4', '#ff7f0e'])
    axes[1, 1].set_title('推理内存使用')
    axes[1, 1].set_ylabel('内存 (MB)')
    for i, v in enumerate(memory_usage):
        axes[1, 1].text(i, v + max(memory_usage)*0.01, f'{v:.1f}', ha='center', va='bottom')
    
    # 6. 相对性能对比
    if len(results) == 2:
        models = list(results.keys())
        gcn_metrics = results[models[0]] if "GCN" in models[0] else results[models[1]]
        gat_metrics = results[models[1]] if "GAT" in models[1] else results[models[0]]
        
        ratios = [
            gat_metrics['num_parameters'] / gcn_metrics['num_parameters'],
            gat_metrics['avg_inference_time'] / gcn_metrics['avg_inference_time'],
            gat_metrics['inference_memory_mb'] / gcn_metrics['inference_memory_mb']
        ]
        ratio_labels = ['参数数量', '推理时间', '内存使用']
        
        bars6 = axes[1, 2].bar(ratio_labels, ratios, color=['#ff7f0e', '#2ca02c', '#d62728'])
        axes[1, 2].set_title('SANN-WA相对SANN-WoA的比率')
        axes[1, 2].set_ylabel('比率')
        axes[1, 2].axhline(y=1, color='black', linestyle='--', alpha=0.5)
        for i, v in enumerate(ratios):
            axes[1, 2].text(i, v + 0.01, f'{v:.2f}x', ha='center', va='bottom')
    
    # 调整布局
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "efficiency_comparison.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"效率对比图表已保存到: {os.path.join(report_dir, 'efficiency_comparison.png')}")


if __name__ == "__main__":
    create_efficiency_report()
