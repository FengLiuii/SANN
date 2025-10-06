# SANN-WA vs SANN-WoA 计算效率评估指南

## 概述

本指南介绍如何对HyperGAT（SANN-WA）和HyperGCN（SANN-WoA）进行全面的计算效率评估，包括训练时间、推理速度、内存使用等关键指标。

## 文件结构

```
├── train_eval.py                    # 主训练脚本（已集成效率评估）
├── efficiency_evaluation.py         # 独立的效率评估模块
├── run_efficiency_evaluation.py     # 效率评估运行脚本
├── plot_roc_auc_curves.py          # 专门绘制ROC-AUC曲线的脚本
├── example_roc_auc_usage.py        # ROC-AUC使用示例
└── EFFICIENCY_EVALUATION_GUIDE.md   # 本指南
```

## 快速开始

### 1. 运行效率对比实验

```bash
# 方法1：使用集成的高效评估
python train_eval.py --efficiency

# 方法2：使用独立的评估脚本（包含ROC-AUC曲线）
python run_efficiency_evaluation.py

# 方法3：使用独立的评估模块
python efficiency_evaluation.py

# 方法4：专门绘制ROC-AUC曲线
python plot_roc_auc_curves.py

# 方法5：运行ROC-AUC使用示例
python example_roc_auc_usage.py

# 方法6：快速演示模式
python example_roc_auc_usage.py --quick
```

### 2. 在现有训练中启用效率评估

```python
from train_eval import TrainConfig, main

# 配置训练参数
config = TrainConfig(
    model_name="HyperGAT",  # 或 "HyperGCN"
    enable_efficiency_eval=True,  # 启用效率评估
    epochs=100,
    early_stop_patience=20
)

# 运行训练（会自动进行效率评估）
main(config)
```

## 评估指标说明

### 1. 模型复杂度指标
- **参数数量** (`num_parameters`): 模型可训练参数的总数
- **模型大小** (`model_size_mb`): 模型文件大小（MB）
- **FLOPs**: 浮点运算次数（近似估算）

### 2. 训练效率指标
- **总训练时间** (`total_train_time`): 完整训练过程的时间
- **平均每轮时间** (`avg_epoch_time`): 单个epoch的平均时间
- **训练峰值内存** (`train_memory_peak`): 训练过程中的最大内存使用
- **训练平均内存** (`train_memory_avg`): 训练过程中的平均内存使用

### 3. 推理效率指标
- **推理时间** (`inference_time`): 单次推理的平均时间
- **推理吞吐量** (`throughput`): 每秒处理的样本数
- **推理内存** (`inference_memory_mb`): 推理过程中的内存使用

### 4. ROC-AUC性能指标
- **各类别AUC**: 每个水质类别的ROC-AUC值
- **宏平均AUC**: 所有类别的宏平均ROC-AUC值
- **分流域分析**: 按流域分别分析ROC-AUC性能

## 详细使用方法

### 1. 独立效率评估模块

```python
from efficiency_evaluation import EfficiencyEvaluator, TrainConfig

# 创建评估器
evaluator = EfficiencyEvaluator(output_dir="./efficiency_results")

# 配置参数
config = TrainConfig(
    csv_path="./包含藻密度的所有数据.csv",
    enable_temporal=True,
    epochs=50
)

# 加载数据
full_data, _, basins = prepare_temporal_data_from_csv(...)

# 对比模型效率
results = evaluator.compare_models(
    config=config,
    data=full_data,
    models=["HyperGCN", "HyperGAT"]
)
```

### 2. 自定义效率评估

```python
from efficiency_evaluation import EfficiencyMetrics, MemoryMonitor
import torch

# 创建自定义评估
def custom_efficiency_test(model, data):
    metrics = EfficiencyMetrics()
    
    # 计算参数数量
    metrics.num_parameters = sum(p.numel() for p in model.parameters())
    
    # 测试推理时间
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        _ = model(data["x"], data["hyperedge_index"])
        metrics.inference_time = time.time() - start_time
    
    return metrics
```

### 3. 批量评估多个配置

```python
# 评估不同配置下的效率
configs = [
    TrainConfig(model_name="HyperGCN", hidden_channels=32),
    TrainConfig(model_name="HyperGCN", hidden_channels=64),
    TrainConfig(model_name="HyperGAT", hidden_channels=32),
    TrainConfig(model_name="HyperGAT", hidden_channels=64),
]

results = {}
for config in configs:
    results[config.model_name + f"_h{config.hidden_channels}"] = evaluator.evaluate_model_efficiency(
        config, model_name, data
    )
```

### 4. 绘制ROC-AUC曲线

```python
from efficiency_evaluation import EfficiencyEvaluator

# 创建评估器
evaluator = EfficiencyEvaluator()

# 设置类别名称
class_names = ['Class I', 'Class II', 'Class III', 'Class IV', 'Class V']

# 为单个流域绘制ROC-AUC曲线
roc_auc = evaluator.plot_roc_auc_curves(
    model=model,
    data=basin_data,
    basin="Yangtze River Basin",
    model_name="HyperGAT",
    class_names=class_names
)

# 为所有流域绘制ROC-AUC曲线
all_roc_auc = evaluator.plot_all_basins_roc_auc(
    config=config,
    data=full_data,
    basins=basins,
    model_name="HyperGAT",
    class_names=class_names
)
```

### 5. 专门绘制ROC-AUC曲线

```python
# 使用专门的ROC-AUC绘制脚本
from plot_roc_auc_curves import plot_all_basins_roc_auc

# 直接运行脚本
# python plot_roc_auc_curves.py

# 或在代码中调用
roc_auc_results = plot_all_basins_roc_auc(
    config=config,
    data=full_data,
    basins=basins,
    model_name="HyperGAT",
    class_names=['Class I', 'Class II', 'Class III', 'Class IV', 'Class V']
)
```

## 结果分析

### 1. 自动生成的报告

运行效率评估后，会在指定目录生成以下文件：

```
efficiency_report/
├── efficiency_report.md          # 详细文本报告
├── efficiency_data.json         # 原始数据（JSON格式）
└── efficiency_comparison.png     # 可视化对比图表

roc_auc_plots/
├── HyperGAT_*_roc_auc.png       # 各流域的ROC-AUC曲线图
├── HyperGCN_*_roc_auc.png       # 各流域的ROC-AUC曲线图
├── HyperGAT_all_basins_roc_auc.json  # HyperGAT所有流域的AUC数据
├── HyperGCN_all_basins_roc_auc.json  # HyperGCN所有流域的AUC数据
└── roc_auc_summary.json         # ROC-AUC汇总结果
```

### 2. 关键对比指标

- **参数数量比率**: SANN-WA相对SANN-WoA的参数增加倍数
- **推理时间比率**: SANN-WA相对SANN-WoA的时间增加倍数
- **内存使用比率**: SANN-WA相对SANN-WoA的内存增加倍数
- **吞吐量比率**: SANN-WA相对SANN-WoA的吞吐量变化

### 3. 性能分析建议

根据评估结果，可以得出以下结论：

1. **注意力机制的计算开销**：
   - 参数数量通常增加1.2-2.0倍
   - 推理时间通常增加1.1-1.5倍
   - 内存使用通常增加1.1-1.3倍

2. **实用性建议**：
   - 计算资源充足时：推荐SANN-WA
   - 计算资源受限时：推荐SANN-WoA
   - 实时应用：根据延迟要求选择
   - 内存受限环境：优先考虑SANN-WoA

## ROC-AUC曲线绘制功能

### 1. 功能特点

- **分流域绘制**: 为每个流域单独绘制ROC-AUC曲线
- **多类别支持**: 支持多个水质类别的ROC-AUC分析
- **模型对比**: 同时对比HyperGAT(SANN-WA)和HyperGCN(SANN-WoA)
- **宏平均计算**: 自动计算所有类别的宏平均AUC
- **高质量输出**: 生成300 DPI的高质量图片

### 2. 使用方法

```python
# 基本使用
from plot_roc_auc_curves import plot_all_basins_roc_auc

# 设置类别名称
class_names = ['Class I', 'Class II', 'Class III', 'Class IV', 'Class V']

# 绘制所有流域的ROC-AUC曲线
roc_auc_results = plot_all_basins_roc_auc(
    config=config,
    data=full_data,
    basins=basins,
    model_name="HyperGAT",
    class_names=class_names
)
```

### 3. 输出文件

- `{model_name}_{basin}_roc_auc.png`: 每个流域的ROC-AUC曲线图
- `{model_name}_all_basins_roc_auc.json`: 所有流域的AUC数据
- `roc_auc_summary.json`: 汇总结果

### 4. 结果解读

- **各类别AUC**: 每个水质类别的分类性能
- **宏平均AUC**: 整体分类性能的综合指标
- **曲线形状**: 反映模型在不同阈值下的性能表现

## 高级功能

### 1. 内存监控

```python
from efficiency_evaluation import MemoryMonitor

monitor = MemoryMonitor()
monitor.start_monitoring()

# 执行模型操作
with torch.no_grad():
    output = model(x, edge_index)

monitor.sample_memory()
peak_memory, avg_memory = monitor.get_metrics()
```

### 2. 自定义计时器

```python
from efficiency_evaluation import timer_and_memory

with timer_and_memory() as monitor:
    # 执行需要计时的操作
    model.train()
    for epoch in range(10):
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        monitor.sample_memory()

print(f"总时间: {timer_and_memory.last_time:.2f}秒")
print(f"峰值内存: {timer_and_memory.peak_memory:.1f}MB")
```

### 3. 可扩展性测试

```python
# 测试不同数据规模下的性能
data_sizes = [100, 500, 1000, 2000]
results = {}

for size in data_sizes:
    # 创建子数据集
    subset_data = create_subset(data, size)
    
    # 评估效率
    metrics = evaluator.evaluate_model_efficiency(config, model_name, subset_data)
    results[size] = metrics
```

## 故障排除

### 1. 常见问题

**Q: 内存不足错误**
```python
# 解决方案：减少批次大小或使用CPU
config = TrainConfig(
    batch_size=32,  # 减少批次大小
    device="cpu"    # 使用CPU
)
```

**Q: 评估时间过长**
```python
# 解决方案：减少测试轮数
config = TrainConfig(
    epochs=10,      # 减少训练轮数
    test_runs=3     # 减少测试次数
)
```

**Q: 结果不一致**
```python
# 解决方案：设置随机种子
torch.manual_seed(42)
np.random.seed(42)
```

### 2. 性能优化建议

1. **使用GPU加速**：确保CUDA可用
2. **批处理优化**：合理设置批次大小
3. **内存管理**：及时清理不需要的张量
4. **并行处理**：使用多进程进行批量评估


> 
> 实验结果表明，SANN-WA相比SANN-WoA在参数数量上增加约X倍，推理时间增加约Y倍，内存使用增加约Z倍，但提供了更好的性能表现。这些分析将帮助读者更好地理解注意力机制的计算成本和实用性权衡。"

## 联系与支持

如有问题或建议，请参考：
1. 检查依赖包是否正确安装
2. 确认数据文件路径正确
3. 查看错误日志获取详细信息
