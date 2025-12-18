import os
import json
import time
import psutil
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score

from data_processing import prepare_data_from_csv, prepare_temporal_data_from_csv
from models import HyperGCN, HyperGAT, HyperTemporalModel


@dataclass
class TrainConfig:
    csv_path: str = "./包含藻密度的所有数据.csv"
    label_col: str = "Water_quality_class"
    drop_cols: Tuple[str, ...] = ()
    group_col: str = "Basin"
    features: Tuple[str, ...] = (
        'Water_temperature(°C)',
        'pH',
        'Dissolved_oxygen(mg/L)',
        'Electrical_conductivity(μS/cm)',
        'Turbidity(NTU)',
        'COD_Mn(mg/L)',
        'NH3-N(mg/L)',
        'TP(mg/L)',
        'TN(mg/L)',
        'Chlorophyll_a(mg/L)',
        'Algae_density(cells/L)'
    )
    k_neighbors: int = 3  # match original knn_graph(k=3)
    val_size: float = 0.2  # original 70/10/20 split
    test_size: float = 0.1
    random_state: int = 42
    model_name: str = "HyperGAT"  # or "HyperGAT"
    hidden_channels: int = 64
    heads: int = 4
    dropout: float = 0.5
    epochs: int = 1000  # max_epochs
    early_stop_patience: int = 20
    lr: float = 1e-3  # 0.001
    weight_decay: float = 5e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 50
    output_dir: str = "./results"
    # temporal sequence settings
    enable_temporal: bool = True
    id_col: str = "Section_name"
    time_col: str = "Monitoring_time"
    interval_days: int = 30
    steps: int = 1 # <=0: auto steps from date span / interval
    # efficiency evaluation settings
    enable_efficiency_eval: bool = False
    efficiency_output_dir: str = "./efficiency_results"


def build_model(config: TrainConfig, in_channels: int, out_channels: int) -> nn.Module:
    if config.model_name.lower() == "hypergcn":
        return HyperGCN(in_channels, config.hidden_channels, out_channels, dropout=config.dropout)
    elif config.model_name.lower() == "hypergat":
        return HyperGAT(in_channels, config.hidden_channels, out_channels, dropout=config.dropout)
    else:
        raise ValueError("model_name must be 'HyperGCN' or 'HyperGAT'")


def train_one_epoch(model: nn.Module, data: Dict[str, torch.Tensor], optimizer: optim.Optimizer, criterion: nn.Module) -> float:
    model.train()
    optimizer.zero_grad()
    if "x_seq" in data:
        logits = model(data["x"], data["hyperedge_index"], data["x_seq"])  # temporal
    else:
        logits = model(data["x"], data["hyperedge_index"])  # spatial only
    loss = criterion(logits[data["train_mask"]], data["y"][data["train_mask"]])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model: nn.Module, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
    model.eval()
    if "x_seq" in data:
        logits = model(data["x"], data["hyperedge_index"], data["x_seq"])  # [N, C]
    else:
        logits = model(data["x"], data["hyperedge_index"])  # [N, C]
    y_true = data["y"].cpu().numpy()
    y_prob = torch.softmax(logits, dim=1).cpu().numpy()
    y_pred = y_prob.argmax(axis=1)

    # Masks
    splits = {
        "train": data["train_mask"].cpu().numpy(),
        "val": data["val_mask"].cpu().numpy(),
        "test": data["test_mask"].cpu().numpy(),
    }

    results: Dict[str, float] = {}
    for split, mask in splits.items():
        yt = y_true[mask]
        yp = y_pred[mask]
        yp_prob = y_prob[mask]
        results[f"{split}_acc"] = accuracy_score(yt, yp)
        results[f"{split}_f1_macro"] = f1_score(yt, yp, average="macro")
        # For multi-class ROC-AUC, use one-vs-rest when possible
        try:
            results[f"{split}_roc_auc_ovr"] = roc_auc_score(yt, yp_prob, multi_class="ovr")
        except Exception:
            results[f"{split}_roc_auc_ovr"] = float("nan")
    return results


def save_run(output_dir: str, config: TrainConfig, data: Dict[str, torch.Tensor], model: nn.Module, history: Dict[str, list], metrics: Dict[str, float]):
    os.makedirs(output_dir, exist_ok=True)
    # Save config and metrics
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Save predictions for later visualization
    model.eval()
    with torch.no_grad():
        if "x_seq" in data:
            logits = model(data["x"], data["hyperedge_index"], data["x_seq"]).cpu()
        else:
            logits = model(data["x"], data["hyperedge_index"]).cpu()
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)
        np.save(os.path.join(output_dir, "y_true.npy"), data["y"].cpu().numpy())
        np.save(os.path.join(output_dir, "y_pred.npy"), preds)
        np.save(os.path.join(output_dir, "y_prob.npy"), probs)
        np.save(os.path.join(output_dir, "train_mask.npy"), data["train_mask"].cpu().numpy())
        np.save(os.path.join(output_dir, "val_mask.npy"), data["val_mask"].cpu().numpy())
        np.save(os.path.join(output_dir, "test_mask.npy"), data["test_mask"].cpu().numpy())


def evaluate_model_efficiency(model: nn.Module, data: Dict[str, torch.Tensor], config: TrainConfig) -> Dict[str, float]:
    """评估模型的计算效率"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 移动数据到设备
    for key in data:
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    
    efficiency_metrics = {}
    
    # 1. 模型参数数量
    efficiency_metrics["num_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 2. 模型大小估算
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    efficiency_metrics["model_size_mb"] = (param_size + buffer_size) / 1024 / 1024
    
    # 3. 推理时间测试
    model.eval()
    inference_times = []
    
    with torch.no_grad():
        # 预热
        for _ in range(5):
            if "x_seq" in data:
                _ = model(data["x"], data["hyperedge_index"], data["x_seq"])
            else:
                _ = model(data["x"], data["hyperedge_index"])
        
        # 正式测试
        for _ in range(20):
            start_time = time.time()
            if "x_seq" in data:
                _ = model(data["x"], data["hyperedge_index"], data["x_seq"])
            else:
                _ = model(data["x"], data["hyperedge_index"])
            inference_times.append(time.time() - start_time)
    
    efficiency_metrics["avg_inference_time"] = np.mean(inference_times)
    efficiency_metrics["throughput"] = data["x"].size(0) / efficiency_metrics["avg_inference_time"]
    
    # 4. 内存使用测试
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            if "x_seq" in data:
                _ = model(data["x"], data["hyperedge_index"], data["x_seq"])
            else:
                _ = model(data["x"], data["hyperedge_index"])
        efficiency_metrics["inference_memory_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        with torch.no_grad():
            if "x_seq" in data:
                _ = model(data["x"], data["hyperedge_index"], data["x_seq"])
            else:
                _ = model(data["x"], data["hyperedge_index"])
        memory_after = process.memory_info().rss / 1024 / 1024
        efficiency_metrics["inference_memory_mb"] = memory_after - memory_before
    
    return efficiency_metrics


def main(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import pandas as pd
    df_all = pd.read_csv(config.csv_path)
    # Load data (optionally temporal sequences)
    if config.enable_temporal:
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
    else:
        full_data, _ = prepare_data_from_csv(
            csv_path=config.csv_path,
            label_col=config.label_col,
            drop_cols=list(config.drop_cols),
            group_col=config.group_col,
            features=list(config.features) if config.features else None,
            k_neighbors=config.k_neighbors,
            val_size=config.val_size,
            test_size=config.test_size,
            random_state=config.random_state,
        )
        if config.group_col not in df_all.columns:
            raise ValueError(f"Group column '{config.group_col}' not found in CSV.")
        basins = df_all[config.group_col].values

    X = full_data["x"]
    y = full_data["y"]
    num_nodes = X.size(0)
    in_channels = X.size(1)
    out_channels = int(y.max().item()) + 1

    # Global 70/10/20 random split (no stratification), like original notebook
    g = torch.Generator().manual_seed(config.random_state)
    indices = torch.randperm(num_nodes, generator=g)
    n_train = int(num_nodes * 0.7)
    n_val = int(num_nodes * 0.1)
    train_nodes = indices[:n_train]
    val_nodes = indices[n_train:n_train + n_val]
    test_nodes = indices[n_train + n_val:]
    global_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    global_val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    global_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    global_train_mask[train_nodes] = True
    global_val_mask[val_nodes] = True
    global_test_mask[test_nodes] = True

    # Class weights from global labels
    y_np = y.cpu().numpy()
    class_counts = np.bincount(y_np)
    class_weights = 1.0 / np.clip(class_counts, a_min=1, a_max=None)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Prepare output root
    root_dir = os.path.join(config.output_dir, f"{config.model_name}")
    os.makedirs(root_dir, exist_ok=True)

    unique_basins = np.unique(basins)
    for basin in unique_basins:
        print(f"\n========== Processing Basin: {basin} ==========")
        basin_indices = np.where(basins == basin)[0]
        if len(basin_indices) == 0:
            continue
        basin_dir = os.path.join(root_dir, str(basin))
        os.makedirs(basin_dir, exist_ok=True)

        idx_tensor = torch.tensor(basin_indices, dtype=torch.long)
        sub_x = X[idx_tensor].to(device)
        sub_y = y[idx_tensor].to(device)

        # Build per-basin hypergraph via KNN with k=3, include self
        from data_processing import build_knn_edge_index
        sub_x_np = sub_x.cpu().numpy()
        sub_edge_index = build_knn_edge_index(sub_x_np, k_neighbors=config.k_neighbors, include_self=True, make_undirected=True).to(device)

        # Masks per basin from global masks
        sub_train_mask = global_train_mask[idx_tensor].to(device)
        sub_val_mask = global_val_mask[idx_tensor].to(device)
        sub_test_mask = global_test_mask[idx_tensor].to(device)
        if sub_train_mask.sum().item() == 0 or sub_test_mask.sum().item() == 0:
            print(f"Skip basin {basin}: insufficient train/test samples.")
            continue

        # Build model (optionally with temporal encoder)
        base_model = build_model(config, in_channels, out_channels).to(device)
        if config.enable_temporal:
            model = HyperTemporalModel(base_model, feature_dim=in_channels, temporal_dim=128, fuse_dim=128, dropout=config.dropout).to(device)
        else:
            model = base_model
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience, verbose=True
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

        # Wrap data dict for this basin
        sub_data = {
            "x": sub_x,
            "y": sub_y,
            "hyperedge_index": sub_edge_index,  # edge_index tensor used by both models
            "train_mask": sub_train_mask,
            "val_mask": sub_val_mask,
            "test_mask": sub_test_mask,
        }
        if config.enable_temporal:
            sub_data["x_seq"] = full_data["x_seq"][idx_tensor].to(device)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        history = {"loss": [], "val_loss": []}

        for epoch in range(config.epochs):
            model.train()
            optimizer.zero_grad()
            if config.enable_temporal:
                logits = model(sub_data["x"], sub_data["hyperedge_index"], sub_data["x_seq"])  # forward
            else:
                logits = model(sub_data["x"], sub_data["hyperedge_index"])  # forward
            loss = criterion(logits[sub_data["train_mask"]], sub_data["y"][sub_data["train_mask"]])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                if config.enable_temporal:
                    val_logits = model(sub_data["x"], sub_data["hyperedge_index"], sub_data["x_seq"]) 
                else:
                    val_logits = model(sub_data["x"], sub_data["hyperedge_index"]) 
                val_loss = criterion(val_logits[sub_data["val_mask"]], sub_data["y"][sub_data["val_mask"]])
            scheduler.step(val_loss)

            history["loss"].append(float(loss.item()))
            history["val_loss"].append(float(val_loss.item()))

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1} for basin {basin}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Evaluate and save per-basin
        # Evaluate (test split) and print in required format
        model.eval()
        with torch.no_grad():
            if config.enable_temporal:
                logits = model(sub_data["x"], sub_data["hyperedge_index"], sub_data["x_seq"])  
            else:
                logits = model(sub_data["x"], sub_data["hyperedge_index"])  
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
        import numpy as _np
        from sklearn.metrics import confusion_matrix as _cm, accuracy_score as _acc, f1_score as _f1, precision_score as _prec, recall_score as _rec
        mask = sub_data["test_mask"].cpu().numpy()
        y_true_test = sub_data["y"].cpu().numpy()[mask]
        y_pred_test = preds[mask]
        cm = _cm(y_true_test, y_pred_test)
        acc = _acc(y_true_test, y_pred_test)
        f1 = _f1(y_true_test, y_pred_test, average='macro')
        prec = _prec(y_true_test, y_pred_test, average='macro', zero_division=0)
        rec = _rec(y_true_test, y_pred_test, average='macro')

        print("\nMetrics for Basin {}:".format(basin))
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")

        # Compute per-basin total days (contiguous span on daily-aggregated dates)
        try:
            from data_processing import _parse_monitoring_time as _parse_time
            df_basin = df_all.iloc[basin_indices]
            times = _parse_time(df_basin[config.time_col]).dropna()
            if len(times) > 0:
                dmin = times.min().normalize()
                dmax = times.max().normalize()
                total_days_basin = int((dmax - dmin).days) + 1
            else:
                total_days_basin = 0
        except Exception:
            total_days_basin = 0

        metrics = evaluate(model, sub_data)
        metrics["total_days"] = int(total_days_basin)
        
        # 计算效率评估
        if config.enable_efficiency_eval:
            print(f"评估 {basin} 的计算效率...")
            efficiency_metrics = evaluate_model_efficiency(model, sub_data, config)
            metrics.update(efficiency_metrics)
            
            # 打印效率指标
            print(f"  参数数量: {efficiency_metrics['num_parameters']:,}")
            print(f"  模型大小: {efficiency_metrics['model_size_mb']:.2f} MB")
            print(f"  平均推理时间: {efficiency_metrics['avg_inference_time']*1000:.2f} ms")
            print(f"  推理吞吐量: {efficiency_metrics['throughput']:.1f} 样本/秒")
            print(f"  推理内存使用: {efficiency_metrics['inference_memory_mb']:.1f} MB")
        
        save_run(basin_dir, config, sub_data, model, history, metrics)

        # Save temporal attention weights if present
        try:
            if config.enable_temporal and hasattr(model, 'temporal') and getattr(model.temporal, 'last_attn', None) is not None:
                attn = model.temporal.last_attn  # [N, heads, T, T]
                np.save(os.path.join(basin_dir, "temporal_last_attn.npy"), attn.detach().cpu().numpy())
        except Exception:
            pass


def run_efficiency_comparison():
    """运行效率对比实验"""
    print("="*60)
    print("开始运行 SANN-WA vs SANN-WoA 效率对比实验")
    print("="*60)
    
    # 运行两个模型并收集效率数据
    efficiency_results = {}
    
    for model_name in ["HyperGCN", "HyperGAT"]:
        print(f"\n运行 {model_name} ({'SANN-WoA' if 'GCN' in model_name else 'SANN-WA'})...")
        
        cfg = TrainConfig(
            model_name=model_name,
            enable_efficiency_eval=True,
            epochs=50,  # 减少epochs用于快速测试
            early_stop_patience=10
        )
        
        # 运行训练并收集效率数据
        main(cfg)
        
        # 从结果目录读取效率数据
        results_dir = os.path.join(cfg.output_dir, model_name)
        if os.path.exists(results_dir):
            basin_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
            if basin_dirs:
                # 读取第一个流域的结果作为示例
                sample_basin = basin_dirs[0]
                metrics_file = os.path.join(results_dir, sample_basin, "metrics.json")
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    
                    efficiency_results[model_name] = {
                        'num_parameters': metrics.get('num_parameters', 0),
                        'model_size_mb': metrics.get('model_size_mb', 0),
                        'avg_inference_time': metrics.get('avg_inference_time', 0),
                        'throughput': metrics.get('throughput', 0),
                        'inference_memory_mb': metrics.get('inference_memory_mb', 0)
                    }
    
    # 生成对比报告
    if efficiency_results:
        print("\n" + "="*60)
        print("效率对比结果")
        print("="*60)
        
        for model_name, metrics in efficiency_results.items():
            sann_type = "SANN-WoA" if "GCN" in model_name else "SANN-WA"
            print(f"\n{model_name} ({sann_type}):")
            print(f"  参数数量: {metrics['num_parameters']:,}")
            print(f"  模型大小: {metrics['model_size_mb']:.2f} MB")
            print(f"  平均推理时间: {metrics['avg_inference_time']*1000:.2f} ms")
            print(f"  推理吞吐量: {metrics['throughput']:.1f} 样本/秒")
            print(f"  推理内存使用: {metrics['inference_memory_mb']:.1f} MB")
        
        # 计算相对性能
        if len(efficiency_results) == 2:
            models = list(efficiency_results.keys())
            gcn_metrics = efficiency_results[models[0]] if "GCN" in models[0] else efficiency_results[models[1]]
            gat_metrics = efficiency_results[models[1]] if "GAT" in models[1] else efficiency_results[models[0]]
            
            print(f"\n相对性能对比 (SANN-WA vs SANN-WoA):")
            print(f"  参数数量比率: {gat_metrics['num_parameters'] / gcn_metrics['num_parameters']:.2f}x")
            print(f"  推理时间比率: {gat_metrics['avg_inference_time'] / gcn_metrics['avg_inference_time']:.2f}x")
            print(f"  内存使用比率: {gat_metrics['inference_memory_mb'] / gcn_metrics['inference_memory_mb']:.2f}x")
            print(f"  吞吐量比率: {gat_metrics['throughput'] / gcn_metrics['throughput']:.2f}x")
    
    return efficiency_results


if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--efficiency":
        # 运行效率对比实验
        run_efficiency_comparison()
    else:
        # 运行常规训练
        for m in ["HyperGCN", "HyperGAT"]:
            cfg = TrainConfig(model_name=m)
            main(cfg)


