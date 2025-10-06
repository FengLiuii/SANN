import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def load_results(run_dir: str):
    with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(os.path.join(run_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    y_true = np.load(os.path.join(run_dir, "y_true.npy"))
    y_pred = np.load(os.path.join(run_dir, "y_pred.npy"))
    y_prob = np.load(os.path.join(run_dir, "y_prob.npy"))
    train_mask = np.load(os.path.join(run_dir, "train_mask.npy"))
    val_mask = np.load(os.path.join(run_dir, "val_mask.npy"))
    test_mask = np.load(os.path.join(run_dir, "test_mask.npy"))
    return metrics, config, y_true, y_pred, y_prob, train_mask, val_mask, test_mask


def plot_confusion(run_dir: str, split: str = "test"):
    metrics, config, y_true, y_pred, y_prob, train_mask, val_mask, test_mask = load_results(run_dir)
    mask = {"train": train_mask, "val": val_mask, "test": test_mask}[split]
    cm = confusion_matrix(y_true[mask], y_pred[mask])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{config['model_name']} - {split} confusion matrix")
    out = os.path.join(run_dir, f"confusion_{split}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", out)


def plot_roc_ovr(run_dir: str, split: str = "test"):
    metrics, config, y_true, y_pred, y_prob, train_mask, val_mask, test_mask = load_results(run_dir)
    mask = {"train": train_mask, "val": val_mask, "test": test_mask}[split]
    yt = y_true[mask]
    yp = y_prob[mask]
    num_classes = yp.shape[1]
    plt.figure()
    for c in range(num_classes):
        y_true_bin = (yt == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, yp[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{config['model_name']} - {split} ROC (OvR)")
    plt.legend(loc="lower right")
    out = os.path.join(run_dir, f"roc_{split}.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", out)


if __name__ == "__main__":
    # Example usage
    for model in ["HyperGCN", "HyperGAT"]:
        rd = os.path.join("./results", model)
        if os.path.isdir(rd):
            plot_confusion(rd, split="test")
            plot_roc_ovr(rd, split="test")


