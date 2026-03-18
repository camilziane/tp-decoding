from __future__ import annotations

import copy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


@dataclass
class DNNTrainingContext:
    device: torch.device
    seed: int
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_weights_torch: torch.Tensor
    class_counts: np.ndarray
    class_weights: np.ndarray
    results: list = field(default_factory=list)
    histories: dict = field(default_factory=dict)
    trained_models: dict = field(default_factory=dict)


def make_optimizer(model, optimizer_name="adamw", lr=2e-4, weight_decay=1e-3):
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def plot_dnn_learning_curves(model_name, history, figsize=(50, 20)):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    loss_values = np.asarray(history["train_loss"] + history["val_loss"], dtype=float)
    f1_values = np.asarray(history["val_macro_f1"], dtype=float)

    loss_padding = max(0.03, 0.05 * max(loss_values.max() - loss_values.min(), 1e-6))
    loss_y_min = max(0.0, loss_values.min() - loss_padding)
    loss_y_max = loss_values.max() + loss_padding

    f1_y_min = max(0.0, f1_values.min() - 0.03)
    f1_y_max = min(1.0, f1_values.max() + 0.03)
    if f1_y_max <= f1_y_min:
        f1_y_max = min(1.0, f1_y_min + 0.06)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    axes[0].set_title(f"{model_name}: train/validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_ylim(loss_y_min, loss_y_max)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["val_macro_f1"], marker="o", color="tab:green")
    axes[1].set_title(f"{model_name}: validation macro F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(f1_y_min, f1_y_max)
    axes[1].grid(alpha=0.3)

    fig.suptitle(f"Learning curves for {model_name}")
    plt.show()


@torch.no_grad()
def predict_labels(model, loader, device):
    model.eval()
    preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


@torch.no_grad()
def predict_probabilities(model, loader, device):
    model.eval()
    y_true_all = []
    proba_all = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        proba = torch.softmax(logits, dim=1)
        y_true_all.append(y_batch.cpu().numpy())
        proba_all.append(proba.cpu().numpy())

    return np.concatenate(y_true_all), np.concatenate(proba_all)


@torch.no_grad()
def evaluate_loader(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true_all = []
    y_pred_all = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        y_pred = torch.argmax(logits, dim=1)

        losses.append(loss.item())
        y_true_all.append(y_batch.cpu().numpy())
        y_pred_all.append(y_pred.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    metrics = {
        "loss": float(np.mean(losses)),
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
    return metrics, y_true, y_pred


def build_class_weights_torch(y, n_classes, device):
    class_counts = np.bincount(y, minlength=n_classes)
    class_weights = class_counts.sum() / (len(class_counts) * np.maximum(class_counts, 1))
    class_weights_torch = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return class_weights_torch, class_counts, class_weights


def make_training_context(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    *,
    batch_size,
    n_classes,
    device,
    seed,
    results=None,
    histories=None,
    trained_models=None,
):
    train_loader = DataLoader(WindowDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    class_weights_torch, class_counts, class_weights = build_class_weights_torch(y_train, n_classes, device)

    return DNNTrainingContext(
        device=device,
        seed=seed,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_weights_torch=class_weights_torch,
        class_counts=class_counts,
        class_weights=class_weights,
        results=[] if results is None else results,
        histories={} if histories is None else histories,
        trained_models={} if trained_models is None else trained_models,
    )


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    *,
    device,
    n_epochs=6,
    optimizer_name="adamw",
    lr=2e-4,
    weight_decay=1e-3,
    checkpoint_metric="bal_acc",
):
    optimizer = make_optimizer(model, optimizer_name=optimizer_name, lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_bal_acc": [], "val_macro_f1": []}
    best_score = -np.inf
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_metrics, _, _ = evaluate_loader(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_bal_acc"].append(val_metrics["bal_acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        if checkpoint_metric not in val_metrics:
            raise ValueError(f"Unknown checkpoint metric: {checkpoint_metric}")

        if val_metrics[checkpoint_metric] > best_score:
            best_score = val_metrics[checkpoint_metric]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{n_epochs} | "
            f"train loss={train_loss:.4f} | "
            f"val loss={val_metrics['loss']:.4f} | "
            f"val bal_acc={val_metrics['bal_acc']:.3f} | "
            f"val macro_f1={val_metrics['macro_f1']:.3f}"
        )

    model.load_state_dict(best_state)
    return model, history


def train_and_record(
    *,
    context,
    model_name,
    model,
    n_epochs=6,
    optimizer_name="adamw",
    lr=2e-4,
    weight_decay=1e-3,
    checkpoint_metric="bal_acc",
    store_results=True,
    show_learning_curves=True,
    learning_curve_figsize=(50, 20),
    return_probabilities=False,
):
    print("\n" + "=" * 90)
    print(f"Training {model_name}")
    print("=" * 90)

    torch.manual_seed(context.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(context.seed)

    model = model.to(context.device)
    criterion = nn.CrossEntropyLoss(weight=context.class_weights_torch)

    model, history = train_model(
        model,
        context.train_loader,
        context.val_loader,
        criterion,
        device=context.device,
        n_epochs=n_epochs,
        optimizer_name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        checkpoint_metric=checkpoint_metric,
    )

    val_metrics, y_true_val, y_pred_val = evaluate_loader(model, context.val_loader, criterion, context.device)
    test_metrics, y_true_test, y_pred_test = evaluate_loader(model, context.test_loader, criterion, context.device)

    result = {
        "model_name": model_name,
        "model": model,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "y_val_true": y_true_val,
        "y_val_pred": y_pred_val,
        "y_test_true": y_true_test,
        "y_test_pred": y_pred_test,
        "batch_size": getattr(context.train_loader, "batch_size", None),
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer_name": optimizer_name,
    }

    if return_probabilities:
        _, result["val_proba"] = predict_probabilities(model, context.val_loader, context.device)
        _, result["test_proba"] = predict_probabilities(model, context.test_loader, context.device)

    row = {
        "model": model_name,
        "val_acc": val_metrics["acc"],
        "val_bal_acc": val_metrics["bal_acc"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_loss": val_metrics["loss"],
        "test_acc": test_metrics["acc"],
        "test_bal_acc": test_metrics["bal_acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_loss": test_metrics["loss"],
    }
    result["summary_row"] = row

    if store_results:
        context.results.append(row)
        context.histories[model_name] = history
        context.trained_models[model_name] = model

    print(
        f"Best validation checkpoint ({model_name}) | "
        f"bal_acc={row['val_bal_acc']:.3f} | "
        f"macro_f1={row['val_macro_f1']:.3f} | "
        f"acc={row['val_acc']:.3f}"
    )
    print(
        f"Test metrics ({model_name}) | "
        f"bal_acc={row['test_bal_acc']:.3f} | "
        f"macro_f1={row['test_macro_f1']:.3f} | "
        f"acc={row['test_acc']:.3f}"
    )

    if show_learning_curves:
        plot_dnn_learning_curves(model_name, history, figsize=learning_curve_figsize)

    return result
