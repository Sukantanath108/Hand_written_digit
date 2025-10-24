"""
Helper functions for PyTorch deep learning projects.

Includes utilities for:
- File and directory management
- Visualization (loss, accuracy, predictions, decision boundaries)
- Model evaluation
- Reproducibility and data downloading
"""

import os
import zipfile
from pathlib import Path
from typing import List
from collections import defaultdict
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn


# ======================
# FILE / DIRECTORY UTILS
# ======================

def walk_through_dir(dir_path: str):
    """
    Walks through a directory and prints its subdirectories and file counts.

    Args:
        dir_path (str): target directory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def download_data(source: str, destination: str, remove_source: bool = True) -> Path:
    """
    Downloads a zipped dataset from a URL and extracts it to a destination folder.

    Args:
        source (str): URL of the zip file
        destination (str): target directory to extract the dataset
        remove_source (bool): whether to delete the downloaded zip file

    Returns:
        pathlib.Path: Path to the extracted dataset
    """
    data_path = Path("data/")
    image_path = data_path / destination

    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
        return image_path

    image_path.mkdir(parents=True, exist_ok=True)
    target_file = Path(source).name

    print(f"[INFO] Downloading {target_file} from {source}...")
    with open(data_path / target_file, "wb") as f:
        request = requests.get(source)
        f.write(request.content)

    print(f"[INFO] Unzipping {target_file}...")
    with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
        zip_ref.extractall(image_path)

    if remove_source:
        os.remove(data_path / target_file)

    return image_path


# ======================
# VISUALIZATION
# ======================

def plot_decision_boundary(model: nn.Module, X: torch.Tensor, y: torch.Tensor):
    """
    Plots decision boundaries of a model against true labels.

    Args:
        model (nn.Module): trained model
        X (torch.Tensor): input features (2D)
        y (torch.Tensor): true labels
    """
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101),
                         np.linspace(y_min, y_max, 101))

    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))

    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots training and testing data with optional model predictions.

    Args:
        train_data: training features
        train_labels: training labels
        test_data: testing features
        test_labels: testing labels
        predictions: optional predictions to overlay on test data
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


def plot_loss_curves(results: dict):
    """
    Plots training and test loss and accuracy curves.

    Args:
        results (dict): dictionary containing 'train_loss', 'train_acc', 'test_loss', 'test_acc'
    """
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["test_loss"], label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="train_accuracy")
    plt.plot(epochs, results["test_acc"], label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


def pred_and_plot_image(model: nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Predicts the class of an image and plots it with the prediction and probability.

    Args:
        model: trained PyTorch model
        image_path: path to image file
        class_names: list of class names
        transform: optional transform to apply
        device: torch device to run model on
    """
    target_image = torchvision.io.read_image(str(image_path)).float() / 255.0
    if transform:
        target_image = transform(target_image)

    model.to(device)
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(0).to(device)
        logits = model(target_image)
        probs = torch.softmax(logits, dim=1)
        label = torch.argmax(probs, dim=1)

    plt.imshow(target_image.squeeze().permute(1, 2, 0))
    title = (f"Pred: {class_names[label.cpu()]} | Prob: {probs.max().cpu():.3f}"
             if class_names else
             f"Pred: {label} | Prob: {probs.max().cpu():.3f}")
    plt.title(title)
    plt.axis(False)
    plt.show()


# ======================
# METRICS / UTILITIES
# ======================

def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Calculates accuracy as a percentage.

    Args:
        y_true: true labels
        y_pred: predicted labels

    Returns:
        Accuracy percentage
    """
    return (torch.eq(y_true, y_pred).sum().item() / len(y_pred)) * 100


def print_train_time(start: float, end: float, device=None) -> float:
    """
    Prints and returns training time.

    Args:
        start: start time
        end: end time
        device: device used for training
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


def set_seeds(seed: int = 42):
    """
    Sets random seeds for reproducibility.

    Args:
        seed: seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)




class MetricsTracker:
    """
    Tracks training metrics and provides visualization utilities.

    Example usage:
        tracker = MetricsTracker()
        tracker.update({'train_loss': 0.5, 'val_loss': 0.6, 'train_acc': 80, 'val_acc': 78})
        tracker.plot_history()
    """

    def __init__(self):
        self.history = defaultdict(list)

    def update(self, metrics_dict):
        """
        Add new metrics from the current epoch.

        Args:
            metrics_dict (dict): {'metric_name': value, ...}
        """
        for key, value in metrics_dict.items():
            self.history[key].append(value)

    def plot_history(self, save_path='visualizations/training_history.png'):
        """
        Plots training/validation loss, accuracy, learning rate (if tracked),
        and loss gap to monitor overfitting.

        Args:
            save_path (str): Path to save the figure
        """
        n_rows, n_cols = 2, 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
        axes = axes.flatten()

        def plot_line(y_keys, ax, title, ylabel, colors=None):
            for i, key in enumerate(y_keys):
                if key in self.history:
                    c = colors[i] if colors else None
                    ax.plot(self.history[key], label=key.replace('_', ' ').title(), linewidth=2, color=c)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Loss
        plot_line(['train_loss', 'val_loss'], axes[0], 'Training and Validation Loss', 'Loss')

        # Accuracy
        plot_line(['train_acc', 'val_acc'], axes[1], 'Training and Validation Accuracy', 'Accuracy (%)')

        # Learning rate (if exists)
        if 'learning_rate' in self.history:
            axes[2].plot(self.history['learning_rate'], color='green', linewidth=2)
            axes[2].set_title('Learning Rate Schedule')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')

        # Loss gap
        if 'train_loss' in self.history and 'val_loss' in self.history:
            loss_gap = np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
            axes[3].plot(loss_gap, color='red', linewidth=2)
            axes[3].axhline(0, color='black', linestyle='--', alpha=0.5)
            axes[3].fill_between(range(len(loss_gap)), 0, loss_gap, alpha=0.3,
                                 color='red' if loss_gap[-1] > 0 else 'green')
            axes[3].set_title('Overfitting Indicator (Val - Train Loss)')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Loss Gap')
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
