from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Run inference on val_loader and compute accuracy and confusion matrix.

    Returns:
        y_true       : ground-truth labels (numpy array)
        y_pred       : predicted labels   (numpy array)
        accuracy     : overall accuracy   (float, 0-1)
        conf_matrix  : confusion matrix   (numpy array, shape [C, C])
    """
    model.eval()
    model.to(device)

    all_labels: list[int] = []
    all_preds: list[int] = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    accuracy = float((y_true == y_pred).mean())
    conf_matrix = confusion_matrix(y_true, y_pred)

    return y_true, y_pred, accuracy, conf_matrix
