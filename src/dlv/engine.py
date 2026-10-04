"""Training loop and evaluation for image classifiers."""

from math import cos, pi

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import LambdaLR


def train_all(model: nn.Module) -> None:
    """Puts every layer of the model in training mode."""
    model.train()


def run_epoch(
    model, loader, device, optimizer=None, scheduler=None, use_amp: bool = False, scaler=None
) -> tuple[float, float]:
    """Runs one pass over a loader and returns mean loss and accuracy.

    The model is updated only when an optimizer is given. An empty loader
    yields zeros. The scheduler, when given, is stepped once per training
    batch. ``use_amp`` turns on mixed precision with gradient scaling on CUDA
    and does nothing on other devices. Pass a ``scaler`` to keep one gradient
    scale across epochs; without it a new one is made for this pass.
    """
    training = optimizer is not None
    amp_enabled = use_amp and device.type == "cuda"
    if scaler is None:
        scaler = GradScaler(device.type, enabled=amp_enabled)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast(device.type, enabled=amp_enabled):
                logits = model(images)
                loss = cross_entropy(logits, labels)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / max(count, 1), correct / max(count, 1)


def fit(
    model,
    train_loader,
    validation_loader,
    optimizer,
    epochs: int,
    device,
    set_train_mode=train_all,
    scheduler=None,
    use_amp: bool = False,
) -> dict[str, list[float]]:
    """Trains a classifier and returns loss and accuracy for every epoch.

    ``set_train_mode`` decides which layers train. Pass a custom function to
    keep a frozen backbone in evaluation mode. ``scheduler`` and ``use_amp``
    are passed to every training pass.
    """
    scaler = GradScaler(device.type, enabled=use_amp and device.type == "cuda")
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(epochs):
        set_train_mode(model)
        train_loss, train_accuracy = run_epoch(model, train_loader, device, optimizer, scheduler, use_amp, scaler)
        model.eval()
        val_loss, val_accuracy = run_epoch(model, validation_loader, device, use_amp=use_amp)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train loss {train_loss:.4f} acc {train_accuracy:.4f} "
            f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
        )
    return history


def evaluate(model, loader, device, num_classes: int, use_amp: bool = False) -> dict:
    """Measures overall accuracy, accuracy per class and the confusion matrix.

    Confusion rows are true classes and columns are predicted classes. The
    predicted and true classes of every sample are also returned as lists of
    ints in loader order. ``use_amp`` runs the forward pass in mixed precision
    on CUDA and does nothing on other devices.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for images, labels in loader:
            with autocast(device.type, enabled=use_amp and device.type == "cuda"):
                logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            for truth, prediction in zip(labels.tolist(), predictions.tolist()):
                confusion[truth, prediction] += 1
                all_targets.append(truth)
                all_predictions.append(prediction)
    hits = confusion.diag()
    per_class = (hits / confusion.sum(dim=1).clamp(min=1)).tolist()
    accuracy = (hits.sum() / confusion.sum().clamp(min=1)).item()
    return {
        "accuracy": accuracy,
        "per_class_accuracy": per_class,
        "confusion": confusion.tolist(),
        "predictions": all_predictions,
        "targets": all_targets,
    }


def macro_f1(confusion: list[list[int]]) -> float:
    """Averages the F1 score of every class of a confusion matrix.

    Rows are true classes and columns are predicted classes. A class with no
    samples and no predictions scores zero.
    """
    size = len(confusion)
    scores = []
    for index in range(size):
        hits = confusion[index][index]
        predicted = sum(row[index] for row in confusion)
        actual = sum(confusion[index])
        denominator = predicted + actual
        scores.append(2 * hits / denominator if denominator else 0.0)
    return sum(scores) / max(size, 1)


def mean_class_distance(confusion: list[list[int]]) -> float:
    """Averages |true index - predicted index| over all samples.

    Useful when classes are ordered, such as periods in time. An empty matrix
    gives zero.
    """
    total = sum(sum(row) for row in confusion)
    distance = sum(
        count * abs(truth - prediction)
        for truth, row in enumerate(confusion)
        for prediction, count in enumerate(row)
    )
    return distance / max(total, 1)


def within_distance_accuracy(confusion: list[list[int]], distance: int = 1) -> float:
    """Returns the share of samples predicted at most ``distance`` classes from the truth.

    Rows are true classes and columns are predicted classes, in an order that
    has meaning, such as periods in time. An empty matrix gives zero.
    """
    total = sum(sum(row) for row in confusion)
    near = sum(
        count
        for truth, row in enumerate(confusion)
        for prediction, count in enumerate(row)
        if abs(truth - prediction) <= distance
    )
    return near / max(total, 1)


def warmup_cosine_schedule(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Schedules a linear warm-up followed by a cosine decay, counted in steps.

    The rate of every parameter group rises in equal steps to its base value
    over ``warmup_steps``, then falls along half a cosine and reaches zero at
    ``total_steps``. Steps past the total keep the rate at zero. The scheduler
    is meant to be stepped once per training batch, as ``fit`` does.
    """
    decay_steps = max(total_steps - warmup_steps, 1)

    def factor(step: int) -> float:
        """Returns the multiplier of the base rate at one step."""
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min((step - warmup_steps) / decay_steps, 1.0)
        return 0.5 * (1 + cos(pi * progress))

    return LambdaLR(optimizer, factor)
