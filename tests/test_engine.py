"""Tests for the training loop and evaluation."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dlv.engine import evaluate, fit, run_epoch


def separable_loader():
    """Returns a loader over two well separated two-dimensional classes."""
    torch.manual_seed(0)
    features = torch.cat([torch.randn(40, 2) + 4, torch.randn(40, 2) - 4])
    labels = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    return DataLoader(TensorDataset(features, labels), batch_size=16, shuffle=True)


class AlwaysFirstClass(nn.Module):
    """Predicts class zero for every input."""

    def forward(self, inputs):
        """Returns logits that favour class zero."""
        logits = torch.zeros(inputs.size(0), 2)
        logits[:, 0] = 1.0
        return logits


def test_fit_reduces_loss_and_records_every_epoch():
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = separable_loader()
    history = fit(model, loader, loader, optimizer, epochs=5, device=torch.device("cpu"))
    assert set(history) == {"train_loss", "train_accuracy", "val_loss", "val_accuracy"}
    assert all(len(values) == 5 for values in history.values())
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["val_accuracy"][-1] > 0.95


def test_fit_calls_train_mode_setter_each_epoch():
    calls = []
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = separable_loader()
    fit(
        model,
        loader,
        loader,
        optimizer,
        epochs=3,
        device=torch.device("cpu"),
        set_train_mode=lambda module: calls.append(module),
    )
    assert calls == [model, model, model]


def test_run_epoch_without_optimizer_leaves_weights_untouched():
    model = nn.Linear(2, 2)
    before = model.weight.clone()
    run_epoch(model, separable_loader(), torch.device("cpu"))
    assert torch.equal(model.weight, before)


def test_run_epoch_on_empty_loader_returns_zeros():
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    assert run_epoch(nn.Linear(2, 2), empty, torch.device("cpu")) == (0.0, 0.0)


def test_evaluate_reports_per_class_accuracy_and_confusion():
    loader = separable_loader()
    report = evaluate(AlwaysFirstClass(), loader, torch.device("cpu"), num_classes=2)
    assert report["accuracy"] == 0.5
    assert report["per_class_accuracy"] == [1.0, 0.0]
    assert report["confusion"] == [[40, 0], [40, 0]]
