"""Tests for the training loop and evaluation."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import pytest

from dlv import engine
from dlv.engine import (
    evaluate,
    fit,
    macro_f1,
    mean_class_distance,
    run_epoch,
    warmup_cosine_schedule,
    within_distance_accuracy,
)


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
    """Training lowers the loss and records four series with one value per epoch."""
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
    """The train mode setter is called once per epoch with the model."""
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
    """A pass without an optimizer does not update the weights."""
    model = nn.Linear(2, 2)
    before = model.weight.clone()
    run_epoch(model, separable_loader(), torch.device("cpu"))
    assert torch.equal(model.weight, before)


def test_run_epoch_on_empty_loader_returns_zeros():
    """A loader with no batches gives zero loss and zero accuracy."""
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    assert run_epoch(nn.Linear(2, 2), empty, torch.device("cpu")) == (0.0, 0.0)


def test_evaluate_reports_per_class_accuracy_and_confusion():
    """A model that always answers class zero gets the expected accuracy and matrix."""
    loader = separable_loader()
    report = evaluate(AlwaysFirstClass(), loader, torch.device("cpu"), num_classes=2)
    assert report["accuracy"] == 0.5
    assert report["per_class_accuracy"] == [1.0, 0.0]
    assert report["confusion"] == [[40, 0], [40, 0]]


def test_evaluate_returns_predictions_and_targets_in_loader_order():
    """Predictions and targets are int lists that follow the loader order."""
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1, 1, 1, 0])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
    report = evaluate(model, loader, torch.device("cpu"), num_classes=2)
    assert report["targets"] == [0, 1, 1, 1, 0]
    assert report["predictions"] == [0, 1, 0, 1, 0]
    assert all(isinstance(value, int) for value in report["predictions"] + report["targets"])


def test_fit_steps_the_scheduler_once_per_training_batch():
    """With 80 samples in batches of 16 and 3 epochs the scheduler steps 15 times."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loader = separable_loader()
    fit(model, loader, loader, optimizer, epochs=3, device=torch.device("cpu"), scheduler=scheduler)
    assert scheduler.last_epoch == 15
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1 * 0.5**15)


def test_run_epoch_does_not_step_the_scheduler_without_an_optimizer():
    """A validation pass leaves the scheduler alone."""
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    run_epoch(model, separable_loader(), torch.device("cpu"), scheduler=scheduler)
    assert scheduler.last_epoch == 0


def test_use_amp_is_a_no_op_on_cpu():
    """Training with mixed precision requested on CPU matches full precision."""
    weights = []
    for use_amp in (False, True):
        torch.manual_seed(0)
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(separable_loader().dataset, batch_size=16, shuffle=False)
        fit(model, loader, loader, optimizer, epochs=2, device=torch.device("cpu"), use_amp=use_amp)
        weights.append(model.weight.detach().clone())
    assert torch.equal(weights[0], weights[1])


def test_macro_f1_averages_the_f1_of_every_class():
    """Class 0 has F1 0.8 and class 1 has F1 2/3, so the mean is 11/15."""
    assert macro_f1([[2, 1], [0, 1]]) == pytest.approx(11 / 15)


def test_macro_f1_counts_a_class_without_samples_or_predictions_as_zero():
    """A silent empty class pulls the mean down: (1 + 1 + 0) / 3."""
    assert macro_f1([[1, 0, 0], [0, 1, 0], [0, 0, 0]]) == pytest.approx(2 / 3)


def test_macro_f1_is_one_for_a_perfect_matrix():
    """A diagonal matrix scores one."""
    assert macro_f1([[5, 0], [0, 3]]) == 1.0


def test_mean_class_distance_averages_index_gaps_over_all_samples():
    """Three samples miss by two classes and one is correct, so the first matrix gives 6 / 4 = 1.5."""
    assert mean_class_distance([[1, 0, 2], [0, 0, 0], [1, 0, 0]]) == pytest.approx(1.5)
    assert mean_class_distance([[2, 1], [0, 1]]) == pytest.approx(0.25)


def test_mean_class_distance_of_an_empty_matrix_is_zero():
    """With no samples the distance is zero."""
    assert mean_class_distance([[0, 0], [0, 0]]) == 0.0


def test_evaluate_with_amp_requested_on_cpu_matches_full_precision():
    """Mixed precision in evaluation is a no-op on CPU."""
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    loader = DataLoader(separable_loader().dataset, batch_size=16, shuffle=False)
    plain = evaluate(model, loader, torch.device("cpu"), num_classes=2)
    amp = evaluate(model, loader, torch.device("cpu"), num_classes=2, use_amp=True)
    assert plain == amp


def test_fit_shares_one_gradient_scaler_across_epochs(monkeypatch):
    """The same scaler object reaches every training pass of a fit call."""
    seen = []
    real = engine.run_epoch

    def recording(model, loader, device, optimizer=None, scheduler=None, use_amp=False, scaler=None):
        """Records the scaler and delegates to the real pass."""
        if optimizer is not None:
            seen.append(scaler)
        return real(model, loader, device, optimizer, scheduler, use_amp, scaler)

    monkeypatch.setattr(engine, "run_epoch", recording)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = separable_loader()
    fit(model, loader, loader, optimizer, epochs=3, device=torch.device("cpu"), use_amp=True)
    assert len(seen) == 3
    assert seen[0] is not None
    assert seen[0] is seen[1] is seen[2]


def learning_rates(warmup_steps, total_steps, base=1.0):
    """Returns the learning rate used at every step of a warm-up cosine schedule."""
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=base)
    scheduler = warmup_cosine_schedule(optimizer, warmup_steps, total_steps)
    rates = []
    for _ in range(total_steps):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


def test_warmup_cosine_schedule_rises_linearly_then_decays_to_near_zero():
    """The rate climbs in equal steps to the base value, then follows a cosine down."""
    rates = learning_rates(warmup_steps=4, total_steps=12)
    assert rates[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert rates[4] == pytest.approx(1.0)
    assert rates[8] == pytest.approx(0.5)
    assert all(later < earlier for earlier, later in zip(rates[4:], rates[5:]))
    assert 0 < rates[-1] < 0.1


def test_warmup_cosine_schedule_scales_each_parameter_group():
    """Groups with different base rates keep their ratio along the schedule."""
    optimizer = torch.optim.SGD(
        [
            {"params": [nn.Parameter(torch.zeros(1))], "lr": 1.0},
            {"params": [nn.Parameter(torch.zeros(1))], "lr": 0.1},
        ]
    )
    scheduler = warmup_cosine_schedule(optimizer, warmup_steps=2, total_steps=6)
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        first, second = (group["lr"] for group in optimizer.param_groups)
        assert second == pytest.approx(first / 10)


def test_warmup_cosine_schedule_without_warmup_starts_at_the_base_rate():
    """Zero warm-up steps give a plain cosine decay."""
    rates = learning_rates(warmup_steps=0, total_steps=4)
    assert rates[0] == pytest.approx(1.0)
    assert rates[2] == pytest.approx(0.5)


def test_warmup_cosine_schedule_stays_at_zero_or_above_past_the_end():
    """Stepping past the planned total never gives a negative or rising rate."""
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = warmup_cosine_schedule(optimizer, warmup_steps=1, total_steps=3)
    for _ in range(6):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_within_distance_accuracy_counts_neighbouring_classes_as_hits():
    """Predictions at most one class away from the truth count as correct."""
    confusion = [[2, 1, 1], [0, 3, 0], [2, 1, 0]]
    assert within_distance_accuracy(confusion) == pytest.approx(7 / 10)
    assert within_distance_accuracy(confusion, distance=0) == pytest.approx(5 / 10)
    assert within_distance_accuracy(confusion, distance=2) == pytest.approx(1.0)


def test_within_distance_accuracy_of_an_empty_matrix_is_zero():
    """A matrix with no samples gives zero instead of dividing by zero."""
    assert within_distance_accuracy([[0, 0], [0, 0]]) == 0.0
