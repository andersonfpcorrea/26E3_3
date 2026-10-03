"""Tests for the frozen ResNet-50 feature extractor."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from dlv.cnn import (
    build_feature_extractor,
    set_feature_extraction_mode,
    trainable_parameters,
)
from dlv.engine import fit


def test_only_the_new_head_is_trainable():
    """Only the weight and bias of the new head receive gradients."""
    model = build_feature_extractor(num_classes=5, pretrained=False)
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert names == {"fc.weight", "fc.bias"}
    assert len(trainable_parameters(model)) == 2


def test_head_matches_number_of_classes():
    """The head has one output per class and the model returns that shape."""
    model = build_feature_extractor(num_classes=5, pretrained=False)
    model.eval()
    assert model.fc.out_features == 5
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5)


def test_feature_extraction_mode_keeps_backbone_in_eval():
    """The backbone stays in evaluation mode while the head trains."""
    model = build_feature_extractor(num_classes=5, pretrained=False)
    set_feature_extraction_mode(model)
    assert model.layer1.training is False
    assert model.bn1.training is False
    assert model.fc.training is True


def test_training_changes_head_and_leaves_backbone_untouched():
    """One epoch changes the head and leaves backbone weights and statistics as they were."""
    torch.manual_seed(0)
    model = build_feature_extractor(num_classes=2, pretrained=False)
    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    optimizer = torch.optim.Adam(trainable_parameters(model), lr=0.01)
    running_mean = model.bn1.running_mean.clone()
    first_filter = model.conv1.weight.clone()
    head = model.fc.weight.clone()
    fit(
        model,
        loader,
        loader,
        optimizer,
        epochs=1,
        device=torch.device("cpu"),
        set_train_mode=set_feature_extraction_mode,
    )
    assert torch.equal(model.bn1.running_mean, running_mean)
    assert torch.equal(model.conv1.weight, first_filter)
    assert not torch.equal(model.fc.weight, head)
