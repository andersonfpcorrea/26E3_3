"""Tests for the conditional GAN, its losses, diversity measures, training loop and sampling."""

from json import dumps, loads
from math import exp, isfinite, log

import pytest
import torch
from torch import nn

from dlv.gan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    choose_by_mean_ratio,
    diagnose_gan,
    differentiable_augment,
    discriminator_loss,
    generate,
    generator_loss,
    gradient_penalty,
    nearest_real_distance,
    pairwise_diversity,
    summarise_diagnoses,
    train_gan,
    train_gan_version,
)

DEVICE = torch.device("cpu")


def small_generator(**overrides):
    """Builds a small generator for fast tests."""
    settings = dict(latent_dim=8, num_classes=3, base_channels=8)
    settings.update(overrides)
    return ConditionalGenerator(**settings)


def small_discriminator(**overrides):
    """Builds a small discriminator for fast tests."""
    settings = dict(num_classes=3, base_channels=8)
    settings.update(overrides)
    return ConditionalDiscriminator(**settings)


def random_images(count, seed=0):
    """Draws a reproducible uint8 batch of (count, 1, 64, 64) images."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (count, 1, 64, 64), generator=generator, dtype=torch.uint8)


def parameters_of(model):
    """Copies every parameter of a model into one flat tensor."""
    return torch.cat([parameter.detach().flatten().clone() for parameter in model.parameters()])


def test_generator_output_shape_and_range():
    """The generator returns (n, 1, 64, 64) values in -1 to 1."""
    generator = small_generator()
    output = generator(torch.randn(4, 8), torch.tensor([0, 1, 2, 0]))
    assert output.shape == (4, 1, 64, 64)
    assert output.min() >= -1.0 and output.max() <= 1.0


def test_generator_changes_with_label_only():
    """The same noise with another label gives another image."""
    generator = small_generator().eval()
    noise = torch.randn(1, 8)
    first = generator(noise, torch.tensor([0]))
    second = generator(noise, torch.tensor([2]))
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("bad", [3, -1])
def test_generator_rejects_out_of_range_label(bad):
    """A label outside the class range raises a clear ValueError."""
    generator = small_generator()
    with pytest.raises(ValueError, match="labels must be in"):
        generator(torch.randn(2, 8), torch.tensor([0, bad]))


def test_discriminator_output_shape():
    """The discriminator returns one logit per image."""
    discriminator = small_discriminator()
    logits = discriminator(torch.randn(5, 1, 64, 64), torch.tensor([0, 1, 2, 0, 1]))
    assert logits.shape == (5,)


@pytest.mark.parametrize("spectral", [False, True])
def test_discriminator_logits_depend_on_label(spectral):
    """The same images get different logits under different labels."""
    torch.manual_seed(0)
    discriminator = small_discriminator(spectral_norm=spectral).eval()
    images = torch.randn(4, 1, 64, 64)
    with torch.no_grad():
        first = discriminator(images, torch.zeros(4, dtype=torch.long))
        second = discriminator(images, torch.full((4,), 2))
    assert not torch.allclose(first, second)


def test_discriminator_rejects_out_of_range_label():
    """The discriminator also refuses unknown labels."""
    with pytest.raises(ValueError, match="labels must be in"):
        small_discriminator()(torch.randn(1, 1, 64, 64), torch.tensor([7]))


def test_spectral_discriminator_has_spectral_norm_and_no_batch_norm():
    """Every convolution is spectrally normalised and no batch norm exists."""
    discriminator = small_discriminator(spectral_norm=True)
    layers = [m for m in discriminator.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    assert layers
    assert all(hasattr(layer, "parametrizations") for layer in layers)
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in discriminator.modules())


def test_plain_discriminator_has_batch_norm_and_no_spectral_norm():
    """The plain discriminator uses batch norm and plain weights."""
    discriminator = small_discriminator()
    layers = [m for m in discriminator.modules() if isinstance(m, nn.Conv2d)]
    assert not any(hasattr(layer, "parametrizations") for layer in layers)
    assert any(isinstance(m, nn.BatchNorm2d) for m in discriminator.modules())


def sigmoid(value):
    """Logistic function on a Python float."""
    return 1 / (1 + exp(-value))


def test_discriminator_loss_bce_matches_hand_computation():
    """BCE discriminator loss equals -log sigmoid(real) - log(1 - sigmoid(fake)) averaged."""
    real = torch.tensor([0.0, 2.0])
    fake = torch.tensor([0.0, -1.0])
    expected = (
        -(log(sigmoid(0.0)) + log(sigmoid(2.0))) / 2
        - (log(1 - sigmoid(0.0)) + log(1 - sigmoid(-1.0))) / 2
    )
    assert discriminator_loss(real, fake, "bce").item() == pytest.approx(expected, rel=1e-5)


def test_discriminator_loss_hinge_matches_hand_computation():
    """Hinge discriminator loss is mean relu(1 - real) plus mean relu(1 + fake)."""
    real = torch.tensor([0.5, 2.0])
    fake = torch.tensor([-2.0, 0.5])
    assert discriminator_loss(real, fake, "hinge").item() == pytest.approx(0.25 + 0.75)


def test_generator_loss_bce_matches_hand_computation():
    """Non-saturating generator loss is the mean of -log sigmoid(fake)."""
    fake = torch.tensor([0.0, 1.0])
    expected = -(log(0.5) + log(1 / (1 + exp(-1.0)))) / 2
    assert generator_loss(fake, "bce").item() == pytest.approx(expected, rel=1e-5)


def test_generator_loss_hinge_matches_hand_computation():
    """Hinge generator loss is minus the mean fake logit."""
    assert generator_loss(torch.tensor([1.0, 3.0]), "hinge").item() == pytest.approx(-2.0)


def test_unknown_loss_kind_raises():
    """Both losses refuse a kind other than bce and hinge."""
    logits = torch.zeros(2)
    with pytest.raises(ValueError):
        discriminator_loss(logits, logits, "wasserstein")
    with pytest.raises(ValueError):
        generator_loss(logits, "wasserstein")


def test_diversity_zero_for_identical_and_positive_otherwise():
    """Diversity is zero for copies of one image and positive for different images."""
    image = random_images(1)
    assert pairwise_diversity(image.repeat(4, 1, 1, 1)) == 0.0
    assert pairwise_diversity(random_images(4)) > 0.0


def test_diversity_hand_computed_for_two_images():
    """An all-black and an all-white image are at distance 1."""
    images = torch.stack([torch.zeros(1, 64, 64), torch.full((1, 64, 64), 255)]).to(torch.uint8)
    assert pairwise_diversity(images) == pytest.approx(1.0)


def test_diversity_of_single_image_is_zero():
    """A batch of one image has no pairs and diversity zero."""
    assert pairwise_diversity(random_images(1)) == 0.0


def test_nearest_real_distance_zero_for_a_real_image():
    """A synthetic image equal to a real one has distance zero, others do not."""
    real = random_images(5)
    synthetic = torch.cat([real[2:3], random_images(1, seed=9)])
    distances = nearest_real_distance(synthetic, real)
    assert distances.shape == (2,)
    assert distances[0].item() == 0.0
    assert distances[1].item() > 0.0


def test_nearest_real_distance_picks_closest():
    """The distance is to the closest real image, not the mean."""
    real = torch.stack([torch.zeros(1, 64, 64), torch.full((1, 64, 64), 255)]).to(torch.uint8)
    synthetic = torch.zeros(1, 1, 64, 64, dtype=torch.uint8)
    assert nearest_real_distance(synthetic, real).item() == 0.0


def run_training(seed=1, images=None, **overrides):
    """Trains small networks for one epoch and returns both networks and the history."""
    torch.manual_seed(100)
    generator, discriminator = small_generator(), small_discriminator()
    images = random_images(16) if images is None else images
    labels = torch.arange(images.shape[0]) % 3
    settings = dict(epochs=1, batch_size=8, device=DEVICE, seed=seed)
    settings.update(overrides)
    history = train_gan(generator, discriminator, images, labels, **settings)
    return generator, discriminator, history


@pytest.mark.parametrize("loss", ["bce", "hinge"])
def test_train_gan_updates_both_networks_and_records_history(loss):
    """One epoch changes both networks and fills every history key."""
    torch.manual_seed(100)
    generator, discriminator = small_generator(), small_discriminator()
    before = parameters_of(generator), parameters_of(discriminator)
    labels = torch.arange(16) % 3
    history = train_gan(
        generator, discriminator, random_images(16), labels, epochs=2, batch_size=8,
        device=DEVICE, loss=loss,
    )
    assert not torch.equal(before[0], parameters_of(generator))
    assert not torch.equal(before[1], parameters_of(discriminator))
    for key in ("discriminator_loss", "generator_loss", "real_score", "fake_score"):
        assert len(history[key]) == 2
        assert all(isfinite(value) for value in history[key])
    assert set(history["diversity"]) == {0, 1, 2}
    assert all(len(values) == 2 for values in history["diversity"].values())
    assert "snapshots" in history
    assert all(0.0 <= value <= 1.0 for value in history["real_score"] + history["fake_score"])


def test_train_gan_is_reproducible_for_the_same_seed():
    """The same seed gives identical parameters and history."""
    first_generator, _, first_history = run_training(seed=5)
    second_generator, _, second_history = run_training(seed=5)
    assert torch.equal(parameters_of(first_generator), parameters_of(second_generator))
    assert first_history["generator_loss"] == second_history["generator_loss"]
    assert first_history["diversity"] == second_history["diversity"]


def test_train_gan_leaves_global_random_state_untouched():
    """Training draws only from local generators, so the global state is unchanged."""
    torch.manual_seed(100)
    generator, discriminator = small_generator(), small_discriminator()
    before = torch.get_rng_state()
    train_gan(
        generator, discriminator, random_images(16), torch.arange(16) % 3,
        epochs=1, batch_size=8, device=DEVICE, seed=11, snapshot_epochs=(1,),
    )
    assert torch.equal(before, torch.get_rng_state())


def test_train_gan_restores_discriminator_gradients_and_input_checks():
    """After training the discriminator is trainable again and direct label checks are back."""
    generator, discriminator, _ = run_training()
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    with pytest.raises(ValueError, match="labels must be in"):
        generator(torch.randn(1, 8), torch.tensor([9]))
    with pytest.raises(ValueError, match="labels must be in"):
        discriminator(torch.randn(1, 1, 64, 64), torch.tensor([9]))


def test_train_gan_restores_flags_after_an_interrupted_generator_step(monkeypatch):
    """An exception in the generator step leaves gradients and label checks as they were."""

    def interrupt(fake_logits, kind):
        """Stands in for generator_loss and aborts the step."""
        raise KeyboardInterrupt

    monkeypatch.setattr("dlv.gan.generator_loss", interrupt)
    generator, discriminator = small_generator(), small_discriminator()
    with pytest.raises(KeyboardInterrupt):
        train_gan(
            generator, discriminator, random_images(8), torch.zeros(8, dtype=torch.long),
            epochs=1, batch_size=8, device=DEVICE,
        )
    assert all(parameter.requires_grad for parameter in discriminator.parameters())
    assert generator.validate_labels and discriminator.validate_labels


def test_train_gan_keeps_a_frozen_discriminator_parameter_frozen():
    """A parameter frozen by the caller before training is still frozen afterwards."""
    discriminator = small_discriminator()
    frozen = next(discriminator.parameters())
    frozen.requires_grad_(False)
    snapshot = frozen.detach().clone()
    train_gan(
        small_generator(), discriminator, random_images(8), torch.zeros(8, dtype=torch.long),
        epochs=1, batch_size=8, device=DEVICE,
    )
    assert not frozen.requires_grad
    assert torch.equal(snapshot, frozen)
    assert any(p.requires_grad for p in discriminator.parameters())


def test_train_gan_differs_for_another_seed():
    """A different seed gives different parameters."""
    first_generator, _, _ = run_training(seed=5)
    second_generator, _, _ = run_training(seed=6)
    assert not torch.equal(parameters_of(first_generator), parameters_of(second_generator))


def test_each_optimizer_steps_only_its_own_network():
    """A zero learning rate freezes exactly the network that owns it."""
    torch.manual_seed(100)
    generator, discriminator = small_generator(), small_discriminator()
    generator_before, discriminator_before = parameters_of(generator), parameters_of(discriminator)
    train_gan(
        generator, discriminator, random_images(8), torch.zeros(8, dtype=torch.long),
        epochs=1, batch_size=8, device=DEVICE, generator_lr=0.0,
    )
    assert torch.equal(generator_before, parameters_of(generator))
    assert not torch.equal(discriminator_before, parameters_of(discriminator))
    torch.manual_seed(100)
    generator, discriminator = small_generator(), small_discriminator()
    generator_before, discriminator_before = parameters_of(generator), parameters_of(discriminator)
    train_gan(
        generator, discriminator, random_images(8), torch.zeros(8, dtype=torch.long),
        epochs=1, batch_size=8, device=DEVICE, discriminator_lr=0.0,
    )
    assert not torch.equal(generator_before, parameters_of(generator))
    assert torch.equal(discriminator_before, parameters_of(discriminator))


def test_train_gan_handles_small_last_batch():
    """A last batch of one image does not break training or the averages."""
    _, _, history = run_training(images=random_images(9), batch_size=8)
    assert all(isfinite(value) for value in history["discriminator_loss"])


def test_train_gan_snapshots_and_mode_restoration():
    """Snapshots exist for the requested epochs and the generator ends in training mode."""
    noise = torch.randn(6, 8)
    snapshot_labels = torch.tensor([0, 1, 2, 0, 1, 2])
    generator, discriminator, history = run_training(
        epochs=3, snapshot_epochs=(1, 3), snapshot_noise=noise, snapshot_labels=snapshot_labels
    )
    assert sorted(history["snapshots"]) == [1, 3]
    snapshot = history["snapshots"][1]
    assert snapshot.dtype == torch.uint8 and snapshot.dim() == 3 and snapshot.shape[0] == 1
    assert generator.training and discriminator.training


def test_train_gan_default_snapshot_inputs_and_pairing_rule():
    """Snapshots work without explicit inputs, and noise without labels is refused."""
    _, _, history = run_training(snapshot_epochs=(1,))
    assert list(history["snapshots"]) == [1]
    with pytest.raises(ValueError):
        run_training(snapshot_epochs=(1,), snapshot_noise=torch.randn(2, 8))


def test_train_gan_rejects_unknown_loss_and_labels():
    """Unknown loss kinds and out-of-range training labels fail before training."""
    with pytest.raises(ValueError):
        run_training(loss="wasserstein")
    with pytest.raises(ValueError, match="labels must be in"):
        train_gan(
            small_generator(), small_discriminator(), random_images(4),
            torch.tensor([0, 1, 2, 3]), epochs=1, batch_size=4, device=DEVICE,
        )


def test_train_gan_history_is_json_serialisable_without_snapshots():
    """Dropping the snapshots key leaves a history json can encode."""
    _, _, history = run_training(snapshot_epochs=(1,))
    history.pop("snapshots")
    assert loads(dumps(history))["real_score"] == history["real_score"]


def test_generate_shape_dtype_and_determinism():
    """generate returns uint8 images, the same for the same seed, and keeps the training mode."""
    generator = small_generator()
    for module in generator.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 1.0
    generator(torch.randn(16, 8), torch.arange(16) % 3)
    labels = torch.tensor([0, 1, 2, 1])
    first = generate(generator, labels, DEVICE, seed=3)
    second = generate(generator, labels, DEVICE, seed=3)
    other = generate(generator, labels, DEVICE, seed=4)
    assert first.shape == (4, 1, 64, 64) and first.dtype == torch.uint8
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    assert generator.training


def test_generate_rejects_out_of_range_label():
    """Asking for a class the generator does not know raises a clear error."""
    with pytest.raises(ValueError, match="labels must be in"):
        generate(small_generator(), torch.tensor([3]), DEVICE, seed=0)


def fake_history(epochs, discriminator=1.0, generator=1.0, diversity=(0.2, 0.2, 0.2)):
    """Builds a constant training history with the given final loss and diversity values."""
    return {
        "discriminator_loss": [discriminator] * epochs,
        "generator_loss": [generator] * epochs,
        "diversity": {index: [value] * epochs for index, value in enumerate(diversity)},
    }


def test_diagnose_gan_flags_a_collapsed_class_by_its_ratio_to_real_diversity():
    """A class whose generated diversity is under half of the real one counts as collapsed."""
    history = fake_history(20, diversity=(0.2, 0.01, 0.15))
    found = diagnose_gan(history, {0: 0.2, 1: 0.2, 2: 0.2})
    assert found["collapsed_classes"] == [1]
    assert found["diversity_ratio"][2] == pytest.approx(0.75)
    assert found["final_diversity"][0] == pytest.approx(0.2)


def test_diagnose_gan_reads_only_the_last_tenth_of_the_epochs():
    """Early values do not change the result, and epoch 1 is never inside the window."""
    history = fake_history(20, discriminator=1.0)
    history["discriminator_loss"][0] = 50.0
    history["diversity"][0][0] = 0.0
    history["diversity"][0][5] = 0.0
    found = diagnose_gan(history, {0: 0.2, 1: 0.2, 2: 0.2})
    assert found["window_epochs"] == 2
    assert found["discriminator_loss_final"] == pytest.approx(1.0)
    assert found["collapsed_classes"] == []


def test_diagnose_gan_flags_a_winning_discriminator_and_non_finite_losses():
    """A near-zero discriminator loss and a NaN anywhere are reported."""
    real = {0: 0.2, 1: 0.2, 2: 0.2}
    assert diagnose_gan(fake_history(20, discriminator=0.05), real)["discriminator_wins"] is True
    assert diagnose_gan(fake_history(20, discriminator=1.0), real)["discriminator_wins"] is False
    history = fake_history(20)
    history["generator_loss"][3] = float("nan")
    assert diagnose_gan(history, real)["non_finite_loss"] is True
    assert diagnose_gan(fake_history(20), real)["non_finite_loss"] is False


def test_diagnose_gan_works_on_a_single_epoch_history():
    """A history with one epoch still gives a result from that epoch."""
    found = diagnose_gan(fake_history(1), {0: 0.2, 1: 0.2, 2: 0.2})
    assert found["window_epochs"] == 1
    assert found["discriminator_loss_final"] == pytest.approx(1.0)


def test_wgan_losses_match_hand_computation():
    """The critic loss is the mean fake logit minus the mean real logit and the generator loss is minus the mean fake logit."""
    real, fake = torch.tensor([1.0, 3.0]), torch.tensor([0.0, 2.0])
    assert discriminator_loss(real, fake, "wgan").item() == pytest.approx(1.0 - 2.0)
    assert generator_loss(fake, "wgan").item() == pytest.approx(-1.0)


def test_gradient_penalty_is_zero_for_a_critic_with_unit_gradient_norm():
    """A critic whose input gradient has norm one is not penalised."""

    class UnitSlope(nn.Module):
        """Critic whose logit is the first pixel, so its input gradient has norm one."""

        def forward(self, images, labels):
            """Returns the first pixel of every image."""
            return images[:, 0, 0, 0]

    real, fake = torch.rand(4, 1, 64, 64), torch.rand(4, 1, 64, 64)
    penalty = gradient_penalty(UnitSlope(), real, fake, torch.zeros(4, dtype=torch.long), torch.Generator().manual_seed(0))
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_penalty_grows_with_the_slope_of_the_critic():
    """A critic with slope three has a penalty of (3 - 1) squared."""

    class Slope(nn.Module):
        """Critic whose logit is three times the first pixel."""

        def forward(self, images, labels):
            """Returns three times the first pixel of every image."""
            return 3 * images[:, 0, 0, 0]

    real, fake = torch.rand(4, 1, 64, 64), torch.rand(4, 1, 64, 64)
    penalty = gradient_penalty(Slope(), real, fake, torch.zeros(4, dtype=torch.long), torch.Generator().manual_seed(0))
    assert penalty.item() == pytest.approx(4.0, rel=1e-5)


def test_discriminator_without_batch_norm_has_none():
    """batch_norm=False leaves the plain discriminator without any normalisation layer."""
    discriminator = small_discriminator(batch_norm=False)
    assert not any(isinstance(module, nn.BatchNorm2d) for module in discriminator.modules())


def test_train_gan_with_wgan_penalty_updates_both_networks():
    """The Wasserstein loss with gradient penalty trains and records finite histories."""
    torch.manual_seed(100)
    generator, discriminator = small_generator(), small_discriminator(batch_norm=False)
    before = parameters_of(generator), parameters_of(discriminator)
    history = train_gan(
        generator, discriminator, random_images(16), torch.arange(16) % 3, epochs=1, batch_size=8,
        device=DEVICE, loss="wgan", gradient_penalty_weight=10.0,
    )
    assert not torch.equal(before[0], parameters_of(generator))
    assert not torch.equal(before[1], parameters_of(discriminator))
    assert all(isfinite(value) for value in history["discriminator_loss"] + history["generator_loss"])


def test_train_gan_rejects_a_negative_penalty_weight():
    """A negative gradient penalty weight is refused."""
    with pytest.raises(ValueError):
        run_training(loss="wgan", gradient_penalty_weight=-1.0)


def test_differentiable_augment_keeps_shape_and_changes_the_images():
    """Translation and brightness change the batch and keep its shape."""
    images = torch.rand(4, 1, 64, 64) * 2 - 1
    augmented = differentiable_augment(images, torch.Generator().manual_seed(0))
    assert augmented.shape == images.shape
    assert not torch.allclose(augmented, images)


def test_differentiable_augment_without_ranges_is_the_identity():
    """Zero shift and zero brightness leave the images as they were."""
    images = torch.rand(4, 1, 64, 64) * 2 - 1
    same = differentiable_augment(images, torch.Generator().manual_seed(0), max_shift=0.0, brightness=0.0)
    assert torch.allclose(same, images, atol=1e-5)


def test_differentiable_augment_passes_gradients_and_is_reproducible():
    """Gradients reach the input and the same generator seed gives the same result."""
    images = (torch.rand(4, 1, 64, 64) * 2 - 1).requires_grad_(True)
    first = differentiable_augment(images, torch.Generator().manual_seed(3))
    second = differentiable_augment(images, torch.Generator().manual_seed(3))
    assert torch.equal(first, second)
    first.sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_train_gan_with_augmentation_trains_and_stays_reproducible():
    """Augmented training updates both networks and repeats exactly for one seed."""
    first_generator, first_discriminator, history = run_training(augment=True, seed=4)
    second_generator, _, _ = run_training(augment=True, seed=4)
    assert torch.equal(parameters_of(first_generator), parameters_of(second_generator))
    assert all(isfinite(value) for value in history["discriminator_loss"])
    plain_generator, _, _ = run_training(seed=4)
    assert not torch.equal(parameters_of(first_generator), parameters_of(plain_generator))


def diagnosis_with(ratios, collapsed=(), non_finite=False, wins=False):
    """Builds a diagnosis with the given per-class diversity ratios and flags."""
    return {
        "diversity_ratio": dict(enumerate(ratios)),
        "collapsed_classes": list(collapsed),
        "non_finite_loss": non_finite,
        "discriminator_wins": wins,
    }


def test_summarise_diagnoses_gives_mean_and_sample_standard_deviation_per_class():
    """Each class gets the mean and the n-1 standard deviation of its ratio over the seeds."""
    found = summarise_diagnoses(
        [diagnosis_with((0.9, 0.2)), diagnosis_with((0.7, 0.4)), diagnosis_with((0.8, 0.3))]
    )
    assert found["seeds"] == 3
    assert found["diversity_ratio_mean"][0] == pytest.approx(0.8)
    assert found["diversity_ratio_std"][0] == pytest.approx(0.1)
    assert found["diversity_ratio_mean"][1] == pytest.approx(0.3)
    assert found["diversity_ratio"][1] == pytest.approx([0.2, 0.4, 0.3])


def test_summarise_diagnoses_counts_collapsed_seeds_and_non_finite_losses_per_class():
    """The collapse count is per class, and one non-finite seed raises the overall flag."""
    found = summarise_diagnoses(
        [
            diagnosis_with((0.9, 0.2), collapsed=[1]),
            diagnosis_with((0.4, 0.3), collapsed=[0, 1], non_finite=True),
            diagnosis_with((0.8, 0.9), wins=True),
        ]
    )
    assert found["collapsed_seeds"] == {0: 1, 1: 2}
    assert found["non_finite_seeds"] == 1
    assert found["any_non_finite_loss"] is True
    assert found["discriminator_wins_seeds"] == 1


def test_summarise_diagnoses_handles_one_seed_and_no_seeds():
    """A single seed has a zero spread and an empty list gives an empty summary."""
    one = summarise_diagnoses([diagnosis_with((0.6,))])
    assert one["diversity_ratio_std"][0] == 0.0
    assert one["any_non_finite_loss"] is False
    none = summarise_diagnoses([])
    assert none["seeds"] == 0
    assert none["diversity_ratio_mean"] == {}
    assert none["any_non_finite_loss"] is False


def test_summarise_diagnoses_survives_a_json_round_trip():
    """The summary holds only plain Python values."""
    found = summarise_diagnoses([diagnosis_with((0.9, 0.2)), diagnosis_with((0.7, 0.4))])
    assert loads(dumps(found))["seeds"] == 2


def test_choose_by_mean_ratio_picks_the_higher_mean_of_the_class_and_reports_the_means():
    """The version with the larger mean ratio for the class wins and both means are returned."""
    summaries = {
        "a": summarise_diagnoses([diagnosis_with((0.9, 0.5)), diagnosis_with((0.9, 0.7))]),
        "b": summarise_diagnoses([diagnosis_with((0.1, 0.8)), diagnosis_with((0.1, 0.9))]),
    }
    found = choose_by_mean_ratio(summaries, 1)
    assert found["chosen"] == "b"
    assert found["label"] == 1
    assert found["means"] == pytest.approx({"a": 0.6, "b": 0.85})
    assert choose_by_mean_ratio(summaries, 0)["chosen"] == "a"


def test_choose_by_mean_ratio_keeps_the_first_version_on_a_tie():
    """Equal means resolve to the version listed first."""
    summary = summarise_diagnoses([diagnosis_with((0.5,))])
    assert choose_by_mean_ratio({"first": summary, "second": summary}, 0)["chosen"] == "first"


VERSION_SETTINGS = {
    "spectral_norm": False,
    "loss": "bce",
    "batch_norm": True,
    "gradient_penalty_weight": 0.0,
    "augment": False,
    "betas": (0.5, 0.999),
    "generator_lr": 2e-4,
    "discriminator_lr": 2e-4,
}


def run_version(seed, **overrides):
    """Trains the tiny version for one epoch on 16 random images and returns the generator and history."""
    images = random_images(16)
    labels = torch.arange(16) % 3
    settings = dict(
        epochs=1, batch_size=8, base_channels=8, device=DEVICE, snapshot_epochs=(1,), latent_dim=8
    )
    settings.update(overrides)
    return train_gan_version(VERSION_SETTINGS, images, labels, seed, **settings)


def test_train_gan_version_is_reproducible_for_a_seed_and_differs_between_seeds():
    """The initialisation and the training both follow the seed, so equal seeds give equal generators."""
    first, history_first = run_version(7)
    again, history_again = run_version(7)
    other, _ = run_version(8)
    assert history_first["generator_loss"] == history_again["generator_loss"]
    assert torch.equal(parameters_of(first), parameters_of(again))
    assert not torch.equal(parameters_of(first), parameters_of(other))


def test_train_gan_version_returns_a_generator_for_every_class_and_a_history_with_snapshots():
    """The generator can sample all classes and the history carries the requested snapshot."""
    generator, history = run_version(7)
    assert generate(generator, torch.arange(3), DEVICE, 0).shape == (3, 1, 64, 64)
    assert set(history["snapshots"]) == {1}


def test_train_gan_version_uses_the_settings_of_the_version():
    """The Wasserstein settings build a critic without batch normalisation and train with them."""
    settings = {**VERSION_SETTINGS, "loss": "wgan", "batch_norm": False, "gradient_penalty_weight": 10.0}
    images = random_images(16)
    generator, history = train_gan_version(
        settings, images, torch.arange(16) % 3, 7,
        epochs=1, batch_size=8, base_channels=8, device=DEVICE, latent_dim=8,
    )
    assert history["discriminator_loss"][0] != 0.0
    assert isfinite(history["generator_loss"][0])
