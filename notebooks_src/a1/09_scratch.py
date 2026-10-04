from dlv.data import TensorBatches, channel_statistics
from dlv.engine import (
    evaluate,
    fit,
    macro_f1,
    mean_class_distance,
    warmup_cosine_schedule,
    within_distance_accuracy,
)
from dlv.vit import VisionTransformer

channel_mean, channel_std = channel_statistics(small_images[train_indices])


def make_batches(indices, training):
    """Builds in-memory batches of the 64x64 images selected by the given indices."""
    return TensorBatches(
        small_images[indices],
        label_tensor[indices],
        SCRATCH_BATCH_SIZE,
        channel_mean,
        channel_std,
        augment=training,
        shuffle=training,
        seed=SEED,
        device=DEVICE,
    )


def summarize(model, history, report, train_seconds):
    """Gathers the parameters, history and test metrics of one model."""
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": train_seconds,
        "history": history,
        "accuracy": report["accuracy"],
        "within_one_period": within_distance_accuracy(report["confusion"], distance=1),
        "macro_f1": macro_f1(report["confusion"]),
        "mean_distance": mean_class_distance(report["confusion"]),
        "per_class_accuracy": dict(zip(PERIOD_NAMES, report["per_class_accuracy"])),
        "confusion": report["confusion"],
    }


def train_scratch(name, use_positional_embedding):
    """Trains a Vision Transformer from scratch and evaluates it on the test split."""
    seed_everything(SEED)
    model = VisionTransformer(
        image_size=IMAGE_SIZE,
        num_classes=len(PERIOD_NAMES),
        use_positional_embedding=use_positional_embedding,
    ).to(DEVICE)
    train_batches = make_batches(train_indices, training=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=SCRATCH_LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = warmup_cosine_schedule(
        optimizer,
        warmup_steps=SCRATCH_WARMUP_EPOCHS * len(train_batches),
        total_steps=SCRATCH_EPOCHS * len(train_batches),
    )
    with stopwatch.measure(f"treino, {name}"):
        history = fit(
            model,
            train_batches,
            make_batches(validation_indices, training=False),
            optimizer,
            epochs=SCRATCH_EPOCHS,
            device=DEVICE,
            scheduler=scheduler,
        )
    with stopwatch.measure(f"avaliação, {name}"):
        report = evaluate(
            model, make_batches(test_indices, training=False), DEVICE, num_classes=len(PERIOD_NAMES)
        )
    return model, report, summarize(model, history, report, stopwatch.seconds[f"treino, {name}"])


SCRATCH_NAME = "ViT do zero"
scratch_model, scratch_report, scratch_summary = train_scratch(SCRATCH_NAME, use_positional_embedding=True)
print("parâmetros:", scratch_summary["parameters"])
print(f"acurácia no teste: {scratch_summary['accuracy']:.4f}")
