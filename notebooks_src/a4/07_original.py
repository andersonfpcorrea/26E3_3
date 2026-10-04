from dlv.engine import evaluate, fit
from dlv.figures import plot_history
from dlv.xray import XrayBatches, build_resnet18, class_report


def make_batches(images, targets, batch_size, seed, shuffle=False, augment=False, output_size=None):
    """Builds in-memory batches of X-rays, optionally shuffled and augmented."""
    return XrayBatches(
        images,
        targets,
        batch_size,
        output_size=output_size,
        augment=augment,
        shuffle=shuffle,
        max_shift=MAX_SHIFT_PIXELS,
        max_degrees=MAX_ROTATION_DEGREES,
        seed=seed,
        device=DEVICE,
    )


original_train, original_validation = train_test_split(
    list(range(len(project_indices))), test_size=VALIDATION_SHARE, shuffle=True, random_state=SEED
)
original_validation_labels = project_labels[original_validation]

seed_everything(SEED)
original_model = build_resnet18(len(CLASS_NAMES), pretrained=False).to(DEVICE)
original_optimizer = torch.optim.SGD(original_model.parameters(), lr=ORIGINAL_LEARNING_RATE)
with stopwatch.measure("projeto original: treino"):
    original_history = fit(
        original_model,
        make_batches(
            project_images[original_train], project_labels[original_train], ORIGINAL_BATCH_SIZE, SEED, shuffle=True
        ),
        make_batches(
            project_images[original_validation], original_validation_labels, ORIGINAL_BATCH_SIZE, SEED
        ),
        original_optimizer,
        epochs=ORIGINAL_EPOCHS,
        device=DEVICE,
    )
original_validation_report = evaluate(
    original_model,
    make_batches(project_images[original_validation], original_validation_labels, ORIGINAL_BATCH_SIZE, SEED),
    DEVICE,
    len(CLASS_NAMES),
)
original_test_report = evaluate(
    original_model,
    make_batches(test_images, test_labels, ORIGINAL_BATCH_SIZE, SEED),
    DEVICE,
    len(CLASS_NAMES),
)
original_validation_classes = class_report(original_validation_report["confusion"])
original_test_classes = class_report(original_test_report["confusion"])

validation_counts_original = counts_by_name(original_validation_labels.tolist())
majority_share = max(validation_counts_original.values()) / len(original_validation)
print("acurácia de treino na última época:", round(original_history["train_accuracy"][-1], 4))
print("acurácia de validação na última época:", round(original_history["val_accuracy"][-1], 4))
print("imagens por classe na validação deles:", validation_counts_original)
print("imagens de COVID-19 na validação deles:", validation_counts_original.get("COVID-19", 0))
print("acurácia de quem sempre responde a classe mais comum:", round(majority_share, 4))
for name, on_validation, on_test in zip(CLASS_NAMES, original_validation_classes, original_test_classes):
    print(f"{name}: recall na validação {on_validation['recall']:.4f}, no teste real {on_test['recall']:.4f}")
print("acurácia no teste real:", round(original_test_report["accuracy"], 4))

original_history_figure = plot_history(original_history, run.figures_dir / "original_history.png")
