from dlv.figures import plot_arm_metric, plot_confusion
from dlv.xray import inverse_frequency_weights, mix_synthetic, summarise_runs

ARMS = {
    "Só reais": {"synthetic": False, "weighted": False},
    "Reais e sintéticas": {"synthetic": True, "weighted": False},
    "Reais com pesos de classe": {"synthetic": False, "weighted": True},
}


def train_arm(settings, seed):
    """Trains and evaluates one classifier of an arm, and returns its history and test results."""
    seed_everything(seed)
    images, targets = train_images, train_labels.tolist()
    if settings["synthetic"]:
        images, targets = mix_synthetic(train_images, targets, synthetic_by_seed[seed], COVID_LABEL)
    weights = inverse_frequency_weights(targets, len(CLASS_NAMES)) if settings["weighted"] else None
    model = build_resnet18(len(CLASS_NAMES), pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LEARNING_RATE)
    history = fit(
        model,
        make_batches(
            images, targets, CLASSIFIER_BATCH_SIZE, seed, shuffle=True, augment=True, output_size=CLASSIFIER_INPUT_SIZE
        ),
        make_batches(
            validation_images, validation_labels, CLASSIFIER_BATCH_SIZE, seed, output_size=CLASSIFIER_INPUT_SIZE
        ),
        optimizer,
        epochs=CLASSIFIER_EPOCHS,
        device=DEVICE,
        class_weights=weights,
    )
    evaluation = evaluate(
        model,
        make_batches(test_images, test_labels, CLASSIFIER_BATCH_SIZE, seed, output_size=CLASSIFIER_INPUT_SIZE),
        DEVICE,
        len(CLASS_NAMES),
    )
    return {
        "seed": seed,
        "training_images": int(images.shape[0]),
        "training_counts": counts_by_name(list(targets)),
        "history": history,
        "accuracy": evaluation["accuracy"],
        "confusion": evaluation["confusion"],
        "report": class_report(evaluation["confusion"]),
    }


arm_runs = {}
for arm, settings in ARMS.items():
    with stopwatch.measure(f"braço: {arm}"):
        arm_runs[arm] = [train_arm(settings, seed) for seed in CLASSIFIER_SEEDS]

arm_summaries = {
    arm: {
        metric: summarise_runs([result["report"] for result in results], metric)
        for metric in ("recall", "precision", "f1")
    }
    for arm, results in arm_runs.items()
}
for arm, results in arm_runs.items():
    print(f"{arm}: imagens de treino {results[0]['training_counts']}")
    accuracies = torch.tensor([result["accuracy"] for result in results])
    print(f"  acurácia no teste: média {accuracies.mean():.4f}, desvio {accuracies.std():.4f}")
    for metric in ("recall", "precision", "f1"):
        for name, row in zip(CLASS_NAMES, arm_summaries[arm][metric]):
            print(f"  {name}, {metric}: média {row['mean']:.4f}, desvio {row['std']:.4f}")
print("teste real por classe:", test_counts)

recall_figure = plot_arm_metric(
    {arm: arm_summaries[arm]["recall"] for arm in ARMS},
    CLASS_NAMES,
    "Recall no teste real (média e desvio padrão, 3 seeds)",
    run.figures_dir / "recall_arms.png",
)
confusion_figures = {
    arm: plot_confusion(
        arm_runs[arm][0]["confusion"], CLASS_NAMES, run.figures_dir / f"confusion_arm{number}.png"
    )
    for number, arm in enumerate(ARMS, start=1)
}
