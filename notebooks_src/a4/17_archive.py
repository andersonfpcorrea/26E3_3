from dlv.run import save_run

gan_metrics = {
    key: {
        "name": settings["name"],
        "hyperparameters": {**settings, "epochs": GAN_EPOCHS, "batch_size": GAN_BATCH_SIZE, "base_channels": GAN_BASE_CHANNELS},
        "history": {name: values for name, values in histories[key].items() if name != "snapshots"},
        "final_diversity": diagnosis[key]["final_diversity"],
        "diagnosis": diagnosis[key],
        "seconds_per_epoch": round(stopwatch.seconds[f"treino da GAN {key}"] / GAN_EPOCHS, 3),
    }
    for key, settings in GAN_VERSIONS.items()
}
metrics = {
    "class_names": CLASS_NAMES,
    "files_found": files_found,
    "duplicates_removed": duplicates_removed,
    "images_per_class": images_per_class,
    "project_counts": project_counts,
    "test_counts": test_counts,
    "train_counts": train_counts,
    "validation_counts": validation_counts,
    "original_project": {
        "train_accuracy": original_history["train_accuracy"][-1],
        "validation_accuracy": original_history["val_accuracy"][-1],
        "test_accuracy": original_test_report["accuracy"],
        "covid_in_validation": validation_counts_original.get("COVID-19", 0),
        "validation_counts": validation_counts_original,
        "majority_class_accuracy": majority_share,
        "recall_validation": dict(zip(CLASS_NAMES, [row["recall"] for row in original_validation_classes])),
        "recall_test": dict(zip(CLASS_NAMES, [row["recall"] for row in original_test_classes])),
        "confusion_validation": original_validation_report["confusion"],
        "confusion_test": original_test_report["confusion"],
        "history": original_history,
    },
    "gan": gan_metrics,
    "real_diversity": real_diversity,
    "memorisation": memorisation,
    "arms": {
        arm: {"runs": arm_runs[arm], "summary": arm_summaries[arm]} for arm in ARMS
    },
    "hyperparameters": {
        "seed": SEED,
        "input_size": INPUT_SIZE,
        "project_counts": PROJECT_COUNTS,
        "test_per_class": TEST_PER_CLASS,
        "validation_share": VALIDATION_SHARE,
        "original": {
            "model": "resnet18, random weights",
            "optimizer": "SGD, no momentum",
            "learning_rate": ORIGINAL_LEARNING_RATE,
            "epochs": ORIGINAL_EPOCHS,
            "batch_size": ORIGINAL_BATCH_SIZE,
            "split": "plain shuffle, no stratification",
        },
        "classifier": {
            "model": "resnet18 IMAGENET1K_V1, all layers trained",
            "optimizer": "Adam",
            "learning_rate": CLASSIFIER_LEARNING_RATE,
            "epochs": CLASSIFIER_EPOCHS,
            "batch_size": CLASSIFIER_BATCH_SIZE,
            "input_size": CLASSIFIER_INPUT_SIZE,
            "seeds": list(CLASSIFIER_SEEDS),
            "augmentation": f"shift up to {MAX_SHIFT_PIXELS} px, rotation up to {MAX_ROTATION_DEGREES} degrees, no flip",
            "synthetic_covid": synthetic_count,
        },
    },
    "step_seconds": stopwatch.seconds,
}
archive = save_run(run, metrics)
saved = json.loads((run.output_dir / "metrics.json").read_text(encoding="utf-8"))
print("arquivo de saída:", archive)
print("tempo total (s):", saved["runtime_seconds"])
print("pico de memória de GPU (MB):", saved["peak_gpu_memory_mb"])
print("pico de memória RAM (MB):", saved["peak_ram_mb"])
