from dlv.figures import plot_confusion, plot_histories

summaries = {
    SCRATCH_NAME: scratch_summary,
    NO_POSITION_NAME: no_position_summary,
    PRETRAINED_NAME: pretrained_summary,
}
for name, summary in summaries.items():
    print(
        f"{name}: acurácia {summary['accuracy']:.4f}, "
        f"com tolerância de um período {summary['within_one_period']:.4f}, "
        f"F1 macro {summary['macro_f1']:.4f}, "
        f"distância média {summary['mean_distance']:.4f}, "
        f"parâmetros {summary['parameters']}, "
        f"treino {summary['train_seconds']} s"
    )
for name, summary in summaries.items():
    print(f"acurácia por período, {name}:")
    for period, value in summary["per_class_accuracy"].items():
        print(f"  {period}: {value:.4f}")

histories_figure = plot_histories(
    {name: summary["history"] for name, summary in summaries.items()},
    run.figures_dir / "histories.png",
)
scratch_confusion_figure = plot_confusion(
    scratch_summary["confusion"], PERIOD_NAMES, run.figures_dir / "confusion_scratch.png"
)
pretrained_confusion_figure = plot_confusion(
    pretrained_summary["confusion"], PERIOD_NAMES, run.figures_dir / "confusion_pretrained.png"
)
