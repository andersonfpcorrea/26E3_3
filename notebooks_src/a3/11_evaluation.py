from dlv.engine import evaluate
from dlv.figures import plot_confusion, plot_history, plot_per_class_accuracy

report = evaluate(model, test_loader, DEVICE, num_classes=len(class_names))
print(f"acurácia global: {report['accuracy']:.4f}")
for name, value in zip(class_names, report["per_class_accuracy"]):
    print(f"{name}: {value:.4f}")

plot_history(history, FIGURES_DIR / "history.png")
plot_confusion(report["confusion"], class_names, FIGURES_DIR / "confusion.png")
plot_per_class_accuracy(report["per_class_accuracy"], class_names, FIGURES_DIR / "per_class.png")
