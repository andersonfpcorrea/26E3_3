from dlv.figures import plot_group_accuracy
from dlv.paintings import accuracy_by_group

test_styles = [styles[index] for index in test_indices]
by_style = {
    name: accuracy_by_group(
        report["targets"], report["predictions"], test_styles, minimum=MINIMUM_PER_STYLE
    )
    for name, report in [(SCRATCH_NAME, scratch_report), (PRETRAINED_NAME, pretrained_report)]
}
for name, entries in by_style.items():
    print(f"acurácia por movimento, {name}:")
    for entry in entries:
        print(
            f"  {entry['group']}: {entry['accuracy']:.4f} "
            f"(n={entry['count']}, distância média {entry['mean_distance']:.2f})"
        )

scratch_styles_figure = plot_group_accuracy(
    by_style[SCRATCH_NAME],
    run.figures_dir / "styles_scratch.png",
    f"Acurácia por movimento, {SCRATCH_NAME}",
)
pretrained_styles_figure = plot_group_accuracy(
    by_style[PRETRAINED_NAME],
    run.figures_dir / "styles_pretrained.png",
    f"Acurácia por movimento, {PRETRAINED_NAME}",
)
