from dlv.figures import plot_gan_curves, plot_gan_diversity, plot_gan_snapshots
from dlv.gan import diagnose_gan, generate, nearest_real_distance, pairwise_diversity, summarise_diagnoses, train_gan_version

snapshot_epochs = sorted(
    {1, max(1, GAN_EPOCHS // 30), max(1, GAN_EPOCHS // 10), GAN_EPOCHS // 3, 2 * GAN_EPOCHS // 3, GAN_EPOCHS}
)
generators = {key: {} for key in GAN_VERSIONS}
histories = {key: {} for key in GAN_VERSIONS}

for key, settings in GAN_VERSIONS.items():
    for gan_seed in GAN_SEEDS:
        step = f"treino da GAN {key}, seed {gan_seed}"
        with stopwatch.measure(step):
            generators[key][gan_seed], histories[key][gan_seed] = train_gan_version(
                settings,
                train_images,
                train_labels,
                gan_seed,
                epochs=GAN_EPOCHS,
                batch_size=GAN_BATCH_SIZE,
                base_channels=GAN_BASE_CHANNELS,
                device=DEVICE,
                snapshot_epochs=snapshot_epochs,
                num_classes=len(CLASS_NAMES),
            )
        print(f"{key}, seed {gan_seed}: {stopwatch.seconds[step] / GAN_EPOCHS:.2f} s por época")

real_diversity = {
    label: pairwise_diversity(train_images[train_labels == label]) for label in range(len(CLASS_NAMES))
}
diagnoses = {
    key: {gan_seed: diagnose_gan(histories[key][gan_seed], real_diversity) for gan_seed in GAN_SEEDS}
    for key in GAN_VERSIONS
}
summaries = {key: summarise_diagnoses([diagnoses[key][gan_seed] for gan_seed in GAN_SEEDS]) for key in GAN_VERSIONS}

print("diversidade das imagens reais de treino:", {CLASS_NAMES[i]: round(v, 4) for i, v in real_diversity.items()})
for key in GAN_VERSIONS:
    for gan_seed, found in diagnoses[key].items():
        print(key, f"seed {gan_seed}, razão para a diversidade real:", {CLASS_NAMES[i]: round(v, 3) for i, v in found["diversity_ratio"].items()})
        print(
            key,
            f"seed {gan_seed}, perda do discriminador no fim: {found['discriminator_loss_final']:.3f}, "
            f"perda do gerador no fim: {found['generator_loss_final']:.3f}, "
            f"discriminador vencendo: {found['discriminator_wins']}, perda não finita: {found['non_finite_loss']}",
        )
    found = summaries[key]
    for label, name in enumerate(CLASS_NAMES):
        print(
            key,
            f"{name}: razão média {found['diversity_ratio_mean'][label]:.3f}, "
            f"desvio padrão {found['diversity_ratio_std'][label]:.3f}, "
            f"seeds com colapso {found['collapsed_seeds'][label]} de {found['seeds']}",
        )
    print(key, "alguma seed com perda não finita:", found["any_non_finite_loss"])

version_names = {key: settings["name"] for key, settings in GAN_VERSIONS.items()}
figure_seed = GAN_SEEDS[0]
curves_figure = plot_gan_curves(
    {f"{version_names[key]}, seed {figure_seed}": histories[key][figure_seed] for key in GAN_VERSIONS},
    run.figures_dir / "gan_curves.png",
)
diversity_figure = plot_gan_diversity(
    {
        version_names[key]: [histories[key][gan_seed]["diversity"] for gan_seed in GAN_SEEDS]
        for key in GAN_VERSIONS
    },
    real_diversity,
    CLASS_NAMES,
    run.figures_dir / "gan_diversity.png",
)
snapshot_figures = {
    key: plot_gan_snapshots(
        histories[key][figure_seed]["snapshots"],
        run.figures_dir / f"gan_snapshots_{key}.png",
        f"{version_names[key]}, seed {figure_seed}: mesmas entradas em cada época. Linhas de cada grade: {', '.join(CLASS_NAMES)}",
    )
    for key in GAN_VERSIONS
}
