from dlv.figures import plot_gan_curves, plot_gan_diversity, plot_gan_snapshots
from dlv.gan import (
    ConditionalDiscriminator,
    ConditionalGenerator,
    diagnose_gan,
    generate,
    nearest_real_distance,
    pairwise_diversity,
    train_gan,
)

snapshot_epochs = sorted(
    {1, max(1, GAN_EPOCHS // 30), max(1, GAN_EPOCHS // 10), GAN_EPOCHS // 3, 2 * GAN_EPOCHS // 3, GAN_EPOCHS}
)
generators = {}
histories = {}


def train_version(key, settings):
    """Trains one GAN version on the training split and keeps its generator and history."""
    seed_everything(SEED)
    generator = ConditionalGenerator(num_classes=len(CLASS_NAMES), base_channels=GAN_BASE_CHANNELS)
    seed_everything(SEED)
    discriminator = ConditionalDiscriminator(
        num_classes=len(CLASS_NAMES),
        base_channels=GAN_BASE_CHANNELS,
        spectral_norm=settings["spectral_norm"],
        batch_norm=settings["batch_norm"],
    )
    with stopwatch.measure(f"treino da GAN {key}"):
        history = train_gan(
            generator,
            discriminator,
            train_images,
            train_labels,
            epochs=GAN_EPOCHS,
            batch_size=GAN_BATCH_SIZE,
            device=DEVICE,
            loss=settings["loss"],
            gradient_penalty_weight=settings["gradient_penalty_weight"],
            betas=settings["betas"],
            augment=settings["augment"],
            generator_lr=settings["generator_lr"],
            discriminator_lr=settings["discriminator_lr"],
            seed=SEED,
            snapshot_epochs=snapshot_epochs,
        )
    print(f"{key}: {stopwatch.seconds[f'treino da GAN {key}'] / GAN_EPOCHS:.2f} s por época")
    generators[key] = generator
    histories[key] = history


for key, settings in GAN_VERSIONS.items():
    train_version(key, settings)

real_diversity = {
    label: pairwise_diversity(train_images[train_labels == label]) for label in range(len(CLASS_NAMES))
}
diagnosis = {key: diagnose_gan(histories[key], real_diversity) for key in GAN_VERSIONS}
print("diversidade das imagens reais de treino:", {CLASS_NAMES[i]: round(v, 4) for i, v in real_diversity.items()})
for key, found in diagnosis.items():
    print(key, "diversidade no fim:", {CLASS_NAMES[i]: round(v, 4) for i, v in found["final_diversity"].items()})
    print(key, "razão para a diversidade real:", {CLASS_NAMES[i]: round(v, 3) for i, v in found["diversity_ratio"].items()})
    print(key, "classes com colapso:", [CLASS_NAMES[i] for i in found["collapsed_classes"]])
    print(
        key,
        f"perda do discriminador no fim: {found['discriminator_loss_final']:.3f}, "
        f"perda do gerador no fim: {found['generator_loss_final']:.3f}, "
        f"discriminador vencendo: {found['discriminator_wins']}, perda não finita: {found['non_finite_loss']}",
    )

version_names = {key: settings["name"] for key, settings in GAN_VERSIONS.items()}
curves_figure = plot_gan_curves(
    {version_names[key]: histories[key] for key in GAN_VERSIONS}, run.figures_dir / "gan_curves.png"
)
diversity_figure = plot_gan_diversity(
    {version_names[key]: histories[key]["diversity"] for key in GAN_VERSIONS},
    real_diversity,
    CLASS_NAMES,
    run.figures_dir / "gan_diversity.png",
)
snapshot_figures = {
    key: plot_gan_snapshots(
        histories[key]["snapshots"],
        run.figures_dir / f"gan_snapshots_{key}.png",
        f"{version_names[key]}: mesmas entradas em cada época. Linhas de cada grade: {', '.join(CLASS_NAMES)}",
    )
    for key in GAN_VERSIONS
}
