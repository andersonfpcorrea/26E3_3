from dlv.figures import plot_attention_maps
from dlv.paintings import select_examples
from dlv.vit import class_token_attention

example_positions = select_examples(
    scratch_report["targets"], scratch_report["predictions"], per_outcome=EXAMPLES_PER_OUTCOME
)
example_indices = [test_indices[position] for position in example_positions]
example_batches = TensorBatches(
    small_images[example_indices],
    label_tensor[example_indices],
    len(example_indices),
    channel_mean,
    channel_std,
    device=DEVICE,
)
example_inputs, _ = next(iter(example_batches))
scratch_model.eval()
with torch.no_grad():
    _, example_attention = scratch_model(example_inputs, return_attention=True)
grid_size = scratch_model.patch_embedding.grid_size
last_block_maps = class_token_attention(example_attention[-1], grid_size).cpu()
first_block_maps = class_token_attention(example_attention[0], grid_size).mean(dim=1, keepdim=True).cpu()

attention_examples = [
    {
        "title": titles[index],
        "painter": artists[index],
        "year": years[index],
        "true_period": PERIOD_NAMES[labels[index]],
        "predicted_period": PERIOD_NAMES[scratch_report["predictions"][position]],
    }
    for position, index in zip(example_positions, example_indices)
]
for number, example in enumerate(attention_examples, start=1):
    print(
        f"pintura {number}: {example['title']}, {example['painter']}, {example['year']}, "
        f"real {example['true_period']}, previsto {example['predicted_period']}"
    )
example_titles = [
    f"real: {example['true_period']}\nprevisto: {example['predicted_period']}"
    for example in attention_examples
]
last_block_figure = plot_attention_maps(
    small_images[example_indices],
    last_block_maps,
    example_titles,
    run.figures_dir / "attention_last_block.png",
    title="Atenção do token de classe no último bloco, por cabeça\n(cada painel tem sua própria escala de cores)",
)
first_block_figure = plot_attention_maps(
    small_images[example_indices],
    first_block_maps,
    example_titles,
    run.figures_dir / "attention_first_block.png",
    panel_titles=["Média das cabeças"],
    title="Atenção do token de classe\nno primeiro bloco\n(escala de cores própria\nem cada painel)",
)
