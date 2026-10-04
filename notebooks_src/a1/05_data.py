from dlv.paintings import (
    PERIODS,
    decode_images,
    download_wikiart,
    is_unknown_artist,
    period_of,
    read_images,
    read_metadata,
    select_balanced,
    split_by_group,
)
from dlv.figures import plot_class_counts

PERIOD_NAMES = [name for name, _, _ in PERIODS]
SPLIT_NAMES = ["Treino", "Validação", "Teste"]

with stopwatch.measure("download"):
    parquet_paths = download_wikiart(DATA_DIR)
with stopwatch.measure("leitura dos metadados"):
    metadata = read_metadata(parquet_paths)

all_labels = [None if year is None else period_of(year) for year in metadata["year"]]
rows_total = len(all_labels)
rows_with_year = sum(label is not None for label in all_labels)
rows_per_period = {
    PERIOD_NAMES[label]: count
    for label, count in sorted(Counter(label for label in all_labels if label is not None).items())
}

selected = select_balanced(all_labels, PER_CLASS, SEED)
labels = [all_labels[index] for index in selected]
artists = [metadata["artist"][index] for index in selected]
styles = [metadata["style"][index] or "sem estilo" for index in selected]
titles = [metadata["title"][index] for index in selected]
years = [metadata["year"][index] for index in selected]

splits = split_by_group(artists, labels, seed=SEED)
train_indices, validation_indices, test_indices = splits

with stopwatch.measure("leitura das imagens"):
    encoded_images = read_images(parquet_paths, [metadata["row"][index] for index in selected])
with stopwatch.measure("redução das imagens para 64x64"):
    small_images = decode_images(encoded_images, IMAGE_SIZE)
label_tensor = torch.tensor(labels)

class_counts = {
    split_name: {
        name: sum(labels[index] == label for index in part) for label, name in enumerate(PERIOD_NAMES)
    }
    for split_name, part in zip(SPLIT_NAMES, splits)
}
split_sizes = {split_name: len(part) for split_name, part in zip(SPLIT_NAMES, splits)}
known_painters = [
    {artists[index] for index in part if not is_unknown_artist(artists[index])} for part in splits
]
painters_per_split = {split_name: len(names) for split_name, names in zip(SPLIT_NAMES, known_painters)}
unknown_painter_rows = {
    split_name: sum(is_unknown_artist(artists[index]) for index in part)
    for split_name, part in zip(SPLIT_NAMES, splits)
}
shared_painters = (
    (known_painters[0] & known_painters[1])
    | (known_painters[0] & known_painters[2])
    | (known_painters[1] & known_painters[2])
)

print("linhas no dataset:", rows_total)
print("linhas com ano:", rows_with_year)
print("linhas com ano por período, antes da amostra:", rows_per_period)
print("pinturas na amostra:", len(selected))
print("divisão:", split_sizes)
for split_name in SPLIT_NAMES:
    print(f"pinturas por período, {split_name}:", class_counts[split_name])
print("pintores conhecidos por parte:", painters_per_split)
print("pinturas sem pintor conhecido por parte:", unknown_painter_rows)
print("pintores presentes em mais de uma parte:", len(shared_painters))

class_counts_figure = plot_class_counts(class_counts, run.figures_dir / "class_counts.png")
