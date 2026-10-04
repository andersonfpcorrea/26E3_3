from collections import Counter

from dlv.data import download_kaggle_dataset
from dlv.figures import plot_class_counts
from dlv.xray import count_xray_files, list_xray_images, load_grayscale, sample_per_class

with stopwatch.measure("download"):
    dataset_root = download_kaggle_dataset(DATASET_SLUG, DATA_DIR)
with stopwatch.measure("listagem e remoção de cópias"):
    files_found = count_xray_files(dataset_root)
    paths, labels = list_xray_images(dataset_root)
duplicates_removed = files_found - len(paths)
images_per_class = {CLASS_NAMES[label]: count for label, count in sorted(Counter(labels).items())}

project_indices = sample_per_class(labels, PROJECT_COUNTS, SEED)
test_indices = sample_per_class(
    labels, {label: TEST_PER_CLASS for label in PROJECT_COUNTS}, SEED, exclude=project_indices
)
with stopwatch.measure("leitura das imagens"):
    project_images = load_grayscale([paths[i] for i in project_indices], INPUT_SIZE)
    test_images = load_grayscale([paths[i] for i in test_indices], INPUT_SIZE)
project_labels = torch.tensor([labels[i] for i in project_indices])
test_labels = torch.tensor([labels[i] for i in test_indices])


def counts_by_name(values):
    """Counts how many times each class appears, keyed by class name."""
    return {CLASS_NAMES[label]: count for label, count in sorted(Counter(values).items())}


project_counts = counts_by_name(project_labels.tolist())
test_counts = counts_by_name(test_labels.tolist())
print("arquivos de imagem nas pastas das três classes:", files_found)
print("arquivos descartados (cópias exatas ou ilegíveis):", duplicates_removed)
print("imagens por classe depois da limpeza:", images_per_class)
print("conjunto do projeto:", project_counts)
print("conjunto de teste real:", test_counts)
print("imagens em comum entre projeto e teste:", len(set(project_indices) & set(test_indices)))
print("formato das imagens:", tuple(project_images.shape), project_images.dtype)

class_counts_figure = plot_class_counts(
    {"Base limpa": images_per_class, "Projeto": project_counts, "Teste real": test_counts},
    run.figures_dir / "class_counts.png",
)
