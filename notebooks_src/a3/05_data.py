from collections import Counter

from dlv.cnn import imagenet_transform
from dlv.data import IMAGE_SUFFIXES, ImageListDataset, download_kaggle_dataset, list_image_folder, stratified_split

dataset_root = download_kaggle_dataset(DATASET_SLUG, DATA_DIR)
paths, labels, class_names = list_image_folder(dataset_root)
files_found = sum(
    1
    for path in dataset_root.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")
)
duplicates_removed = files_found - len(paths)
train_indices, validation_indices, test_indices = stratified_split(labels, seed=SEED)
transform = imagenet_transform()


def make_loader(indices, shuffle):
    """Builds a loader over the images selected by the given indices."""
    dataset = ImageListDataset(
        [paths[i] for i in indices], [labels[i] for i in indices], transform
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=2)


train_loader = make_loader(train_indices, shuffle=True)
validation_loader = make_loader(validation_indices, shuffle=False)
test_loader = make_loader(test_indices, shuffle=False)

class_counts = {class_names[label]: count for label, count in sorted(Counter(labels).items())}
split_sizes = {
    "train": len(train_indices),
    "validation": len(validation_indices),
    "test": len(test_indices),
}
print("arquivos de imagem no download:", files_found)
print("cópias exatas removidas:", duplicates_removed)
print("classes:", class_names)
print("imagens por classe:", class_counts)
print("divisão:", split_sizes)
