train_positions, validation_positions = train_test_split(
    list(range(len(project_indices))),
    test_size=VALIDATION_SHARE,
    stratify=project_labels.tolist(),
    random_state=SEED,
)
train_images = project_images[train_positions]
train_labels = project_labels[train_positions]
validation_images = project_images[validation_positions]
validation_labels = project_labels[validation_positions]

train_counts = counts_by_name(train_labels.tolist())
validation_counts = counts_by_name(validation_labels.tolist())
print("treino (só imagens reais):", train_counts)
print("validação (só imagens reais):", validation_counts)
print("teste (só imagens reais):", test_counts)
print("imagens em comum entre treino e validação:", len(set(train_positions) & set(validation_positions)))
print("imagens em comum entre treino e teste:", len(set(project_indices[i] for i in train_positions) & set(test_indices)))
print("imagens que a GAN vai ver:", train_images.shape[0], "(todas do treino)")
