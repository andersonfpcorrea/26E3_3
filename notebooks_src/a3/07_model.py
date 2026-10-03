from dlv.cnn import build_feature_extractor, trainable_parameters

model = build_feature_extractor(num_classes=len(class_names)).to(DEVICE)
total_parameters = sum(parameter.numel() for parameter in model.parameters())
trainable_count = sum(parameter.numel() for parameter in trainable_parameters(model))
print("parâmetros totais:", total_parameters)
print("parâmetros treináveis:", trainable_count)
print("camada final:", model.fc)
