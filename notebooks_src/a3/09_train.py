from dlv.cnn import set_feature_extraction_mode, trainable_parameters
from dlv.engine import fit

optimizer = torch.optim.Adam(trainable_parameters(model), lr=LEARNING_RATE)
history = fit(
    model,
    train_loader,
    validation_loader,
    optimizer,
    epochs=EPOCHS,
    device=DEVICE,
    set_train_mode=set_feature_extraction_mode,
)
