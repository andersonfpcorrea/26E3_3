NO_POSITION_NAME = "ViT do zero sem posição"
no_position_model, no_position_report, no_position_summary = train_scratch(
    NO_POSITION_NAME, use_positional_embedding=False
)
print("parâmetros:", no_position_summary["parameters"])
print(f"acurácia no teste: {no_position_summary['accuracy']:.4f}")
