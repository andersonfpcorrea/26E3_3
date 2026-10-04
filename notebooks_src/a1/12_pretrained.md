## Modelo B: ViT-B/16 pré-treinado com fine-tuning

- ViT-B/16 do torchvision com pesos `IMAGENET1K_V1`
- A camada final é trocada por uma nova, com 8 saídas. Todas as camadas são treinadas
- Entrada igual à do pré-treinamento: redimensionar para 256, recorte central de 224x224 e normalização com média e desvio padrão da ImageNet
- AdamW, taxa de aprendizado 0,00003 no backbone e 0,001 na camada nova, weight decay 0,05
- Lotes de 32 imagens, precisão mista, 2 épocas
