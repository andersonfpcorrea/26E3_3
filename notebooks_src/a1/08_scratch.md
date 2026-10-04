## Modelo A: Vision Transformer treinado do zero

- Imagem de 64x64 cortada em patches de 8x8, o que dá 64 patches e um token de classe
- Embedding de 192 dimensões, 6 blocos, 3 cabeças de atenção por bloco
- Augmentation no treino: recorte aleatório com margem de 4 pixels e espelhamento horizontal
- Normalização com a média e o desvio padrão de cada canal, medidos no treino
- AdamW, taxa de aprendizado 0,001, weight decay 0,05, lotes de 256 imagens, 40 épocas
- A taxa de aprendizado sobe de forma linear nas 5 primeiras épocas e depois cai seguindo um cosseno
