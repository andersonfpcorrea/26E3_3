## Parte D: as imagens sintéticas aumentam o recall de COVID-19?

- Classificador: `resnet18` do torchvision com pesos da ImageNet, todas as camadas treinadas
- Entrada: 64x64 em tons de cinza, repetida em três canais, ampliada para 128 e normalizada com a média e o desvio padrão da ImageNet
- Adam com taxa de aprendizado 1e-4, 10 épocas, lotes de 32
- Augmentation leve e igual nos três braços: pequena translação e rotação de até 7 graus. Sem espelhamento horizontal, porque esquerda e direita importam em uma radiografia de tórax
- **Braço 1:** só imagens reais de treino
- **Braço 2:** imagens reais mais imagens sintéticas de COVID-19 da versão escolhida da GAN, o bastante para igualar a contagem de Normal. O classificador da seed i recebe as imagens da GAN de seed i
- **Braço 3:** só imagens reais, com perda ponderada pelo inverso da frequência das classes. É a alternativa simples, sem GAN
- Três seeds por braço. A avaliação é no teste real de 300 imagens por classe. Aparecem média e desvio padrão entre as seeds
- O braço 2 tem mais imagens de treino que os outros dois, então faz mais atualizações de pesos nas mesmas 10 épocas. A comparação não é controlada para isso: uma diferença a favor do braço 2 pode vir do treino mais longo
