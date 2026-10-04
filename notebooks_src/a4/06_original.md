## Parte A: o projeto que falhou, reproduzido

- Divisão 80/20 do conjunto do projeto, embaralhada e sem estratificação, seed 42
- ResNet-18 com pesos aleatórios, sem pré-treinamento
- SGD com taxa de aprendizado 0,01, sem momentum, 15 épocas, lotes de 32, sem augmentation
- O time reportou só a acurácia de treino e de validação. Aqui aparecem também o recall por classe na validação deles, o recall no teste real e quantas imagens de COVID-19 caíram na validação
- As acurácias desta reprodução são diferentes dos 93% e 61% citados no enunciado. As imagens são outras e a entrada tem 64x64 pixels. O que interessa aqui é o recall por classe, que a acurácia global não mostra
