## Parte A: o projeto que falhou, reproduzido

- Divisão 80/20 do conjunto do projeto, embaralhada e sem estratificação, seed 42
- ResNet-18 com pesos aleatórios, sem pré-treinamento
- SGD com taxa de aprendizado 0,01, sem momentum, 15 épocas, lotes de 32, sem augmentation
- O time reportou só a acurácia de treino e de validação. Aqui aparecem também a revocação por classe na validação deles, a revocação no teste real e quantas imagens de COVID-19 caíram na validação
