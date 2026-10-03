## Modelo

- ResNet-50 com pesos da ImageNet
- Todos os pesos do backbone congelados
- A camada final `fc` é trocada por uma nova, com uma saída por classe do dataset
- Só a nova camada recebe gradiente
