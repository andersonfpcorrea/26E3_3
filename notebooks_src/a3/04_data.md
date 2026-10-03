## Dados

- Download do dataset público do Kaggle
- A classe de cada imagem é o nome da pasta onde ela está
- O download traz o dataset duas vezes; cópias exatas são removidas comparando o conteúdo dos arquivos, para que a mesma imagem nunca apareça no treino e no teste
- Divisão estratificada: 70% treino, 15% validação, 15% teste
- Pré-processamento igual ao do pré-treinamento: redimensionar, recorte central de 224x224 e normalização com média e desvio padrão da ImageNet
