## Embeddings

- CLIP ViT-B/32 da OpenAI, em modo de avaliação, sem gradiente e sem nenhum ajuste de pesos
- Cada imagem passa uma vez pelo codificador de imagem e vira um vetor de comprimento 1
- Os vetores ficam em disco: se o arquivo existir e tiver o número certo de linhas, é reaproveitado
- Os conceitos e as frases de busca passam pelo codificador de texto
