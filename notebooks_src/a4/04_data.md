## Dados

- Download do COVID-19 Radiography Database, do Kaggle, que não exige login
- Só as pastas `images/` das classes Normal, Pneumonia viral e COVID-19 entram. As máscaras de segmentação dos pulmões são outro tipo de imagem e nunca são carregadas
- Cópias exatas por conteúdo de arquivo são removidas antes de qualquer sorteio, para que a mesma imagem não apareça no treino e no teste
- **Conjunto do projeto:** 840 Normal, 240 Pneumonia e 120 COVID-19, sorteados com a seed 42. É o tamanho do projeto que falhou
- **Conjunto de teste real:** 300 imagens de cada classe, sorteadas entre as imagens que sobraram, sem nenhuma do conjunto do projeto. O conjunto do projeto deixa só cerca de 24 imagens de COVID-19 para validação. A base pública tem milhares de imagens reais a mais, então o recall pode ser medido em 300 imagens por classe e não em cerca de 24
- Imagens lidas em tons de cinza e reduzidas para 64x64
- Só cópias exatas são removidas. Quase-duplicatas, como a mesma radiografia regravada ou redimensionada, não foram procuradas. Os arquivos usados não trazem identificador de paciente, então duas imagens do mesmo paciente podem cair em conjuntos diferentes
