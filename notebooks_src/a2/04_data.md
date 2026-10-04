## Dados

- Download do ADS-16 no Kaggle, sem login
- O arquivo traz dois blocos (`part1` e `part2`). Cada um tem uma pasta `Ads`, com os anúncios em pastas numeradas por categoria, e uma pasta `Corpus`, com as imagens enviadas pelos participantes do estudo
- Os arquivos ZIP dentro do download (`Ads.zip`, `Corpus.zip`, `Documents.zip`) já vêm extraídos ao lado deles, então são ignorados
- Só entram imagens que o Pillow consegue abrir. Cópias exatas, comparadas pelo conteúdo, são removidas para que a mesma imagem não conte duas vezes na frequência
- Todos os resultados saem para o corpus inteiro e separados em anúncios e imagens de participantes
