## Tokenização

- Três frases de tamanhos diferentes passam pelo tokenizador do CLIP e pelo do BERT (`bert-base-uncased`)
- O tokenizador completa as frases curtas até o tamanho da mais longa, com preenchimento (padding), e a máscara de atenção marca com 0 as posições de preenchimento
- O CLIP aceita no máximo 77 tokens por texto. Um texto maior é cortado, e o notebook avisa quando isso acontece
