## Leitura dos resultados

A discussão completa, com os números desta execução, está na seção da Atividade 2 do relatório. Aqui fica um resumo do que as células acima mostram.

**Corpus**

- O corpus tem duas partes: anúncios, em pastas numeradas por categoria, e imagens enviadas pelos participantes do estudo. As imagens de participantes são a maior parte.
- Saem do corpus as cópias exatas e as miniaturas (`<nome>_th_<número>.png` ao lado de `<nome>.png`). As contagens estão na saída da célula de dados.
- A regra das miniaturas olha só o nome do arquivo. Em uma pequena parte dos pares a miniatura mostra uma figura diferente da do original, e essas imagens se perdem.
- Sem a remoção, as buscas devolviam a imagem e a sua miniatura lado a lado, e as frequências das imagens de participantes ficavam infladas.

**Limiar**

- Um conceito conta como presente quando a sua probabilidade chega a 0,5, ou seja, quando ele tem mais probabilidade que todos os outros juntos. Cada imagem tem no máximo um conceito presente.
- O histograma da maior probabilidade por imagem não tem vale nem degrau. Nenhum ponto da curva indica sozinho o limiar.
- A decisão veio das imagens sorteadas em cada lado de 0,3, 0,5 e 0,7. O acerto dos rótulos sobe aos poucos com o limiar, e a parcela de imagens com algum conceito cai.
- 0,5 é uma escolha de julgamento, sobre uma amostra pequena avaliada por mim.

**Ranking**

- A frequência mede quantas vezes um conceito vence a disputa entre os 26 prompts. Outra lista de prompts pode dar outra ordem.
- Entre os anúncios, o prompt de anúncio com muito texto domina. Entre as imagens de participantes, os primeiros lugares são animal, comida e paisagem.
- A frequência do anúncio de texto no corpus inteiro depende da mistura das duas partes. Quase todas as imagens com esse conceito são anúncios, e a remoção das miniaturas encolheu só a parte dos participantes. A contagem mudou pouco e o total de imagens caiu, por isso a frequência subiu.
- Prompts parecidos dividem a probabilidade: os quatro prompts de pessoas entre si, e os dois prompts de texto entre si. A regra conta pessoas e texto para menos.
- Uma parte grande das imagens fica sem nenhum conceito presente.
- Nas figuras dos cinco conceitos mais frequentes, anúncio de texto, animal, comida e paisagem são confirmados pelos exemplos. "text and a logo" é confirmado em parte: um dos exemplos é uma imagem toda preta e outro é uma arte abstrata sem texto.

**Busca por texto**

- Consultas concretas acham o que descrevem: "something healthy to eat", "an advertisement with a lot of text". Em "something healthy to eat" o mesmo brócolis aparece duas vezes, uma em baixa resolução, em arquivos sem ligação pelo nome.
- Detalhes se perdem. Em "a red sports car on a road" nenhuma imagem é um carro esportivo vermelho: há um conversível cinza, um carro amarelo em uma rua alagada e um Mini vermelho, entre outras. Em "a happy family at home" há famílias, mas nenhuma dentro de casa.
- Consultas abstratas ("luxury", "freedom", "loneliness", "technology that makes life easier") dão resultados fracos, simbólicos ou literais, como uma foto com a palavra FREEDOM escrita em um muro.
- "um cachorro brincando na praia" devolve cachorros, mas só uma cena de praia. A quinta imagem é um borrão escuro e pixelado, que não dá para reconhecer como cachorro.
- "uma pessoa sorrindo" devolve três pessoas sorrindo, uma mulher com a boca coberta por um cobertor e um rosto de meme de terror com um sorriso largo.

**Tokenização**

- Os dois tokenizers dão o mesmo número de posições para cada frase. As frases curtas recebem padding até o tamanho da mais longa, e a attention mask tem 0 nessas posições.
- O CLIP usa o token de fim de texto como padding. O BERT usa `[PAD]`.

**Limites**

- Uma execução, com um modelo, uma redação por conceito, em inglês, e consultas fixadas antes de ver os resultados.
- Não há rótulos de objeto no corpus. A avaliação das figuras é a leitura de poucas imagens por caso.
