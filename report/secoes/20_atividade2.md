# Atividade 2: Reconhecimento semântico em publicidade com CLIP

## Problema

- Tarefa: usar o CLIP pré-treinado para ordenar conceitos pela frequência no corpus ADS-16 e para buscar imagens a partir de uma frase.
- Dataset: `groffo/ads16-dataset`, público no Kaggle. Modelo: `openai/clip-vit-base-patch32`, sem ajuste de pesos e sem uso de rótulos do ADS-16.

**O corpus como veio**

- O download tem 2.697 arquivos de imagem, todos legíveis, em duas partes: pastas `Ads`, com anúncios, e pastas `Corpus`, com imagens enviadas pelos participantes do estudo que gerou o dataset.
- Os anúncios estão em 20 pastas numeradas por categoria: 19 com 15 anúncios e 1 com 16.
- 47 cópias exatas foram removidas, comparando o conteúdo dos arquivos.
- 1.149 miniaturas foram removidas pela regra abaixo.

| Parte | Imagens | Parcela |
| --- | --- | --- |
| Anúncios | 301 | 20,1% |
| Imagens de participantes | 1.200 | 79,9% |
| Total | 1.501 | 100% |

**Miniaturas: a regra e o seu limite**

- Nas pastas dos participantes, um arquivo com `_th_` no nome é a cópia reduzida de outro arquivo da mesma pasta: `4_th_1448374800.png` é a miniatura de `4.png`.
- Regra: a miniatura sai quando o original está no corpus. Sem a regra, a mesma figura contaria duas vezes na frequência e viria em dobro nas buscas.
- Conferência: os 1.173 pares formados pelo nome foram comparados pelos pixels. Em 1.159 as duas imagens são a mesma figura. Das 1.173 miniaturas, 24 já tinham saído como cópias exatas.
- Limite: em 14 pares as figuras são diferentes. A regra olha só o nome, então 14 imagens distintas ficaram fora do corpus.

## Como o CLIP alinha imagem e texto

**O que o pré-treinamento faz**

- O CLIP tem dois encoders: um para imagens (aqui um ViT-B/32) e um para texto (um Transformer). Uma projeção no fim de cada um leva as duas saídas para o mesmo espaço de embedding.
- O treino usou cerca de 400 milhões de pares de imagem e legenda coletados na web. Ninguém anotou classes. A legenda que já acompanhava a imagem serve de supervisão.
- O objetivo é contrastive. Em um batch com N pares, o modelo calcula a cosine similarity de cada imagem com cada texto. A perda aumenta a similaridade dos N pares corretos e reduz a dos pares trocados.
- As similaridades são multiplicadas por uma temperatura aprendida antes da softmax. Neste modelo o valor é 100.
- Resultado: uma imagem e um texto que combinam ficam próximos, e a cosine similarity entre os dois vetores mede esse acordo.

**Por que isso permite busca sem treino supervisionado**

- A legenda da web é texto livre. O modelo aprende a ligar imagens a frases, sem lista fechada de categorias, e qualquer frase nova já tem um lugar no espaço.
- Classificação zero-shot: cada classe vira um prompt, e a imagem recebe a classe do prompt mais próximo. Os prompts fazem o papel dos pesos de um classificador, sem nenhum exemplo rotulado.
- Busca semântica é a mesma conta no sentido contrário: uma frase, muitas imagens, e a resposta são as imagens de maior similaridade.

**Neste projeto**

- Cada uma das 1.501 imagens passou uma vez pelo encoder de imagem e virou um vetor de comprimento 1. Isso levou 37,0 segundos na T4.
- Os mesmos embeddings servem ao ranking e à busca. Muda só o texto comparado: 26 prompts de conceito no ranking, 11 frases na busca.

## Ranking de conceitos

### Prompts

- 26 conceitos, cada um escrito como "a photo of" seguido do conceito. A lista está na tabela do ranking.
- Ela cobre objetos de anúncio, pessoas, cenas e formatos de imagem. Foi fixada antes de ver os resultados, com uma redação por conceito, em inglês.

### Regra de presença e escolha do threshold

- Para cada imagem, as 26 similaridades são multiplicadas por 100 e passam por uma softmax. Cada conceito recebe uma probabilidade e as 26 somam 1.
- Regra: um conceito está presente na imagem quando a sua probabilidade é igual ou maior que 0,5. A frequência é a parcela das imagens em que ele está presente.

![Distribuição da maior probabilidade por imagem e de todas as probabilidades](figures/a2_distribution.png)

- A distribuição não aponta um threshold. A maior probabilidade de cada imagem se espalha de 0,2 a 1, sem vale: os quartis são 0,404, 0,602 e 0,832.
- A escolha veio de olhar 8 imagens sorteadas logo abaixo e 8 logo acima de três candidatos: 0,3, 0,5 e 0,7. Perto de 0,3 os erros são comuns. Perto de 0,7 quase todos os rótulos combinam.
- Em 0,5 a maioria combina. Acima da linha, os erros vistos são uma chave como "a car" e uma maquete de castelo como "a house". Logo abaixo há rótulos certos que a regra descarta, como um macaco em "an animal" com 0,46.

![Imagens sorteadas abaixo (linha de cima) e acima (linha de baixo) de 0,5](figures/a2_threshold_050.png)

- Decisão: 0,5, a regra de maioria. O conceito tem mais probabilidade que os outros 25 juntos. É uma escolha de julgamento, sobre uma amostra pequena avaliada por mim.
- Custo: têm algum conceito presente 86,3% das imagens em 0,3, 62,6% em 0,5 e 40,0% em 0,7.

### Resultado

Frequência com threshold 0,5. "Imagens" é a contagem no corpus inteiro. Empates são desfeitos pela similaridade média.

| # | Conceito | Imagens | Corpus | Anúncios | Participantes |
| --- | --- | --- | --- | --- | --- |
| 1 | an online advertisement with a lot of text | 232 | 15,5% | 70,1% | 1,8% |
| 2 | an animal | 120 | 8,0% | 0,0% | 10,0% |
| 3 | food | 92 | 6,1% | 0,7% | 7,5% |
| 4 | outdoor scenery | 73 | 4,9% | 0,0% | 6,1% |
| 5 | text and a logo | 45 | 3,0% | 1,0% | 3,5% |
| 6 | a child | 42 | 2,8% | 0,0% | 3,5% |
| 7 | a product on a plain background | 39 | 2,6% | 7,0% | 1,5% |
| 8 | a man | 38 | 2,5% | 0,0% | 3,2% |
| 9 | sports | 37 | 2,5% | 0,7% | 2,9% |
| 10 | an illustration or cartoon | 35 | 2,3% | 0,3% | 2,8% |
| 11 | jewellery | 18 | 1,2% | 1,7% | 1,1% |
| 12 | a car | 18 | 1,2% | 0,3% | 1,4% |
| 13 | clothes | 17 | 1,1% | 0,0% | 1,4% |
| 14 | a beach | 17 | 1,1% | 0,0% | 1,4% |
| 15 | cosmetics | 16 | 1,1% | 0,7% | 1,2% |
| 16 | a chart or diagram | 16 | 1,1% | 0,0% | 1,3% |
| 17 | a person | 12 | 0,8% | 0,0% | 1,0% |
| 18 | a mobile phone | 12 | 0,8% | 1,0% | 0,8% |
| 19 | a smiling woman | 10 | 0,7% | 0,0% | 0,8% |
| 20 | furniture | 9 | 0,6% | 1,0% | 0,5% |
| 21 | a musical instrument | 9 | 0,6% | 1,3% | 0,4% |
| 22 | a drink | 9 | 0,6% | 0,0% | 0,8% |
| 23 | a house | 9 | 0,6% | 0,0% | 0,8% |
| 24 | shoes | 7 | 0,5% | 0,3% | 0,5% |
| 25 | a computer | 4 | 0,3% | 0,3% | 0,2% |
| 26 | a city street | 4 | 0,3% | 0,0% | 0,3% |

![Os 15 conceitos mais frequentes no corpus, nos anúncios e nas imagens de participantes](figures/a2_ranking.png)

- A similaridade média fica entre 0,162 e 0,209 nos 26 conceitos e quase não os separa.

### Os cinco conceitos mais frequentes

Cada figura mostra as seis imagens de maior similaridade com o prompt.

![1. an online advertisement with a lot of text](figures/a2_top_1.png)

- Confirmado: seis anúncios de busca, só com título, endereço e linhas de texto.

![2. an animal](figures/a2_top_2.png)

- Confirmado: seis animais, entre eles dois besouros, um porco e um guaxinim.

![3. food](figures/a2_top_3.png)

- Confirmado: seis pratos e alimentos, entre eles um peixe, um lanche com batata frita e uma pizza.

![4. outdoor scenery](figures/a2_top_4.png)

- Confirmado: seis paisagens sem pessoas, com neve, lago, montanha e floresta.

![5. text and a logo](figures/a2_top_5.png)

- Confirmado em parte. Três logotipos (imgur, THE 1975, eBay) e um anúncio de texto combinam.
- A imagem de maior similaridade é um retângulo todo preto. A sexta é uma arte abstrata colorida, sem texto.

### O que o ranking diz e onde a regra falha

**Sobre o corpus**

- Os anúncios do ADS-16 são, em sua maioria, anúncios de texto. O prompt de anúncio com muito texto está presente em 211 dos 301. O segundo lugar entre os anúncios, "a product on a plain background", tem 21.
- Objetos anunciados quase não aparecem como conceito presente nos anúncios: "jewellery" tem 5 imagens, "a car" tem 1.
- O primeiro lugar do corpus inteiro depende da mistura das duas partes, por isso a tabela traz uma coluna para cada uma.

**Onde a regra falha**

- A frequência mede quantas vezes um conceito vence a disputa entre os 26 prompts. Ela vale para esta lista e para esta regra. Quantas imagens mostram cada objeto é outra pergunta, que exigiria rótulos.
- Cada imagem tem no máximo um conceito presente. Uma imagem com um homem e um carro conta para um dos dois ou para nenhum.
- 561 imagens (37,4%) ficam sem conceito. Têm um conceito 260 dos 301 anúncios e 680 das 1.200 imagens de participantes.
- Prompts parecidos dividem a probabilidade. Os quatro prompts de pessoas ("a person", "a man", "a child", "a smiling woman") somam 15,2% de probabilidade média, mas só 102 imagens (6,8%) têm um deles presente. Os dois prompts de texto disputam da mesma forma. A regra conta pessoas e texto para menos.
- Efeito visível: "a person" fica em 17º lugar, abaixo de "a child" e de "a man".

## Busca por texto

- 11 consultas, fixadas antes de ver os resultados. Nenhuma foi trocada depois.
- Cada consulta vira um embedding de texto. O resultado são as 5 imagens de maior cosine similarity entre as 1.501.
- A busca sempre devolve 5 imagens, mesmo quando nenhuma combina com a frase.

### 1. "a car"

![Resultados de "a car"](figures/a2_query_01.png)

- Há carro ou referência a carro nas cinco imagens. Ele é o assunto em duas: a colagem de um carro esportivo prateado e o carro azul.
- As outras são inesperadas: um meme com a frase "a dog driving a car", um túnel visto de dentro de um carro e um anúncio de texto com "Accessories For Car".

### 2. "a red sports car on a road"

![Resultados de "a red sports car on a road"](figures/a2_query_02.png)

- Nenhuma imagem mostra um carro esportivo vermelho em uma estrada.
- Cada resultado atende a um pedaço da frase: um conversível esportivo cinza, um carro amarelo em uma rua alagada, o meme do cachorro ao volante, crianças em um carrinho vermelho de brinquedo e um Mini vermelho sobre fundo preto.
- O primeiro resultado tem 0,245, o menor valor entre os primeiros resultados das 11 consultas. O corpus pode não ter a cena pedida.

### 3. "a happy family at home"

![Resultados de "a happy family at home"](figures/a2_query_03.png)

- Quatro resultados são famílias alegres: um desenho 3D no sofá, uma foto sobre fundo branco, três pessoas de capacete amarelo e uma silhueta ao pôr do sol. Nenhuma foto é dentro de casa.
- O segundo resultado são dois gatos no canto de uma sala. Combina com "at home". Falta a família.

### 4. "luxury"

![Resultados de "luxury"](figures/a2_query_04.png)

- Os dois primeiros resultados são carros: o conversível cinza e a colagem do carro esportivo.
- O quinto é um anúncio de texto com "Luxury Pens" no título. A palavra escrita puxou a imagem.
- Os outros são um anúncio de cadeira de escritório de 39,99 libras e uma mesa diante de uma janela para o mar.

### 5. "freedom"

![Resultados de "freedom"](figures/a2_query_05.png)

- O primeiro resultado é literal: uma foto em preto e branco de uma multidão sobre um muro com a palavra FREEDOM pichada.
- Os outros são símbolos comuns de liberdade: cabelo ao vento, dois cavalos pretos correndo e uma pessoa de snowboard na neve.

### 6. "something healthy to eat"

![Resultados de "something healthy to eat"](figures/a2_query_06.png)

- Os cinco resultados combinam: um prato de frutas, cenouras, uma pirâmide alimentar e brócolis.
- A descrição é indireta, sem citar fruta nem verdura, e o modelo a resolveu.
- O brócolis aparece duas vezes, uma em baixa resolução, em arquivos sem ligação pelo nome.

### 7. "technology that makes life easier"

![Resultados de "technology that makes life easier"](figures/a2_query_07.png)

- Resultado fraco. As cinco similaridades ficam entre 0,250 e 0,254, quase empatadas.
- Quatro têm ligação com tecnologia: um anúncio de texto de "Innovative Baby Gadgets", um anúncio do aplicativo Tinder, um disquete com braços desenhados e um ícone de computador.
- O segundo resultado, um passo a passo com um molde azul e um anel de borracha, não tem relação visível com a frase.

### 8. "an advertisement with a lot of text"

![Resultados de "an advertisement with a lot of text"](figures/a2_query_08.png)

- Os cinco resultados são anúncios, todos da parte de anúncios do corpus.
- Quatro têm bastante texto. O banner de hambúrguer tem só uma frase curta.

### 9. "loneliness"

![Resultados de "loneliness"](figures/a2_query_09.png)

- O primeiro resultado é de novo texto: um cartaz com a frase "Learn to be Alone and to like it".
- Seguem cenas vazias e escuras: um boneco pendurado, um céu estrelado com uma figura pequena e um trilho de trem na névoa.
- O quinto é a mesma foto de cabelo ao vento de "freedom". Uma imagem que serve a duas ideias indica uma ligação vaga.

### 10. "um cachorro brincando na praia"

![Resultados de "um cachorro brincando na praia"](figures/a2_query_10.png)

- Quatro resultados mostram cachorros. O modelo entendeu "cachorro" em português.
- Só o terceiro tem praia: um homem, uma criança e cachorros na areia. Outros três contrariam "brincando": um cachorro magro preso a uma corrente, um com o focinho amarrado com fita e um de coleira com pontas.
- O quinto é um borrão escuro e pixelado. Não dá para dizer que é um cachorro.

### 11. "uma pessoa sorrindo"

![Resultados de "uma pessoa sorrindo"](figures/a2_query_11.png)

- Três resultados são pessoas sorrindo: uma mulher em uma varanda, um homem com restos de folha verde nos dentes e um rapaz loiro.
- No segundo, a mulher cobre a boca com um cobertor. O quarto é um rosto branco de meme de terror, com um sorriso largo: o modelo achou o sorriso e ignorou o resto.

### Resumo por tipo de consulta

**Concreta contra abstrata**

- Consultas sobre coisas visíveis funcionam: "a car", "something healthy to eat", "an advertisement with a lot of text", "uma pessoa sorrindo".
- As abstratas ("luxury", "freedom", "loneliness", "technology that makes life easier") dão resultados fracos, literais ou simbólicos.
- A similaridade não separa os dois casos. O primeiro resultado de "loneliness" tem 0,281 e o de "a red sports car on a road" tem 0,245. Todas as similaridades das buscas ficam entre 0,235 e 0,301.

**Genérica contra específica**

- A genérica "a car" acha carros. Nas específicas ("a red sports car on a road", "a happy family at home", "um cachorro brincando na praia"), o assunto principal aparece e a cor, o lugar ou a ação ficam de fora na maior parte dos resultados.
- Parte disso é o corpus: com 1.501 imagens, a cena pedida pode não existir.

**Inglês contra português**

- As duas consultas em português acharam o assunto certo: cachorros e sorrisos.
- A perda de detalhes é parecida com a das consultas específicas em inglês. Com duas consultas, e sem a mesma frase nas duas línguas, este experimento não mede a diferença entre elas.

## Consulta de texto no CLIP e tokenização no BERT

**O que os dois tokenizers fazem com o mesmo texto**

Três frases passaram pelo tokenizer do CLIP e pelo do `bert-base-uncased`, com padding até a mais longa. Saída para "a car":

- CLIP, tokens: `<|startoftext|>`, `a</w>`, `car</w>`, `<|endoftext|>` e mais 23 vezes `<|endoftext|>`.
- BERT, tokens: `[CLS]`, `a`, `car`, `[SEP]` e mais 23 vezes `[PAD]`.
- Attention mask, igual nos dois: 1, 1, 1, 1 e depois 23 zeros.

| Frase | Tokens reais | Padding |
| --- | --- | --- |
| "a car" | 4 | 23 |
| "a red sports car on a road" | 9 | 18 |
| Frase longa sobre uma família no café da manhã | 27 | 0 |

- Os números da tabela são os mesmos nos dois tokenizers. Nestas frases, cada palavra virou um único token.
- Os dois passam o texto para minúsculas, cortam palavras raras em unidades menores e trocam cada unidade por um id do vocabulário. O CLIP usa byte-pair encoding e marca o fim de cada palavra com `</w>`. O BERT usa WordPiece.
- Os dois acrescentam dois tokens especiais: início e fim de texto no CLIP, `[CLS]` e `[SEP]` no BERT. O CLIP aceita até 77 tokens por texto e o BERT até 512.

**Padding**

- O modelo processa um batch como uma matriz, e todas as linhas precisam do mesmo comprimento. O padding completa as frases curtas até o tamanho da mais longa.
- O BERT tem um token próprio para isso, `[PAD]`. O tokenizer do CLIP repete o token de fim de texto.

**Attention mask**

- Ela marca os tokens reais com 1 e o padding com 0. A attention usa a máscara para tirar o peso das posições com 0.
- No BERT a attention é bidirecional: cada token olha para todos os outros. Sem a máscara, os 23 tokens de padding entrariam na conta, e a representação de "a car" mudaria conforme a frase mais longa do batch.
- O encoder de texto do CLIP usa também uma máscara causal: cada posição só olha para si e para as anteriores. Como o padding vem depois do fim do texto, ele já fica fora do alcance das posições reais.

**Da sequência a um vetor**

- CLIP: o vetor do texto é a saída na posição do token de fim de texto. Com a máscara causal, essa é a posição que viu a frase inteira. O vetor passa pela projeção e cai no espaço comum com as imagens, pronto para a cosine similarity.
- BERT: o vetor da sequência é a saída na posição do `[CLS]`, no início. Com attention bidirecional, a primeira posição também vê a frase inteira. Esse vetor alimenta uma classification head, treinada com rótulos na tarefa final.

## Análise crítica

**O modelo lê o que está escrito na imagem**

- Em quatro consultas, algum resultado tem escrita a palavra pedida, ou uma próxima: "car" no meme e no anúncio, "Luxury" no anúncio de canetas, FREEDOM no muro, "Alone" no cartaz.
- Os anúncios do corpus são quase todos texto. Neles, a busca tende a achar o anúncio que cita a palavra antes do anúncio que mostra o objeto.

**Viés do modelo e dos dados de treino**

- O CLIP aprendeu com legendas da web, e as associações dele repetem as de quem as escreveu. Aqui, "luxury" virou carro esportivo e "freedom" virou cavalo correndo.
- As legendas de treino são em sua maior parte em inglês. As consultas em português usaram palavras comuns. Termos raros ou regionais e o viés sobre pessoas ficaram sem teste.

**Prompts**

- Há uma redação por conceito, em inglês. A ordem do ranking pode mudar com outra redação, e isso não foi medido.
- O molde "a photo of" foi aplicado a todos os conceitos, inclusive a banners e ilustrações.

**Imagens que não são anúncios**

- Quatro em cada cinco imagens vieram dos participantes: animais, comida, paisagens, memes. O ranking do corpus inteiro descreve principalmente esse material.
- Para falar de publicidade, vale a coluna de anúncios. Nela a lista de 26 conceitos serve pouco, porque um único prompt de formato domina.

**Sem gabarito**

- O ADS-16 não traz rótulos de objeto, então não há precisão nem recall. A avaliação das figuras é a minha leitura de 5 ou 6 imagens por caso, em uma execução.

**O que eu mudaria**

- Várias redações por conceito, com a média dos embeddings de texto.
- Decisão independente por conceito, comparando cada prompt com um prompt neutro. Uma imagem poderia ter dois conceitos, e prompts parecidos parariam de disputar.
- Rotular à mão uma amostra de cerca de 200 imagens e medir a precisão da regra em cada threshold.
- Remover miniaturas e repetições pela comparação de pixels, o que evitaria a perda das 14 imagens e pegaria o brócolis repetido.
- Um CLIP multilíngue para consultas em português.
