## Leitura dos resultados

**Corpus**

- O download tem 2.697 arquivos de imagem. Foram removidas 47 cópias exatas, todas entre as imagens dos participantes, e 1.149 miniaturas de imagens que já estão no corpus. Ficaram 1.501 imagens: 301 anúncios e 1.200 imagens de participantes.
- Uma miniatura é um arquivo `<nome>_th_<número>.png` ao lado do arquivo `<nome>.png`, na mesma pasta. Nas 1.173 miniaturas do download, a imagem original existe em todas. Em uma amostra de 200 pares, 197 têm a mesma figura em tamanho menor. Na lista completa, 14 de 1.173 (1,2%) apontam para uma figura diferente, por isso a regra pelo nome remove uma pequena parte de imagens que não eram cópias.
- As miniaturas inflavam as frequências e faziam os resultados da busca virem em pares. Com a remoção, as buscas devolvem cinco imagens diferentes. Resta um par parecido: o brócolis da busca "something healthy to eat" aparece em duas versões, uma delas em resolução baixa, e os nomes dos arquivos não mostram a ligação.
- Os anúncios estão em 20 pastas numeradas, com 15 ou 16 anúncios cada. A atividade fala em 16 categorias, mas o arquivo tem 20 pastas.

**Limiar**

- Limiar usado: 0,5. Um conceito conta como presente quando a sua probabilidade chega a 0,5, isto é, quando ele fica com a maioria da probabilidade entre os 26 conceitos. Por isso cada imagem tem no máximo um conceito presente.
- A distribuição da maior probabilidade por imagem não tem vale nem degrau. Ela é quase plana entre 0,2 e 0,9 e sobe perto de 1. Na distribuição de todas as probabilidades a contagem cai até cerca de 0,4 e depois fica num patamar. Nenhum ponto da curva indica sozinho o limiar.
- Por isso a decisão veio das imagens sorteadas em cada lado dos candidatos. Em 0,3, de 16 imagens, cerca de 10 têm conceito aceitável, com erros como uma moça de fita na boca como cosmético, uma sala colorida e uma turma de alunos como anúncio ou ilustração, e dedos dos pés como pessoa. Em 0,5, cerca de 12 das 16 têm conceito aceitável, com erros como uma colagem de monstros de filmes de terror como cosmético e uma chave como carro. Em 0,7, as 16 têm conceito aceitável.
- A qualidade sobe aos poucos com o limiar e não há um salto. O custo também sobe: com algum conceito presente ficam 86,3% das imagens em 0,3, 62,6% em 0,5 e 40,0% em 0,7. Fiquei com 0,5 por ser o ponto em que a maior parte das imagens inspecionadas já está certa, por manter a maioria do corpus e por dar a regra simples de maioria. Em 0,7 o ranking seria mais limpo, mas 60% das imagens ficariam sem conceito.
- A inspeção usou 8 imagens por lado e por candidato, sorteadas com seed fixa. É uma amostra pequena e a avaliação de cada imagem é minha.

**Ranking**

- Corpus inteiro (1.501 imagens): anúncio online com muito texto (15,5%), animal (8,0%), comida (6,1%), paisagem ao ar livre (4,9%) e texto e logotipo (3,0%). A frequência do anúncio com texto subiu de 9,7% para 15,5% porque as miniaturas saíram só do lado dos participantes e os anúncios passaram a pesar mais no corpus.
- Só anúncios: anúncio online com muito texto em 70,1% (211 de 301), produto sobre fundo liso em 7,0% e depois conceitos com 1,7% ou menos. Os anúncios do corpus são em grande parte banners de texto.
- Só participantes (1.200 imagens): animal (10,0%), comida (7,5%), paisagem (6,1%), criança (3,5%) e texto e logotipo (3,5%). A ordem é a mesma de antes e as frequências mudaram pouco.
- A similaridade média fica entre 0,16 e 0,21 em todos os conceitos do corpus inteiro e quase não separa um conceito do outro. A ordem útil é a da frequência.
- Cerca de 37% das imagens não têm nenhum conceito presente em 0,5. A lista de 26 conceitos não cobre o corpus.
- Os cinco conceitos mais frequentes têm exemplos nas figuras `top_1` a `top_5`. Os de anúncio de texto, animal, comida e paisagem mostram seis imagens diferentes que combinam com o conceito. Em texto e logotipo, um dos seis exemplos é uma imagem totalmente preta, e os outros são logotipos (imgur, THE 1975, eBay), um anúncio de texto e uma arte abstrata.

**Busca por texto**

- Sem as miniaturas, cada busca devolve cinco imagens diferentes. A única exceção é "something healthy to eat", que traz duas fotos do mesmo brócolis, uma em resolução baixa.
- As frases concretas e específicas funcionam melhor: "an advertisement with a lot of text" devolve cinco anúncios com muito texto e "something healthy to eat" devolve frutas, cenouras, uma pirâmide alimentar e brócolis.
- "a car" devolve um meme com um cachorro dentro de um carro, um túnel visto de dentro de um carro, uma colagem de carros esportivos, um carro azul de passeio e um anúncio de texto sobre acessórios para carro. Há carro nas cinco, mas ele é o assunto principal só na colagem e no carro azul.
- Em "a red sports car on a road" só um dos cinco é vermelho, um Mini Cooper, e nenhum é esportivo e vermelho ao mesmo tempo. Os outros são um cupê cinza, um carro amarelo numa enchente, o meme do cachorro no carro e crianças num carrinho vermelho.
- "a happy family at home" devolve uma família de desenho 3D, dois gatos em um canto, uma família real sobre fundo branco, uma mulher, um homem e uma criança de capacete e uma família em silhueta ao pôr do sol. Quatro dos cinco são famílias, nenhuma em casa.
- As frases abstratas dão resultados fracos ou ambíguos. "freedom" traz uma foto com a palavra pichada no muro, cabelo ao vento, dois cavalos diferentes e um snowboarder. "loneliness" traz uma frase escrita sobre ficar sozinho, um boneco pendurado, um céu estrelado, uma ferrovia na névoa e cabelo ao vento. "luxury" traz um carro esportivo, uma colagem de carros, um anúncio de cadeira de escritório, uma mesa diante do mar e um anúncio de canetas de luxo. "technology that makes life easier" traz anúncios de gadgets e de aplicativo, um disquete desenhado, um ícone de computador e uma imagem sem relação com a frase.
- Em português, "uma pessoa sorrindo" devolve pessoas, a maioria sorrindo, e "um cachorro brincando na praia" devolve cachorros. Só uma das imagens tem cachorros numa praia, e ela mostra um homem, uma criança e cachorros. As outras mostram cachorros presos, amarrados ou em posição de ataque. O CLIP entendeu o assunto principal e só às vezes a cena.
- As pontuações ficam entre 0,23 e 0,30.

**Tokenização**

- As três frases têm 4, 9 e 27 posições no CLIP, e o mesmo número no BERT. Com padding até a mais longa, a mais curta tem 23 posições de preenchimento e a máscara de atenção tem 0 nelas.
- O CLIP usa o mesmo símbolo de fim de texto como preenchimento. O BERT usa `[PAD]`.

**Limites**

- Uma execução, com um modelo, um conjunto de conceitos e um conjunto de frases fixados antes de ver os resultados.
- A frequência de cada conceito depende da lista de conceitos, porque a softmax divide a probabilidade entre eles. Outra lista daria outra ordem.
