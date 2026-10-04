## Leitura dos resultados

**Corpus**

- O download tem 2.697 arquivos de imagem. Foram removidas 47 cópias exatas, todas entre as imagens dos participantes, e ficaram 2.650 imagens: 301 anúncios e 2.349 imagens de participantes.
- Os anúncios estão em 20 pastas numeradas, com 15 ou 16 anúncios cada. A atividade fala em 16 categorias, mas o arquivo tem 20 pastas.
- Cerca de metade dos arquivos dos participantes (a contagem está acima, nos arquivos com `_th_` no nome) é uma miniatura de outra imagem do corpus. Elas não são cópias exatas e por isso não foram removidas. O efeito aparece nas figuras: muitos resultados vêm em pares, a imagem e a sua miniatura.

**Limiar**

- Limiar usado: 0,5. Um conceito conta como presente quando a sua probabilidade chega a 0,5, isto é, quando ele fica com a maioria da probabilidade entre os 26 conceitos. Por isso cada imagem tem no máximo um conceito presente.
- A distribuição da maior probabilidade por imagem não tem vale nem degrau. Ela é quase plana entre 0,2 e 0,9 e sobe perto de 1. Na distribuição de todas as probabilidades a contagem cai até cerca de 0,4 e depois fica num patamar. Nenhum ponto da curva indica sozinho o limiar.
- Por isso a decisão veio das imagens sorteadas em cada lado dos candidatos. Em 0,3, de 16 imagens, a metade tem o conceito errado ou fraco (lava como joia, ferida como joia, livros como ilustração, multidão como celular). Em 0,5, cerca de 12 das 16 têm um conceito aceitável, com erros como símbolos religiosos como joia e gatos como móvel. Em 0,7, as 16 têm conceito aceitável.
- A qualidade sobe aos poucos com o limiar e não há um salto. O custo também sobe: com algum conceito presente ficam 85% das imagens em 0,3, 59,7% em 0,5 e 37% em 0,7. Fiquei com 0,5 por ser o ponto em que a maior parte das imagens inspecionadas já está certa, por manter a maioria do corpus e por dar a regra simples de maioria. Em 0,7 o ranking seria mais limpo, mas 63% das imagens ficariam sem conceito.
- A inspeção usou 8 imagens por lado e por candidato, sorteadas com seed fixa. É uma amostra pequena e a avaliação de cada imagem é minha.

**Ranking**

- Corpus inteiro: anúncio online com muito texto (9,7%), animal (8,6%), comida (6,6%), paisagem ao ar livre (5,4%) e texto e logotipo (3,2%).
- Só anúncios: anúncio online com muito texto em 70,1% (211 de 301), produto sobre fundo liso em 7,0% e depois conceitos com 1,7% ou menos. Os anúncios do corpus são em grande parte banners de texto.
- Só participantes: animal (9,7%), comida (7,4%), paisagem (6,1%), criança (3,4%) e texto e logotipo (3,4%).
- A similaridade média fica entre 0,16 e 0,25 em todos os conceitos e quase não separa um conceito do outro. A ordem útil é a da frequência.
- Cerca de 40% das imagens não têm nenhum conceito presente em 0,5. A lista de 26 conceitos não cobre o corpus.
- Os cinco conceitos mais frequentes têm exemplos nas figuras `top_1` a `top_5`. Os de anúncio de texto, animal, comida e paisagem mostram imagens que combinam com o conceito. Em texto e logotipo, dois dos seis exemplos são imagens totalmente pretas e os outros são logotipos e um anúncio de texto.

**Busca por texto**

- As frases concretas e específicas funcionam melhor: "an advertisement with a lot of text" devolve cinco anúncios com muito texto, "something healthy to eat" devolve frutas, cenouras e uma pirâmide alimentar, "a red sports car on a road" devolve carros.
- "a car" devolve um meme com um cachorro dentro de um carro, uma colagem de carros esportivos e um túnel visto de dentro de um carro. Os carros não são o assunto principal da maioria.
- Em "a red sports car on a road" nenhum carro é vermelho e esportivo ao mesmo tempo: há um Mini vermelho, um cupê cinza e um carro amarelo numa enchente.
- "a happy family at home" devolve uma família de desenho 3D, dois gatos em um canto e uma família real sobre fundo branco, só um dos cinco é uma família real.
- As frases abstratas dão resultados fracos ou ambíguos. "freedom" traz uma foto com a palavra pichada no muro, uma pessoa de cabelo ao vento e um cavalo. "loneliness" traz uma frase escrita sobre ficar sozinho e um boneco pendurado. "luxury" traz um carro esportivo, uma cadeira de escritório num anúncio e uma mesa diante do mar. "technology that makes life easier" traz anúncios de gadgets e de aplicativo, um disquete desenhado e imagens que não têm relação com a frase.
- Em português, "uma pessoa sorrindo" devolve pessoas, a maioria sorrindo, e "um cachorro brincando na praia" devolve cachorros, mas nenhum está na praia nem brincando. O CLIP entendeu o assunto principal e não a cena inteira.
- As pontuações ficam entre 0,24 e 0,30. As primeiras posições têm pares de imagem e miniatura, então cada busca mostra só 3 ou 4 imagens diferentes.

**Tokenização**

- As três frases têm 4, 9 e 27 posições no CLIP, e o mesmo número no BERT. Com padding até a mais longa, a mais curta tem 23 posições de preenchimento e a máscara de atenção tem 0 nelas.
- O CLIP usa o mesmo símbolo de fim de texto como preenchimento. O BERT usa `[PAD]`.

**Limites**

- Uma execução, com um modelo, um conjunto de conceitos e um conjunto de frases fixados antes de ver os resultados.
- A frequência de cada conceito depende da lista de conceitos, porque a softmax divide a probabilidade entre eles. Outra lista daria outra ordem.
