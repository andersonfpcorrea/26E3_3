# Atividade 1: Datação de pinturas com Vision Transformers

## Problema

- Tarefa: dizer em qual de 8 períodos uma pintura foi feita, olhando só a imagem.
- Dataset: WikiArt, na cópia `Artificio/WikiArt` do Hugging Face. São 103.250 linhas, com imagens já em 256x256.
- Origem e licença: é um upload de terceiros, sem licença declarada. O uso aqui é só para estudo.
- O ano vem da coluna `date`, que é texto em formatos variados (`1920`, `1997.0`, `c.1904`, `None`). 76.881 linhas têm um ano que pode ser lido, 74,5% do total. As outras foram descartadas.
- Três modelos são comparados:
    - um Vision Transformer construído à mão e treinado do zero;
    - o mesmo modelo sem positional encoding;
    - um ViT-B/16 pré-treinado na ImageNet, com fine-tuning.

**Os oito períodos**

| Período | Pinturas com ano no dataset | Pinturas na amostra |
| --- | --- | --- |
| até 1599 | 5.427 | 2.000 |
| 1600 a 1799 | 6.686 | 2.000 |
| 1800 a 1849 | 4.467 | 2.000 |
| 1850 a 1874 | 5.453 | 2.000 |
| 1875 a 1899 | 13.640 | 2.000 |
| 1900 a 1924 | 17.235 | 2.000 |
| 1925 a 1949 | 9.927 | 2.000 |
| 1950 em diante | 14.046 | 2.000 |
| Total | 76.881 | 16.000 |

- Os períodos antigos são mais largos porque têm menos pinturas. O primeiro cobre séculos, o segundo 200 anos e o terceiro 50. Os quatro seguintes cobrem 25 anos cada.
- A amostra é balanceada: 2.000 pinturas por período, sorteadas com seed 42. São 16.000 pinturas, 20,8% das que têm ano.
- Com 8 classes do mesmo tamanho, um chute ao acaso acerta 12,5%.

**Divisão por pintor**

| Conjunto | Pinturas | Por período | Pintores |
| --- | --- | --- | --- |
| Treino | 11.200 | 1.400 | 1.107 |
| Validação | 2.400 | 300 | 223 |
| Teste | 2.400 | 300 | 223 |

- Todas as pinturas de um pintor ficam no mesmo conjunto. A amostra tem 1.553 pintores, e nenhum aparece em dois conjuntos.
- Motivo: um pintor trabalha poucas décadas e tem um jeito próprio. Se o mesmo pintor estivesse no treino e no teste, o modelo poderia acertar o período só por reconhecer o pintor. A divisão por pintor mede o que interessa: datar a obra de um pintor que o modelo nunca viu.
- Nenhuma pintura da amostra está sem pintor conhecido. Nenhum pintor da amostra aparece com duas grafias.
- Os conjuntos ficaram balanceados por período, como mostra a figura.
- Um pintor muito produtivo pesa no teste de dois períodos: Max Ernst tem 56 das 300 pinturas de teste de 1925 a 1949 (18,7%), e Sam Francis tem 46 das 300 de 1950 em diante (15,3%).

![Pinturas por período em cada conjunto](figures/a1_class_counts.png)

**Estilo e data são coisas diferentes**

- O rótulo é o ano da pintura. O que a imagem mostra é o estilo.
- Um pintor pode trabalhar em um estilo mais antigo que o seu tempo. Nesses casos a imagem aponta para um período e o rótulo para outro.
- Os anos vêm do upload do WikiArt e não foram conferidos. Datas aproximadas, como `c.1904`, entram como se fossem exatas.
- Por isso existe um teto de acerto abaixo de 100%, que este trabalho não mediu.

## Decisões técnicas

### Attention e encoder construídos do zero

Tudo está em `src/dlv/attention.py`, escrito com camadas lineares e softmax. Os módulos prontos de attention do PyTorch (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`) não são usados.

- **Scaled dot-product attention: função `scaled_dot_product_attention`.**
    - Calcula `softmax(Q K^T / sqrt(d)) V` e devolve a saída e os pesos.
    - A divisão por `sqrt(d)` evita que os produtos internos cresçam com a dimensão e saturem o softmax.
    - Aceita uma máscara opcional.
- **Multi-head attention: módulo `MultiHeadAttention`.**
    - É o módulo que usa a função acima, uma vez por head.
    - Cada head tem as suas próprias projeções de query, key e value: três camadas lineares de 192 para 64 dimensões.
    - As saídas das 3 heads são concatenadas (3 x 64 = 192) e passam por uma projeção final de 192 para 192.
    - O módulo devolve também os pesos de cada head, no formato (lote, heads, tokens, tokens).
- **Encoder: módulo `TransformerEncoderBlock`.**
    - LayerNorm, multi-head attention e soma com a entrada (residual connection).
    - LayerNorm, feedforward de duas camadas (192 para 768, GELU, 768 para 192) e outra soma com a entrada.
    - A LayerNorm vem antes de cada parte (pre-norm), como no ViT original.

**Como é testado (`tests/test_attention.py`, 13 testes em CPU)**

- A saída da função é igual à fórmula calculada à mão.
- Os pesos são não negativos e cada linha soma 1.
- Uma posição mascarada recebe peso zero. Uma linha toda mascarada não gera NaN, nem na saída nem no gradiente.
- Keys iguais dão pesos uniformes.
- O módulo tem 3 x 3 + 1 = 10 camadas lineares: uma por head e por papel, mais a projeção final.
- Mudar as projeções de uma head muda só os pesos dessa head. As outras duas ficam idênticas. Isso prova que as projeções são independentes.
- Com as últimas projeções zeradas, o bloco devolve a própria entrada. Isso prova que as residual connections existem.
- O gradiente chega a todos os parâmetros do bloco, menos aos vieses das projeções de key. O gradiente deles é zero por construção, porque o softmax ignora uma constante somada à linha.

### Vision Transformer

O modelo está em `src/dlv/vit.py`, classe `VisionTransformer`. O caminho da imagem até os logits:

1. **Patch embedding** (`PatchEmbedding`): a imagem de 64x64 é cortada em patches de 8x8 sem sobreposição. São 8 x 8 = 64 patches. Cada patch vira um vetor de 192 dimensões por uma projeção linear, feita com uma convolução de kernel 8 e passo 8.
2. **CLS token**: um vetor de 192 dimensões, aprendido no treino, é colocado no início da sequência. A sequência passa a ter 64 + 1 = 65 tokens.
3. **Positional encoding**: um vetor aprendido por posição, somado a cada token. São 65 x 192 = 12.480 parâmetros.
4. **Encoder**: 6 blocos `TransformerEncoderBlock`, com 3 heads cada e dropout de 0,1.
5. **Saída**: LayerNorm final, depois uma camada linear de 192 para 8 aplicada só ao CLS token. O resultado são 8 logits, um por período.

- Total: 2.720.840 parâmetros.
- O modelo também devolve os pesos de attention de cada bloco. A função `class_token_attention` pega a linha do CLS token e a organiza na grade de 8x8 patches.

**Como é testado (`tests/test_vit.py`, 14 testes em CPU)**

- Uma imagem de 64x64 vira 64 tokens de 192 dimensões. Cada token depende só do seu patch.
- O modelo devolve 8 logits por imagem.
- O CLS token e o positional encoding são parâmetros treináveis, e o positional encoding tem 65 posições.
- O mapa de attention lê a linha do CLS token, tem o formato da grade e soma no máximo 1.
- No modelo pré-treinado, a classification head é trocada e todas as camadas continuam treináveis.

### Por que attention precisa de positional encoding

**O argumento**

- A attention compara cada token com todos os outros por produto interno. Nenhuma parte da conta usa a posição do token na sequência.
- Se os tokens de entrada trocam de lugar, as saídas trocam de lugar do mesmo jeito, com os mesmos valores. A attention trata a entrada como um conjunto, sem ordem.
- A feedforward e a LayerNorm agem em cada token sozinho, então também não enxergam ordem.
- No ViT, o CLS token soma informação de todos os patches. Sem posição, o que ele recebe é igual para qualquer arrumação dos patches.
- O positional encoding resolve isso: soma a cada token um vetor que depende da posição. Dois patches iguais em lugares diferentes passam a ser tokens diferentes.

**O teste que prova o mecanismo**

- `test_without_positional_embedding_patch_order_does_not_matter`: no modelo sem positional encoding, embaralhar os 64 patches deixa os logits iguais.
- `test_with_positional_embedding_patch_order_matters`: no modelo com positional encoding, o mesmo embaralhamento muda os logits.

**O experimento**

- O modelo foi treinado uma segunda vez sem positional encoding. Mesma seed, mesmos dados, mesmo treino.
- Ele tem 2.708.360 parâmetros, 12.480 a menos que o modelo completo: os do positional encoding.
- Resultado no teste: 780 acertos em 2.400 com posição, 779 sem posição. Diferença de 1 pintura.
- Os dois modelos são diferentes de fato: as matrizes de confusão não são iguais. Em 1925 a 1949, por exemplo, um acerta 48 e o outro 65.
- Com 2.400 pinturas e acurácia perto de 32%, o erro padrão é de cerca de 1 ponto percentual: raiz de (0,325 x 0,675 / 2.400) = 0,0096. A diferença medida, de 0,04 ponto, é muito menor que isso. E há uma única seed por modelo.
- Leitura: nesta tarefa, a 64x64 e com este modelo pequeno, a informação de posição não ajudou de forma mensurável. Outras tarefas e resoluções não foram testadas.

**Por que isso pode acontecer aqui**

- Hipótese: as pistas de período que sobrevivem a 64x64 são em boa parte globais, como paleta de cores, contraste e textura da pincelada. Um conjunto de patches sem ordem já carrega essas pistas.

### Hiperparâmetros

**ViT treinado do zero (com e sem posição)**

- **Entrada de 64x64 e patch de 8x8.** Dá 65 tokens, o que deixa o treino rápido na T4: cerca de 7 segundos por época. As imagens reduzidas cabem na memória da GPU.
- **192 dimensões, 6 blocos, 3 heads.** Um modelo pequeno, de 2,7 milhões de parâmetros, para 11.200 imagens de treino. São 243 parâmetros por imagem. Um modelo maior tenderia a ter ainda mais overfitting.
- **AdamW, weight decay 0,05.** É o otimizador usual para Transformers. O weight decay é a regularização principal.
- **Taxa de aprendizado 0,001, com 5 épocas de subida linear e depois queda em cosseno.** Transformers treinados do zero são instáveis no começo. A subida gradual evita passos grandes enquanto os pesos ainda são aleatórios.
- **Lotes de 256 imagens.** São 44 passos por época.
- **Augmentation: recorte aleatório com margem de 4 pixels e flip horizontal.** São leves e não mudam as cores, que são pista de período.
- **Normalização com a média e o desvio padrão de cada canal, medidos no treino.**
- **Dropout 0,1. Perda: entropia cruzada. Seed 42.**
- **40 épocas.** A primeira execução usou 60 épocas e levou 27,7 minutos no total, perto do limite de 30 minutos do notebook. O corte para 40 foi por custo. Nessa primeira execução, a acurácia de validação era 32,9% na época 40 e 32,0% na época 60. Na execução final, a melhor acurácia de validação (32,2%) aparece nas épocas 27 e 37, e a época 40 fica em 31,4%. As duas execuções usam agendas de taxa de aprendizado diferentes, então a comparação é indicativa.

**ViT-B/16 pré-treinado**

- **Pesos `IMAGENET1K_V1` do torchvision.** O modelo tem 85.804.808 parâmetros.
- **Classification head trocada.** A camada final, de 1.000 saídas, deu lugar a uma camada linear nova de 8 saídas.
- **Fine-tuning de todas as camadas.** Pinturas são um domínio distante das fotos da ImageNet, e há 11.200 imagens de treino. Congelar o backbone, como na Atividade 3, poderia deixar as features presas às fotos.
- **Duas taxas de aprendizado: 0,00003 no backbone e 0,001 na camada nova.** A camada nova começa aleatória e precisa de passos grandes. O backbone já está treinado, e passos grandes apagariam o que ele sabe.
- **Entrada igual à do pré-treinamento:** redimensionar para 256, recorte central de 224x224, média e desvio padrão da ImageNet.
- **Lotes de 32 imagens e precisão mista.** Mantêm o pico de memória da GPU em 3.971,6 MB.
- **Sem augmentation e com taxa de aprendizado constante.** Foi a configuração mais simples. É um limite: ver a análise crítica.
- **2 épocas.** A primeira execução usou 3, com acurácia de validação de 49,2% na época 2 e 47,8% na época 3. O corte foi por custo. Na execução final, a perda de validação já sobe da época 1 para a 2, como mostram as curvas de treino.

**Custo da execução final**

- Notebook inteiro: 866,6 segundos (14,4 minutos) na T4.
- Pico de memória: 3.971,6 MB na GPU e 2.867,3 MB de RAM.

## Resultados

**Comparação dos três modelos no teste (2.400 pinturas)**

| Modelo | Acurácia | Acertos | Acerto com tolerância de 1 período | F1 macro | Distância média | Parâmetros | Treino |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT do zero | 32,50% | 780 | 64,0% | 0,320 | 1,49 | 2.720.840 | 277,4 s |
| ViT do zero sem posição | 32,46% | 779 | 64,5% | 0,322 | 1,47 | 2.708.360 | 280,1 s |
| ViT-B/16 pré-treinado | 47,38% | 1.137 | 80,2% | 0,458 | 0,93 | 85.804.808 | 236,0 s |
| Chute ao acaso | 12,50% | 300 | 34,4% | | 2,63 | | |

- **Acerto com tolerância de 1 período:** conta como acerto o período exato ou um vizinho.
- **Distância média:** quantos períodos, em média, separam a previsão do período real.
- **Linha do chute ao acaso:** valores calculados por conta, sem execução.
    - Os 2 períodos das pontas têm 2 respostas aceitas e os outros 6 têm 3: (2 x 2 + 6 x 3) / 64 = 34,4%.
    - A distância média entre dois períodos sorteados é 168 / 64 = 2,63.
- O modelo pré-treinado acerta 1.137 - 780 = 357 pinturas a mais que o ViT do zero: 14,9 pontos percentuais.
- Os dois modelos do zero ficam cerca de 20 pontos acima do acaso. Eles aprendem algo, mas pouco.

**Acurácia por período no teste (300 pinturas por período)**

| Período | ViT do zero | Sem posição | ViT-B/16 pré-treinado |
| --- | --- | --- | --- |
| até 1599 | 51,0% | 51,3% | 89,7% |
| 1600 a 1799 | 45,7% | 45,0% | 47,7% |
| 1800 a 1849 | 25,7% | 24,3% | 38,7% |
| 1850 a 1874 | 24,7% | 24,0% | 42,3% |
| 1875 a 1899 | 28,0% | 23,0% | 9,3% |
| 1900 a 1924 | 17,3% | 17,7% | 33,7% |
| 1925 a 1949 | 16,0% | 21,7% | 56,3% |
| 1950 em diante | 51,7% | 52,7% | 61,3% |

- Os três modelos acertam mais nas pontas. São os períodos mais diferentes dos outros, e o primeiro é o mais largo.

**Curvas de treino**

- **Os dois modelos do zero têm overfitting.**
    - A acurácia de treino sobe até o fim: 46,6% e 45,2% na época 40.
    - A acurácia de validação para perto de 31%.
    - A perda de validação do ViT do zero tem o mínimo na época 20 (1,713) e sobe até 1,781. A de treino continua caindo.
    - Diferença final entre treino e validação no ViT do zero: 46,56% - 31,42% = 15,1 pontos.
- **As curvas com e sem posição andam juntas** do início ao fim. A diferença entre elas é do tamanho da oscilação de uma época para outra. A diferença final na validação, 31,4% com posição e 30,3% sem, está dentro do ruído.
- **O modelo pré-treinado também tem overfitting, e mais rápido.** Na época 2 o treino chega a 67,7% e a validação fica em 49,0%. A perda de validação sobe de 1,332 para 1,431.
- Os pesos avaliados no teste são os da última época, nos três modelos.

![Curvas de perda e acurácia por época dos três modelos](figures/a1_histories.png)

**Onde caem os erros**

| Modelo | Período exato | Vizinho (1 período) | Longe (2 ou mais) |
| --- | --- | --- | --- |
| ViT do zero | 780 (32,5%) | 756 (31,5%) | 864 (36,0%) |
| ViT-B/16 pré-treinado | 1.137 (47,4%) | 788 (32,8%) | 475 (19,8%) |

- **ViT do zero:** os erros se espalham pela matriz. Mais de um terço das previsões cai a 2 períodos ou mais do real.
    - Há erros de séculos: 48 pinturas de 1900 a 1924 e 36 de 1925 a 1949 foram datadas como "até 1599".
    - O miolo é confuso: de 1800 a 1949, nenhum período passa de 84 acertos em 300.
- **ViT-B/16 pré-treinado:** os erros ficam junto da diagonal.
    - 1600 a 1799 perde 98 pinturas para 1800 a 1849.
    - 1900 a 1924 perde 102 para 1925 a 1949, mais do que acerta (101).
    - 1950 em diante perde 80 para 1925 a 1949.
    - "até 1599" é quase resolvido: 269 de 300.
- **O buraco em 1875 a 1899.** Das 300 pinturas desse período, o modelo pré-treinado acerta 28 (9,3%, abaixo do acaso e pior que os 28,0% do ViT do zero), manda 94 para 1850 a 1874 e 96 para 1900 a 1924. No teste inteiro, ele só responde "1875 a 1899" 64 vezes em 2.400. O modelo quase não usa essa classe e divide as pinturas dela entre os dois vizinhos. Como os erros caem nos vizinhos, a métrica de 1 período de tolerância esconde esse defeito. A acurácia por período mostra.

![Matriz de confusão do ViT do zero no teste](figures/a1_confusion_scratch.png)

![Matriz de confusão do ViT-B/16 pré-treinado no teste](figures/a1_confusion_pretrained.png)

## O que o modelo observa

**O que as figuras mostram**

- Modelo: o ViT treinado do zero, com positional encoding.
- Seis pinturas do teste: três que o modelo acertou e três que errou, de períodos diferentes. A escolha é fixa (a primeira pintura de cada período escolhido), sem seleção à mão.
- Primeira figura: attention do CLS token para cada um dos 64 patches, em cada uma das 3 heads do último bloco. É a attention que alimenta a classificação.
- Segunda figura: a mesma attention no primeiro bloco, na média das 3 heads.
- Cada painel tem a sua própria escala de cores. O vermelho marca o maior valor daquele painel. As cores não servem para comparar um painel com outro.
- A grade é de 8x8 patches, ampliada sobre a imagem. Cada mancha cobre uma região de pelo menos 8x8 pixels.

![Attention do CLS token no último bloco, uma coluna por head](figures/a1_attention_last_block.png)

**Último bloco, pintura por pintura**

| Pintura | Head 1 | Head 2 | Head 3 |
| --- | --- | --- | --- |
| 1. The Nymph of the Spring, Lucas Cranach, 1540 (acerto) | Cantos inferiores, mais forte no direito, com pontos fracos no canto superior direito e na borda esquerda | Um único ponto forte, sobre o quadril da figura deitada | Canto inferior esquerdo e borda esquerda |
| 2. Torso of a Dancer, Edgar Degas, 1899 (acerto) | Só o canto superior esquerdo, no fundo liso | Um ponto ao lado do rosto, na altura do queixo e do ombro | Base da imagem, no centro, e faixa difusa no alto à direita |
| 3. Pastorale, Ernst Wilhelm Nay, 1954 (acerto) | Base da imagem, sobre um disco verde | Borda esquerda, com faixa fraca que atravessa o centro | Borda esquerda, mais abaixo, e pontos menores à direita |
| 4. Jane Maxwell com o filho, George Romney, 1778 (erro: previsto 1800 a 1849) | Ponto forte na área escura à esquerda da mulher, junto ao casaco do rapaz, e pontos mais fracos no vestido | Um ponto no braço erguido da mulher, à direita | Canto superior direito e borda direita, no fundo |
| 5. Madame Gobillard, Edgar Degas, 1869 (erro: previsto 1900 a 1924) | Um ponto acima e à esquerda da cabeça, no fundo | Espalhada pela parede e pelas bordas, com um ponto na altura do pescoço da figura | Parte de baixo do vestido, à direita |
| 6. Milking the Cow, Yasuo Kuniyoshi, 1927 (erro: previsto 1950 em diante) | Topo do desenho | Centro inferior do desenho, e difusa no papel branco | Difusa no papel branco, mais forte na borda direita e no canto inferior direito |

A pintura 3 é abstrata. Não há figura para a attention seguir.

**Regiões que emergem**

- **Head 2 cai mais vezes sobre a figura**, nas pinturas 1, 2, 4, 5 e 6. É um ponto pequeno, de um ou dois patches, e muda de parte do corpo a cada pintura.
- **Heads 1 e 3 caem mais vezes em cantos, bordas e fundo.**
- **O padrão não é estável.** Na pintura 3 a head 2 está na borda, e na 5 ela é a mais espalhada das três. A head 1 cai sobre a cena nas pinturas 4 e 6. Seis pinturas é pouco para afirmar um papel fixo por head.
- **Não há diferença visível entre os mapas dos acertos e os dos erros.**

![Attention do CLS token no primeiro bloco, média das heads](figures/a1_attention_first_block.png)

**Primeiro bloco**

- Os mapas são manchas locais. Parecem seguir pontos de contraste forte:
    - pintura 1: borda direita no alto, ao lado do céu claro, com um ponto mais fraco no céu;
    - pintura 2: braço da figura, à esquerda;
    - pintura 3: vários pontos entre as formas coloridas;
    - pintura 4: parte de baixo do vestido branco, à esquerda;
    - pintura 5: parede no alto à esquerda;
    - pintura 6: traços escuros do desenho, sem nada no papel branco.
- No primeiro bloco os tokens ainda são quase só o patch e a posição. A attention parece seguir cor e contraste do próprio patch.

**Interpretação**

- Os mapas não mostram uma história limpa, do tipo "o modelo olha para o rosto". É o esperado para um modelo pequeno, que acerta 32,5% no teste, com patches de 8x8 em uma imagem de 64x64. Mapas com contorno de objeto são relatados em ViTs treinados com muito mais dados, sobretudo com self-supervised learning.
- A attention em cantos e fundo combina com a hipótese da seção sobre positional encoding: o modelo pode estar lendo cor e textura do fundo, que são pistas globais de período. Os mapas não provam isso.
- Limite do método: o mapa mostra para onde o CLS token olha em um bloco. Ele não mostra o quanto cada região pesou na decisão final.

## Erros por movimento artístico

- As previsões do teste foram agrupadas pela coluna `style` do dataset.
- Entram os 21 movimentos com pelo menos 20 pinturas no teste. Eles somam 2.102 das 2.400 pinturas.

**Movimentos com mais de 50 pinturas no teste**

| Movimento | Pinturas | Acurácia, ViT do zero | Acurácia, pré-treinado | Distância média, pré-treinado |
| --- | --- | --- | --- | --- |
| Romanticism | 437 | 25,6% | 34,3% | 1,09 |
| Realism | 244 | 28,3% | 34,4% | 1,05 |
| Northern Renaissance | 212 | 50,9% | 90,6% | 0,43 |
| Impressionism | 200 | 27,5% | 25,0% | 1,20 |
| Rococo | 152 | 43,4% | 49,3% | 0,57 |
| Expressionism | 128 | 14,8% | 49,2% | 1,29 |
| Surrealism | 121 | 19,0% | 54,5% | 1,19 |
| Symbolism | 98 | 16,3% | 14,3% | 1,70 |
| Post-Impressionism | 75 | 26,7% | 34,7% | 1,12 |
| Naïve Art (Primitivism) | 65 | 29,2% | 52,3% | 1,06 |
| Abstract Expressionism | 61 | 63,9% | 75,4% | 0,31 |
| Early Renaissance | 54 | 50,0% | 87,0% | 0,44 |

**Mais bem e mais mal datados pelo modelo pré-treinado**

- Os mais bem datados já aparecem na tabela. Fora dela, por terem poucas pinturas, estão High Renaissance (90,5%, 19 de 21), Lyrical Abstraction (82,6%, 19 de 23) e Art Informel (81,0%, 17 de 21). O pior é Neoclassicism, com 9,7% (3 de 31). Romanticism e Realism, com 34,3% e 34,4%, somam 681 pinturas, 28% do teste.

**Razão provável: quanto tempo o movimento durou em relação à largura dos períodos**

- Os movimentos do Renascimento cabem inteiros no período "até 1599", que é o mais largo. Reconhecer o movimento já dá o período certo.
- A abstração do pós-guerra cabe quase inteira em "1950 em diante", que não tem limite final.
- Romanticism, Realism, Impressionism e Symbolism são do século XIX e do começo do XX, onde os períodos têm 50 ou 25 anos. Um movimento que dura várias décadas fica partido em 2 ou 3 classes. O modelo reconhece o movimento e ainda precisa adivinhar a fatia.
- A distância média apoia essa leitura para o Impressionism: a acurácia é de 25,0%, mas o erro médio é de 1,20 período. O modelo erra por perto.
- O Neoclassicism é um caso diferente: o erro médio é de 2,10 períodos, ou seja, erra para longe. Uma razão plausível é que o estilo imita de propósito a pintura de séculos anteriores. É o exemplo mais direto de que estilo e data são coisas diferentes. São só 31 pinturas, então a medida é fraca.
- A hipótese não explica tudo: Baroque (42,0%) e Rococo (49,3%) cabem no período 1600 a 1799 e mesmo assim ficam abaixo de 50%. Os valores são do modelo pré-treinado.
- Esta explicação é uma hipótese. O notebook não guardou a distribuição de anos de cada movimento, então ela não foi conferida nos dados.

**Onde o pré-treinamento ajudou mais e onde não ajudou**

- Maiores ganhos, em pontos percentuais, entre os 21 movimentos: Northern Renaissance, High Renaissance (52,4% para 90,5%), Early Renaissance e Surrealism.
- Sem ganho: Impressionism, Symbolism, Baroque (48,0% para 42,0%, com 50 pinturas) e Neoclassicism (9,7% nos dois).
- Uma leitura possível, não conferida nos dados: Impressionism e Symbolism têm muitas pinturas do fim do século XIX, onde fica o buraco descrito em Resultados.

![Acurácia por movimento, ViT do zero](figures/a1_styles_scratch.png)

![Acurácia por movimento, ViT-B/16 pré-treinado](figures/a1_styles_pretrained.png)

## ViT do zero ou ViT pré-treinado

**Tabela comparativa**

| | ViT do zero | ViT-B/16 pré-treinado |
| --- | --- | --- |
| Acurácia no teste | 32,5% (780 de 2.400) | 47,4% (1.137 de 2.400) |
| Acerto com tolerância de 1 período | 64,0% | 80,2% |
| F1 macro | 0,320 | 0,458 |
| Distância média | 1,49 período | 0,93 período |
| Erros a 2 períodos ou mais | 864 | 475 |
| Parâmetros | 2.720.840 | 85.804.808 |
| Entrada | 64x64 | 224x224 |
| Épocas | 40 | 2 |
| Tempo de treino na T4 | 277,4 s | 236,0 s |
| Tempo de avaliação do teste | 0,6 s | 7,7 s |
| Dados vistos antes | nenhum | ImageNet-1k |

**Escolha para este domínio: o ViT pré-treinado**

- Acerta 14,9 pontos percentuais a mais no teste. O erro padrão de cada acurácia é de cerca de 1 ponto, então a diferença está muito acima do ruído.
- Custou menos tempo de treino, mesmo com 31,5 vezes mais parâmetros (85.804.808 / 2.720.840). O ViT do zero não tem dados suficientes: com 11.200 pinturas, tem o overfitting descrito em Resultados.

**O que pesa contra**

- O modelo é grande: avaliar o teste leva 7,7 s contra 0,6 s, e o pico de memória da GPU, de 3.971,6 MB, provavelmente vem dele.
- Ele tem um defeito que o ViT do zero não tem: quase ignora o período 1875 a 1899 (9,3% contra 28,0%).
- Os dois modelos diferem em três coisas ao mesmo tempo: pré-treinamento, tamanho e resolução de entrada (224x224 contra 64x64). O experimento não separa quanto do ganho vem de cada uma.

**Para que o resultado serve**

- Serve para uma primeira triagem: sugerir uma faixa de época para uma pintura sem data, a ser conferida por uma pessoa. Para datar sem conferência humana, o erro é alto demais.

## Pré-treinamento: BERT e ViT

| | BERT | ViT original |
| --- | --- | --- |
| Dados | Texto sem rótulo | Imagens com rótulo |
| Tipo | Self-supervised | Supervisionado |
| Tarefa | Masked language modeling e next sentence prediction | Classificação da imagem |
| De onde vem o alvo | Do próprio texto | De rótulos dados por pessoas |
| O que é maximizado | A probabilidade das palavras escondidas, dado o contexto | A probabilidade da classe correta da imagem |

**BERT**

- **Masked language modeling:** parte dos tokens da frase é escondida. O modelo precisa prever cada token escondido usando as palavras à esquerda e à direita.
- **Next sentence prediction:** o modelo recebe duas frases e diz se a segunda vem depois da primeira no texto original.
- Para ir bem nas duas tarefas, cada token precisa de uma representação que use o contexto dos dois lados.

**ViT**

- O artigo usa conjuntos rotulados grandes, como ImageNet-21k e JFT-300M. Ele mostra que, só com a ImageNet-1k, o ViT fica atrás de CNNs do mesmo porte, e que a vantagem aparece com mais dados.
- Os pesos usados aqui (`IMAGENET1K_V1`) foram treinados com classificação supervisionada na ImageNet-1k.

**Diferenças que importam**

- **Custo dos dados.** BERT aproveita texto bruto. O ViT supervisionado depende de rótulos, que são caros.
- **Onde está o sinal.** BERT recebe um alvo para cada token escondido, então aprende sobre cada parte da frase. O ViT recebe um alvo por imagem inteira.
- **O que a representação guarda.** O ViT supervisionado guarda o que separa as classes do pré-treinamento. Na ImageNet, isso é o objeto da foto. O estilo da imagem pode ser descartado, porque não ajuda a dizer se a foto é de um cão ou de um carro.
- **Ligação com este trabalho.** A tarefa aqui é quase o oposto: o objeto importa pouco e o estilo importa muito. Isso ajuda a entender por que o fine-tuning parou em 47%.

**O equivalente de BERT em imagens: masked image modeling**

- **BEiT:** esconde patches e prevê, para cada um, um token visual discreto produzido por um tokenizador de imagens.
- **MAE:** esconde a maior parte dos patches e reconstrói os pixels que faltam. O encoder só processa os patches visíveis.
- Os dois são self-supervised e dispensam rótulos, como BERT.
- O que maximizam: o acerto na previsão das partes escondidas (tokens visuais no BEiT, pixels no MAE). Para reconstruir um pedaço de pintura, textura e cor importam, o que parece mais próximo do que a datação precisa. Isso é uma hipótese e não foi testado aqui.

## DeiT e Swin Transformer

**DeiT: o problema dos dados**

- Limite do ViT original: precisa de conjuntos enormes de imagens rotuladas para superar CNNs, porque não traz as suposições de uma CNN sobre imagens e precisa aprender tudo dos dados.
- O que o DeiT resolve: treinar um ViT competitivo só com a ImageNet-1k, sem dados extras.
- Como:
    - augmentation forte e regularização forte no treino;
    - um distillation token, que entra na sequência ao lado do CLS token. Ele aprende a imitar a resposta de um modelo professor, em geral uma CNN. O aluno herda parte das suposições da CNN.
- O encoder é o do ViT, com um token a mais. O que muda é a receita de treino.
- **Ligação com este trabalho.** O ViT do zero daqui está no regime que o DeiT ataca, em escala bem menor: 11.200 imagens, augmentation leve, nenhum professor, e o overfitting descrito em Resultados. A receita do DeiT indica o que tentar: augmentation mais forte e distillation a partir de um modelo melhor, como o ViT-B/16 com fine-tuning.

**Swin Transformer: o problema da escala e do custo**

- Limites do ViT original:
    - uma única grade de patches, com a mesma resolução do primeiro ao último bloco. Não há mapas de features em várias escalas;
    - attention global, de todos os tokens com todos. O custo cresce com o quadrado do número de tokens, o que pesa em imagens grandes.
- O que o Swin resolve:
    - **Hierarquia.** Começa com patches pequenos e junta patches vizinhos a cada estágio. A resolução cai e a dimensão sobe, como nos estágios de uma CNN.
    - **Janelas locais.** A attention é calculada dentro de janelas de tamanho fixo. O custo passa a crescer de forma linear com o tamanho da imagem.
    - **Janelas deslocadas.** De uma camada para a seguinte, as janelas mudam de lugar. Isso liga janelas vizinhas e deixa a informação atravessar a imagem.
- Resultado: um backbone de uso geral, que serve também para detecção e segmentação, tarefas que pedem features em várias escalas.
- **Ligação com este trabalho.**
    - O ViT daqui tem uma grade única de 8x8 patches, sem hierarquia. Cada patch de 8x8 pixels mistura pincelada, contorno e cor em um só vetor desde a primeira camada. É o que o Swin muda.
    - O custo quadrático não pesou aqui: são 65 tokens, ou 65 x 65 = 4.225 pares por head. Em uma pintura em alta resolução, onde a pincelada aparece, pesaria.

## Quando usar ViT e quando usar CNN

**O que cada arquitetura supõe**

- **CNN:** supõe que pixels vizinhos se relacionam e que o mesmo filtro serve em qualquer lugar da imagem. Essas suposições valem para a maioria das imagens e economizam dados.
- **ViT:** supõe quase nada. Qualquer patch pode se relacionar com qualquer outro desde o primeiro bloco. Ganha flexibilidade e precisa de mais dados para aprender o que a CNN já traz pronto.

**ViT é a melhor escolha quando**

- existe muito dado, ou um pré-treinamento grande que possa ser aproveitado;
- a tarefa depende de relações entre partes distantes da imagem, como a composição da cena;
- o modelo vai ser combinado com texto ou outras modalidades, onde o Transformer é a arquitetura comum.

**CNN é preferível quando**

- o dataset é pequeno e não há bons pesos pré-treinados para o domínio;
- a pista é local: textura, borda, padrão que se repete;
- o limite é memória, tempo ou imagem em alta resolução.

**Neste domínio**

- Na Atividade 3, uma ResNet-50 pré-treinada, com o backbone congelado, chegou a 100% em um teste de 265 fotos de 7 classes de objetos, um domínio próximo da ImageNet. Aqui o ViT-B/16 pré-treinado ficou em 47,4%. A datação de pinturas é distante, e boa parte da dificuldade vem do domínio e da tarefa.
- As pistas de data são de dois tipos:
    - locais: pincelada, craquelado, textura da tinta. Favorecem a CNN;
    - globais: composição, paleta, uso do espaço. Favorecem o ViT.
- O experimento sem positional encoding sugere que, a 64x64, o ViT do zero não tirou proveito do arranjo espacial.
- **O que não foi medido.** Nenhuma CNN foi treinada nas pinturas, então este trabalho não compara o ViT-B/16 com uma ResNet-50 com fine-tuning.
- **Recomendação para este domínio:**
    - treinar do zero com cerca de 11 mil imagens: uma CNN pequena é a aposta mais segura, pelas suposições que economizam dados. É uma expectativa, sem medida aqui;
    - com pesos pré-treinados disponíveis: o ViT-B/16 com fine-tuning é a escolha apoiada pelos dados deste trabalho.

## Análise crítica e próximos passos

**Limites do experimento**

- **Uma seed por modelo.** A variação entre execuções não foi medida. A diferença de 14,9 pontos entre o modelo pré-treinado e o do zero é grande o bastante para não depender disso. A comparação com e sem posição (1 pintura de diferença) depende.
- **Uma amostra.** São 16.000 pinturas sorteadas uma vez, de 76.881 com ano. Outra amostra daria outros pintores no teste.
- **Grupos pequenos.** Alguns grupos de movimento têm pouco mais de 20 pinturas no teste.
- **Fine-tuning curto e simples.** Duas épocas, sem augmentation, taxa de aprendizado constante. Um treino mais cuidadoso provavelmente passaria de 47,4%, mas isso não foi medido.
- **Rótulos.** Os anos não foram conferidos, e estilo e data podem divergir, como descrito em Problema.
- **Períodos de larguras diferentes.** Errar por 1 período depois de 1850 é errar cerca de 25 anos. Entre "até 1599" e "1600 a 1799" pode ser errar séculos. A distância em períodos não é uma distância em anos.

**O que mudaria**

- Repetir cada modelo com 3 a 5 seeds e informar média e desvio.
- Guardar os pesos da melhor época de validação, em vez dos da última.
- No fine-tuning: augmentation leve, queda da taxa de aprendizado e mais épocas com parada pela validação.
- Olhar o período 1875 a 1899 de perto: ver quais pintores e movimentos caem nele no treino e no teste.
- Treinar o ViT do zero a 128x128 ou mais, com augmentation mais forte, e repetir o teste sem posição. A hipótese a testar: com mais resolução e menos overfitting, a posição passa a fazer diferença.
- Fazer o fine-tuning de uma ResNet-50 nas mesmas pinturas, para ter a comparação com uma CNN neste domínio.
- Testar pesos de masked image modeling (MAE) como ponto de partida.
- Tratar a tarefa como ordinal ou como regressão do ano, já que os períodos têm ordem e errar por um vizinho custa menos que errar por séculos.
- Guardar o ano de cada pintura do teste, para medir o erro em anos e conferir a hipótese dos movimentos longos.
