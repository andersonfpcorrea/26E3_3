# Atividade 4.1: Triagem de COVID-19 em radiografias de tórax

## O projeto analisado

![Atividade 4.1: dos dados à avaliação dos três braços](diagrams/a4_pipeline.png){: width="499"}

- Um grupo anterior montou um sistema de triagem de pneumonia e COVID-19 em radiografias de tórax e encerrou o projeto com resultados "promissores". O time clínico relatou que o modelo "ignora casos positivos".
- Dados: 1.200 radiografias em três classes: Normal (840), Pneumonia (240) e COVID-19 (120). Divisão 80/20 sem estratificação.
- Modelo: ResNet-18 sem pré-treinamento, SGD, taxa de aprendizado fixa de 0,01, 15 épocas, sem augmentation.
- Avaliação: só acurácia global. Resultado reportado: 93% no treino e 61% na validação.

**O que este trabalho fez com o caso**

- Reproduziu o projeto com as mesmas configurações, em imagens públicas, mediu o que o time não reportou e testou uma GAN condicional como resposta à falta de imagens de COVID-19.
- A reprodução não repete os 93% e 61%. As imagens são outras e a entrada tem 64x64 pixels. Ela serve para mostrar o que a métrica do time escondia.

**Sobre a referência do enunciado**

- O enunciado atribui a um estudo da Scientific Reports a observação de que modelos com accuracy alta tinham recall abaixo de 60% na classe COVID-19.
- O artigo que abre no endereço citado é "Automated COVID-19 detection with convolutional neural networks", de Dumakude e Ezugwu (2023). Ele classifica tomografias em COVID-19, pneumonia e normal, trata o desbalanceamento com upsampling e reporta precision, recall, specificity, accuracy e F1. Eu não encontrei nele a observação sobre recall abaixo de 60%, então ela não é usada aqui como dado.

## Diagnóstico: problemas técnicos e impacto clínico

A reprodução do projeto serve de evidência. Os números dela, todos da mesma execução:

| Medida | Valor |
| --- | --- |
| Acurácia de treino na última época | 100% |
| Acurácia de validação na última época | 89,2% (214 de 240) |
| Acurácia de quem sempre responde Normal, na mesma validação | 72,5% (174 de 240) |
| Recall de COVID-19 na validação do projeto | 0,41 (9 de 22) |
| Recall de COVID-19 no teste real | 0,44 (133 de 300) |
| Recall de Normal e de Pneumonia no teste real | 0,97 e 0,94 |

![Projeto reproduzido: perda e acurácia por época](figures/a4_original_history.png)

### 1. Desbalanceamento sem tratamento

**O que está errado**

- Normal tem 70% das imagens e COVID-19 tem 10%. São 7 imagens de Normal para cada imagem de COVID-19.
- A perda dá o mesmo peso a cada imagem. Errar todas as imagens de COVID-19 custa pouco na média, então o modelo aprende a preferir Normal.
- No teste real da reprodução, 154 das 300 imagens de COVID-19 foram classificadas como Normal.

**Impacto clínico**

- O erro mais comum é o pior para uma triagem: um caso positivo recebe o rótulo Normal e sai da fila de prioridade.
- O paciente espera mais pelo laudo e pode circular sem isolamento nesse período.

### 2. Divisão sem estratificação e validação pequena

**O que está errado**

- Sem estratificação, o número de imagens de COVID-19 na validação depende do sorteio. Na reprodução caíram 22. Com estratificação seriam 24.
- Com 22 imagens, cada imagem vale 4,5 pontos percentuais de recall. Uma diferença de duas ou três imagens muda a conclusão.

**Impacto clínico**

- O hospital pode receber um modelo aprovado por duas ou três imagens de sorte. Na fila real, ele deixa passar mais casos de COVID-19 do que o número apresentado indicava, e a equipe só percebe depois, pelos laudos.

### 3. Acurácia global como única métrica

**O que está errado**

- A acurácia mistura as três classes e é dominada pela maior.
- Na reprodução, a acurácia de validação foi 89,2% e parece aceitável. Responder sempre Normal já daria 72,5%.
- Na mesma validação, a recall de COVID-19 foi 0,41. O modelo perdeu mais da metade dos casos da classe que motivou o projeto.

**Impacto clínico**

- É a queixa do time clínico: o modelo "ignora casos positivos". A métrica escolhida não podia mostrar o problema.

### 4. Sem pré-treinamento em um conjunto pequeno

**O que está errado**

- A ResNet-18 tem cerca de 11 milhões de parâmetros. O treino tinha 960 imagens, 98 delas de COVID-19.
- Com pesos aleatórios, a rede precisa aprender bordas e texturas a partir dessas imagens. Ela decora o treino: 100% de acurácia na época 9 da reprodução.
- Pesos da ImageNet dão filtros iniciais prontos. Radiografia é um domínio distante, então o caso pede fine-tuning com taxa de aprendizado baixa. Aqui todas as camadas foram treinadas com taxa de aprendizado baixa, e treinar só os últimos blocos, como a regra da Atividade 3 indica para dados pequenos e distantes, é a alternativa a testar.

**Impacto clínico**

- O impacto é o descrito no problema 5: os dois vêm de um modelo que decorou o treino.

### 5. Overfitting visível, taxa de aprendizado fixa e sem critério de parada

**O que está errado**

- O time reportou 93% no treino e 61% na validação: 32 pontos de diferença. O número já dizia que o modelo não generalizava.
- A reprodução mostra o mesmo padrão em escala menor. A perda de validação teve o mínimo na época 6 (0,304) e subiu até 0,352 na época 15, enquanto a de treino caiu para 0,009.
- SGD sem momentum, com taxa de aprendizado fixa de 0,01 e número fixo de épocas. Sem scheduler, sem weight decay, sem early stopping e sem escolha de checkpoint pela validação. O modelo entregue foi o da última época.

**Impacto clínico**

- O modelo decorou 98 imagens de COVID-19 e foi entregue depois do melhor ponto. Um paciente novo, de outro equipamento ou com outro posicionamento, tem a radiografia marcada como Normal e vai para o fim da fila de laudos. O time clínico só descobre quando os casos perdidos aparecem.

### 6. Sem augmentation

**O que está errado**

- Radiografias variam em posição do paciente, rotação leve, contraste e exposição. Sem augmentation, o modelo vê cada imagem sempre igual.
- Com menos de 100 imagens de COVID-19, isso facilita decorar detalhes sem relação com a doença, como marcações na borda da imagem.

**Impacto clínico**

- A taxa de casos perdidos passa a variar com o equipamento e com o posicionamento do paciente.

### 7. Sem conjunto de teste independente

**O que está errado**

- O mesmo conjunto de validação serviu para acompanhar o treino e para reportar o resultado. Não havia teste separado.
- Na reprodução, a acurácia no teste real, que tem 300 imagens por classe, foi 78,6%, abaixo da de validação. A recall de COVID-19 foi parecida nos dois. A acurácia caiu porque o teste tem a mesma quantidade de cada classe e a validação é dominada por Normal.

**Impacto clínico**

- A proporção de casos positivos em um hospital muda com a fase da epidemia. Um número medido em uma proporção fixa não informa o que acontece em outra.

### 8. Cópias de imagens e pacientes repetidos

**O que está errado**

- O projeto não descreve verificação de cópias nem divisão por paciente.
- Na base pública usada aqui havia 54 arquivos com conteúdo idêntico a outro.
- Um paciente internado faz várias radiografias. Se imagens do mesmo paciente caem no treino e na validação, o modelo é avaliado em alguém que já viu.

**Impacto clínico**

- Um paciente internado tem radiografias no treino e na validação. O modelo acerta esse paciente e a recall reportada sobe. Com pacientes novos na fila, mais casos saem como Normal do que a equipe espera, e ela confia em um resultado negativo que não merece essa confiança.

## A base de dados

- Fonte: COVID-19 Radiography Database (`tawsifurrahman/covid19-radiography-database`), pública no Kaggle.
- Entraram só as radiografias das classes Normal, Pneumonia viral e COVID-19, em tons de cinza e reduzidas para 64x64. As máscaras de pulmão que acompanham a base nunca são lidas.
- 15.153 arquivos de radiografia. 54 cópias exatas removidas pelo conteúdo do arquivo. Sobram 15.099 imagens distintas.

| Classe | Base limpa | Conjunto do projeto | Treino | Validação | Teste real |
| --- | --- | --- | --- | --- | --- |
| Normal | 10.191 | 840 | 672 | 168 | 300 |
| Pneumonia | 1.338 | 240 | 192 | 48 | 300 |
| COVID-19 | 3.570 | 120 | 96 | 24 | 300 |
| Total | 15.099 | 1.200 | 960 | 240 | 900 |

![Número de imagens por classe na base limpa, no conjunto do projeto e no teste real](figures/a4_class_counts.png)

- **Conjunto do projeto:** 1.200 imagens sorteadas com a seed 42, nas quantidades do projeto que falhou.
- **Teste real:** 300 imagens por classe, sorteadas entre as que sobraram. Nenhuma imagem está nos dois conjuntos. A base tem milhares de imagens a mais que o projeto, então a recall pode ser medida em 300 imagens de COVID-19. A validação tem 24.
- **Treino e validação:** divisão 80/20 estratificada do conjunto do projeto. A reprodução do projeto original usa a divisão dele, sem estratificação.
- **Controles de vazamento:** a GAN só vê o treino. Validação e teste nunca recebem imagem gerada. Todos os modelos são medidos no mesmo teste real.
- Limite: só cópias exatas foram removidas, e os arquivos não trazem identificador de paciente. O problema 8 do diagnóstico vale também para este estudo.

## GAN condicional para a classe COVID-19

**Por que GAN condicional**

- O objetivo é gerar imagens de uma classe específica. Com o rótulo como entrada, uma única rede aprende as três classes e usa as 864 imagens de Normal e Pneumonia para aprender a anatomia do tórax, que as 96 de COVID-19 sozinhas não ensinam.
- Uma CycleGAN tem dois generators e dois discriminators e custa mais tempo na T4. Ficou fora.

**Generator**

- Entrada: ruído de 100 dimensões e o rótulo da classe.
- Condicionamento: o rótulo passa por um embedding de 100 dimensões, concatenado ao ruído. A rede sabe a classe desde a primeira camada.
- Cinco convoluções transpostas levam de 1x1 a 64x64, com 256, 128, 64 e 32 canais, batch normalization e ReLU. A saída tem 1 canal e tanh, na faixa de -1 a 1.

**Discriminator**

- Entrada: a imagem de 64x64 e o rótulo.
- Condicionamento: o rótulo vira um mapa aprendido de 64x64, empilhado na imagem como segundo canal. O discriminator julga se a imagem é real para aquela classe.
- Quatro convoluções com stride 2 levam de 64x64 a 4x4, com 32, 64, 128 e 256 canais, batch normalization a partir da segunda convolução e LeakyReLU. Uma convolução final produz um logit.

**O loop adversarial, a cada batch**

1. Passo do discriminator: recebe as imagens reais do batch com seus rótulos e imagens geradas com os mesmos rótulos. As geradas são produzidas sem gradiente, então esse passo não altera o generator. A perda é a entropia cruzada binária (BCE): alvo 1 para as reais e 0 para as geradas.
2. Passo do generator: sorteia ruído novo, gera imagens e passa pelo discriminator já atualizado, com os pesos dele congelados. A perda é a BCE com alvo 1, a forma non-saturating: o generator ganha quando o discriminator chama as geradas de reais.
3. Cada rede tem o seu otimizador Adam, com taxa de aprendizado 0,0002 e betas (0,5, 0,999). Um passo de cada rede por batch.

- Treino: 286 épocas, batch de 64, 15 batches por época, 0,67 segundo por época na T4.
- O número de épocas veio do limite de tempo do notebook, que rodou em 23,0 minutos com pico de 656,5 MB de GPU.

## Instabilidade do treinamento e mitigação

### Como a instabilidade foi medida

- **Diversidade:** diferença média de pixels entre pares de imagens geradas da mesma classe, com ruído fixo, a cada época. Perto de zero quer dizer que o generator produz quase a mesma imagem para qualquer ruído: mode collapse.
- **Razão:** diversidade gerada nas últimas 29 épocas dividida pela diversidade das imagens reais de treino da classe (0,197 em Normal, 0,160 em Pneumonia, 0,219 em COVID-19). Razão abaixo de 0,5 marca a classe como colapsada.
- **Divergência:** perda não finita, ou perda do discriminator abaixo de 0,2 no fim, sinal de que ele venceu e o generator ficou sem gradiente útil.
- A época 1 não é interpretada. Nela a medida de diversidade não é confiável.

### Diagnóstico da versão 1

A versão 1 é o baseline: a rede e o loop descritos acima, com os batches sorteados como vêm dos dados.

![Diversidade por época, um painel por classe, nas duas versões. Linha grossa: média das três seeds. Linhas finas: cada seed. Linha tracejada: diversidade das imagens reais](figures/a4_gan_diversity.png)

- **Mode collapse no começo, em todas as seeds.** A diversidade fica entre 0,01 e 0,03 até cerca da época 50 e só passa de 0,05 entre as épocas 53 e 101, conforme a seed e a classe. Na grade de imagens, as oito imagens de cada classe são iguais nas épocas 9 e 28.
- **Saída do colapso desigual entre classes.** Normal sobe perto da época 50 e chega ao nível das imagens reais. Pneumonia e COVID-19 sobem depois e param abaixo da linha das reais, COVID-19 mais longe.
- **Sem divergência.** Nenhuma perda não finita. Na seed 42, a perda do discriminator termina em 0,47 e a do generator em 3,10. No gráfico, o escore médio fica perto de 0,82 para imagens reais e 0,18 para geradas no fim do treino.
- **A classe minoritária é a menos diversa.** A razão média de COVID-19 foi 0,439, contra 0,956 em Normal e 0,890 em Pneumonia.
- **Colapso de COVID-19 em duas de três seeds.** As razões foram 0,412 (seed 42), 0,602 (seed 43) e 0,303 (seed 44), com o limite em 0,5. Na seed 42 a curva chega a cerca de 0,17 perto da época 155 e termina perto de 0,09.

![Versão 1, seed 42: imagens geradas com o mesmo ruído em seis épocas](figures/a4_gan_snapshots_v1.png)

![Perdas e escores do discriminator, seed 42, nas duas versões](figures/a4_gan_curves.png)

**Causa provável**

- As imagens falsas de cada batch recebem os rótulos do batch real. COVID-19 tem 96 das 960 imagens, então só cerca de 10% das atualizações do generator são dessa classe.

### Quatro mitigações tentadas

Três foram testadas em execuções anteriores do notebook e saíram dele. Os números são dessas execuções, guardados no arquivo de resultados como `earlier_executions`. São razões de diversidade em COVID-19, com uma seed por versão. Nas três execuções, o baseline BCE teve 0,66, 0,56 e 0,65.

1. **Spectral normalization com hinge loss e taxas de aprendizado diferentes para as duas redes: falhou.** COVID-19 ficou colapsada nas três execuções: 0,04, 0,04 e 0,18.
2. **Augmentation diferenciável (translação e brilho aleatórios antes do discriminator): falhou.** Testada na execução 3, teve 0,45 e ficou abaixo do baseline nas três classes.
3. **WGAN-GP: falhou.** Com uma seed parecia a melhor versão: 0,76 e 0,91 nas execuções 2 e 3. Em uma execução anterior com três seeds e 240 épocas, a razão de COVID-19 foi 0,485 ± 0,370, contra 0,591 ± 0,051 no baseline. Das três seeds, a 43 terminou colapsada (razão 0,061).
4. **Batches balanceados por classe: melhora parcial.** É a versão 2 do notebook, descrita abaixo.

**Por que estabilizadores padrão podem falhar aqui**

- Hipótese, não testada em separado: o treino é curto para essas técnicas. São 960 imagens e 15 batches por época, 4.290 passos em 286 épocas. Elas foram propostas para datasets com dezenas de milhares de imagens e treinos muito mais longos. Com poucos passos, o sinal que chega ao generator pode ficar fraco, e ele não sai do colapso inicial.
- Os valores mudam de uma execução para outra. O baseline com 32 canais teve 0,56 na execução 2 e 0,65 na execução 3, com uma seed em cada. Uma hipótese é que a GPU não repete o treino de uma GAN bit a bit e que pequenas diferenças no começo crescem. O arquivo de resultados não registra se as duas execuções tinham as mesmas configurações. Por isso a comparação final usa três seeds e olha média e desvio padrão.

### Versão 2: batches balanceados por classe

- Mesma rede, mesma perda e mesmas taxas de aprendizado da versão 1. Muda só o sorteio: cada imagem é sorteada com reposição, com probabilidade inversa à frequência da sua classe. Cada classe passa a ter um terço esperado de cada batch.
- O número de batches por época fica igual. A mudança ataca a causa apontada no diagnóstico.

**Razão de diversidade, três seeds (média ± desvio padrão amostral)**

| Classe | Versão 1 | Seeds colapsadas | Versão 2 | Seeds colapsadas |
| --- | --- | --- | --- | --- |
| Normal | 0,956 ± 0,032 | 0 de 3 | 0,522 ± 0,086 | 1 de 3 |
| Pneumonia | 0,890 ± 0,092 | 0 de 3 | 0,588 ± 0,094 | 1 de 3 |
| COVID-19 | 0,439 ± 0,151 | 2 de 3 | 0,700 ± 0,043 | 0 de 3 |

- COVID-19 por seed: versão 1 com 0,412, 0,602 e 0,303. Versão 2 com 0,698, 0,744 e 0,658.

**O que melhorou**

- Nesta execução, a diversidade média de COVID-19 subiu 0,261, de 0,439 para 0,700. As três seeds da versão 2 (0,658 a 0,744) ficaram acima das três da versão 1 (0,303 a 0,602).
- A estabilidade da classe minoritária entre seeds. O desvio padrão de COVID-19 caiu de 0,151 para 0,043, cerca de 3,5 vezes menor.
- Nenhuma seed da versão 2 ficou colapsada em COVID-19. Na versão 1, duas de três ficaram (0,412 e 0,303).

**O que não foi mostrado**

- Que a melhora venha só do balanceamento, nem o tamanho dela. A execução anterior deste notebook, com as mesmas configurações e as mesmas seeds, deu 0,577 ± 0,106 na versão 1 e 0,660 ± 0,021 na versão 2: um ganho de 0,083, menor que o desvio padrão da versão 1. Nesta execução o ganho foi 0,261. A versão 2 mudou pouco entre as duas execuções (0,660 e 0,700). A versão 1 mudou muito (0,577 e 0,439) e ficou abaixo das três execuções anteriores do baseline com uma seed (0,66, 0,56 e 0,65). Parte da distância desta execução vem da variação entre execuções da GAN.
- O que se repetiu nas duas execuções: nenhuma seed da versão 2 colapsou em COVID-19 (0 de 6), contra 3 de 6 na versão 1, e o desvio padrão da versão 2 foi o menor. Os números da execução anterior não estão no arquivo de resultados desta execução.

**O que custou**

- Normal e Pneumonia perderam diversidade, como mostra a tabela. Cada imagem dessas classes passou a aparecer em menos atualizações.
- No gráfico, a versão 2 sai do colapso mais tarde: entre as épocas 84 e 119, contra 53 a 101 na versão 1. Na seed 43, as três classes ficam colapsadas até perto da época 118.
- Nenhuma das duas versões diverge. As curvas de perda da seed 42 são parecidas.

![Versão 2, seed 42: imagens geradas com o mesmo ruído em seis épocas](figures/a4_gan_snapshots_v2.png)

**Versão escolhida para o experimento de recall**

- Regra fixada antes do experimento: a versão com a maior razão média de diversidade em COVID-19 nas três seeds. A regra escolheu a versão 2.

## As imagens geradas são cópias?

- Uma GAN treinada com 96 imagens pode decorar essas imagens, e cópias não acrescentam nada ao classificador.
- Medida: para cada imagem gerada de COVID-19, a diferença média de pixels até a imagem real de COVID-19 mais próxima no treino. A régua é a mesma conta feita com as 24 imagens reais de COVID-19 da validação.

| Distância até a imagem de treino mais próxima | Imagens | Média | Mediana | Mínimo |
| --- | --- | --- | --- | --- |
| Geradas pela versão 2 (três seeds) | 1.728 | 0,119 | 0,119 | 0,077 |
| Reais de validação | 24 | 0,105 | 0,099 | 0,072 |

- Nenhuma imagem gerada ficou mais perto do treino que a imagem real de validação mais próxima. A parcela foi 0. Não há sinal de cópia.
- As geradas ficam um pouco mais longe do treino que as reais de validação. Parte dessa distância é ruído.
- Segundo controle: balancear poderia aumentar a diversidade só por gerar ruído. A distância média por seed foi 0,112, 0,128 e 0,107 na versão 1 e 0,125, 0,098 e 0,134 na versão 2. As médias são 0,116 e 0,119: a diferença é menor que a variação entre seeds.

![Primeira linha: COVID-19 reais do treino. Demais linhas: imagens geradas por cada versão e seed](figures/a4_real_vs_synthetic.png){: width="499"}

**O que se vê nas imagens**

- As reais são nítidas e variam bastante em contraste, enquadramento e posição do paciente.
- As geradas mostram o contorno do tórax e dos pulmões, mas são granuladas, sem costelas nem detalhes finos. Dentro de uma mesma seed, as oito são parecidas entre si na versão 1: o mesmo tórax com pequenas mudanças de brilho. Na versão 2 há mais variação, sobretudo na seed 43.
- Limite da medida: distância de pixels em 64x64 detecta cópia exata ou quase exata. Ela não mede se a imagem gerada mostra os sinais que um radiologista procura.

## Efeito das imagens sintéticas na recall

**Classificador, igual nos três braços**

- ResNet-18 com pesos da ImageNet, todas as camadas treinadas. Adam, taxa de aprendizado 0,0001, 10 épocas, batch de 32.
- Entrada de 64x64 ampliada para 128. Augmentation leve: translação de até 4 pixels e rotação de até 7 graus. Sem flip horizontal, porque esquerda e direita importam em uma radiografia de tórax.

**Os três braços**

- **Braço 1, só reais:** as 960 imagens de treino.
- **Braço 2, reais e sintéticas:** as 960 mais 576 imagens geradas de COVID-19, o que iguala COVID-19 a Normal (672 cada). O classificador da seed i usa a GAN da seed i.
- **Braço 3, reais com pesos de classe:** as 960 imagens, com a perda ponderada pelo inverso da frequência de cada classe. É a alternativa sem GAN.
- Três seeds por braço (42, 43, 44). Medida no teste real, 300 imagens por classe.

**Recall no teste real (média ± desvio padrão amostral, três seeds)**

| Braço | Normal | Pneumonia | COVID-19 | COVID-19 por seed |
| --- | --- | --- | --- | --- |
| 1. Só reais | 0,994 ± 0,005 | 0,873 ± 0,046 | 0,586 ± 0,138 | 0,713 / 0,440 / 0,603 |
| 2. Reais e sintéticas | 0,992 ± 0,004 | 0,916 ± 0,033 | 0,571 ± 0,095 | 0,640 / 0,610 / 0,463 |
| 3. Reais com pesos de classe | 0,977 ± 0,015 | 0,948 ± 0,027 | 0,787 ± 0,074 | 0,810 / 0,703 / 0,847 |

**Nesta execução, o braço com pesos de classe fica acima dos outros dois por mais que o desvio padrão de cada um. Na execução anterior, com as mesmas configurações, o braço com pesos de classe também era o mais alto, mas a diferença ficava dentro do desvio. Os braços 1 e 2 não se distinguem.**

![Recall por classe nos três braços, com o desvio padrão entre seeds](figures/a4_recall_arms.png)

- Braço 2 menos braço 1: -0,014. O desvio padrão do braço 1 é 0,138.
- Braço 3 menos braço 1: 0,201, acima de 0,138.
- Braço 3 menos braço 2: 0,216. Os desvios padrão dos braços 2 e 3 são 0,095 e 0,074.
- Execução anterior, mesmas seeds: 0,583 ± 0,160, 0,663 ± 0,044 e 0,736 ± 0,102. O braço 3 menos o braço 1 dava 0,153, abaixo do desvio padrão do braço 1. O braço 2 ficava 0,080 acima do braço 1 e nesta execução ficou 0,014 abaixo: o sinal trocou.
- Com três seeds a separação não é total: a pior seed do braço 3 (0,703) ficou abaixo da melhor seed do braço 1 (0,713).
- Nesta execução, o braço 3 teve o menor desvio padrão (0,074). Na anterior foi o braço 2.

**Confusão que não foi controlada**

- O braço 2 treina com 1.536 imagens contra 960 nos outros dois. Nas mesmas 10 épocas, são 1,6 vez mais atualizações de pesos.
- Qualquer diferença do braço 2, a favor ou contra, poderia vir do treino mais longo. O experimento não separa as duas coisas.

**Para onde vão os casos perdidos (900 imagens de COVID-19: 300 por seed)**

| Braço | Acertos | Classificadas como Normal | Classificadas como Pneumonia |
| --- | --- | --- | --- |
| 1. Só reais | 527 | 356 | 17 |
| 2. Reais e sintéticas | 514 | 376 | 10 |
| 3. Reais com pesos de classe | 708 | 176 | 16 |

- Quase todo caso perdido de COVID-19 vira Normal, nos três braços.
- A precision de COVID-19 fica entre 0,98 e 0,99 nos três braços. Quando o modelo diz COVID-19, quase sempre acerta. Ele diz pouco.
- A precision de Normal é 0,661, 0,667 e 0,802 no teste balanceado. Um resultado Normal desses modelos não é confiável como resultado negativo.
- Mesmo o melhor braço perde em média 21% dos casos de COVID-19 do teste.

**Conclusão para o plano**

- Pesos de classe custam uma linha de código e nenhum treino extra. São a primeira coisa a tentar, junto com um protocolo correto.
- Este estudo não mostrou que a augmentation sintética aumenta a recall: a do braço 2 (0,571) ficou 0,014 abaixo da do braço 1 (0,586), diferença muito menor que a variação entre seeds. O braço com pesos de classe teve a maior recall de COVID-19 nas duas execuções (0,736 e 0,787).

## Plano de melhoria integrado

| Problema do diagnóstico | Ação |
| --- | --- |
| 1. Desbalanceamento | Pesos de classe na perda como primeira medida. Coletar mais casos reais de COVID-19, que é a correção de fato. |
| 2. Divisão sem estratificação | Divisão estratificada. Validação cruzada com pelo menos 5 partes, para reportar média e intervalo. |
| 3. Só acurácia | Métrica principal: recall de COVID-19 em uma specificity fixa. Reportar também precision, F1 e matriz de confusão por classe, com intervalo de confiança. |
| 4. Sem pré-treinamento | ResNet-18 pré-treinada, com fine-tuning e taxa de aprendizado baixa. Comparar com pesos pré-treinados em radiografias, se houver. |
| 5. Overfitting e otimização | Adam ou SGD com momentum, scheduler de taxa de aprendizado, weight decay, early stopping e escolha de checkpoint pela recall de validação. |
| 6. Sem augmentation | Translação, rotação leve, brilho e contraste. Sem flip horizontal. |
| 7. Sem teste | Teste separado desde o início, usado uma única vez, e o teste externo descrito abaixo. |
| 8. Cópias e pacientes | Remover cópias exatas e quase-duplicatas. Dividir por identificador de paciente. |

**Modelo**

- Entrada em resolução maior que 64x64. É provável que achados sutis de COVID-19 em radiografia se percam nessa redução.
- Limiar de decisão ajustado na validação para atingir a recall pedida.

**Augmentation sintética**

- Entra só depois das ações acima, como experimento, com os controles de vazamento e a verificação de cópias deste estudo.
- Comparação controlada: o braço com sintéticas contra um braço de imagens reais com o mesmo número de atualizações e contra o braço com pesos de classe, com pelo menos 5 seeds.
- Critério para adotar: recall de COVID-19 maior que a do braço com pesos de classe, com diferença acima da variação entre seeds, no teste externo.

**Critério para adoção clínica**

Os valores abaixo são propostas, a discutir e fechar com o time clínico.

1. **Teste externo.** Radiografias de outro hospital, com outros equipamentos, separadas por paciente e nunca usadas no desenvolvimento.
2. **Recall mínima em specificity fixa.** Proposta: recall de COVID-19 de pelo menos 0,90, com o limite inferior do intervalo de confiança de 95% acima de 0,85, em specificity de pelo menos 0,80. O time clínico define os valores conforme o custo de um caso perdido e a carga de revisão que aguenta.
3. **Tamanho do teste.** Casos positivos suficientes para o intervalo ser estreito. Com 300 positivos e recall perto de 0,90, o intervalo de 95% tem cerca de 0,03 para cada lado.
4. **Estabilidade.** A pior seed e o pior subgrupo (equipamento, projeção, faixa etária) também acima do mínimo.
5. **Validação prospectiva.** Um período em que o sistema roda em silêncio, sem influenciar o atendimento, e a saída é comparada com o laudo e com o teste laboratorial.
6. **Radiologista no circuito.** O sistema ordena a fila de laudos. Ele não libera nenhum paciente. Todo exame é lido por um radiologista, e os resultados negativos com incerteza acima de um limiar combinado recebem revisão prioritária.
7. **Monitoramento depois da implantação.** Auditoria mensal de uma amostra, com recall e número de casos perdidos, acompanhamento da proporção de positivos e de mudanças de equipamento, e uma regra escrita para suspender o sistema se a recall cair abaixo do mínimo.

## Limites deste estudo

- **Este estudo não tem validade clínica.** É um exercício de método com imagens públicas.
- **Resolução.** 64x64 pixels é muito menos do que um radiologista usa. A escolha veio do tempo de treino da GAN na T4.
- **Uma base montada de várias fontes.** As classes da base vêm de coleções diferentes. Um classificador pode aprender a reconhecer a origem da imagem e usar isso como atalho. Esse efeito não foi medido, e o teste vem da mesma base.
- **Três seeds.** Bastam para mostrar que a variação é grande. Não bastam para ordenar os braços 1 e 2, nem para medir com precisão a vantagem do braço 3 e a das versões da GAN.
- **Um sorteio do conjunto do projeto.** Outro sorteio daria outras 96 imagens de COVID-19 no treino.
- **Teste balanceado.** O teste tem 300 imagens por classe. A acurácia e a precision medidas nele não valem para a proporção de casos de um hospital. A recall por classe não depende dessa proporção.
- **Rótulos.** São os da base, sem revisão por radiologista.
