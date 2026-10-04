# Atividade 4.2: Transfer learning para análise de tráfego urbano

## O projeto analisado

- Tarefa: classificar o fluxo visto por câmeras urbanas como "livre", "moderado" ou "congestionado".
- Modelo: ResNet-50 pré-treinada na ImageNet, com fine-tuning.
- Dados: 800 frames rotulados de 12 câmeras.
- Resultado: 78% de acurácia na validação. O sistema foi implantado.
- Em produção: falhas sistemáticas em chuva, à noite e em câmeras com ângulos que não estavam no treino.
- Esta seção é só análise. Nada foi implementado.
- A página indicada no enunciado (Meegle, "Transfer Learning For Traffic Analysis") lista quatro desafios de transfer learning nessa área: diferença entre o domínio de origem e o de destino, overfitting por fine-tuning agressivo, custo computacional e dificuldade de interpretar o modelo. Os dois primeiros são plausíveis neste projeto.

## Problemas técnicos e metodológicos

### 1. A validação usa as mesmas câmeras do treino

- **Problema.** Provavelmente os frames de validação vêm das 12 câmeras do treino. Cada câmera tem fundo fixo: os mesmos prédios, faixas e postes. O modelo pode reconhecer a câmera e o seu padrão habitual. Nesse caso, os 78% medem frames novos de câmeras conhecidas e não informam nada sobre uma câmera nova.
- **Impacto operacional.** O sistema foi instalado em câmeras com outros ângulos. Se a validação usou as câmeras do treino, a acurácia nas câmeras novas nunca foi medida.
- **Como abordar.** Dividir por câmera: todas as imagens de uma câmera ficam no mesmo conjunto. Com 12 câmeras, usar validação cruzada deixando uma câmera de fora por vez. É a mesma lógica da divisão por pintor da Atividade 1, que mede o modelo em um pintor nunca visto.

### 2. Domain shift em chuva e à noite

- **Problema.** O treino não cobria essas condições, ou cobria pouco. Na chuva há reflexo no asfalto, gotas na lente e contraste baixo. À noite os veículos viram pares de faróis sobre fundo escuro, com ruído do sensor. A distribuição da entrada muda, e o modelo foi ajustado para outra.
- **Impacto operacional.** Chuva e pico noturno são momentos em que o congestionamento é comum e a central mais precisa do sistema. Ele falha junto com a demanda.
- **Como abordar.** Coletar e rotular frames de chuva, noite, amanhecer e entardecer em todas as câmeras. Reportar acurácia e recall por condição. Usar augmentation ligada a essas condições: brilho e contraste baixos, ruído, desfoque, reflexos. A augmentation complementa os dados reais.

### 3. Poucos frames, e correlacionados no tempo

- **Problema.** São 800 frames para 12 câmeras, cerca de 67 por câmera, divididos em três classes. Se os frames foram tirados com segundos de diferença, são quase a mesma imagem, e o número de exemplos independentes é bem menor que 800. Se frames vizinhos caem no treino e na validação, a validação mede memória.
- **Impacto operacional.** A acurácia reportada fica otimista, e o fine-tuning de uma ResNet-50 com milhões de parâmetros tende a overfitting. A Atividade 3 deste relatório teve o mesmo tipo de vazamento, com imagens duplicadas.
- **Como abordar.** Amostrar frames espaçados no tempo, em dias e horários diferentes. Congelar a maior parte do backbone enquanto houver poucos dados.

### 4. Features de objeto para uma propriedade da cena

- **Problema.** A ImageNet ensina a reconhecer objetos. Fluxo é quantos veículos ocupam a via e a que velocidade andam. O classificador de imagem inteira não conta veículos nem mede ocupação de forma explícita. Ele pode aprender atalhos, como a hora do dia pela luz ou a câmera pelo fundo.
- **Impacto operacional.** O sistema acerta enquanto o atalho vale e erra quando ele deixa de valer: um acidente às 3 da manhã, uma via vazia em feriado no horário de pico.
- **Como abordar.** Medir a quantidade que define o fluxo. Detectar e contar veículos em uma região da via, calcular a ocupação por faixa e classificar a partir desses números. O resultado fica interpretável: o operador vê a contagem que levou ao rótulo.

### 5. Fronteira subjetiva entre as classes

- **Problema.** Não há definição objetiva de onde termina "moderado" e começa "congestionado". Anotadores diferentes rotulam o mesmo frame de formas diferentes. O mesmo número de veículos é fluxo livre em uma avenida de quatro faixas e congestionamento em uma rua de uma faixa. Parte dos erros na fronteira é ruído de rótulo.
- **Impacto operacional.** Os alertas oscilam entre "moderado" e "congestionado" em situações estáveis. O operador deixa de confiar nos alertas.
- **Como abordar.** Definir as classes por medidas: ocupação da via, velocidade média ou comprimento de fila, com limiares por tipo de via. Rotular cada frame com dois anotadores e medir a concordância. Reportar a matriz de confusão, para separar o erro entre classes vizinhas do erro entre "livre" e "congestionado".

### 6. Acurácia global e nenhum monitoramento depois da implantação

- **Problema.** A decisão de implantar se apoiou em um número só. O projeto não informa a proporção das classes nem a recall de "congestionado". Se, em produção, as falhas foram percebidas só por quem usava o sistema, faltou medição contínua.
- **Impacto operacional.** Se "livre" for a classe mais comum, um modelo pode chegar perto de 78% e ainda perder boa parte dos congestionamentos. Na Atividade 4.1, uma acurácia de validação de 89,6% convivia com recall de 0,41 na classe que importava. Sem monitoramento, a degradação por câmera reposicionada, lente suja ou obra na via só aparece como reclamação.
- **Como abordar.** Reportar recall e precision por classe, por câmera e por condição. Depois da implantação, auditar uma amostra rotulada por semana, acompanhar a distribuição das previsões e da confiança por câmera e definir um alerta para quando elas mudarem.

## Riscos de partir da ImageNet para classificar fluxo

- **Objeto contra cena.** As features da ImageNet descrevem, em geral, um objeto central, grande e bem iluminado. Em uma câmera de tráfego, a resposta está na cena inteira: densidade, ocupação e comprimento de fila. Nenhum veículo isolado define a classe.
- **Ponto de vista.** As fotos da ImageNet são, em geral, tiradas na altura dos olhos e de perto. A câmera de tráfego é fixa, alta e de ângulo aberto. Os veículos aparecem pequenos, de cima e parcialmente cobertos uns pelos outros. As camadas finais do backbone foram treinadas para outra escala de objeto.
- **Frame contra estado.** Congestionamento é um estado ao longo do tempo. Um frame com muitos carros pode ser uma fila parada ou um pelotão passando em velocidade normal. Um modelo de imagem única não tem como separar os dois casos.
- **Condições.** Em geral, a ImageNet é formada sobretudo por fotos bem iluminadas, e o fine-tuning com 800 frames não cobre a diferença para chuva, noite e ofuscamento.
- **O que ainda se aproveita.** As primeiras camadas (bordas, texturas) servem em qualquer domínio. O risco está em confiar nas camadas finais e em um conjunto pequeno para adaptá-las. Pela regra da Atividade 3, este é o caso de dataset pequeno e domínio distante, o mais difícil.

## Como eu conduziria o projeto

1. **Definir a tarefa com a central de tráfego.** Classes definidas por ocupação, velocidade ou fila, com limiares por tipo de via. Decidir qual erro custa mais: perder um congestionamento ou gerar um alerta falso.
2. **Coletar dados com cobertura planejada.** Mais câmeras e mais ângulos. Frames espaçados no tempo, em dias diferentes, com chuva, noite, amanhecer e entardecer. Dois anotadores por frame.
3. **Dividir por câmera e por dia.** Validação cruzada por câmera, e um conjunto de teste com câmeras e dias nunca vistos, usado uma única vez.
4. **Começar por um baseline que mede o fluxo.** Um detector de veículos pré-treinado, contagem e ocupação por região da via, e limiares simples. Em seguida, um baseline temporal: a mesma medida em uma janela de frames, ou a diferença entre frames para estimar movimento.
5. **Só depois, o classificador de imagem.** Fine-tuning dos últimos blocos, com taxa de aprendizado baixa e augmentation ligada às condições reais: brilho, contraste, ruído, desfoque, pequenas mudanças de escala e perspectiva. Ele entra se superar os baselines no teste por câmera.
6. **Avaliar por fatia.** Recall, precision e matriz de confusão por classe, por câmera e por condição (dia, noite, chuva), com intervalo de confiança.
7. **Implantar em etapas.** Primeiro em silêncio, ao lado da operação atual, em poucas câmeras. Para cada câmera nova, rotular uma amostra pequena e medir antes de ativar.
8. **Monitorar.** Auditoria semanal de amostras, acompanhamento da distribuição das previsões por câmera, alerta para mudanças e um ciclo para rotular os erros e treinar de novo.
