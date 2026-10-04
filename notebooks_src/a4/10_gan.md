## Parte C: a GAN condicional, duas versões com três seeds cada

Uma GAN tem dois modelos que treinam um contra o outro. O gerador cria imagens a partir de ruído e de um rótulo de classe. O discriminador tenta separar imagens reais das geradas. Aqui as duas redes recebem o rótulo, então o gerador sabe qual classe desenhar. Treino só com as imagens de treino, nas três classes, em 64x64.

- **Versão 1 (a base):** perda de entropia cruzada binária (BCE), discriminador com batch normalization, as duas taxas de aprendizado em 2e-4, lotes sorteados como vêm dos dados
- **Versão 2 (a mitigação, lotes balanceados por classe):** a mesma rede, a mesma perda e as mesmas taxas da versão 1. A única mudança é o sorteio dos lotes. O treino tem 672 imagens Normal, 192 Pneumonia e 96 COVID-19, e os rótulos das imagens falsas seguem os rótulos do lote real, então só cerca de 10% das atualizações do gerador são de COVID-19. Na versão 2 cada índice é sorteado com reposição, com probabilidade inversa à frequência da classe, e cada classe fica com a mesma parcela esperada dos lotes. O número de lotes por época não muda
- **Três seeds por versão (42, 43 e 44).** Para cada seed, a inicialização das redes e o treino usam essa seed. Uma só execução por versão não separa o efeito da versão do acaso da GPU, então o que vale é a média e o desvio padrão entre as seeds
- Mesmo número de épocas, mesmo tamanho de lote (64) e mesmos 32 canais base nas duas versões
- A diversidade é a diferença média de pixels entre pares de imagens da mesma classe. Perto de zero indica colapso de modo. A razão é a diversidade gerada na última décima parte das épocas dividida pela diversidade das imagens reais de treino da classe. Razão abaixo de 0,5 marca colapso. A diversidade lida na época 1 não é confiável e não é interpretada
- Balancear pode aumentar a diversidade só porque o gerador produz ruído. Por isso a distância das imagens geradas de COVID-19 até as imagens reais de treino também é medida, para cada versão e seed, e comparada com a distância das imagens reais de validação

**Conta das épocas.** O tempo total fica em cerca de 24 minutos na T4. Medido na execução anterior: versão 1 gasta cerca de 0,68 s por época, a versão 2 tem o mesmo custo por época (mesma rede e mesmo número de lotes, usado 0,68 s), e o resto do notebook (download, leitura, projeto original, três braços) cerca de 4,5 minutos (270 s). Com três seeds, cada época custa 3 x (0,68 + 0,68) = 4,08 s. Sobram 1.440 - 270 = 1.170 s para a GAN, o que dá 1.170 / 4,08 = 286,8 épocas. O maior valor inteiro entre 200 e 300 que cabe é 286 épocas, usado nas duas versões.

**Outras tentativas.** Em execuções anteriores deste notebook duas outras mitigações foram testadas com uma seed por versão, e uma terceira com três seeds e 240 épocas. Os números estão no relatório e no arquivo de resultados (`earlier_executions`). Não são resultados desta execução.

- Normalização espectral com perda hinge e taxas de aprendizado diferentes: colapsou em todas as execuções
- Aumento diferenciável (translação e brilho aleatórios antes do discriminador): ficou pior que a versão 1
- WGAN-GP (perda de Wasserstein com penalidade de gradiente): com três seeds, razão média de diversidade de COVID-19 de 0,485 com desvio padrão de 0,370 contra 0,591 e 0,051 da versão 1, e uma das três seeds colapsou durante todo o treino
