## Parte C: a GAN condicional, duas versões com três seeds cada

Uma GAN tem dois modelos que treinam um contra o outro. O gerador cria imagens a partir de ruído e de um rótulo de classe. O discriminador tenta separar imagens reais das geradas. Aqui as duas redes recebem o rótulo, então o gerador sabe qual classe desenhar. Treino só com as imagens de treino, nas três classes, em 64x64.

- **Versão 1 (a base):** perda de entropia cruzada binária (BCE), discriminador com batch normalization, as duas taxas de aprendizado em 2e-4
- **Versão 3 (a mitigação, WGAN-GP):** perda de Wasserstein com penalidade de gradiente de peso 10, discriminador sem batch normalization, as duas taxas em 1e-4, Adam com betas 0 e 0,9, uma atualização do discriminador por lote
- **Três seeds por versão (42, 43 e 44).** Para cada seed, a inicialização das redes e o treino usam essa seed. Uma só execução por versão não separa o efeito da versão do acaso da GPU, então o que vale é a média e o desvio padrão entre as seeds
- Mesmo número de épocas, mesmo tamanho de lote (64) e mesmos 32 canais base nas duas versões
- A diversidade é a diferença média de pixels entre pares de imagens da mesma classe. Perto de zero indica colapso de modo. A razão é a diversidade gerada na última décima parte das épocas dividida pela diversidade das imagens reais de treino da classe. Razão abaixo de 0,5 marca colapso. A diversidade lida na época 1 não é confiável e não é interpretada
- Na versão 3, os escores do discriminador são sigmoides de valores sem escala de probabilidade. Servem como sinal relativo

**Conta das épocas.** O tempo total fica em cerca de 24 minutos na T4. Medido: versão 1 gasta cerca de 0,64 s por época, versão 3 cerca de 0,98 s, e o resto do notebook (download, leitura, projeto original, três braços) cerca de 4,5 minutos (270 s). Com três seeds, cada época custa 3 x (0,64 + 0,98) = 4,86 s. Sobram 1.440 - 270 = 1.170 s para a GAN, o que dá 1.170 / 4,86 = 240,7 épocas. O maior valor inteiro entre 200 e 300 que cabe é 240 épocas, usado nas duas versões.

**Outras tentativas.** Em execuções anteriores deste notebook, com uma seed por versão, duas outras mitigações foram testadas. Os números estão no relatório e no arquivo de resultados (`earlier_executions`). Não são resultados desta execução.

- Normalização espectral com perda hinge e taxas de aprendizado diferentes: colapsou em todas as execuções
- Aumento diferenciável (translação e brilho aleatórios antes do discriminador): ficou pior que a versão 1
