## Parte C: a GAN condicional

Uma GAN tem dois modelos que treinam um contra o outro. O gerador cria imagens a partir de ruído e de um rótulo de classe. O discriminador tenta separar imagens reais das geradas. Aqui as duas redes recebem o rótulo, então o gerador sabe qual classe desenhar. Treino só com as imagens de treino, nas três classes, em 64x64.

- **Versão 1:** perda de entropia cruzada binária (BCE), discriminador com batch normalization, as duas taxas de aprendizado em 2e-4
- **Versão 2:** perda hinge, normalização espectral no discriminador (limita o quanto cada camada pode esticar o sinal), sem batch normalization, taxa de aprendizado 4e-4 no discriminador e 1e-4 no gerador
- **Versão 3 (WGAN-GP):** tentada porque a versão 2 não melhorou a diversidade da versão 1 na primeira execução (ver o relatório). Perda de Wasserstein com penalidade de gradiente de peso 10, discriminador sem normalização de lote, as duas taxas em 1e-4, Adam com betas 0 e 0,9, uma atualização do discriminador por lote
- **Versão 4 (aumento diferenciável):** igual à versão 1, com uma mudança. Toda imagem que o discriminador avalia, real ou gerada, passa antes por uma translação aleatória de até 1/8 do tamanho e por uma mudança aleatória de brilho. A operação deixa o gradiente passar até o gerador. Tentada porque a versão 3 também não melhorou a versão 1 (ver o relatório)
- As versões usam o mesmo número de épocas, o mesmo tamanho de lote, a mesma seed, a mesma inicialização e o mesmo ruído nas imagens de acompanhamento
- A diversidade é a diferença média de pixels entre pares de imagens da mesma classe. Perto de zero indica colapso de modo. A linha de referência é a diversidade das imagens reais de treino da classe
- A diversidade lida na época 1 não é confiável e não é interpretada
- Nas versões 2 e 3, os escores do discriminador são sigmoides de valores que a perda hinge não treina como probabilidade. Servem como sinal relativo
