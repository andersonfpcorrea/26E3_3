## Leitura dos resultados

Os números estão nas saídas e nas figuras das células acima. A análise completa está no relatório.

- **Projeto reproduzido (Parte A).** A acurácia de validação parece aceitável e fica acima da acurácia de quem sempre responde Normal. A recall de COVID-19 é baixa, na validação do projeto e no teste real. A métrica que o time reportou não mostrava os casos perdidos
- **GAN (Parte C).** As duas versões começam em colapso de modo e saem dele ao longo do treino, sem perda não finita. Na versão 1, COVID-19 é a classe menos diversa. A versão 2 deixa essa classe mais estável entre seeds. Compare a diferença de média entre as versões com o desvio padrão da versão 1, lembrando que o baseline variou entre execuções. Normal e Pneumonia perdem diversidade
- **Memorização.** As imagens geradas não ficam mais perto do treino que as imagens reais de validação. Elas são mais ruidosas e menos nítidas que as reais, e parecidas entre si dentro de cada seed
- **Recall (Parte D).** Compare a diferença entre as médias dos braços com o desvio padrão entre seeds. Com três seeds, uma diferença menor que esse desvio não permite concluir que um braço é melhor. O braço com pesos de classe é a alternativa sem GAN e serve de referência
- **Limites.** Imagens de 64x64, uma base pública montada de várias fontes, sem identificador de paciente e com três seeds. O resultado de uma GAN pode mudar entre execuções com a mesma seed. Este estudo não tem validade clínica
