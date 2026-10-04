## Dados

- Dataset WikiArt, baixado do Hugging Face (`Artificio/WikiArt`), com imagens já em 256x256
- O ano vem da coluna `date`, que é texto em formatos variados. Linhas sem ano são descartadas
- A classe é o período da pintura, entre oito períodos. Os períodos mais antigos são mais largos porque têm menos pinturas
- Amostra balanceada: até 2.000 pinturas por período, sorteadas com seed fixa
- Divisão por pintor: 70% treino, 15% validação, 15% teste. Todas as pinturas de um pintor ficam na mesma parte, para que o modelo não acerte o período só por reconhecer o pintor
- Pinturas sem pintor conhecido contam cada uma como um pintor separado
- As imagens da amostra são lidas uma única vez. Para o modelo treinado do zero, elas são reduzidas para 64x64 e ficam na memória
