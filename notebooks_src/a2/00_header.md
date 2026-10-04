# Atividade 2: Busca e ranking de conceitos em anúncios com CLIP

**Aluno:** Anderson Corrêa

**Objetivo:** usar o CLIP pré-treinado (ViT-B/32), sem nenhum treino, para duas tarefas sobre o corpus ADS-16: (1) ordenar conceitos pela frequência com que aparecem nas imagens e (2) buscar imagens a partir de uma frase.

**Requisitos de execução**

- Ambiente: Google Colab com GPU T4
- Memória de GPU: cerca de 0,8 GB (medido na T4)
- Memória RAM: cerca de 2 GB (processo principal, medido na T4)
- Disco: 3 GB livres
- Tempo estimado: cerca de 2,5 minutos (medido na T4, sem contar a inicialização da máquina; o download de 0,8 GB pode variar)

**Como executar:** menu *Ambiente de execução*, *Executar tudo*. O notebook baixa os dados sozinho e não precisa de login. A biblioteca `transformers` já vem instalada no Colab. Os embeddings das imagens ficam em disco, então uma segunda execução na mesma sessão não os calcula de novo.
