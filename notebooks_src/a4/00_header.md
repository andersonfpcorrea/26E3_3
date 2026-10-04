# Atividade 4: Estudo de caso, triagem de COVID-19 em radiografias de tórax

**Aluno:** Anderson Corrêa

**Objetivo:** um projeto de triagem de COVID-19 em radiografias chegou a 93% de acurácia no treino e 61% na validação. Este notebook reproduz esse projeto, mostra o que ele não reportou, treina uma GAN condicional em duas versões, com três seeds cada, verifica se as imagens geradas são cópias e mede se imagens sintéticas de COVID-19 aumentam a revocação (recall) da classe.

**Requisitos de execução**

- Ambiente: Google Colab com GPU T4
- Memória de GPU: cerca de 0,7 GB (medido na T4)
- Memória RAM: cerca de 2,2 GB (processo principal, medido na T4)
- Disco: 3 GB livres
- Tempo estimado: cerca de 23 minutos (medido na T4, sem contar a inicialização da máquina; o download de 0,8 GB pode variar)

**Como executar:** menu *Ambiente de execução*, *Executar tudo*. O notebook baixa os dados sozinho e não precisa de login.
