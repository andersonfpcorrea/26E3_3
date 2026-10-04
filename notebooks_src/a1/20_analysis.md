## Leitura dos resultados

- O ViT-B/16 pré-treinado é o melhor dos três modelos, com folga: fica bem acima do ViT do zero na acurácia, no F1 macro e na distância média. Mesmo assim, erra o período exato em cerca de metade das pinturas
- Com tolerância de um período, o modelo pré-treinado acerta a grande maioria das pinturas. O ViT do zero fica bem abaixo, mas acima do acaso (12,5%)
- Os dois modelos treinados do zero têm overfitting: a acurácia de treino sobe até o fim e a de validação para cedo, com a perda de validação voltando a subir
- O modelo sem positional encoding fica praticamente empatado com o modelo completo. A diferença entre os dois é menor que a incerteza de um teste deste tamanho, com uma única seed. Nesta tarefa, a 64x64, a posição não ajudou de forma mensurável
- O teste de embaralhar os patches (`tests/test_vit.py`) prova o mecanismo: sem positional encoding a saída não muda, com ele muda. O empate acima vale só para esta tarefa e este modelo
- Na matriz de confusão do modelo pré-treinado, os erros ficam junto da diagonal, entre períodos vizinhos. Na do ViT do zero, eles se espalham, com erros de vários períodos
- Os períodos das pontas são os mais fáceis para os três modelos. O miolo, do século XIX ao começo do XX, é o mais confuso. Vale conferir a acurácia por período: um período pode ficar quase sem previsões e isso não aparece na acurácia global
- Os mapas de attention do ViT do zero mostram pontos pequenos, muitas vezes em cantos, bordas e fundo, e às vezes sobre a figura. O padrão muda de uma pintura para outra. Para um modelo com esta acurácia e patches de 8x8, não se espera um mapa com contorno de objeto
- Por movimento, os que cabem inteiros em um período largo (Renascimento, abstração do pós-guerra) são os mais bem datados. Os que atravessam vários períodos de 25 anos são os piores
- A discussão completa, com os números da execução final, os limites do experimento e os próximos passos, está no relatório
