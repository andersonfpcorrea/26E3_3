## Leitura dos resultados

- Acurácia global no teste: 100% (265 de 265 imagens).
- As sete classes ficaram em 100%. Nesta execução não há melhor nem pior classe.
- A matriz de confusão não tem nenhum erro, então não há par de classes confundido no teste.
- Na validação, a acurácia ficou em 99,25% da época 2 até a 10: 2 erros em 265 imagens.
- A perda de validação caiu em todas as épocas, de 0,277 para 0,031. A acurácia de treino terminou em 99,92%. A diferença entre treino e validação é menor que 1 ponto percentual, sem sinal de overfitting.
- O teste é pequeno. A menor classe, human, tem 27 imagens: um erro nela baixaria a acurácia da classe para 96,30% e a global para 99,62%.
- Foi uma única execução, com uma única seed. Ela não mede quanto o resultado varia entre execuções.
- A discussão de augmentation e de feature extraction contra fine-tuning está no relatório em PDF.
