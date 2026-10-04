## Ranking de conceitos: escolha do limiar

- A probabilidade de um conceito em uma imagem é a softmax das similaridades entre todos os conceitos, com a temperatura 100 aprendida pelo CLIP
- Um conceito conta como presente quando a sua probabilidade chega ao limiar
- O limiar não é dado pelo modelo. Ele sai dos dados: primeiro a distribuição da probabilidade mais alta de cada imagem, depois imagens sorteadas logo abaixo e logo acima de três limiares candidatos (0,3, 0,5 e 0,7)
- Em cada figura de inspeção, a linha de cima tem imagens abaixo do candidato e a de baixo tem imagens acima dele. O título de cada imagem mostra o conceito mais provável e a sua probabilidade
