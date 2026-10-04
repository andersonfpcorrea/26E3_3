# Uso de IA e referências

## Uso de IA

O enunciado marca este trabalho como "Sinal Verde". Todas as fontes, incluindo ferramentas de IA, devem ser citadas. O resultado de uma IA pode ser tendencioso e impreciso, e o aluno responde pela exatidão do que usa. Esta seção cumpre essa regra.

**Ferramenta**

- Claude Code, da Anthropic. É um assistente de programação que roda no terminal.

**Para que foi usada**

- Planejar o projeto: dividir o trabalho em tarefas e definir a ordem de execução.
- Escrever e revisar o código do pacote `src/dlv`, dos scripts e dos testes.
- Rodar os experimentos no Google Colab, em uma GPU T4.
- Redigir e revisar o texto do relatório.

**Como o resultado foi conferido**

- Testes automáticos. O repositório tem 394 testes, que rodam na CPU com entradas pequenas.
- Revisão independente. Cada tarefa passou por uma revisão independente antes de ser aceita.
- Números do relatório. Cada número foi comparado com os arquivos de resultado em `results/`, gerados pelas execuções dos notebooks.
- Figuras. Eu olhei as figuras e as páginas do PDF para achar cortes, sobreposições e rótulos ilegíveis.
- Problemas de dados achados e corrigidos no caminho:
    - Atividade 3: o download trazia as imagens duplicadas, e as cópias foram removidas antes da divisão.
    - Atividade 2: as miniaturas contavam a mesma figura duas vezes, e foram removidas.
    - Atividade 4: as máscaras de pulmão que acompanham as radiografias nunca são lidas, e o primeiro resultado da GAN não se repetia entre seeds, o que levou ao estudo de instabilidade.
- Resultados fracos foram escritos como saíram. Hiperparâmetros não foram ajustados para chegar a uma meta.

**Responsabilidade**

- Eu, Anderson Corrêa, revisei o trabalho e respondo por ele. As conclusões do relatório são minhas, e os erros que restarem também.

## Referências

### Dados

- WikiArt, cópia `Artificio/WikiArt` no Hugging Face. Página: https://huggingface.co/datasets/Artificio/WikiArt. A página não declara licença. É um upload de terceiros, usado aqui só para estudo.
- ADS-16 Computational Advertising Dataset, de Giorgio Roffo (`groffo/ads16-dataset`). Página: https://www.kaggle.com/datasets/groffo/ads16-dataset. Licença na página: "Other (specified in description)". A descrição pede a citação de Roffo, G. e Vinciarelli, A. (2016), "Personality in computational advertising: A benchmark", 4th Workshop on Emotions and Personality in Personalized Systems (EMPIRE) 2016.
- Images Dataset, de Pavan Sanagapati (`pavansanagapati/images-dataset`). Página: https://www.kaggle.com/datasets/pavansanagapati/images-dataset. Licença na página: "CC0: Public Domain".
- COVID-19 Radiography Database, de Tawsifur Rahman (`tawsifurrahman/covid19-radiography-database`). Página: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database. Licença na página: "Data files © Original Authors". A página pede a citação de dois artigos:
    - M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676. https://ieeexplore.ieee.org/document/9144185
    - Rahman, T., Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, M.E., 2020. Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-ray Images. https://doi.org/10.1016/j.compbiomed.2021.104319

### Modelos e pesos pré-treinados

- ResNet-50 da torchvision, pesos `IMAGENET1K_V2` (Atividade 3).
- ResNet-18 da torchvision, pesos da ImageNet (Atividade 4.1).
- ViT-B/16 da torchvision, pesos `IMAGENET1K_V1` (Atividade 1).
- CLIP, `openai/clip-vit-base-patch32`, no Hugging Face (Atividade 2).
- Tokenizer do `bert-base-uncased`, no Hugging Face (Atividade 2).

### Bibliotecas

- PyTorch 2.11.0.
- torchvision 0.26.0.
- Hugging Face Transformers 5.18.0.
- scikit-learn 1.9.1.
- Matplotlib 3.11.2.

Os resultados do relatório foram produzidos no Google Colab, com as versões instaladas lá na época das execuções. O `uv.lock` fixa as versões usadas nos testes locais.

### Artigos dos métodos

- Vaswani, A. et al. Attention Is All You Need. NeurIPS, 2017.
- Dosovitskiy, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR, 2021.
- Touvron, H. et al. Training data-efficient image transformers and distillation through attention. ICML, 2021.
- Liu, Z. et al. Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV, 2021.
- Bao, H. et al. BEiT: BERT Pre-Training of Image Transformers. ICLR, 2022.
- He, K. et al. Masked Autoencoders Are Scalable Vision Learners. CVPR, 2022.
- Devlin, J. et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL, 2019.
- Radford, A. et al. Learning Transferable Visual Models From Natural Language Supervision. ICML, 2021.
- He, K. et al. Deep Residual Learning for Image Recognition. CVPR, 2016.
- Goodfellow, I. et al. Generative Adversarial Nets. NeurIPS, 2014.
- Mirza, M. e Osindero, S. Conditional Generative Adversarial Nets. arXiv, 2014.
- Miyato, T. et al. Spectral Normalization for Generative Adversarial Networks. ICLR, 2018.
- Gulrajani, I. et al. Improved Training of Wasserstein GANs. NeurIPS, 2017.
- Zhao, S. et al. Differentiable Augmentation for Data-Efficient GAN Training. NeurIPS, 2020.
- Zhu, J.-Y., Park, T., Isola, P. e Efros, A. A. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV, 2017.
- Tan, M. e Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML, 2019.
- Kingma, D. P. e Ba, J. Adam: A Method for Stochastic Optimization. ICLR, 2015.
- Loshchilov, I. e Hutter, F. Decoupled Weight Decay Regularization. ICLR, 2019.
- Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K. e Fei-Fei, L. ImageNet: A Large-Scale Hierarchical Image Database. CVPR, 2009.

### Referências do enunciado

- Meegle, "Transfer Learning For Traffic Analysis", como indicada no enunciado: https://www.meegle.com/en_us/topics/transfer-learning/transfer-learning-for-traffic-analysis
- Dumakude e Ezugwu. Automated COVID-19 detection with convolutional neural networks. Scientific Reports, 2023. https://www.nature.com/articles/s41598-023-37743-4
