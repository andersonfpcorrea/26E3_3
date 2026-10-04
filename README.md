# Visão Computacional: Projeto da Disciplina (INFNET 26E3_3)

**Aluno:** Anderson Corrêa

Projeto da disciplina de Visão Computacional, em PyTorch. São quatro atividades:

1. Datação de pinturas do WikiArt em 8 períodos, com Vision Transformers.
2. Busca e ranking de conceitos em anúncios com CLIP.
3. Classificador de fotos de 7 classes com ResNet-50 pré-treinada.
4. Estudos de caso: triagem de COVID-19 em radiografias, com GAN condicional (4.1), e análise de um projeto de tráfego urbano, só por escrito (4.2).

Cada atividade tem um notebook e uma seção do relatório em PDF.

## Requisitos

- [uv](https://docs.astral.sh/uv/getting-started/installation/), para instalar as dependências e rodar os comandos.
- A CLI `colab`, só para rodar os notebooks em uma GPU T4 remota (`make colab-*`).

## Como rodar

```bash
make setup            # instala as dependências (uv sync --all-groups)
make test             # roda os testes na CPU
make notebooks        # gera os notebooks em build/notebooks
make colab-a3         # roda uma atividade na T4 do Colab (a1, a2, a3 ou a4)
make colab-all        # roda a3, a2, a1 e a4 em sequência
make check-notebooks  # falha se um notebook executado difere de uma geração nova
make report           # gera o PDF do relatório
make zip              # gera o arquivo de entrega em entrega/
make clean            # remove build/ e entrega/
make help             # lista os comandos
```

`make colab-<atividade>` gera os notebooks, executa o escolhido no Colab, e grava o notebook executado em `notebooks/`, as métricas em `results/` e as figuras em `report/figures/`.

## Entregáveis

| Entregável do enunciado | Onde está |
| --- | --- |
| Notebook A1 | `notebooks/A1_vision_transformers.ipynb` |
| Notebook A2 | `notebooks/A2_clip_ads16.ipynb` |
| Notebook A3 | `notebooks/A3_cnn_kaggle.ipynb` |
| Notebook A4.1 | `notebooks/A4_estudo_caso_raio_x.ipynb` |
| Relatório técnico (PDF) | `report/anderson_correa_deep-learning-and-vision_computer-vision.pdf` |
| Fonte do relatório | `report/secoes/*.md` |
| A4.2 (tráfego, só texto) | `report/secoes/45_atividade4_trafego.md` |
| Uso de IA e referências | `report/secoes/90_uso_de_ia_e_referencias.md` |
| Métricas das execuções | `results/a1.json` a `results/a4.json` |
| Arquivo de entrega | `entrega/anderson_correa_computer-vision_pd.zip`, gerado por `make zip` |
| Dependências com versões | `pyproject.toml` e `uv.lock` |

## Estrutura

```
src/dlv/            pacote com toda a lógica: dados, modelos, treino, figuras
notebooks_src/      células de cada notebook (a1 a a4), só configuração, chamadas e prints
scripts/            build_notebooks.py, colab_run.py, build_report.py
notebooks/          notebooks executados no Colab
results/            métricas em JSON, base dos números do relatório
report/             secoes/*.md, figures/*.png, diagrams/ e o PDF final
tests/              testes na CPU, sem baixar dados nem pesos
build/              arquivos gerados (ignorado pelo git)
entrega/            arquivo zip de entrega (ignorado pelo git)
```

## Os notebooks são gerados

Os notebooks de `notebooks/` são gerados a partir de `notebooks_src/` e de `src/dlv/`. Não edite um notebook à mão. Para mudar algo, edite a célula em `notebooks_src/` ou o módulo em `src/dlv/` e rode `make notebooks`. Cada notebook sai autocontido: o código do pacote vai dentro dele, então ele roda no Colab sem clonar o repositório.

## Como abrir um notebook no Colab

1. Abra https://colab.research.google.com e escolha Arquivo, Fazer upload de notebook.
2. Envie um arquivo de `notebooks/` (ou de `build/notebooks/`, depois de `make notebooks`).
3. Em Ambiente de execução, Alterar o tipo de ambiente de execução, escolha a GPU T4.
4. Em Ambiente de execução, escolha Executar tudo.

Os requisitos de memória e o tempo estimado estão no início de cada notebook. Os dados são baixados dentro do notebook.
