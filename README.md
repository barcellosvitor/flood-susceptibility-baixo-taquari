# Suscetibilidade a alagamento com ML — Baixo Taquari (RS)

Projeto da disciplina de Pesquisa Aplicada. Contexto completo, decisões e
pendências estão em [`claude.md`](claude.md) — leia aquele arquivo primeiro.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
earthengine authenticate      # necessário para os scripts src/0*_*.py
```

O `earthengine-api` recente exige um projeto do Google Cloud associado à
conta (o fluxo do `earthengine authenticate` oferece criar um gratuito na
hora, se você ainda não tiver um). Se `ee.Initialize()` reclamar de projeto
ausente, use `ee.Initialize(project="<seu-project-id>")` nos scripts.

## Estrutura

```
src/
  config.py        # caminhos, AOI, eventos, constantes de amostragem
  01_aoi.py         # área de estudo (Baixo Taquari)
  02_labels.py      # manchas de água (rótulo) dos dois eventos
  03_variables.py   # variáveis ambientais empilhadas por evento
  04_sampling.py    # amostragem estratificada + export da base de treino
notebooks/          # exploração, treino dos modelos, figuras
data/
  raw/              # dados baixados, sem alteração (não versionado)
  interim/          # camadas intermediárias do GEE (não versionado)
  processed/        # tabelas finais pontos × variáveis (não versionado)
outputs/
  figures/          # mapas e gráficos gerados
  models/           # modelos treinados (.pkl)
```

Os scripts em `src/` estão numerados na ordem do pipeline e hoje são
stubs (`TODO` + `NotImplementedError`) — a lógica de cada um está descrita
no próprio docstring do arquivo.
