# Suscetibilidade a alagamento com ML — Baixo Taquari (RS)

Projeto da disciplina de Pesquisa Aplicada. Contexto completo, decisões e
pendências estão em [`claude.md`](claude.md) — leia aquele arquivo primeiro.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
earthengine authenticate --auth_mode=notebook --force \
  --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform
```

O `earthengine-api` recente exige um projeto do Google Cloud associado à
conta (crie um e habilite a API do Earth Engine nele) — troque
`config.EE_PROJECT` pelo seu próprio project id. O escopo `cloud-platform`
acima é necessário pros exports pesados (`ee_utils.export_via_asset`,
usado em `04_sampling.py`/`08_mapa_suscetibilidade.py`); sem ele, o GEE
responde "insufficient authentication scopes".

## Dados necessários (⚠️ reprodução não é só clonar o repo)

Este repositório **não inclui os dados** — nem os baixados via Earth
Engine (grandes, regeneráveis pelo código) nem os rasters originais do
paper de referência (privados, os próprios autores não publicam). Pra
reproduzir do zero:

1. **Rasters dos autores (Alcântara et al.)** — NÃO vêm com o repo e não
   são nossos pra redistribuir (~7GB, e o paper explicitamente não
   publica esses dados). Quem quiser reproduzir precisa pedir aos
   autores (contato em `claude.md`) e colocar o conteúdo recebido em
   **`data/raw/taquari_antas_autores/`** (não na raiz do projeto — é
   aí que `src/local_rasters.py` procura), mantendo a estrutura de
   pastas original (`FloodMap_SEP052023/`, `Sentinel 1 SAR/`,
   `Sentinel 2 MSI/`, `Temáticos/`, `EucDist/`, `CN_S_grid/`). Sem isso,
   `04_sampling.py` em diante não roda — não tem substituto público hoje
   pras variáveis de terreno/solo/geologia/geomorfologia na resolução e
   área usadas aqui.
2. **Google Earth Engine** — chuva (CHIRPS), AOI (malha municipal do
   MapBiomas) e rótulo/NDVI/NDWI/LULC de mai/2024 (MapBiomas affected
   area) vêm do GEE em tempo de execução, não precisam de download
   manual — só autenticação válida (acima) e `config.EE_PROJECT` setado
   pro seu projeto.
3. Com os dois acima prontos, o pipeline é regenerável do zero rodando
   os scripts na ordem abaixo — nenhum dado intermediário (`data/interim`,
   `data/processed`, `outputs/`) é versionado.

## Pipeline (rodar em ordem, de dentro de `src/`)

```
python 01_aoi.py                    # AOI (Baixo Taquari) -> data/interim/
python 04_sampling.py               # amostragem -> data/processed/pontos_*.csv
python 05_train.py                  # experimentos A-D -> outputs/models/
python 06_importance.py             # ranking de variáveis -> outputs/models/
python 08_mapa_suscetibilidade.py   # mapa de suscetibilidade -> outputs/figures/
python 09_verificacao_halo.py       # checagem FP x mancha set/2023 -> outputs/figures/
python 10_tabela_final.py           # tabela/figuras consolidadas -> outputs/
```

`02_labels.py` e `03_variables.py` não são rodados diretamente — são
importados por `04_sampling.py` (rótulo/NDVI/NDWI/LULC/chuva via GEE
pra mai/2024; terreno e o rótulo de set/2023 vêm de `local_rasters.py`,
direto dos rasters dos autores). `07_selecao_variaveis.py` existiu só
pra decidir o corte de `r10mm`/`cwd` e foi removido depois — a decisão e
os números ficaram documentados em `claude.md`.

Todo o histórico de decisões, achados e ressalvas metodológicas (o que
cada script faz e por quê, bugs já resolvidos, limitações dos dados)
está em [`claude.md`](claude.md) — leia aquele arquivo pra contexto
completo antes de mexer no pipeline.

## Estrutura

```
src/
  config.py                   # caminhos, AOI, eventos, constantes
  ee_utils.py                 # helper de export pesado via GEE (asset intermediário)
  local_rasters.py            # reprojeta/alinha os rasters dos autores na grade comum
  01_aoi.py                   # área de estudo (Baixo Taquari)
  02_labels.py                # rótulo via GEE (mai/2024) — importado por 04, não roda sozinho
  03_variables.py             # variáveis via GEE (chuva + NDVI/NDWI/LULC mai/2024) — idem
  04_sampling.py               # amostragem estratificada -> CSV de treino/teste
  05_train.py                  # experimentos A-D, 5 modelos
  06_importance.py             # permutation importance
  08_mapa_suscetibilidade.py   # mapa de suscetibilidade + diferenças (AOI inteira)
  09_verificacao_halo.py       # cruza FP de mai/2024 x mancha de set/2023
  10_tabela_final.py           # consolida tudo acima em tabela/figuras finais
data/
  raw/taquari_antas_autores/  # rasters dos autores (não versionado, ver acima)
  interim/                    # AOI e camadas intermediárias (não versionado)
  processed/                  # pontos_<evento>.csv — base de treino/teste (não versionado)
outputs/
  figures/                    # mapas, gráficos e tabelas finais (não versionado)
  models/                     # modelos treinados (.pkl) e tabelas de métricas (não versionado)
```
