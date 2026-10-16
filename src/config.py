"""Constantes e caminhos compartilhados pelo pipeline.

Mantém num só lugar o que os scripts 01-04 precisam em comum, para não
duplicar (e desalinhar) região, eventos e caminhos de dados.
"""

import sys
from pathlib import Path

# O console padrão do Windows (cp1252/cp850) corrompe acentos ao printar
# (ex. "Muçum" vira "Mu?um"). Forçar utf-8 aqui cobre todo script que
# importa config antes de imprimir qualquer coisa.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# --- Caminhos -----------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_INTERIM = ROOT_DIR / "data" / "interim"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
OUTPUTS_FIGURES = ROOT_DIR / "outputs" / "figures"
OUTPUTS_MODELS = ROOT_DIR / "outputs" / "models"

# --- Google Earth Engine ---------------------------------------------
EE_PROJECT = "kunumi-alagamento-rs"

# --- Área de estudo: Baixo Taquari ---------------------------------------
# TODO: confirmar a lista definitiva de municípios com o orientador
# (ver "Pendências" em claude.md).
MUNICIPIOS_BAIXO_TAQUARI = [
    "Muçum",
    "Encantado",
    "Roca Sales",
    "Arroio do Meio",
    "Lajeado",
    "Estrela",
    "Cruzeiro do Sul",
]

# --- Eventos --------------------------------------------------------------
# set/2023 = treino (evento do paper de referência), mai/2024 = teste.
EVENTO_TREINO = "2023-09"
EVENTO_TESTE = "2024-05"

# Picos confirmados: set/2023 em 4-5/09 (nota IPH-UFRGS); mai/2024 em
# 02/05 13h30, 33,66 m em Lajeado (Grupo A Hora).
#
# HISTÓRICO (removido em 15/10/2026, ver claude.md): set/2023 usava um
# fallback via Sentinel-1 (change detection) porque não existe imagem
# Sentinel-1 pré-evento pra essa região (gap jan/2022-13/09/2023, falha do
# Sentinel-1B) — dava só ~1,1 km² alagados, muito subestimado. Resolvido
# com o FloodMap real dos autores (336 km²) — ver local_rasters.py.

# --- Mancha oficial do MapBiomas RS Climate Emergency Toolkit -----------
# https://mapbiomas-workspace.earthengine.app/view/rio-grande-do-sul-climate-emergency-toolkit
# O app é um SPA sem asset documentado publicamente, mas o bootstrap JS
# (rio-grande-do-sul-climate-emergency-toolkit-modules.json, linkado no
# HTML da página) expõe o script real. Dataset "Mapeamento dos danos" tem
# 8 camadas de "área atingida" por categoria; "affected_area-v2" é a geral
# (união das outras). Testado em 24/09/2026: ~149,4 km² na nossa AOI —
# ordem de grandeza plausível pra essa cheia. Só cobre mai/2024 — nada
# equivalente pra set/2023 nesse toolkit (resolvido via o FloodMap real
# dos autores pra set/2023, ver local_rasters.py).
#
# RESSALVA: é "área atingida" (dano em geral — inclui lavoura, floresta,
# deslizamento), não uma máscara de água pura. Sub-camadas disponíveis se
# precisar de algo mais específico (mesmo padrão de nome, trocar o sufixo):
# affected_urban_area-v2, affected_agricultural_area-v2, affected_forest-v3,
# damaged_forest-v3, affected_grassland_area-v2, affected_wetland-v2,
# landslide_areas-v2.
#
# Metodologia real (lida em mapbiomas/2024/toAsset-AREAS-AFFECTED, script
# copiado pelo usuário do Code Editor): soma de 3 métodos (índice de água
# Sentinel-2/SMA, radar Sentinel-1 change detection com baseline "antes" de
# 8 MESES — evita o gap de cobertura que travou nosso Sentinel-1 fallback
# — e PCA/k-means) + fusão com camada da Defesa Civil/IPH-UFRGS + máscara
# HAND por bacia hidrográfica (limiar 25m especificamente pra 'Taquari-
# Antas') + blend com deslizamento. Ver claude.md pros detalhes completos.
#
# Alternativa mais "pura" testada e funcionando, mas não usada (produto
# mais antigo/menos refinado, só ~25 km² na AOI vs ~170 km² do affected_area):
MAPBIOMAS_WATER_V1 = (
    "projects/nexgenmap/MapBiomas2/RS/AREAS-AFFECTED/"
    "mapbiomas_climate_emergency-2024_RS-water-v1"
)
MAPBIOMAS_AFFECTED_AREA = {
    EVENTO_TESTE: (
        "projects/nexgenmap/MapBiomas2/RS/AREAS-AFFECTED/"
        "mapbiomas_climate_emergency-2024_RS-affected_area-v2"
    ),
}

# Malha hidrográfica oficial do RS (26 bacias, campo 'nome' — inclui
# 'Taquari-Antas'). Achada nos mesmos scripts; não integrada no pipeline
# ainda, mas é a fonte certa se algum passo futuro precisar de bacia por
# ponto ou quiser replicar a máscara HAND-por-bacia do MapBiomas.
BACIAS_RS = "projects/mapbiomas-workspace/AUXILIAR/BACIAS_RS/BH_RS_25mil"

# --- Variáveis ambientais via GEE (03_variables.py) ----------------------
# Terreno, geologia/solos/geomorfologia, CN/S e NDVI/NDWI/LULC de set/2023
# agora vêm dos rasters reais dos autores (local_rasters.py, ver
# claude.md 15/10/2026). Só NDVI/NDWI/LULC de MAI/2024 ainda dependem do
# GEE (autores só mandaram a versão de set/2023) — janela PRÉ-evento,
# evitando qualquer imagem durante/depois (senão o índice já enxerga a
# própria água do rótulo, vazamento — ver claude.md "Cuidados"). LULC: ano
# completo mais recente antes do evento.
JANELA_OPTICO_TESTE = ("2024-01-15", "2024-04-20")
LULC_ANO_TESTE = "classification_2023"
CLOUD_SCORE_THRESHOLD = 0.40  # mesmo limiar usado no toolkit do MapBiomas

# Janela de chuva (índices ETCCDI via CHIRPS): 30 dias antecedentes ao
# pico, terminando no dia do pico. Convenção nossa (não confirmada contra
# a seção de métodos do paper de referência) — se for preciso bater
# exatamente com o Alcântara et al., conferir lá antes de usar em produção.
JANELAS_CHUVA = {
    EVENTO_TREINO: ("2023-08-06", "2023-09-05"),
    EVENTO_TESTE: ("2024-04-03", "2024-05-02"),
}

TPI_RAIO_M = 270  # ~9 pixels de 30 m — janela de vizinhança pro TPI

# --- Grade comum das variáveis ---------------------------------------
RESOLUCAO_M = 30
# UTM 22S — mesma projeção que o GEE atribui nativamente às cenas Sentinel-1
# desta região (long. ~-52). `.mosaic()` de uma ImageCollection perde a
# projeção real das imagens de origem (cai pra um identity transform
# degenerado, "1 grau por pixel") — todo raster usado em operação por
# pixel (focal_mean, slope, etc.) precisa de `.reproject(crs=CRS_TRABALHO,
# scale=...)` logo após o mosaic, senão a operação roda sobre pixels de
# ~111 km e produz lixo silenciosamente (sem erro, só máscara tudo).
CRS_TRABALHO = "EPSG:32722"

# --- Amostragem -------------------------------------------------------
N_PONTOS_POR_EVENTO = 3000  # réplica em escala reduzida do paper de referência
DIST_MIN_ENTRE_PONTOS_M = 90  # evita pontos vizinhos autocorrelacionados
