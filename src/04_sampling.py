"""Passo 4: amostragem estratificada — monta o CSV de treino/teste.

Combina duas fontes por evento (ver claude.md, achado de 15/10/2026):
- LOCAL (rasterio, sem GEE): 18 variáveis estáticas + TPI (terreno,
  geologia/solos/geomorfologia, CN/S) pros dois eventos; NDVI/NDWI/LULC +
  rótulo (FloodMap real) só pra set/2023.
- GEE: chuva (5 índices ETCCDI) pros dois eventos; NDVI/NDWI/LULC + rótulo
  (MapBiomas) só pra mai/2024. Peças GEE "pesadas" passam por
  ee_utils.export_via_asset (precisa de `earthengine authenticate` com
  escopo de export — ver README/claude.md).

Depois de reunir todas as bandas na MESMA grade (AOI, 30m, EPSG:32722),
amostra pontos dentro/fora da mancha, respeitando:
- Balanceamento: até N_PONTOS_POR_EVENTO/2 por classe.
- Distância mínima entre pontos (DIST_MIN_ENTRE_PONTOS_M): no máx. 1 ponto
  por bloco de DIST_MIN_ENTRE_PONTOS_M x DIST_MIN_ENTRE_PONTOS_M.
- Só pixels com TODAS as variáveis válidas (sem NaN/nodata em nenhuma banda).

Exporta data/processed/pontos_<evento>.csv: coordenadas (lon/lat),
município (NM_MUN/CD_MUN, pro leave-one-municipality-out do experimento
B), todas as variáveis e o rótulo.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

import ee
import geemap
import importlib

import local_rasters
from ee_utils import export_via_asset
from config import (
    CRS_TRABALHO,
    DATA_INTERIM,
    DATA_PROCESSED,
    DIST_MIN_ENTRE_PONTOS_M,
    EE_PROJECT,
    EVENTO_TESTE,
    EVENTO_TREINO,
    MAPBIOMAS_AFFECTED_AREA,
    N_PONTOS_POR_EVENTO,
    RESOLUCAO_M,
)

variables_gee = importlib.import_module("03_variables")
labels_module = importlib.import_module("02_labels")

AOI_PATH = DATA_INTERIM / "aoi_baixo_taquari.geojson"
SEED = 42

# Nomes de variável que são categóricas (nodata = -1, nunca NaN) — usado
# pra montar a máscara de validade corretamente. Sem isso, um valor -1
# LEGÍTIMO numa variável contínua (ex. "aspect" usa -1 pra terreno plano,
# não é nodata) seria descartado por engano.
VARIAVEIS_CATEGORICAS = {
    nome for nome, (_, categorica) in local_rasters.RASTERS_ESTATICOS.items() if categorica
} | {
    nome for nome, (_, categorica) in local_rasters.RASTERS_SET2023.items() if categorica
} | {"lulc"}


# --- Peças locais (sem GEE) -----------------------------------------------


def _municipio_raster() -> tuple[np.ndarray, dict[int, str]]:
    aoi = gpd.read_file(AOI_PATH).to_crs(CRS_TRABALHO)
    transform, largura, altura = local_rasters.aoi_grid()
    formas = [(geom, int(cd)) for geom, cd in zip(aoi.geometry, aoi["CD_MUN"])]
    raster = rasterize(
        formas, out_shape=(altura, largura), transform=transform, fill=-1, dtype="int32"
    )
    nomes = {int(cd): nm for cd, nm in zip(aoi["CD_MUN"], aoi["NM_MUN"])}
    return raster, nomes


def _bandas_locais(evento: str) -> dict[str, np.ndarray]:
    bandas = dict(local_rasters.build_static_stack())
    if evento == EVENTO_TREINO:
        bandas.update(local_rasters.build_set2023_stack())
    return bandas


def _rotulo_local(evento: str) -> np.ndarray | None:
    if evento == EVENTO_TREINO:
        return local_rasters.build_flood_label_set2023()
    return None


# --- Peças GEE (chuva nos dois eventos; NDVI/NDWI/LULC+rótulo só teste) --


def _baixar_multibanda_gee(
    imagem: ee.Image, aoi_ee: ee.FeatureCollection, nomes_categoricos: set, descricao: str
) -> dict[str, np.ndarray]:
    """Baixa cada banda de uma ee.Image SEPARADAMENTE e já realinha na
    nossa grade local.

    Testado em 16/10/2026: baixar tudo junto (um .tif multi-banda) estoura
    o limite de download direto do GEE ("Total request size... must be
    less than or equal to 50331648 bytes") mesmo depois de materializar
    como asset — esse limite é de TAMANHO da resposta, não de memória de
    computação (que já resolvemos com export_via_asset). Banda por banda
    fica bem abaixo do limite (~15-18 MB cada, testado).
    """
    nomes_bandas = imagem.bandNames().getInfo()
    saida = {}
    for nome in nomes_bandas:
        tmp_path = DATA_INTERIM / f"_tmp_{descricao}_{nome}.tif"
        export_via_asset(
            imagem.select([nome]),
            region=aoi_ee.geometry(),
            scale=RESOLUCAO_M,
            out_path=tmp_path,
            description=f"{descricao}_{nome}",
        )
        categorica = nome in nomes_categoricos
        saida[nome] = local_rasters.reprojetar_arquivo(tmp_path, categorica=categorica)
        tmp_path.unlink(missing_ok=True)
    return saida


def _bandas_gee(aoi_ee: ee.FeatureCollection, evento: str) -> dict[str, np.ndarray]:
    stack = variables_gee.build_gee_stack(aoi_ee, evento)
    return _baixar_multibanda_gee(
        stack, aoi_ee, nomes_categoricos={"lulc"}, descricao=f"variaveis_gee_{evento}".replace("-", "_")
    )


def _rotulo_gee(aoi_ee: ee.FeatureCollection, evento: str) -> np.ndarray:
    mancha = labels_module.build_flood_mask_mapbiomas(aoi_ee, MAPBIOMAS_AFFECTED_AREA[evento])
    tmp_path = DATA_INTERIM / f"_tmp_rotulo_{evento}.tif".replace("-", "_")
    geemap.ee_export_image(
        mancha, filename=str(tmp_path), scale=RESOLUCAO_M, region=aoi_ee.geometry(), file_per_band=False
    )
    arr = local_rasters.reprojetar_arquivo(tmp_path, categorica=True)
    tmp_path.unlink(missing_ok=True)
    return (arr == 1).astype(np.uint8)


# --- Amostragem -------------------------------------------------------


def _bloco_id(altura: int, largura: int, passo_px: int) -> np.ndarray:
    linhas = (np.arange(altura) // passo_px).reshape(-1, 1)
    colunas = (np.arange(largura) // passo_px).reshape(1, -1)
    n_colunas_bloco = largura // passo_px + 1
    return linhas * n_colunas_bloco + colunas


def _amostrar_classe(
    candidatos: np.ndarray, bloco_id: np.ndarray, n_pontos: int, rng: np.random.Generator
) -> np.ndarray:
    """No máx. 1 ponto por bloco de espaçamento mínimo, até n_pontos."""
    if candidatos.shape[0] == 0:
        return candidatos
    ordem = rng.permutation(candidatos.shape[0])
    candidatos = candidatos[ordem]

    blocos_vistos: set[int] = set()
    selecionados = []
    for linha, coluna in candidatos:
        b = int(bloco_id[linha, coluna])
        if b in blocos_vistos:
            continue
        blocos_vistos.add(b)
        selecionados.append((linha, coluna))
        if len(selecionados) >= n_pontos:
            break
    return np.array(selecionados, dtype=int)


def amostrar_evento(evento: str, usar_gee: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    transform, largura, altura = local_rasters.aoi_grid()

    bandas = _bandas_locais(evento)
    rotulo = _rotulo_local(evento)
    municipio, nomes_municipio = _municipio_raster()

    if usar_gee:
        ee.Initialize(project=EE_PROJECT)
        aoi_ee = geemap.vector_to_ee(str(AOI_PATH))
        bandas.update(_bandas_gee(aoi_ee, evento))
        if rotulo is None:
            rotulo = _rotulo_gee(aoi_ee, evento)
    elif rotulo is None:
        raise RuntimeError(
            f"Evento {evento} não tem rótulo local — precisa do GEE (usar_gee=False não serve aqui)."
        )

    # máscara de validade: todas as bandas precisam ter valor nesse pixel.
    # Categóricas usam -1 como nodata; contínuas usam NaN — não trocar
    # (algumas contínuas têm -1 como valor legítimo, ex. "aspect" plano).
    valido = np.ones((altura, largura), dtype=bool)
    for nome, arr in bandas.items():
        if nome in VARIAVEIS_CATEGORICAS:
            valido &= arr != -1
        else:
            valido &= ~np.isnan(arr)
    valido &= municipio != -1

    passo_px = round(DIST_MIN_ENTRE_PONTOS_M / RESOLUCAO_M)
    bloco_id = _bloco_id(altura, largura, passo_px)

    n_por_classe = N_PONTOS_POR_EVENTO // 2
    cand_1 = np.argwhere((rotulo == 1) & valido)
    cand_0 = np.argwhere((rotulo == 0) & valido)

    sel_1 = _amostrar_classe(cand_1, bloco_id, n_por_classe, rng)
    sel_0 = _amostrar_classe(cand_0, bloco_id, n_por_classe, rng)

    print(
        f"[{evento}] candidatos válidos: {len(cand_1)} alagados / {len(cand_0)} não-alagados "
        f"-> amostrados: {len(sel_1)} / {len(sel_0)} (alvo {n_por_classe} cada)"
    )

    indices = np.vstack([sel_1, sel_0]) if len(sel_1) and len(sel_0) else (sel_1 if len(sel_1) else sel_0)
    rotulos = np.concatenate([np.ones(len(sel_1), dtype=int), np.zeros(len(sel_0), dtype=int)])

    linhas, colunas = indices[:, 0], indices[:, 1]
    xs, ys = rasterio.transform.xy(transform, linhas, colunas)
    pontos = gpd.GeoSeries(gpd.points_from_xy(xs, ys), crs=CRS_TRABALHO).to_crs("EPSG:4326")

    dados = {
        "evento": evento,
        "lon": pontos.x.values,
        "lat": pontos.y.values,
        "cod_municipio": municipio[linhas, colunas],
        "flood": rotulos,
    }
    for nome, arr in bandas.items():
        dados[nome] = arr[linhas, colunas]

    df = pd.DataFrame(dados)
    df["municipio"] = df["cod_municipio"].map(nomes_municipio)
    return df


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    for evento in (EVENTO_TREINO, EVENTO_TESTE):
        try:
            df = amostrar_evento(evento)
        except Exception as e:  # noqa: BLE001 — queremos seguir pro próximo evento
            print(f"[{evento}] FALHOU: {e}")
            continue

        out_path = DATA_PROCESSED / f"pontos_{evento}.csv"
        df.to_csv(out_path, index=False)
        print(f"[{evento}] {len(df)} pontos -> {out_path}")


if __name__ == "__main__":
    main()
