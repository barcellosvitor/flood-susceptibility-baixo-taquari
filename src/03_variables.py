"""Passo 3 (parte GEE): variáveis sem equivalente local pros dois eventos.

Terreno (elevação, declive, aspecto, hillshade, TPI, curvaturas, LS factor,
TWI, landforms), geologia/solos/geomorfologia, CN/S, EucDist_Drenagem real
e NDVI/NDWI/LULC de SET/2023 agora vêm dos rasters reais dos autores — ver
local_rasters.py e claude.md (achado de 15/10/2026). Este módulo cobre só
o que ainda não tem fonte local pros dois eventos:

- Chuva (5 índices ETCCDI via CHIRPS) — muda por evento, sem equivalente
  estático, precisa pros dois.
- NDVI/NDWI/LULC de MAI/2024 — os autores só mandaram a versão de
  set/2023; pra mai/2024 seguimos calculando via Sentinel-2/MapBiomas.

Requer `earthengine authenticate` com escopo de export (ver README) — a
mediana/composto do CHIRPS e do Sentinel-2 é pesada demais pro download
direto, usa ee_utils.export_via_asset.
"""

import ee
import geemap

from config import (
    CLOUD_SCORE_THRESHOLD,
    CRS_TRABALHO,
    DATA_INTERIM,
    EE_PROJECT,
    EVENTO_TESTE,
    EVENTO_TREINO,
    JANELA_OPTICO_TESTE,
    JANELAS_CHUVA,
    LULC_ANO_TESTE,
    RESOLUCAO_M,
)

AOI_PATH = DATA_INTERIM / "aoi_baixo_taquari.geojson"

MAPBIOMAS_LULC = (
    "projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/"
    "mapbiomas_10m_collection2_integration_v1"
)
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_PLUS = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"


def lulc_teste(aoi: ee.FeatureCollection) -> ee.Image:
    return ee.Image(MAPBIOMAS_LULC).select(LULC_ANO_TESTE).rename("lulc").clip(aoi)


def ndvi_ndwi_teste(aoi: ee.FeatureCollection) -> ee.Image:
    s2 = ee.ImageCollection(S2_COLLECTION).filterBounds(aoi).filterDate(*JANELA_OPTICO_TESTE)
    cs_plus = ee.ImageCollection(CLOUD_SCORE_PLUS)
    s2 = s2.linkCollection(cs_plus, ["cs"])

    def _mascarar_e_indexar(img):
        limpo = img.updateMask(img.select("cs").gte(CLOUD_SCORE_THRESHOLD))
        ndvi = limpo.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ndwi = limpo.normalizedDifference(["B3", "B8"]).rename("ndwi")
        return ndvi.addBands(ndwi)

    composto = s2.map(_mascarar_e_indexar).median()
    return composto.reproject(crs=CRS_TRABALHO, scale=RESOLUCAO_M).clip(aoi)


def chuva_indices(aoi: ee.FeatureCollection, evento: str) -> ee.Image:
    data_inicio, data_fim = JANELAS_CHUVA[evento]
    diario = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterBounds(aoi)
        .filterDate(data_inicio, data_fim)
        .select("precipitation")
    )

    rx1day = diario.max().rename("rx1day")
    r10mm = diario.map(lambda img: img.gt(10)).sum().rename("r10mm")
    r20mm = diario.map(lambda img: img.gt(20)).sum().rename("r20mm")

    lista = diario.toList(diario.size())
    n_janelas = ee.List.sequence(0, diario.size().subtract(5))

    def _soma_5_dias(i):
        subconjunto = ee.ImageCollection(lista.slice(ee.Number(i), ee.Number(i).add(5)))
        return subconjunto.sum()

    rx5day = ee.ImageCollection(n_janelas.map(_soma_5_dias)).max().rename("rx5day")

    # cwd: maior sequência de dias consecutivos com chuva >= 1mm (streak
    # corrente e streak máximo acumulados dia a dia via iterate).
    molhado = diario.map(lambda img: img.gte(1))
    zero = ee.Image(0)

    def _acumular(atual, anterior):
        anterior = ee.List(anterior)
        streak_atual = ee.Image(anterior.get(0))
        streak_max = ee.Image(anterior.get(1))
        novo_streak = streak_atual.add(1).multiply(ee.Image(atual))
        novo_max = streak_max.max(novo_streak)
        return ee.List([novo_streak, novo_max])

    resultado = molhado.toList(molhado.size()).iterate(_acumular, ee.List([zero, zero]))
    cwd = ee.Image(ee.List(resultado).get(1)).rename("cwd")

    composto = rx1day.addBands([rx5day, r10mm, r20mm, cwd])
    return composto.reproject(crs=CRS_TRABALHO, scale=RESOLUCAO_M).clip(aoi)


def build_gee_stack(aoi: ee.FeatureCollection, evento: str) -> ee.Image:
    """Bandas que só o GEE fornece, pro evento dado.

    set/2023: só chuva (NDVI/NDWI/LULC vêm de local_rasters).
    mai/2024: chuva + NDVI/NDWI/LULC (autores não cobrem esse evento).
    """
    stack = chuva_indices(aoi, evento)
    if evento == EVENTO_TESTE:
        stack = stack.addBands([ndvi_ndwi_teste(aoi), lulc_teste(aoi)])
    return stack


def main() -> None:
    ee.Initialize(project=EE_PROJECT)

    if not AOI_PATH.exists():
        raise FileNotFoundError(f"Rode 01_aoi.py primeiro — {AOI_PATH} não existe")
    aoi = geemap.vector_to_ee(str(AOI_PATH))

    for evento in (EVENTO_TREINO, EVENTO_TESTE):
        stack = build_gee_stack(aoi, evento)
        bandas = stack.bandNames().getInfo()

        stats = stack.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi.geometry(),
            scale=RESOLUCAO_M,
            maxPixels=1e10,
            bestEffort=True,
        ).getInfo()

        print(f"\nEvento {evento} — {len(bandas)} bandas (via GEE): {bandas}")
        for banda in bandas:
            print(f"  média {banda}: {stats.get(banda)}")


if __name__ == "__main__":
    main()
