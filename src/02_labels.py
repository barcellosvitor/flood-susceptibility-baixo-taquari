"""Passo 2: rótulos — manchas de água/dano dos dois eventos.

Duas fontes bem diferentes agora (ver claude.md, achado de 15/10/2026):

1. set/2023 (treino): FloodMap real dos autores do paper de referência
   (local_rasters.py, processado localmente com rasterio) — substitui o
   fallback via Sentinel-1 que tínhamos antes (~1,1 km², muito subestimado
   por falta de imagem pré-evento pra essa região). O FloodMap real dá
   ~336 km² alagados na nossa AOI — ordem de grandeza plausível pra "a
   maior cheia em 150 anos" nessa região.
2. mai/2024 (teste): mancha oficial "área atingida" do MapBiomas RS
   Climate Emergency Toolkit, via GEE (sem mudança) — ~170 km² na AOI.
   É "área atingida" em geral (inclui dano além de água — ver ressalva em
   config.py), não uma máscara de água pura.

RESSALVA pro artigo: os dois rótulos têm definições diferentes (FloodMap =
inundação em si; affected_area = união de 3 métodos + Defesa Civil + HAND,
mais conservador) — comparação entre os dois eventos não é 1:1, documentar
isso.

Requer data/interim/aoi_baixo_taquari.geojson (saída do 01_aoi.py) pros
dois eventos, e `earthengine authenticate` só pro de mai/2024.
"""

import ee
import geemap

import local_rasters
from config import (
    DATA_INTERIM,
    EE_PROJECT,
    EVENTO_TESTE,
    EVENTO_TREINO,
    MAPBIOMAS_AFFECTED_AREA,
    RESOLUCAO_M,
)

AOI_PATH = DATA_INTERIM / "aoi_baixo_taquari.geojson"


def build_flood_mask_mapbiomas(aoi: ee.FeatureCollection, asset_id: str) -> ee.Image:
    """Mancha binária a partir do asset oficial do MapBiomas Toolkit.

    O asset já vem como máscara binária (valor 1 = atingido, resto
    mascarado) na resolução nativa de 10 m — só precisa clip + unmask.
    """
    return ee.Image(asset_id).clip(aoi).unmask(0).rename("flood")


def _label_treino() -> None:
    """set/2023 — local, FloodMap real dos autores."""
    flood = local_rasters.build_flood_label_set2023()
    area_km2 = flood.sum() * (RESOLUCAO_M**2) / 1e6

    out_path = DATA_INTERIM / f"mancha_{EVENTO_TREINO}.tif"
    local_rasters.salvar_geotiff(flood, out_path, dtype="uint8", nodata=255)
    print(f"Mancha {EVENTO_TREINO} [FloodMap autores]: {out_path} (~{area_km2:.1f} km² atingidos)")


def _label_teste() -> None:
    """mai/2024 — GEE, mancha oficial do MapBiomas."""
    ee.Initialize(project=EE_PROJECT)
    aoi = geemap.vector_to_ee(str(AOI_PATH))

    mancha = build_flood_mask_mapbiomas(aoi, MAPBIOMAS_AFFECTED_AREA[EVENTO_TESTE])
    stats = mancha.selfMask().multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi.geometry(),
        scale=RESOLUCAO_M,
        maxPixels=1e10,
    )
    area_km2 = ee.Number(stats.get("flood", 0)).divide(1e6).getInfo()

    out_path = DATA_INTERIM / f"mancha_{EVENTO_TESTE}.tif"
    geemap.ee_export_image(
        mancha,
        filename=str(out_path),
        scale=RESOLUCAO_M,
        region=aoi.geometry(),
        file_per_band=False,
    )
    print(f"Mancha {EVENTO_TESTE} [MapBiomas Toolkit]: {out_path} (~{area_km2:.1f} km² atingidos)")


def main() -> None:
    if not AOI_PATH.exists():
        raise FileNotFoundError(f"Rode 01_aoi.py primeiro — {AOI_PATH} não existe")

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)

    _label_treino()
    _label_teste()


if __name__ == "__main__":
    main()
