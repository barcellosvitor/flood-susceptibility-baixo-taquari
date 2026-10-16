"""Passo 1: área de estudo (Baixo Taquari).

Constrói a AOI (area of interest) a partir da malha municipal oficial do
MapBiomas (`AUXILIAR/MUNICIPIOS/municipios-RS`, mesma base que os assets de
área atingida em 02_labels.py), filtrada para os municípios em
`config.MUNICIPIOS_BAIXO_TAQUARI`, e exporta o polígono que os demais
scripts (02-04) usam para recortar manchas de água e variáveis.

Trocado de FAO/GAUL/2015/level2 pra essa malha em 24/09/2026: nomes batem
exatos (sem normalizar acento) e a área fica mais precisa — 1049 km² contra
1192 km² da FAO/GAUL pros mesmos 7 municípios (ver claude.md).

Requer `earthengine authenticate` já rodado (ver README.md).
"""

import ee
import geemap

from config import DATA_INTERIM, EE_PROJECT, MUNICIPIOS_BAIXO_TAQUARI

MALHA_MUNICIPAL = "projects/mapbiomas-workspace/AUXILIAR/MUNICIPIOS/municipios-RS"


def build_aoi() -> ee.FeatureCollection:
    municipios = ee.FeatureCollection(MALHA_MUNICIPAL)
    aoi = municipios.filter(ee.Filter.inList("NM_MUN", MUNICIPIOS_BAIXO_TAQUARI))

    encontrados = set(aoi.aggregate_array("NM_MUN").getInfo())
    faltando = set(MUNICIPIOS_BAIXO_TAQUARI) - encontrados
    if faltando:
        raise ValueError(
            f"Municípios não encontrados em {MALHA_MUNICIPAL}: {sorted(faltando)}"
        )

    return aoi


def main() -> None:
    ee.Initialize(project=EE_PROJECT)
    aoi = build_aoi()

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    out_path = DATA_INTERIM / "aoi_baixo_taquari.geojson"
    geemap.ee_export_vector(aoi, filename=str(out_path))

    area_km2 = aoi.geometry().area(1).divide(1e6).getInfo()
    print(
        f"AOI exportada: {out_path} "
        f"({len(MUNICIPIOS_BAIXO_TAQUARI)} municípios, ~{area_km2:.0f} km²)"
    )


if __name__ == "__main__":
    main()
