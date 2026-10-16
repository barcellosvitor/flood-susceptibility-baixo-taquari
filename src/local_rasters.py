"""Leitura e alinhamento LOCAL dos rasters originais do paper de referência.

Em 15/10/2026 os autores (Alcântara et al.) passaram os rasters reais
usados no estudo — ver `data/raw/taquari_antas_autores/` e a entrada
correspondente em claude.md. Esse módulo reprojeta/recorta cada um pra
nossa grade comum (AOI do Baixo Taquari, CRS_TRABALHO, RESOLUCAO_M) usando
rasterio — processamento local, não passa pelo Google Earth Engine
(decisão tomada em 15/10/2026 pra evitar a fricção recorrente com o GEE:
limite de memória, autenticação de export, bugs de projeção).

ATENÇÃO — os arquivos de origem declaram CRS EPSG:32718 (UTM zona 18S),
o que é geograficamente errado pra essa região (zona 18S cobre Peru/
Bolívia). Investigado e validado: os números foram genuinamente calculados
sob a fórmula dessa zona (não é só uma etiqueta trocada), o que distorce a
malha bruta — mas reprojetar com ferramenta padrão (rasterio.warp) usando
o CRS DECLARADO recupera a geografia certa (testado: bounds reprojetados
contêm nossa AOI corretamente). NUNCA usar os metros brutos desses
arquivos sem reprojetar primeiro.

Duas categorias de variável:
- ESTÁTICAS (terreno, solo, geologia): mesmo raster serve pros dois
  eventos — terreno não muda entre set/2023 e mai/2024.
- ESPECÍFICAS DE SET/2023 (NDVI, NDWI, LULC_2023, e o próprio rótulo
  FloodMap): os autores só mandaram a versão de set/2023. Pra mai/2024
  continuamos usando as fontes via GEE já implementadas em
  03_variables.py (Sentinel-2/MapBiomas por evento).

TPI não veio nos arquivos dos autores — calculado aqui mesmo (DEM menos
média focal numa vizinhança quadrada ~equivalente ao raio usado antes).

`reprojetar_arquivo()` é genérico (recebe um Path qualquer) — além dos
rasters dos autores, 04_sampling.py reusa pra alinhar nessa mesma grade
qualquer banda baixada do GEE (chuva, NDVI/NDWI/LULC de mai/2024), assim
tudo fica garantidamente no mesmo pixel a pixel não importa a origem.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

from config import CRS_TRABALHO, DATA_INTERIM, DATA_RAW, RESOLUCAO_M, TPI_RAIO_M

AOI_PATH = DATA_INTERIM / "aoi_baixo_taquari.geojson"
RASTER_ROOT = DATA_RAW / "taquari_antas_autores"

# nome da variável -> (caminho relativo dentro de RASTER_ROOT, é categórica?)
# Categórica = resampling nearest (classe não pode ser "misturada" por
# interpolação); contínua = bilinear.
RASTERS_ESTATICOS = {
    "elevation": ("Sentinel 1 SAR/DEM.tif", False),
    "slope_degree": ("Sentinel 1 SAR/Slope_Degree.tif", False),
    "aspect": ("Sentinel 1 SAR/Aspect_Planar.tif", False),
    "hillshade": ("Sentinel 1 SAR/HillSha.tif", False),
    "ls_factor": ("Sentinel 1 SAR/LSFactor.tif", False),
    "plan_curvature": ("Sentinel 1 SAR/PlanCurvature.tif", False),
    "profile_curvature": ("Sentinel 1 SAR/ProfileCurvature.tif", False),
    "twi": ("Sentinel 1 SAR/TopographicWetnessIndex.tif", False),
    "landforms": ("FloodMap_SEP052023/Landforms.tif", True),
    "dist_drenagem": ("EucDist/EucDist_Drenagem.tif", False),
    "dist_edificacoes": ("EucDist/EucDist_Buildings.tif", False),
    "dist_rodovias": ("EucDist/EucDist_Roads.tif", False),
    "geologia": ("Temáticos/Geologia.tif", True),
    "geomorfologia": ("Temáticos/Geomorfologia.tif", True),
    "solos": ("Temáticos/Solos.tif", True),
    # CN/S: valor numérico, mas atrelado a uma combinação discreta de
    # classe de solo x uso — médias intermediárias (bilinear) não
    # correspondem a nenhuma combinação real, então nearest também aqui.
    "cn_grid": ("CN_S_grid/CN_grid.tif", True),
    "s_grid": ("CN_S_grid/S_grid.tif", True),
}

# Só têm a versão de set/2023 — usar GEE (03_variables.py) pra mai/2024.
RASTERS_SET2023 = {
    "ndvi": ("Sentinel 2 MSI/NDVI.tif", False),
    "ndwi": ("Sentinel 2 MSI/NDWI.tif", False),
    "lulc": ("Temáticos/LULC_2023.tif", True),
}

FLOODMAP_SET2023 = "FloodMap_SEP052023/FloodMap.tif"


def aoi_grid():
    """Grade alvo (transform, largura, altura) a partir da nossa AOI."""
    aoi = gpd.read_file(AOI_PATH).to_crs(CRS_TRABALHO)
    left, bottom, right, top = aoi.total_bounds
    largura = int(np.ceil((right - left) / RESOLUCAO_M))
    altura = int(np.ceil((top - bottom) / RESOLUCAO_M))
    transform = from_bounds(left, bottom, right, top, largura, altura)
    return transform, largura, altura


def reprojetar_arquivo(caminho: Path, categorica: bool) -> np.ndarray:
    """Reprojeta+recorta QUALQUER raster georreferenciado pra nossa grade.

    Nodata vira NaN (contínuas) ou -1 (categóricas) — nunca propaga o
    nodata original (que varia por arquivo: 127, 255, 2147483647, etc.).
    """
    transform, largura, altura = aoi_grid()
    resampling = Resampling.nearest if categorica else Resampling.bilinear
    nodata_destino = -1.0 if categorica else np.nan

    destino = np.full((altura, largura), nodata_destino, dtype=np.float32)
    with rasterio.open(caminho) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destino,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=CRS_TRABALHO,
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=nodata_destino,
        )
    return destino


def reprojetar_recortar(caminho_relativo: str, categorica: bool) -> np.ndarray:
    """Atalho de reprojetar_arquivo() pros rasters dos autores (RASTER_ROOT)."""
    return reprojetar_arquivo(RASTER_ROOT / caminho_relativo, categorica)


def salvar_geotiff(arr: np.ndarray, out_path: Path, dtype: str = "float32", nodata=None) -> None:
    """Salva um array já alinhado na nossa grade como GeoTIFF (EPSG:32722)."""
    transform, largura, altura = aoi_grid()
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=altura,
        width=largura,
        count=1,
        dtype=dtype,
        crs=CRS_TRABALHO,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def _tpi(elevacao: np.ndarray) -> np.ndarray:
    """TPI = elevação menos a média numa vizinhança quadrada ao redor.

    Aproximação de janela quadrada (não circular) do `focal_mean(..., "circle")`
    que usávamos via GEE — diferença pequena, técnica padrão equivalente.
    """
    raio_px = round(TPI_RAIO_M / RESOLUCAO_M)
    tamanho = 2 * raio_px + 1
    media_local = uniform_filter(elevacao, size=tamanho, mode="nearest")
    return elevacao - media_local


def build_static_stack() -> dict[str, np.ndarray]:
    """Todas as variáveis estáticas (terreno + temáticas), já alinhadas."""
    bandas = {
        nome: reprojetar_recortar(caminho, categorica)
        for nome, (caminho, categorica) in RASTERS_ESTATICOS.items()
    }
    bandas["tpi"] = _tpi(bandas["elevation"])
    return bandas


def build_set2023_stack() -> dict[str, np.ndarray]:
    """NDVI/NDWI/LULC específicos de set/2023, já alinhados."""
    return {
        nome: reprojetar_recortar(caminho, categorica)
        for nome, (caminho, categorica) in RASTERS_SET2023.items()
    }


def build_flood_label_set2023() -> np.ndarray:
    """Rótulo binário (1 = alagado) de set/2023, já alinhado."""
    bruto = reprojetar_recortar(FLOODMAP_SET2023, categorica=True)
    return (bruto == 1).astype(np.uint8)


def main() -> None:
    transform, largura, altura = aoi_grid()
    n_pixels = largura * altura
    print(f"Grade alvo: {largura}x{altura} ({n_pixels} pixels, {RESOLUCAO_M} m, {CRS_TRABALHO})")

    flood = build_flood_label_set2023()
    area_km2 = flood.sum() * (RESOLUCAO_M**2) / 1e6
    print(f"\nFloodMap set/2023: {int(flood.sum())} pixels alagados (~{area_km2:.1f} km²)")

    print("\nVariáveis estáticas:")
    for nome, arr in build_static_stack().items():
        valido = arr[(~np.isnan(arr)) & (arr != -1)]
        cobertura = 100 * valido.size / n_pixels
        if valido.size:
            print(f"  {nome}: cobertura {cobertura:.0f}%  min={valido.min():.3f}  max={valido.max():.3f}")
        else:
            print(f"  {nome}: SEM PIXELS VÁLIDOS NA AOI")

    print("\nVariáveis específicas de set/2023 (NDVI/NDWI/LULC):")
    for nome, arr in build_set2023_stack().items():
        valido = arr[(~np.isnan(arr)) & (arr != -1)]
        cobertura = 100 * valido.size / n_pixels
        if valido.size:
            print(f"  {nome}: cobertura {cobertura:.0f}%  min={valido.min():.3f}  max={valido.max():.3f}")
        else:
            print(f"  {nome}: SEM PIXELS VÁLIDOS NA AOI")


if __name__ == "__main__":
    main()
