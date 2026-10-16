"""Utilidades compartilhadas para exportar imagens "pesadas" do Earth Engine.

`geemap.ee_export_image` (e `ee.Image.getDownloadURL`) rodam a computação de
forma síncrona/interativa, que tem um orçamento de memória bem menor que um
export em lote — qualquer coisa com várias bandas empilhadas, mediana de
muitas cenas, etc. estoura com "User memory limit exceeded" (sem erro até
tentar baixar). A saída é materializar o resultado como asset (export em
lote, sem esse limite) e só então baixar — nesse ponto já é só ler um
raster pronto, uma operação leve.
"""

import time

import ee
import geemap

from config import EE_PROJECT


def export_via_asset(
    image: ee.Image,
    region: ee.Geometry,
    scale: int,
    out_path,
    description: str,
    delete_asset: bool = True,
) -> None:
    """Materializa `image` como asset temporário, baixa e (por padrão) apaga.

    Use no lugar de `geemap.ee_export_image` sempre que a imagem envolver
    computação pesada (mediana/composição de muitas cenas, várias bandas
    empilhadas, etc.) — para uma imagem já "pronta" (ex. um asset existente
    só clipado), `geemap.ee_export_image` direto é mais simples e mais
    rápido.
    """
    asset_id = f"projects/{EE_PROJECT}/assets/tmp_{description}"

    task = ee.batch.Export.image.toAsset(
        image=image,
        description=description,
        assetId=asset_id,
        region=region,
        scale=scale,
        maxPixels=1e10,
    )
    task.start()

    print(f"Exportando '{description}' em lote (asset temporário)...")
    while True:
        status = task.status()
        state = status["state"]
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(5)

    if state != "COMPLETED":
        raise RuntimeError(f"Export '{description}' terminou como {state}: {status}")

    geemap.ee_export_image(
        ee.Image(asset_id),
        filename=str(out_path),
        scale=scale,
        region=region,
        file_per_band=False,
    )

    if delete_asset:
        ee.data.deleteAsset(asset_id)
