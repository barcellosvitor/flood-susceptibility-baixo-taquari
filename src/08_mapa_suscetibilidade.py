"""Mapa de suscetibilidade + mapa de diferenças (semana 9 do cronograma).

Aplica o `best_model_experimento_c.pkl` (treinado em set/2023, ver
05_train.py) em TODOS os pixels válidos da AOI, pros dois eventos, e
compara contra o rótulo real de cada um (FloodMap local em set/2023,
MapBiomas affected_area-v2 em mai/2024).

Reaproveita a primeira metade de `amostrar_evento` (04_sampling.py):
monta exatamente o mesmo stack de bandas (locais + GEE) e a mesma
máscara de validade — só que em vez de amostrar 3.000 pontos, prediz em
todos os pixels válidos (~1-2 milhões, dependendo do evento).

Saídas em outputs/figures/: suscetibilidade_<evento>.tif (probabilidade,
float32, nodata=NaN), diferencas_<evento>.tif (0=TN, 1=TP, 2=FP, 3=FN,
255=inválido) e um PNG lado a lado dos dois mapas por evento.
"""

import importlib
import sys
import warnings
from pathlib import Path

import ee
import geemap
import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
config = importlib.import_module("config")
treino = importlib.import_module("05_train")
sampling = importlib.import_module("04_sampling")

LIMIAR = 0.5
TAMANHO_LOTE = 200_000  # processa a predição em lotes p/ não estourar memória


def montar_stack_completo(evento: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Monta o stack de bandas + rótulo + máscara de validade pra TODA a
    AOI — mesma lógica das primeiras linhas de `amostrar_evento`, mas
    expondo o resultado em vez de amostrar pontos dele."""
    bandas = sampling._bandas_locais(evento)
    rotulo = sampling._rotulo_local(evento)
    municipio, _ = sampling._municipio_raster()

    ee.Initialize(project=config.EE_PROJECT)
    aoi_ee = geemap.vector_to_ee(str(sampling.AOI_PATH))
    bandas.update(sampling._bandas_gee(aoi_ee, evento))
    if rotulo is None:
        rotulo = sampling._rotulo_gee(aoi_ee, evento)

    for col in treino.COLS_EXCLUIDAS:
        bandas.pop(col, None)

    valido = np.ones_like(municipio, dtype=bool)
    for nome, arr in bandas.items():
        if nome in sampling.VARIAVEIS_CATEGORICAS:
            valido &= arr != -1
        else:
            valido &= ~np.isnan(arr)
    valido &= municipio != -1

    return bandas, rotulo, valido


def prever_grade(pipeline, bandas: dict, valido: np.ndarray, colunas: list[str]) -> np.ndarray:
    altura, largura = valido.shape
    prob = np.full((altura, largura), np.nan, dtype=np.float32)
    linhas, colunas_px = np.where(valido)
    n = len(linhas)
    print(f"  prevendo {n:,} pixels válidos em lotes de {TAMANHO_LOTE:,}...")

    for inicio in range(0, n, TAMANHO_LOTE):
        fim = min(inicio + TAMANHO_LOTE, n)
        li, co = linhas[inicio:fim], colunas_px[inicio:fim]
        dados = {}
        for col in colunas:
            arr = bandas[col][li, co]
            if col in treino.COLS_CATEGORICAS:
                arr = arr.round().astype(int).astype(str)
            dados[col] = arr
        X = pd.DataFrame(dados)
        prob[li, co] = pipeline.predict_proba(X)[:, 1]

    return prob


def mapa_diferencas(prob: np.ndarray, rotulo: np.ndarray, valido: np.ndarray, limiar: float = LIMIAR) -> np.ndarray:
    """0=acerto não-alagado (TN), 1=acerto alagado (TP), 2=falso positivo,
    3=falso negativo, 255=inválido/sem predição."""
    pred = prob >= limiar
    diff = np.full(prob.shape, 255, dtype=np.uint8)
    mask = valido & ~np.isnan(prob)
    diff[mask & ~pred & (rotulo == 0)] = 0
    diff[mask & pred & (rotulo == 1)] = 1
    diff[mask & pred & (rotulo == 0)] = 2
    diff[mask & ~pred & (rotulo == 1)] = 3
    return diff


def salvar_geotiffs(prob: np.ndarray, diff: np.ndarray, transform, evento: str) -> None:
    perfil_base = dict(
        driver="GTiff", height=prob.shape[0], width=prob.shape[1],
        crs=config.CRS_TRABALHO, transform=transform, count=1, compress="lzw",
    )
    caminho_prob = config.OUTPUTS_FIGURES / f"suscetibilidade_{evento}.tif"
    with rasterio.open(caminho_prob, "w", dtype="float32", nodata=np.nan, **perfil_base) as dst:
        dst.write(prob, 1)

    caminho_diff = config.OUTPUTS_FIGURES / f"diferencas_{evento}.tif"
    with rasterio.open(caminho_diff, "w", dtype="uint8", nodata=255, **perfil_base) as dst:
        dst.write(diff, 1)
    print(f"  GeoTIFFs salvos: {caminho_prob.name}, {caminho_diff.name}")


def salvar_png(prob: np.ndarray, diff: np.ndarray, evento: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    im0 = axes[0].imshow(prob, cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_title(f"Suscetibilidade a alagamento — {evento}")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.04, label="Probabilidade (modelo C)")

    cores = [
        (0.85, 0.85, 0.85, 1.0),  # TN
        (0.0, 0.4, 0.8, 1.0),     # TP
        (0.9, 0.1, 0.1, 1.0),     # FP
        (1.0, 0.7, 0.0, 1.0),     # FN
        (1.0, 1.0, 1.0, 0.0),     # inválido
    ]
    cmap = ListedColormap(cores)
    norm = BoundaryNorm([0, 1, 2, 3, 4, 256], cmap.N)
    diff_idx = np.where(diff == 255, 4, diff)
    axes[1].imshow(diff_idx, cmap=cmap, norm=norm)
    axes[1].set_title(f"Mapa de diferenças — {evento} (limiar {LIMIAR})")
    axes[1].axis("off")
    legenda = [
        mpatches.Patch(color=cores[0], label="Não alagado (acerto)"),
        mpatches.Patch(color=cores[1], label="Alagado (acerto)"),
        mpatches.Patch(color=cores[2], label="Falso positivo"),
        mpatches.Patch(color=cores[3], label="Falso negativo"),
    ]
    axes[1].legend(handles=legenda, loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle(f"Evento: {evento}", fontsize=13)
    fig.tight_layout()
    caminho = config.OUTPUTS_FIGURES / f"mapa_{evento}.png"
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f"  PNG salvo: {caminho.name}")


def processar_evento(pipeline, colunas: list[str], evento: str, transform) -> dict:
    print(f"\n=== Evento {evento} ===")
    bandas, rotulo, valido = montar_stack_completo(evento)
    print(f"  pixels válidos: {valido.sum():,} / {valido.size:,} ({100 * valido.sum() / valido.size:.1f}%)")

    prob = prever_grade(pipeline, bandas, valido, colunas)
    diff = mapa_diferencas(prob, rotulo, valido)

    salvar_geotiffs(prob, diff, transform, evento)
    salvar_png(prob, diff, evento)

    mask = valido & ~np.isnan(prob)
    n_tn, n_tp, n_fp, n_fn = (int((diff[mask] == c).sum()) for c in (0, 1, 2, 3))
    total = n_tn + n_tp + n_fp + n_fn
    resumo = {
        "evento": evento,
        "n_pixels_validos": total,
        "acerto_nao_alagado": n_tn,
        "acerto_alagado": n_tp,
        "falso_positivo": n_fp,
        "falso_negativo": n_fn,
        "acuracia_pixel": (n_tn + n_tp) / total,
    }
    print(
        f"  TN={n_tn:,} TP={n_tp:,} FP={n_fp:,} FN={n_fn:,} "
        f"(acurácia por pixel: {resumo['acuracia_pixel']:.4f})"
    )
    return resumo


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    config.OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    caminho_modelo = config.OUTPUTS_MODELS / "best_model_experimento_c.pkl"
    pipeline = joblib.load(caminho_modelo)
    colunas = list(pipeline.named_steps["prep"].feature_names_in_)
    print(f"Modelo carregado de {caminho_modelo} ({len(colunas)} variáveis)")

    import local_rasters
    transform, _, _ = local_rasters.aoi_grid()

    resumos = []
    for evento in (config.EVENTO_TREINO, config.EVENTO_TESTE):
        resumos.append(processar_evento(pipeline, colunas, evento, transform))

    resumo_df = pd.DataFrame(resumos)
    caminho_resumo = config.OUTPUTS_FIGURES / "resumo_mapas.csv"
    resumo_df.to_csv(caminho_resumo, index=False)
    print(f"\nResumo salvo em {caminho_resumo}")
    print(resumo_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
