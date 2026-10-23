"""Verificação quantitativa da hipótese do "halo" (semana 9, 22/10/2026,
ver claude.md): os falsos positivos do modelo em mai/2024 coincidem,
espacialmente, com a área que alagou no evento maior de set/2023
(FloodMap real, 336 km²) mas não chegou a alagar em mai/2024
(affected_area-v2, 170 km²)? Se sim, reforça a leitura de que o modelo
está identificando corretamente terreno suscetível que esse evento
específico não atingiu — não "errando" ao acaso.

Reaproveita o GeoTIFF já gerado por 08_mapa_suscetibilidade.py
(diferencas_2024-05.tif) e o rótulo real de set/2023 (local_rasters,
mesma grade — nenhuma reprojeção extra necessária, ambos vêm de
local_rasters.aoi_grid()).
"""

import importlib
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
config = importlib.import_module("config")
import local_rasters

RESOLUCAO_M = config.RESOLUCAO_M


def carregar_diferencas_2024() -> np.ndarray:
    caminho = config.OUTPUTS_FIGURES / "diferencas_2024-05.tif"
    with rasterio.open(caminho) as src:
        return src.read(1)


def km2(mask: np.ndarray) -> float:
    return mask.sum() * (RESOLUCAO_M**2) / 1e6


def main():
    diff = carregar_diferencas_2024()
    flood_2023 = local_rasters.build_flood_label_set2023()
    assert diff.shape == flood_2023.shape, "grades desalinhadas — não deveria acontecer"

    valido = diff != 255
    atual_2024 = np.where(np.isin(diff, [1, 3]), 1, 0)  # rótulo real mai/2024 (TP/FN=1)
    pred_2024 = np.where(np.isin(diff, [1, 2]), 1, 0)  # predição do modelo (TP/FP=1)
    a2023 = flood_2023 == 1

    # --- Estatística principal: FP de mai/2024 x extensão de set/2023 ---
    fp = valido & (diff == 2)
    n_fp = int(fp.sum())
    n_fp_dentro_2023 = int((fp & a2023).sum())
    pct_fp_dentro_2023 = 100 * n_fp_dentro_2023 / n_fp
    pct_aoi_dentro_2023 = 100 * (valido & a2023).sum() / valido.sum()

    print("=== Falsos positivos de mai/2024 x extensão de set/2023 ===")
    print(f"Falsos positivos (mai/2024): {n_fp:,} pixels ({km2(fp):.1f} km²)")
    print(
        f"  ...dos quais dentro da mancha de set/2023: {n_fp_dentro_2023:,} "
        f"({pct_fp_dentro_2023:.1f}%)"
    )
    print(f"Referência: {pct_aoi_dentro_2023:.1f}% da AOI válida está dentro da mancha de set/2023")
    print(f"Razão de enriquecimento: {pct_fp_dentro_2023 / pct_aoi_dentro_2023:.2f}x")

    # --- Comparação de taxa de FP por zona (teste mais rigoroso) ---
    zona_so_2023 = valido & a2023 & (atual_2024 == 0)  # alagou em 2023, não em 2024
    zona_nunca = valido & (~a2023) & (atual_2024 == 0)  # não alagou em nenhum dos dois
    zona_ambos = valido & a2023 & (atual_2024 == 1)  # alagou nos dois
    zona_so_2024 = valido & (~a2023) & (atual_2024 == 1)  # alagou só em 2024

    def taxa_fp(zona: np.ndarray) -> tuple[float, int]:
        n = int(zona.sum())
        if n == 0:
            return float("nan"), 0
        return 100 * (zona & (pred_2024 == 1)).sum() / n, n

    taxa_so2023, n_so2023 = taxa_fp(zona_so_2023)
    taxa_nunca, n_nunca = taxa_fp(zona_nunca)

    print("\n=== Taxa de falso-positivo do modelo por zona (restrito a onde mai/2024 = não-alagado) ===")
    print(
        f"Zona 'alagou em set/2023 mas não em mai/2024' ({km2(zona_so_2023):.1f} km², {n_so2023:,} px): "
        f"taxa de FP = {taxa_so2023:.1f}%"
    )
    print(
        f"Zona 'nunca alagou (nenhum dos dois eventos)' ({km2(zona_nunca):.1f} km², {n_nunca:,} px): "
        f"taxa de FP = {taxa_nunca:.1f}%"
    )
    print(f"Razão: {taxa_so2023 / taxa_nunca:.2f}x")

    print("\nContexto — tamanho das zonas:")
    print(f"  alagou nos dois eventos: {km2(zona_ambos):.1f} km²")
    print(f"  alagou só em mai/2024 (não em set/2023): {km2(zona_so_2024):.1f} km²")

    # --- Visualização ---
    categorias = np.full(diff.shape, 5, dtype=np.uint8)  # 5 = inválido
    categorias[valido & (diff == 0)] = 0  # TN
    categorias[valido & (diff == 1)] = 1  # TP
    categorias[zona_so_2023 & (pred_2024 == 1)] = 2  # FP dentro da mancha 2023 ("halo esperado")
    categorias[zona_nunca & (pred_2024 == 1)] = 3  # FP fora da mancha 2023 (FP "genuíno")
    categorias[valido & (diff == 3)] = 4  # FN

    cores = [
        (0.85, 0.85, 0.85, 1.0),  # TN
        (0.0, 0.4, 0.8, 1.0),  # TP
        (1.0, 0.55, 0.0, 1.0),  # FP dentro da mancha 2023
        (0.9, 0.1, 0.1, 1.0),  # FP fora (genuíno)
        (1.0, 0.85, 0.0, 1.0),  # FN
        (1.0, 1.0, 1.0, 0.0),  # inválido
    ]
    cmap = ListedColormap(cores)
    norm = BoundaryNorm([0, 1, 2, 3, 4, 5, 6], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(categorias, cmap=cmap, norm=norm)
    ax.set_title("mai/2024 — falsos positivos decompostos pela extensão de set/2023")
    ax.axis("off")
    legenda = [
        mpatches.Patch(color=cores[0], label="Acerto não-alagado"),
        mpatches.Patch(color=cores[1], label="Acerto alagado"),
        mpatches.Patch(
            color=cores[2],
            label=f"FP DENTRO da mancha set/2023 ({pct_fp_dentro_2023:.0f}% dos FP)",
        ),
        mpatches.Patch(
            color=cores[3],
            label=f"FP FORA da mancha set/2023 ({100 - pct_fp_dentro_2023:.0f}% dos FP)",
        ),
        mpatches.Patch(color=cores[4], label="Falso negativo"),
    ]
    ax.legend(handles=legenda, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    caminho_png = config.OUTPUTS_FIGURES / "verificacao_halo_2024-05.png"
    fig.savefig(caminho_png, dpi=150)
    plt.close(fig)
    print(f"\nPNG salvo: {caminho_png}")


if __name__ == "__main__":
    main()
