"""Consolidação final dos resultados pro artigo (semana 9, 22/10/2026).

Junta o que já foi gerado por 05_train.py (experimentos A-D),
06_importance.py (ranking de variáveis), 08_mapa_suscetibilidade.py
(mapa cheio) e 09_verificacao_halo.py (verificação do "halo") numa
tabela única + duas figuras resumo.

Saídas em outputs/models/ e outputs/figures/:
- tabela_final_resultados_longa.csv  (experimento x modelo x 6 métricas)
- tabela_final_resultados_wide.csv   (experimento x modelo, só AUC/Acc/MCC)
- resumo_final_mapas.csv             (pixel-level + verificação do halo)
- comparacao_experimentos.png        (AUC e MCC por experimento x modelo)
- importancia_variaveis_top10.png    (ranking consenso)
"""

import importlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
config = importlib.import_module("config")
import local_rasters

# Paleta categórica fixa (skill dataviz, ordem validada p/ CVD) — um slot
# por modelo, sempre a mesma cor em toda figura.
CORES_MODELOS = {
    "RandomForest": "#2a78d6",  # slot 1 azul
    "GradientBoosting": "#eb6834",  # slot 2 laranja
    "LightGBM": "#1baf7a",  # slot 3 aqua
    "XGBoost": "#eda100",  # slot 4 amarelo
    "CatBoost": "#e87ba4",  # slot 5 magenta
}
ORDEM_MODELOS = list(CORES_MODELOS)
AZUL_SEQUENCIAL = "#2a78d6"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"
COR_TEXTO = "#52514e"

ORDEM_EXPERIMENTOS = [
    ("A", "A — split aleatório\n(set/2023)"),
    ("B_2023", "B — LOMO\n(set/2023)"),
    ("B_2024", "B — LOMO\n(mai/2024)"),
    ("C", "C — set/2023→mai/2024\n(principal)"),
    ("D", "D — sem chuva\n(set/2023→mai/2024)"),
]


def _chave_experimento(row) -> str:
    if row["experimento"] == "B":
        return f"B_{str(row['evento_teste'])[:4]}"
    return row["experimento"]


def montar_tabelas() -> tuple[pd.DataFrame, pd.DataFrame]:
    resumo = pd.read_csv(config.OUTPUTS_MODELS / "model_comparison_resumo.csv")
    resumo["chave_experimento"] = resumo.apply(_chave_experimento, axis=1)

    longa = resumo[
        ["chave_experimento", "modelo", "accuracy_mean", "f1_mean", "precision_mean",
         "recall_mean", "roc_auc_mean", "mcc_mean"]
    ].rename(columns={
        "chave_experimento": "experimento",
        "accuracy_mean": "accuracy",
        "f1_mean": "f1",
        "precision_mean": "precision",
        "recall_mean": "recall",
        "roc_auc_mean": "roc_auc",
        "mcc_mean": "mcc",
    })

    ordem_chaves = [k for k, _ in ORDEM_EXPERIMENTOS]
    longa["experimento"] = pd.Categorical(longa["experimento"], categories=ordem_chaves, ordered=True)
    longa["modelo"] = pd.Categorical(longa["modelo"], categories=ORDEM_MODELOS, ordered=True)
    longa = longa.sort_values(["experimento", "modelo"]).reset_index(drop=True)

    wide = longa.pivot_table(index="modelo", columns="experimento", values=["roc_auc", "accuracy", "mcc"])
    wide.columns = [f"{exp}_{metrica}" for metrica, exp in wide.columns]
    ordem_colunas = [f"{exp}_{m}" for exp in ordem_chaves for m in ("roc_auc", "accuracy", "mcc")]
    wide = wide[ordem_colunas].reset_index()

    return longa, wide


def grafico_comparacao_experimentos(longa: pd.DataFrame) -> None:
    fig, (ax_auc, ax_mcc) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    chaves = [k for k, _ in ORDEM_EXPERIMENTOS]
    rotulos = [r for _, r in ORDEM_EXPERIMENTOS]
    n_modelos = len(ORDEM_MODELOS)
    largura_barra = 0.8 / n_modelos
    x = np.arange(len(chaves))

    for ax, metrica, titulo in ((ax_auc, "roc_auc", "ROC-AUC"), (ax_mcc, "mcc", "MCC")):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=COR_GRADE, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COR_EIXO)

        for i, modelo in enumerate(ORDEM_MODELOS):
            valores = []
            for chave in chaves:
                linha = longa[(longa["experimento"] == chave) & (longa["modelo"] == modelo)]
                valores.append(linha[metrica].iloc[0] if len(linha) else np.nan)
            offset = (i - (n_modelos - 1) / 2) * largura_barra
            ax.bar(
                x + offset, valores, width=largura_barra * 0.9,
                color=CORES_MODELOS[modelo], label=modelo if metrica == "roc_auc" else None,
            )

        ax.set_ylabel(titulo, color=COR_TEXTO)
        ax.tick_params(colors=COR_TEXTO)
        ax.set_ylim(0, 1.02)

    ax_mcc.set_xticks(x)
    ax_mcc.set_xticklabels(rotulos, fontsize=9, color=COR_TEXTO)
    fig.suptitle(
        "Comparação entre experimentos A-D (ROC-AUC e MCC por modelo)",
        y=0.98, fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    ax_auc.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=n_modelos,
        frameon=False, fontsize=9,
    )

    caminho = config.OUTPUTS_FIGURES / "comparacao_experimentos.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva: {caminho}")


def grafico_importancia(top_n: int = 10) -> None:
    resumo = pd.read_csv(config.OUTPUTS_MODELS / "feature_importance_resumo.csv")
    top = resumo.sort_values("importancia_pct_media", ascending=False).head(top_n)

    fig, (ax_tudo, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5))

    for ax in (ax_tudo, ax_zoom):
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=COR_GRADE, linewidth=0.8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COR_EIXO)
        ax.tick_params(colors=COR_TEXTO)

    ax_tudo.barh(top["variavel"][::-1], top["importancia_pct_media"][::-1], color=AZUL_SEQUENCIAL)
    ax_tudo.set_xlabel("Importância (% média, 5 modelos)", color=COR_TEXTO)
    ax_tudo.set_title(f"Top {top_n} — escala completa", fontsize=10, color=COR_TEXTO)

    resto = top.iloc[1:]  # exclui elevation p/ ver estrutura das demais
    ax_zoom.barh(resto["variavel"][::-1], resto["importancia_pct_media"][::-1], color=AZUL_SEQUENCIAL)
    ax_zoom.set_xlabel("Importância (% média, 5 modelos)", color=COR_TEXTO)
    ax_zoom.set_title(f"Ranks 2-{top_n} — zoom (sem elevation, 91,1%)", fontsize=10, color=COR_TEXTO)

    fig.suptitle("Importância de variáveis (permutation importance, consenso 5 modelos)", fontsize=13)
    fig.tight_layout()

    caminho = config.OUTPUTS_FIGURES / "importancia_variaveis_top10.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura salva: {caminho}")


def tabela_mapas_com_halo() -> pd.DataFrame:
    """resumo_mapas.csv (pixel-level) + verificação do halo (09_verificacao_halo)."""
    import rasterio

    resumo = pd.read_csv(config.OUTPUTS_FIGURES / "resumo_mapas.csv")

    diff_path = config.OUTPUTS_FIGURES / "diferencas_2024-05.tif"
    with rasterio.open(diff_path) as src:
        diff = src.read(1)
    flood_2023 = local_rasters.build_flood_label_set2023()

    valido = diff != 255
    atual_2024 = np.where(np.isin(diff, [1, 3]), 1, 0)
    pred_2024 = np.where(np.isin(diff, [1, 2]), 1, 0)
    a2023 = flood_2023 == 1

    fp = valido & (diff == 2)
    pct_fp_dentro_2023 = 100 * (fp & a2023).sum() / fp.sum()

    zona_so_2023 = valido & a2023 & (atual_2024 == 0)
    zona_nunca = valido & (~a2023) & (atual_2024 == 0)
    taxa_so2023 = 100 * (zona_so_2023 & (pred_2024 == 1)).sum() / zona_so_2023.sum()
    taxa_nunca = 100 * (zona_nunca & (pred_2024 == 1)).sum() / zona_nunca.sum()

    resumo["pct_fp_dentro_mancha_set2023"] = [np.nan, round(pct_fp_dentro_2023, 1)]
    resumo["taxa_fp_zona_so_set2023_pct"] = [np.nan, round(taxa_so2023, 1)]
    resumo["taxa_fp_zona_nunca_alagou_pct"] = [np.nan, round(taxa_nunca, 1)]

    caminho = config.OUTPUTS_FIGURES / "resumo_final_mapas.csv"
    resumo.to_csv(caminho, index=False)
    print(f"Tabela salva: {caminho}")
    return resumo


def main():
    config.OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)
    config.OUTPUTS_FIGURES.mkdir(parents=True, exist_ok=True)

    longa, wide = montar_tabelas()
    caminho_longa = config.OUTPUTS_MODELS / "tabela_final_resultados_longa.csv"
    caminho_wide = config.OUTPUTS_MODELS / "tabela_final_resultados_wide.csv"
    longa.to_csv(caminho_longa, index=False)
    wide.to_csv(caminho_wide, index=False)
    print(f"Tabela salva: {caminho_longa}")
    print(f"Tabela salva: {caminho_wide}")

    print("\n=== Tabela final (AUC / Accuracy / MCC por experimento x modelo) ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(wide.round(3).to_string(index=False))

    grafico_comparacao_experimentos(longa)
    grafico_importancia()
    tabela_mapas_com_halo()


if __name__ == "__main__":
    main()
