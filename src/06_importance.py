"""Ranking de importância das variáveis (semana 8 do cronograma).

Usa permutation importance (mais robusta e comparável entre modelos do
que a importância nativa de árvore, que precisaria agregar de volta as
colunas do one-hot) sobre os 5 modelos do experimento C (treino em
set/2023 -> teste em mai/2024, ver 05_train.py). Importância medida como
queda de ROC-AUC no conjunto de TESTE (mai/2024) ao embaralhar cada
variável original — mede o que cada variável contribui pra generalizar
entre eventos, não só pro ajuste em set/2023 (C é o resultado principal
do projeto, ver claude.md).

Como a permutação embaralha as colunas ORIGINAIS do DataFrame (antes do
OneHotEncoder, que mora dentro do Pipeline treinado), as categóricas
(geologia/geomorfologia/solos/lulc/landforms) são embaralhadas como uma
coluna só — sem precisar agregar manualmente as dummies do one-hot de
volta ao nome da variável original.

Saídas em outputs/models/: feature_importance.csv (longa, por modelo) e
feature_importance_resumo.csv (ranking consenso, média entre os 5
modelos).
"""

import importlib
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parent))
config = importlib.import_module("config")
treino = importlib.import_module("05_train")

N_REPEATS = 10


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    config.OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    dados = treino.carregar_dados()
    df_treino = dados[config.EVENTO_TREINO]
    df_teste = dados[config.EVENTO_TESTE]

    X_tr, y_tr = treino.preparar_x_y(df_treino, usar_chuva=True)
    X_te, y_te = treino.preparar_x_y(df_teste, usar_chuva=True)
    modelos = treino.construir_modelos(list(X_tr.columns))

    linhas = []
    for nome, pipeline in modelos.items():
        print(f"Treinando {nome} e calculando permutation importance (n_repeats={N_REPEATS})...")
        pipeline.fit(X_tr, y_tr)
        resultado = permutation_importance(
            pipeline,
            X_te,
            y_te,
            scoring="roc_auc",
            n_repeats=N_REPEATS,
            random_state=treino.SEED,
            n_jobs=-1,
        )
        for i, variavel in enumerate(X_te.columns):
            linhas.append(
                {
                    "modelo": nome,
                    "variavel": variavel,
                    "importancia_media": resultado.importances_mean[i],
                    "importancia_desvio": resultado.importances_std[i],
                }
            )

    resultado_df = pd.DataFrame(linhas)

    # Percentual normalizado por modelo (negativo = ruído, não contribuição
    # real — embaralhar a variável "melhorou" a métrica por acaso; zera
    # antes de normalizar, igual à convenção do paper de referência, que
    # reporta importância nativa de árvore, sempre >= 0).
    resultado_df["importancia_pct"] = resultado_df.groupby("modelo")["importancia_media"].transform(
        lambda s: s.clip(lower=0) / s.clip(lower=0).sum() * 100
    )

    caminho = config.OUTPUTS_MODELS / "feature_importance.csv"
    resultado_df.to_csv(caminho, index=False)
    print(f"\nTabela completa salva em {caminho}")

    resumo = (
        resultado_df.groupby("variavel", as_index=False)["importancia_pct"]
        .mean()
        .rename(columns={"importancia_pct": "importancia_pct_media"})
        .sort_values("importancia_pct_media", ascending=False)
        .reset_index(drop=True)
    )
    resumo["ranking"] = resumo.index + 1

    caminho_resumo = config.OUTPUTS_MODELS / "feature_importance_resumo.csv"
    resumo.to_csv(caminho_resumo, index=False)
    print(f"Ranking consenso (média entre os 5 modelos) salvo em {caminho_resumo}")

    print("\n=== Top 10 variáveis (consenso entre os 5 modelos) ===")
    print(resumo.head(10).round(2).to_string(index=False))

    for nome in modelos:
        print(f"\n=== Top 5 variáveis — {nome} ===")
        top = (
            resultado_df[resultado_df["modelo"] == nome]
            .sort_values("importancia_pct", ascending=False)
            .head(5)
        )
        print(top[["variavel", "importancia_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
