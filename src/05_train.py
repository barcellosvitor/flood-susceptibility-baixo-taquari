"""Treino e avaliação dos 5 modelos nos 4 experimentos (A-D).

Carrega os dois datasets gerados por 04_sampling.py (pontos_2023-09.csv,
pontos_2024-05.csv) e roda:

  A. Split aleatório 70/30 dentro de set/2023 — reproduz o protocolo do
     paper de referência (Alcântara et al. 2025). Esperado: métricas
     infladas por vazamento espacial (autocorrelação entre pixels
     vizinhos no split aleatório) — ver claude.md "Problemas a corrigir".
  B. Leave-one-municipality-out dentro de cada evento — validação
     espacial: a diferença entre A e B é uma contribuição do artigo.
  C. Treino em set/2023, teste em mai/2024 — resultado principal.
     Prioriza ROC-AUC sobre acurácia/limiar fixo de 0,5 (mai/2024 foi
     maior que set/2023 na nossa AOI, um limiar fixo tende a
     subestimar).
  D. Igual a C, mas sem as variáveis de chuva (rx1day/rx5day/r20mm) —
     testa se o terreno sozinho generaliza melhor.

Variáveis categóricas nominais (geologia, geomorfologia, solos, lulc,
landforms) recebem one-hot encoding, igual pros 5 modelos — ver
claude.md "Problemas a corrigir": no código original do paper elas
eram tratadas como números, o que é um bug. cn_grid/s_grid são
discretas por construção (lookup CN x uso do solo) mas numericamente
ordenadas (maior CN = mais escoamento) — tratadas como contínuas.

`r10mm` e `cwd` são removidas em `carregar_dados()`, antes de qualquer
experimento: em set/2023 (evento de TREINO) são EXATAMENTE constantes —
um único valor nos 3.000 pontos amostrados — então não podem contribuir
com nenhum split em nenhum modelo de árvore, por construção matemática
(não é uma observação de "baixa importância" feita olhando o teste,
é verificável só com dado de treino). Confirmado empiricamente em
leave-one-municipality-out dentro de set/2023 e no experimento C: cortar
as duas não muda os resultados (ver claude.md, semana 8). `r20mm` fica —
tem variância (pequena, mas real) e não se qualifica pelo mesmo critério.

Saídas em outputs/models/: model_comparison.csv (tabela longa, uma
linha por experimento/modelo/contexto), model_comparison_resumo.csv
(médias agregadas, útil pro artigo) e best_model_experimento_c.pkl
(pipeline completo do melhor modelo do resultado principal).
"""

import importlib
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
config = importlib.import_module("config")

SEED = 42

# Mesma contagem de árvores/iterações do código de referência do paper
# ("50 árvores/iterações, hiperparâmetros padrão") — usada nos 4
# experimentos pra manter a comparação A-D apples-to-apples. Tuning de
# hiperparâmetros é trabalho futuro, fora do escopo desta entrega.
N_ESTIMATORS = 50

COLS_ID = ["evento", "lon", "lat", "cod_municipio", "municipio"]
TARGET = "flood"
COLS_CATEGORICAS = ["geologia", "geomorfologia", "solos", "lulc", "landforms"]
COLS_CHUVA = ["rx1day", "rx5day", "r20mm"]
# Constantes por construção em set/2023 (treino) — ver docstring do módulo.
COLS_EXCLUIDAS = ["r10mm", "cwd"]


# --- Dados ----------------------------------------------------------------

def carregar_dados() -> dict[str, pd.DataFrame]:
    caminhos = {
        config.EVENTO_TREINO: config.DATA_PROCESSED / f"pontos_{config.EVENTO_TREINO}.csv",
        config.EVENTO_TESTE: config.DATA_PROCESSED / f"pontos_{config.EVENTO_TESTE}.csv",
    }
    dados = {}
    for evento, caminho in caminhos.items():
        if not caminho.exists():
            raise FileNotFoundError(
                f"{caminho} não existe — rode 04_sampling.py antes de treinar."
            )
        df = pd.read_csv(caminho)
        for col in COLS_CATEGORICAS:
            df[col] = df[col].round().astype(int).astype(str)
        df = df.drop(columns=COLS_EXCLUIDAS)
        dados[evento] = df
    return dados


def preparar_x_y(df: pd.DataFrame, usar_chuva: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    excluir = set(COLS_ID) | {TARGET}
    if not usar_chuva:
        excluir |= set(COLS_CHUVA)
    cols = [c for c in df.columns if c not in excluir]
    return df[cols].copy(), df[TARGET].copy()


# --- Modelos ---------------------------------------------------------------

def construir_preprocessador(colunas: list[str]) -> ColumnTransformer:
    categoricas = [c for c in colunas if c in COLS_CATEGORICAS]
    numericas = [c for c in colunas if c not in COLS_CATEGORICAS]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
            ("num", "passthrough", numericas),
        ]
    )


def construir_modelos(colunas: list[str]) -> dict[str, Pipeline]:
    prep = construir_preprocessador(colunas)
    base = {
        "RandomForest": RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=SEED, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, random_state=SEED
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=N_ESTIMATORS, random_state=SEED, verbosity=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=N_ESTIMATORS, random_state=SEED, verbose=False
        ),
    }
    # Cada modelo tem sua própria instância do preprocessador (clonada pelo
    # Pipeline no fit), senão o OneHotEncoder ajustado por um modelo seria
    # reaproveitado (já ajustado) pelos outros.
    return {
        nome: Pipeline([("prep", construir_preprocessador(colunas)), ("modelo", modelo)])
        for nome, modelo in base.items()
    }


# --- Avaliação --------------------------------------------------------------

def avaliar(y_true, y_pred, y_proba) -> dict:
    if len(set(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def treinar_e_avaliar(pipeline: Pipeline, X_treino, y_treino, X_teste, y_teste) -> dict:
    pipeline.fit(X_treino, y_treino)
    y_pred = pipeline.predict(X_teste)
    y_proba = pipeline.predict_proba(X_teste)[:, 1]
    metricas = avaliar(y_teste, y_pred, y_proba)
    metricas["n_treino"] = len(y_treino)
    metricas["n_teste"] = len(y_teste)
    return metricas


def linha_base(experimento, evento_treino, evento_teste, contexto, modelo) -> dict:
    return {
        "experimento": experimento,
        "evento_treino": evento_treino,
        "evento_teste": evento_teste,
        "contexto": contexto,
        "modelo": modelo,
    }


# --- Experimento A: split aleatório dentro de set/2023 --------------------

def experimento_a(df_treino: pd.DataFrame) -> list[dict]:
    print("\n=== Experimento A: split aleatório 70/30 em set/2023 (reprodução) ===")
    X, y = preparar_x_y(df_treino, usar_chuva=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    modelos = construir_modelos(list(X.columns))
    linhas = []
    for nome, pipeline in modelos.items():
        m = treinar_e_avaliar(pipeline, X_tr, y_tr, X_te, y_te)
        linha = linha_base("A", config.EVENTO_TREINO, config.EVENTO_TREINO, "split_aleatorio_70_30", nome)
        linha.update(m)
        linhas.append(linha)
        print(f"  {nome:18s} AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}  MCC={m['mcc']:.4f}")
    return linhas


# --- Experimento B: leave-one-municipality-out -----------------------------

def experimento_b(dados: dict[str, pd.DataFrame]) -> list[dict]:
    print("\n=== Experimento B: leave-one-municipality-out (por evento) ===")
    linhas = []
    for evento, df in dados.items():
        municipios = sorted(df["municipio"].unique())
        X_completo, y_completo = preparar_x_y(df, usar_chuva=True)
        modelos = construir_modelos(list(X_completo.columns))
        for municipio in municipios:
            mascara_teste = df["municipio"] == municipio
            X_tr, y_tr = X_completo[~mascara_teste], y_completo[~mascara_teste]
            X_te, y_te = X_completo[mascara_teste], y_completo[mascara_teste]
            for nome, pipeline in modelos.items():
                # Reaproveita o mesmo objeto Pipeline entre os folds: cada
                # .fit() treina do zero e sobrescreve o estado anterior
                # (OneHotEncoder e os 5 modelos não acumulam entre chamadas).
                m = treinar_e_avaliar(pipeline, X_tr, y_tr, X_te, y_te)
                linha = linha_base("B", evento, evento, f"loo_municipio={municipio}", nome)
                linha.update(m)
                linhas.append(linha)
        print(f"  evento {evento}: {len(municipios)} municípios avaliados")
    return linhas


# --- Experimentos C/D: treino set/2023 -> teste mai/2024 -------------------

def experimento_cd(df_treino: pd.DataFrame, df_teste: pd.DataFrame, usar_chuva: bool, rotulo: str) -> list[dict]:
    contexto = "com_chuva" if usar_chuva else "sem_chuva"
    print(f"\n=== Experimento {rotulo}: treino {config.EVENTO_TREINO} -> teste {config.EVENTO_TESTE} ({contexto}) ===")
    X_tr, y_tr = preparar_x_y(df_treino, usar_chuva=usar_chuva)
    X_te, y_te = preparar_x_y(df_teste, usar_chuva=usar_chuva)
    modelos = construir_modelos(list(X_tr.columns))
    linhas = []
    melhor_auc = -1.0
    melhor_nome, melhor_pipeline = None, None
    for nome, pipeline in modelos.items():
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)[:, 1]
        m = avaliar(y_te, y_pred, y_proba)
        m["n_treino"] = len(y_tr)
        m["n_teste"] = len(y_te)
        linha = linha_base(rotulo, config.EVENTO_TREINO, config.EVENTO_TESTE, contexto, nome)
        linha.update(m)
        linhas.append(linha)
        print(f"  {nome:18s} AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}  MCC={m['mcc']:.4f}")
        if m["roc_auc"] > melhor_auc:
            melhor_auc, melhor_nome, melhor_pipeline = m["roc_auc"], nome, pipeline
    if rotulo == "C" and melhor_pipeline is not None:
        caminho = config.OUTPUTS_MODELS / "best_model_experimento_c.pkl"
        joblib.dump(melhor_pipeline, caminho)
        print(f"  melhor modelo ({melhor_nome}, AUC={melhor_auc:.4f}) salvo em {caminho}")
    return linhas


# --- Main --------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    config.OUTPUTS_MODELS.mkdir(parents=True, exist_ok=True)

    dados = carregar_dados()
    df_treino = dados[config.EVENTO_TREINO]
    df_teste = dados[config.EVENTO_TESTE]

    linhas = []
    linhas += experimento_a(df_treino)
    linhas += experimento_b(dados)
    linhas += experimento_cd(df_treino, df_teste, usar_chuva=True, rotulo="C")
    linhas += experimento_cd(df_treino, df_teste, usar_chuva=False, rotulo="D")

    resultado = pd.DataFrame(linhas)
    caminho_completo = config.OUTPUTS_MODELS / "model_comparison.csv"
    resultado.to_csv(caminho_completo, index=False)
    print(f"\nTabela completa ({len(resultado)} linhas) salva em {caminho_completo}")

    resumo = (
        resultado.groupby(["experimento", "evento_teste", "contexto", "modelo"], as_index=False)[
            ["accuracy", "f1", "precision", "recall", "roc_auc", "mcc"]
        ]
        .agg(["mean", "std"])
    )
    resumo.columns = ["_".join(c).rstrip("_") for c in resumo.columns]
    # Experimento B tem um contexto por município (muitas linhas) — a média
    # por modelo/evento é o que interessa pro resumo; A/C/D já têm 1 linha
    # por modelo, então agrupar por contexto ali só identifica a linha.
    resumo_b = (
        resultado[resultado["experimento"] == "B"]
        .groupby(["evento_teste", "modelo"], as_index=False)[
            ["accuracy", "f1", "precision", "recall", "roc_auc", "mcc"]
        ]
        .agg(["mean", "std"])
    )
    resumo_b.columns = ["_".join(c).rstrip("_") for c in resumo_b.columns]
    resumo_b.insert(0, "experimento", "B")

    resumo_outros = resultado[resultado["experimento"] != "B"].copy()
    for col in ["accuracy", "f1", "precision", "recall", "roc_auc", "mcc"]:
        resumo_outros[f"{col}_mean"] = resumo_outros[col]
        resumo_outros[f"{col}_std"] = 0.0
    resumo_outros = resumo_outros[
        ["experimento", "evento_teste", "contexto", "modelo"]
        + [f"{c}_mean" for c in ["accuracy", "f1", "precision", "recall", "roc_auc", "mcc"]]
        + [f"{c}_std" for c in ["accuracy", "f1", "precision", "recall", "roc_auc", "mcc"]]
    ]

    caminho_resumo = config.OUTPUTS_MODELS / "model_comparison_resumo.csv"
    resumo_final = pd.concat([resumo_outros, resumo_b], ignore_index=True)
    resumo_final.to_csv(caminho_resumo, index=False)
    print(f"Resumo agregado salvo em {caminho_resumo}")

    print("\n=== Resumo do experimento B (média ± desvio entre municípios) ===")
    print(resumo_b.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
