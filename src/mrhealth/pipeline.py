"""Pipeline de ingestão, qualidade, features e modelo de demanda da Mr. Health.

Extraído do notebook `notebooks/mr_health_demand_forecast.py` (change
`mr-health-demand-forecast-notebook`) para ser reaproveitado tanto pelo
notebook quanto pelo app Streamlit (change `mr-health-stock-decision-app`),
sem duplicar a lógica de dados/modelo em dois lugares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

RANDOM_STATE = 42

# Dias da semana em português (índice 0 = segunda-feira, padrão pandas .dayofweek)
DIAS_SEMANA_PT = [
    "Segunda-feira",
    "Terça-feira",
    "Quarta-feira",
    "Quinta-feira",
    "Sexta-feira",
    "Sábado",
    "Domingo",
]

# DIA_1..DIA_6, referência (categoria omitida) = segunda-feira (0)
DIAS_DUMMY_COLS = [f"DIA_{i}" for i in range(1, 7)]

N_TESTE_DIAS = 14
Z_NIVEL_SERVICO = 1.65  # ~95% de nível de serviço


# --------------------------------------------------------------------------
# 1. Ingestão e padronização
# --------------------------------------------------------------------------


def carregar_dados(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega e padroniza PEDIDO, ITEM_PEDIDO e ITENS a partir de `data_dir`."""
    pedido_path = data_dir / "PEDIDO-_1_.xlsx"
    item_pedido_path = data_dir / "ITEM_PEDIDO-_2_.xlsx"
    itens_path = data_dir / "ITENS-_3_.xlsx"

    df_pedido_raw = pd.read_excel(pedido_path)
    df_item_pedido_raw = pd.read_excel(item_pedido_path)

    # Remove a coluna de índice técnico gerada na exportação ("Unnamed: 0")
    df_pedido = df_pedido_raw.drop(columns=[c for c in df_pedido_raw.columns if c.startswith("Unnamed")])
    df_item_pedido = df_item_pedido_raw.drop(
        columns=[c for c in df_item_pedido_raw.columns if c.startswith("Unnamed")]
    )

    # ITENS é exportado sem cabeçalho de dados válido: a primeira linha já é um dado.
    df_itens = pd.read_excel(itens_path, header=None)
    df_itens.columns = ["ID_ITEM", "PRECO_UNITARIO"]
    df_itens = df_itens.dropna(subset=["ID_ITEM"])
    df_itens["ID_ITEM"] = df_itens["ID_ITEM"].str.strip()
    df_itens["PRECO_UNITARIO"] = df_itens["PRECO_UNITARIO"].astype(float)
    df_itens = df_itens.reset_index(drop=True)

    # Padronização de tipos
    df_pedido["DATA"] = pd.to_datetime(df_pedido["DATA"])
    df_pedido["ID_PEDIDO"] = df_pedido["ID_PEDIDO"].astype("int64")

    df_item_pedido["ID_PEDIDO"] = df_item_pedido["ID_PEDIDO"].astype("int64")
    df_item_pedido["ID_ITEM"] = df_item_pedido["ID_ITEM"].str.strip()
    df_item_pedido["QUANTIDADE"] = df_item_pedido["QUANTIDADE"].astype(int)

    return df_pedido, df_item_pedido, df_itens


# --------------------------------------------------------------------------
# 2. Qualidade de dados e reconstrução do valor total
# --------------------------------------------------------------------------


@dataclass
class DiagnosticoQualidade:
    pct_valor_total_nulo: float
    duplicados_pedido: int
    duplicados_item_pedido: int
    duplicados_exatos: int
    pedidos_sem_itens: int
    itens_sem_pedido: int
    itens_desconhecidos: set[str] = field(default_factory=set)


def diagnosticar_qualidade(
    df_pedido: pd.DataFrame, df_item_pedido: pd.DataFrame, df_itens: pd.DataFrame
) -> DiagnosticoQualidade:
    """Reproduz os achados de qualidade de dados já documentados no notebook."""
    pct_valor_nulo = df_pedido["VALOR_TOTAL"].isna().mean() * 100
    duplicados_pedido = int(df_pedido["ID_PEDIDO"].duplicated().sum())
    duplicados_item_pedido = int(df_item_pedido.duplicated(subset=["ID_PEDIDO", "ID_ITEM"]).sum())
    duplicados_exatos = int(
        df_item_pedido.duplicated(subset=["ID_PEDIDO", "ID_ITEM", "QUANTIDADE"]).sum()
    )
    pedidos_sem_itens = set(df_pedido["ID_PEDIDO"]) - set(df_item_pedido["ID_PEDIDO"])
    itens_sem_pedido = set(df_item_pedido["ID_PEDIDO"]) - set(df_pedido["ID_PEDIDO"])
    itens_desconhecidos = set(df_item_pedido["ID_ITEM"]) - set(df_itens["ID_ITEM"])

    return DiagnosticoQualidade(
        pct_valor_total_nulo=pct_valor_nulo,
        duplicados_pedido=duplicados_pedido,
        duplicados_item_pedido=duplicados_item_pedido,
        duplicados_exatos=duplicados_exatos,
        pedidos_sem_itens=len(pedidos_sem_itens),
        itens_sem_pedido=len(itens_sem_pedido),
        itens_desconhecidos=itens_desconhecidos,
    )


def reconstruir_valor_total(
    df_pedido: pd.DataFrame, df_item_pedido: pd.DataFrame, df_itens: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstrói VALOR_TOTAL por pedido a partir de ITEM_PEDIDO x ITENS.

    Retorna (df_pedido_atualizado, df_item_valor), onde df_item_valor traz uma
    linha por item vendido com DATA, PRECO_UNITARIO e SUBTOTAL — usado também
    pela EDA e pela previsão por item.
    """
    df_item_valor = df_item_pedido.merge(df_itens, on="ID_ITEM", how="left").merge(
        df_pedido[["ID_PEDIDO", "DATA"]], on="ID_PEDIDO", how="left"
    )
    df_item_valor["SUBTOTAL"] = df_item_valor["QUANTIDADE"] * df_item_valor["PRECO_UNITARIO"]

    valor_por_pedido = df_item_valor.groupby("ID_PEDIDO", as_index=False).agg(
        VALOR_TOTAL_CALC=("SUBTOTAL", "sum"), QUANTIDADE_TOTAL=("QUANTIDADE", "sum")
    )

    df_pedido_atualizado = df_pedido.drop(columns=["VALOR_TOTAL"]).merge(
        valor_por_pedido, on="ID_PEDIDO", how="left"
    )
    df_pedido_atualizado = df_pedido_atualizado.rename(columns={"VALOR_TOTAL_CALC": "VALOR_TOTAL"})

    return df_pedido_atualizado, df_item_valor


def montar_vendas_diarias(df_pedido: pd.DataFrame) -> pd.DataFrame:
    """Agrega quantidade e valor total vendidos por dia, com dia da semana em PT-BR."""
    vendas_diarias = (
        df_pedido.dropna(subset=["VALOR_TOTAL"])
        .groupby("DATA", as_index=False)
        .agg(QUANTIDADE_TOTAL=("QUANTIDADE_TOTAL", "sum"), VALOR_TOTAL=("VALOR_TOTAL", "sum"))
        .sort_values("DATA")
    )
    vendas_diarias["DIA_SEMANA"] = vendas_diarias["DATA"].dt.dayofweek.map(lambda i: DIAS_SEMANA_PT[i])
    return vendas_diarias


def montar_mix_produtos(df_item_valor: pd.DataFrame) -> pd.DataFrame:
    """Participação de cada item no total vendido (quantidade e valor)."""
    mix_produtos = (
        df_item_valor.groupby("ID_ITEM", as_index=False)
        .agg(QUANTIDADE_TOTAL=("QUANTIDADE", "sum"), VALOR_TOTAL=("SUBTOTAL", "sum"))
        .sort_values("VALOR_TOTAL", ascending=False)
    )
    mix_produtos["PARTICIPACAO_VALOR_%"] = (
        mix_produtos["VALOR_TOTAL"] / mix_produtos["VALOR_TOTAL"].sum() * 100
    )
    return mix_produtos


# --------------------------------------------------------------------------
# 3. Features e modelos
# --------------------------------------------------------------------------


def construir_features(serie_quantidade: pd.Series) -> pd.DataFrame:
    """Monta a matriz de features de calendário (dia da semana one-hot) + lags."""
    df = pd.DataFrame({"QUANTIDADE": serie_quantidade})
    dummies_dia = pd.get_dummies(df.index.dayofweek, prefix="DIA", drop_first=True, dtype=float)
    dummies_dia.index = df.index
    df = pd.concat([df, dummies_dia.reindex(columns=DIAS_DUMMY_COLS, fill_value=0.0)], axis=1)
    df["TENDENCIA"] = np.arange(len(df))
    for lag in (1, 2, 7):
        df[f"LAG_{lag}"] = df["QUANTIDADE"].shift(lag)
    return df.dropna()


def features_colunas() -> list[str]:
    return DIAS_DUMMY_COLS + ["TENDENCIA", "LAG_1", "LAG_2", "LAG_7"]


def construir_modelos(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Instâncias novas dos 4 modelos candidatos ao benchmark."""
    return {
        "Regressão Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=random_state),
        "XGBoost": XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=random_state, verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            min_child_samples=5,
            random_state=random_state,
            verbose=-1,
        ),
    }


def dividir_treino_teste(
    df_features: pd.DataFrame, n_teste_dias: int = N_TESTE_DIAS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal: as últimas `n_teste_dias` linhas viram o holdout de teste."""
    corte = df_features.index.max() - pd.Timedelta(days=n_teste_dias)
    treino = df_features[df_features.index <= corte]
    teste = df_features[df_features.index > corte]
    return treino, teste


def benchmark_modelos(
    X_treino: pd.DataFrame, y_treino: pd.Series, n_splits: int = 4, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """Avalia os modelos candidatos por validação cruzada temporal no treino.

    Retorna um DataFrame ordenado por MAE_CV crescente (melhor modelo primeiro).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    resultados_cv = []
    for nome_modelo, modelo_cv in construir_modelos(random_state).items():
        maes_fold, rmses_fold = [], []
        for idx_treino, idx_val in tscv.split(X_treino):
            modelo_cv.fit(X_treino.iloc[idx_treino], y_treino.iloc[idx_treino])
            pred_val = modelo_cv.predict(X_treino.iloc[idx_val])
            maes_fold.append(mean_absolute_error(y_treino.iloc[idx_val], pred_val))
            rmses_fold.append(np.sqrt(mean_squared_error(y_treino.iloc[idx_val], pred_val)))
        resultados_cv.append(
            {"Modelo": nome_modelo, "MAE_CV": np.mean(maes_fold), "RMSE_CV": np.mean(rmses_fold)}
        )
    return pd.DataFrame(resultados_cv).sort_values("MAE_CV").reset_index(drop=True)


# --------------------------------------------------------------------------
# 4. Previsão por item e recomendação de estoque
# --------------------------------------------------------------------------


def montar_pivot_item(df_item_valor: pd.DataFrame) -> pd.DataFrame:
    """Série diária de quantidade vendida por item (colunas = ID_ITEM)."""
    vendas_item_diarias = (
        df_item_valor.dropna(subset=["DATA"])
        .groupby(["DATA", "ID_ITEM"], as_index=False)["QUANTIDADE"]
        .sum()
    )
    return (
        vendas_item_diarias.pivot(index="DATA", columns="ID_ITEM", values="QUANTIDADE")
        .asfreq("D")
        .fillna(0.0)
    )


def prever_por_item(
    pivot_item: pd.DataFrame,
    melhor_modelo_nome: str,
    n_teste_dias: int = N_TESTE_DIAS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.Series, np.ndarray]]]:
    """Treina o modelo vencedor do benchmark, individualmente, para cada item.

    Retorna (df_resultado_item, previsoes_por_item), onde previsoes_por_item
    mapeia ID_ITEM -> (y_teste, previsao) para uso em gráficos.
    """
    features = features_colunas()
    resultados_por_item = []
    previsoes_por_item: dict[str, tuple[pd.Series, np.ndarray]] = {}

    for item in pivot_item.columns:
        df_feat_item = construir_features(pivot_item[item])
        treino_item, teste_item = dividir_treino_teste(df_feat_item, n_teste_dias)

        X_treino_item, y_treino_item = treino_item[features], treino_item["QUANTIDADE"]
        X_teste_item, y_teste_item = teste_item[features], teste_item["QUANTIDADE"]

        modelo_item = construir_modelos(random_state)[melhor_modelo_nome]
        modelo_item.fit(X_treino_item, y_treino_item)
        previsao_item = modelo_item.predict(X_teste_item)
        previsoes_por_item[item] = (y_teste_item, previsao_item)

        erro_item = y_teste_item.to_numpy() - previsao_item
        resultados_por_item.append(
            {
                "ID_ITEM": item,
                "MAE": mean_absolute_error(y_teste_item, previsao_item),
                "RMSE": np.sqrt(mean_squared_error(y_teste_item, previsao_item)),
                "ERRO_PADRAO": erro_item.std(ddof=1) if len(erro_item) > 1 else 0.0,
            }
        )

    df_resultado_item = pd.DataFrame(resultados_por_item).sort_values("ID_ITEM").reset_index(drop=True)
    return df_resultado_item, previsoes_por_item


def treinar_modelo_final_item(
    pivot_item: pd.DataFrame, item: str, melhor_modelo_nome: str, random_state: int = RANDOM_STATE
) -> tuple[Any, pd.DataFrame]:
    """Treina o modelo vencedor com todo o histórico disponível do item (sem
    reservar holdout), para uso em previsão de dias futuros além do fim do
    histórico (ver `prever_proximos_dias`).
    """
    df_feat_item = construir_features(pivot_item[item])
    features = features_colunas()
    modelo = construir_modelos(random_state)[melhor_modelo_nome]
    modelo.fit(df_feat_item[features], df_feat_item["QUANTIDADE"])
    return modelo, df_feat_item


def prever_proximos_dias(
    df_features_treino: pd.DataFrame, serie_original: pd.Series, modelo: Any, n_dias: int = 7
) -> pd.Series:
    """Previsão recursiva para os `n_dias` seguintes ao fim do histórico.

    A cada passo, usa a previsão do dia anterior como novo lag (não há dado
    real futuro disponível). `df_features_treino` é o resultado de
    `construir_features` sobre a mesma série, usado apenas para ancorar a
    escala da feature TENDENCIA no ponto onde o histórico termina.
    """
    ultima_tendencia = int(df_features_treino["TENDENCIA"].iloc[-1])
    historico = serie_original.astype(float).copy()
    features = features_colunas()
    previsoes: dict[pd.Timestamp, float] = {}

    for passo in range(1, n_dias + 1):
        proxima_data = historico.index.max() + pd.Timedelta(days=1)
        dia_semana = proxima_data.dayofweek
        linha = {col: 0.0 for col in DIAS_DUMMY_COLS}
        if dia_semana != 0:  # 0 = segunda-feira, categoria de referência (sem dummy)
            linha[f"DIA_{dia_semana}"] = 1.0
        linha["TENDENCIA"] = ultima_tendencia + passo
        linha["LAG_1"] = historico.iloc[-1]
        linha["LAG_2"] = historico.iloc[-2]
        linha["LAG_7"] = historico.iloc[-7]

        X_futuro = pd.DataFrame([linha])[features]
        pred = max(0.0, float(modelo.predict(X_futuro)[0]))
        previsoes[proxima_data] = pred
        historico.loc[proxima_data] = pred

    return pd.Series(previsoes, name="PREVISAO")


def calcular_estoque_sugerido(
    df_resultado_item: pd.DataFrame,
    previsoes_por_item: dict[str, tuple[pd.Series, np.ndarray]],
    pivot_item: pd.DataFrame,
    z_nivel_servico: float = Z_NIVEL_SERVICO,
) -> pd.DataFrame:
    """Estoque de segurança e estoque-alvo sugeridos por item.

    Estoque de segurança = z * desvio-padrão do erro de previsão no holdout.
    Estoque-alvo = previsão do último dia de teste + estoque de segurança.
    """
    df = df_resultado_item.copy()
    df["ESTOQUE_SEGURANCA_SUGERIDO"] = np.ceil(z_nivel_servico * df["ERRO_PADRAO"]).astype(int)

    previsao_ultimo_dia = {
        item: max(0, round(previsoes_por_item[item][1][-1])) for item in pivot_item.columns
    }
    df["PREVISAO_ULTIMO_DIA_TESTE"] = df["ID_ITEM"].map(previsao_ultimo_dia)
    df["ESTOQUE_ALVO_SUGERIDO"] = df["PREVISAO_ULTIMO_DIA_TESTE"] + df["ESTOQUE_SEGURANCA_SUGERIDO"]
    return df


# --------------------------------------------------------------------------
# 5. Pipeline completo (conveniência para o app)
# --------------------------------------------------------------------------


@dataclass
class ResultadoPipeline:
    df_pedido: pd.DataFrame
    df_item_pedido: pd.DataFrame
    df_itens: pd.DataFrame
    df_item_valor: pd.DataFrame
    diagnostico_qualidade: DiagnosticoQualidade
    vendas_diarias: pd.DataFrame
    mix_produtos: pd.DataFrame
    pivot_item: pd.DataFrame
    df_benchmark_cv: pd.DataFrame
    melhor_modelo_nome: str
    df_resultado_item: pd.DataFrame
    previsoes_por_item: dict[str, tuple[pd.Series, np.ndarray]]


def executar_pipeline_completo(data_dir: Path, n_teste_dias: int = N_TESTE_DIAS) -> ResultadoPipeline:
    """Executa a pipeline inteira (ingestão -> qualidade -> EDA -> benchmark -> previsão por item).

    Usado tanto pelo notebook (célula a célula, reaproveitando as funções
    individuais acima) quanto pelo app Streamlit (chamada única, cacheada).
    """
    df_pedido, df_item_pedido, df_itens = carregar_dados(data_dir)
    diagnostico = diagnosticar_qualidade(df_pedido, df_item_pedido, df_itens)
    df_pedido, df_item_valor = reconstruir_valor_total(df_pedido, df_item_pedido, df_itens)

    vendas_diarias = montar_vendas_diarias(df_pedido)
    mix_produtos = montar_mix_produtos(df_item_valor)

    serie = vendas_diarias.set_index("DATA")["QUANTIDADE_TOTAL"].asfreq("D").fillna(0.0)
    df_features = construir_features(serie)
    treino, teste = dividir_treino_teste(df_features, n_teste_dias)
    features = features_colunas()

    df_benchmark_cv = benchmark_modelos(treino[features], treino["QUANTIDADE"])
    melhor_modelo_nome = df_benchmark_cv.iloc[0]["Modelo"]

    pivot_item = montar_pivot_item(df_item_valor)
    df_resultado_item, previsoes_por_item = prever_por_item(
        pivot_item, melhor_modelo_nome, n_teste_dias
    )
    df_resultado_item = calcular_estoque_sugerido(df_resultado_item, previsoes_por_item, pivot_item)

    return ResultadoPipeline(
        df_pedido=df_pedido,
        df_item_pedido=df_item_pedido,
        df_itens=df_itens,
        df_item_valor=df_item_valor,
        diagnostico_qualidade=diagnostico,
        vendas_diarias=vendas_diarias,
        mix_produtos=mix_produtos,
        pivot_item=pivot_item,
        df_benchmark_cv=df_benchmark_cv,
        melhor_modelo_nome=melhor_modelo_nome,
        df_resultado_item=df_resultado_item,
        previsoes_por_item=previsoes_por_item,
    )
