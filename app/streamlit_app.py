"""Painel de decisão de estoque — Mr. Health (case DataLakers).

Como executar (a partir da raiz do projeto, com o ambiente virtual ativo):

    python -m venv .venv                        # se ainda não existir
    .venv\\Scripts\\pip install -r requirements.txt
    .venv\\Scripts\\streamlit run app\\streamlit_app.py

O app reaproveita a mesma lógica de ingestão, qualidade de dados e modelo
preditivo do notebook `notebooks/mr_health_demand_forecast.py`, via o
módulo compartilhado `src/mrhealth/pipeline.py` — não há lógica de dados
duplicada aqui, apenas apresentação e interação.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

# Raiz do projeto: este arquivo vive em app/, os dados e o src/ ficam um nível acima.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
import mrhealth.pipeline as pl  # noqa: E402

st.set_page_config(page_title="Mr. Health — Decisão de Estoque", layout="wide")

# Tema padrão do app é dark (ver .streamlit/config.toml); os gráficos Plotly
# usam o template escuro correspondente para não destoar do fundo.
pio.templates.default = "plotly_dark"


# --------------------------------------------------------------------------
# Carga e treino, cacheados
# --------------------------------------------------------------------------


@st.cache_data(show_spinner="Carregando e tratando os dados de vendas...")
def carregar_dados_tratados(data_dir: str) -> dict:
    df_pedido, df_item_pedido, df_itens = pl.carregar_dados(Path(data_dir))
    diagnostico = pl.diagnosticar_qualidade(df_pedido, df_item_pedido, df_itens)
    df_pedido, df_item_valor = pl.reconstruir_valor_total(df_pedido, df_item_pedido, df_itens)
    vendas_diarias = pl.montar_vendas_diarias(df_pedido)
    mix_produtos = pl.montar_mix_produtos(df_item_valor)
    pivot_item = pl.montar_pivot_item(df_item_valor)
    return {
        "diagnostico": diagnostico,
        "vendas_diarias": vendas_diarias,
        "mix_produtos": mix_produtos,
        "df_item_valor": df_item_valor,
        "pivot_item": pivot_item,
    }


@st.cache_resource(show_spinner="Treinando e comparando modelos de previsão de demanda...")
def treinar_modelo(data_dir: str) -> dict:
    dados = carregar_dados_tratados(data_dir)
    vendas_diarias = dados["vendas_diarias"]
    pivot_item = dados["pivot_item"]

    serie = vendas_diarias.set_index("DATA")["QUANTIDADE_TOTAL"].asfreq("D").fillna(0.0)
    df_features = pl.construir_features(serie)
    treino, _teste = pl.dividir_treino_teste(df_features)
    features = pl.features_colunas()

    df_benchmark_cv = pl.benchmark_modelos(treino[features], treino["QUANTIDADE"])
    melhor_modelo_nome = df_benchmark_cv.iloc[0]["Modelo"]

    df_resultado_item, previsoes_por_item = pl.prever_por_item(pivot_item, melhor_modelo_nome)
    df_resultado_item = pl.calcular_estoque_sugerido(df_resultado_item, previsoes_por_item, pivot_item)

    return {
        "df_benchmark_cv": df_benchmark_cv,
        "melhor_modelo_nome": melhor_modelo_nome,
        "df_resultado_item": df_resultado_item,
        "previsoes_por_item": previsoes_por_item,
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Mr. Health")
st.sidebar.caption("Painel de decisão de estoque — case DataLakers")

if st.sidebar.button("🔄 Recarregar dados"):
    carregar_dados_tratados.clear()
    treinar_modelo.clear()
    st.rerun()

data_dir_str = str(ROOT_DIR)

try:
    dados = carregar_dados_tratados(data_dir_str)
    st.sidebar.success("Dados carregados com sucesso.")
except FileNotFoundError as exc:
    st.sidebar.error(f"Não foi possível carregar os dados de origem: {exc}")
    st.error(
        "Arquivos de origem não encontrados na raiz do projeto "
        "(`PEDIDO-_1_.xlsx`, `ITEM_PEDIDO-_2_.xlsx`, `ITENS-_3_.xlsx`)."
    )
    st.stop()

modelo_info = treinar_modelo(data_dir_str)

pagina = st.sidebar.radio(
    "Navegação",
    ["Visão geral de vendas", "Previsão de demanda por item", "Painel de decisão de estoque"],
)

# --------------------------------------------------------------------------
# Página 1 — Visão geral de vendas
# --------------------------------------------------------------------------

if pagina == "Visão geral de vendas":
    st.title("Visão geral de vendas")

    vendas_diarias = dados["vendas_diarias"]
    mix_produtos = dados["mix_produtos"]
    diagnostico = dados["diagnostico"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Quantidade total vendida", f"{int(vendas_diarias['QUANTIDADE_TOTAL'].sum())} un.")
    col2.metric("Valor total vendido", f"R$ {vendas_diarias['VALOR_TOTAL'].sum():,.2f}")
    col3.metric("Dias com venda registrada", f"{len(vendas_diarias)}")

    with st.expander("Qualidade dos dados de origem"):
        st.write(
            f"- `VALOR_TOTAL` chegou {diagnostico.pct_valor_total_nulo:.0f}% nulo da origem "
            "e foi reconstruído a partir de `ITEM_PEDIDO` × `ITENS`."
        )
        st.write(f"- Pedidos duplicados: {diagnostico.duplicados_pedido}.")
        st.write(
            f"- Combinações (pedido, item) repetidas: {diagnostico.duplicados_item_pedido} "
            f"(sendo {diagnostico.duplicados_exatos} linhas 100% idênticas — mantidas e somadas)."
        )

    st.subheader("Série temporal de vendas")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Quantidade total vendida por dia", "Valor total vendido por dia (R$)"),
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=vendas_diarias["DATA"],
            y=vendas_diarias["QUANTIDADE_TOTAL"],
            mode="lines+markers",
            name="Quantidade",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vendas_diarias["DATA"],
            y=vendas_diarias["VALOR_TOTAL"],
            mode="lines+markers",
            name="Valor (R$)",
            line=dict(color="darkorange"),
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(tickformat="%d/%m/%Y", tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickformat="%d/%m/%Y", tickangle=-45, row=2, col=1)
    fig.update_layout(height=550, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Participação por item")
    col_a, col_b = st.columns(2)
    with col_a:
        st.dataframe(
            mix_produtos[["ID_ITEM", "QUANTIDADE_TOTAL", "VALOR_TOTAL", "PARTICIPACAO_VALOR_%"]],
            hide_index=True,
        )
    with col_b:
        fig2 = px.bar(
            mix_produtos,
            x="ID_ITEM",
            y="QUANTIDADE_TOTAL",
            title="Quantidade total vendida por item",
            labels={"ID_ITEM": "Item", "QUANTIDADE_TOTAL": "Quantidade"},
        )
        fig2.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------------------------------
# Página 2 — Previsão de demanda por item
# --------------------------------------------------------------------------

elif pagina == "Previsão de demanda por item":
    st.title("Previsão de demanda por item")

    st.caption(
        f"Modelo vencedor do benchmark (validação cruzada temporal): "
        f"**{modelo_info['melhor_modelo_nome']}**"
    )
    with st.expander("Benchmark de modelos (MAE/RMSE de validação)"):
        st.dataframe(modelo_info["df_benchmark_cv"], hide_index=True)

    pivot_item = dados["pivot_item"]
    itens = list(pivot_item.columns)
    item_selecionado = st.selectbox("Selecione o item", itens)

    y_real, y_prev = modelo_info["previsoes_por_item"][item_selecionado]
    y_prev_int = np.round(y_prev).astype(int)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_real.index, y=y_real.values.astype(int), mode="lines+markers", name="Real"))
    fig.add_trace(go.Scatter(x=y_real.index, y=y_prev_int, mode="lines+markers", name="Previsão"))
    fig.update_layout(
        title=f"Demanda diária — {item_selecionado} (conjunto de teste)",
        xaxis=dict(tickformat="%d/%m/%Y", tickangle=-45),
        yaxis_title="Quantidade",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    n_dias_futuro = st.slider("Dias a projetar após o fim do histórico", min_value=3, max_value=14, value=7)
    modelo_final, df_feat_item = pl.treinar_modelo_final_item(
        pivot_item, item_selecionado, modelo_info["melhor_modelo_nome"]
    )
    previsao_futura = pl.prever_proximos_dias(
        df_feat_item, pivot_item[item_selecionado], modelo_final, n_dias=n_dias_futuro
    )
    # Quantidade vendida é sempre inteira: arredondamos só para exibição (tabela e gráfico).
    previsao_futura_exibicao = previsao_futura.round().astype(int)

    st.subheader(f"Previsão para os próximos {n_dias_futuro} dias — {item_selecionado}")
    st.dataframe(previsao_futura_exibicao.rename("Quantidade prevista"))

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=pivot_item.index[-30:],
            y=pivot_item[item_selecionado].iloc[-30:].astype(int),
            mode="lines+markers",
            name="Histórico (últimos 30 dias)",
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=previsao_futura_exibicao.index,
            y=previsao_futura_exibicao.values,
            mode="lines+markers",
            name="Previsão futura",
            line=dict(color="darkorange", dash="dash"),
        )
    )
    fig2.update_layout(
        title=f"Histórico recente + previsão futura — {item_selecionado}",
        xaxis=dict(tickformat="%d/%m/%Y", tickangle=-45),
        yaxis_title="Quantidade",
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig2, width="stretch")

# --------------------------------------------------------------------------
# Página 3 — Painel de decisão de estoque
# --------------------------------------------------------------------------

else:
    st.title("Painel de decisão de estoque")
    st.caption(
        "Informe o estoque atual de cada item para comparar com o estoque-alvo "
        "sugerido pelo modelo (previsão de demanda + margem de segurança)."
    )

    df_resultado_item = modelo_info["df_resultado_item"]

    if "estoque_atual" not in st.session_state:
        st.session_state["estoque_atual"] = {item: 0 for item in df_resultado_item["ID_ITEM"]}

    st.subheader("Estoque atual informado")
    cols = st.columns(len(df_resultado_item))
    for col, (_, linha) in zip(cols, df_resultado_item.iterrows()):
        item = linha["ID_ITEM"]
        with col:
            st.session_state["estoque_atual"][item] = st.number_input(
                item, min_value=0, step=1, value=st.session_state["estoque_atual"][item], key=f"estoque_{item}"
            )

    tabela = df_resultado_item[["ID_ITEM", "ESTOQUE_ALVO_SUGERIDO"]].copy()
    tabela["ESTOQUE_ATUAL"] = tabela["ID_ITEM"].map(st.session_state["estoque_atual"])
    tabela["STATUS"] = tabela.apply(
        lambda linha: "🔴 Reabastecer" if linha["ESTOQUE_ATUAL"] < linha["ESTOQUE_ALVO_SUGERIDO"] else "🟢 Estoque adequado",
        axis=1,
    )

    st.subheader("Decisão de reposição")
    st.dataframe(
        tabela[["ID_ITEM", "ESTOQUE_ATUAL", "ESTOQUE_ALVO_SUGERIDO", "STATUS"]],
        hide_index=True,
        width="stretch",
    )

    itens_criticos = tabela[tabela["STATUS"].str.contains("Reabastecer")]["ID_ITEM"].tolist()
    if itens_criticos:
        st.warning(f"Itens que precisam de reposição agora: {', '.join(itens_criticos)}")
    else:
        st.success("Todos os itens estão com estoque adequado em relação ao estoque-alvo sugerido.")
