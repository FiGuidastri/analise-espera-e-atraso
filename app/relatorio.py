"""Relatório interativo do projeto (Streamlit).

Lê só os artefatos que os notebooks geram e que vão para o Git:
reports/base_app.parquet, reports/previsoes.parquet, reports/resultados_causais.json,
models/modelo_lgbm.joblib e models/atributos.json.

Rodar da raiz do projeto:  streamlit run app/relatorio.py
"""
import json
import os
import sys
from pathlib import Path

# O joblib tenta contar núcleos físicos via `wmic` (ausente no Windows 11) e imprime um
# traceback inofensivo; informar o limite de núcleos evita a consulta.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max((os.cpu_count() or 2) // 2, 1)))

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
FIG = REPORTS / "figuras"
sys.path.insert(0, str(ROOT))

from src.estilo import AZUL, CINZA, LARANJA
from src.features import REGIAO_POR_UF, preparar_x

st.set_page_config(page_title="Atraso de entrega — Olist", layout="wide")


# ---------------------------------------------------------------- dados
@st.cache_data
def carregar_base():
    base = pd.read_parquet(REPORTS / "base_app.parquet")
    base["mes_compra"] = base["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()
    return base


@st.cache_data
def carregar_previsoes():
    return pd.read_parquet(REPORTS / "previsoes.parquet")


@st.cache_data
def carregar_json(caminho):
    return json.load(open(caminho, encoding="utf-8"))


@st.cache_resource
def carregar_modelo():
    return joblib.load(MODELS / "modelo_lgbm.joblib")


base = carregar_base()
previsoes = carregar_previsoes()
causal = carregar_json(REPORTS / "resultados_causais.json")
meta = carregar_json(MODELS / "atributos.json")
TAXA_GLOBAL = base["atrasou"].mean()


# ---------------------------------------------------------------- gráficos
def pct(v, casas=1):
    return f"{v:.{casas}%}".replace(".", ",")


def num(v, casas=2):
    return f"{v:.{casas}f}".replace(".", ",")


def barras_taxa(df, categoria, titulo, horizontal=False, ordem=None, media=TAXA_GLOBAL):
    """Taxa de atraso por grupo, com tooltip e linha da média."""
    eixo_cat = alt.X if not horizontal else alt.Y
    eixo_val = alt.Y if not horizontal else alt.X
    barras = alt.Chart(df).mark_bar(color=AZUL, cornerRadiusEnd=4).encode(
        eixo_cat(f"{categoria}:N", sort=ordem or ("-x" if horizontal else None), title=None),
        eixo_val("taxa:Q", title="taxa de atraso", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip(f"{categoria}:N", title="grupo"),
                 alt.Tooltip("taxa:Q", format=".1%", title="taxa de atraso"),
                 alt.Tooltip("pedidos:Q", format=",", title="pedidos")],
    )
    regra = alt.Chart(pd.DataFrame({"m": [media]})).mark_rule(color=CINZA, strokeDash=[4, 4]).encode(
        **({"x": "m:Q"} if horizontal else {"y": "m:Q"}))
    return (barras + regra).properties(title=titulo, height=520 if horizontal else 300)


def taxa_por(df, coluna):
    return (df.groupby(coluna, observed=True)["atrasou"].agg(taxa="mean", pedidos="size")
            .reset_index())


# ---------------------------------------------------------------- páginas
def resumo():
    st.title("Atraso de entrega e satisfação do cliente")
    st.caption("Olist · 96 mil pedidos entregues entre jan/2017 e ago/2018 · "
               "[código no GitHub](https://github.com/FiGuidastri/analise-espera-e-atraso)")

    r = causal["nota"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taxa de atraso", pct(TAXA_GLOBAL, 2))
    c2.metric("Efeito causal do atraso na nota", num(r["theta"]),
              help=f"Double Machine Learning, IC 95% {num(r['ic_inf'])} a {num(r['ic_sup'])}")
    c3.metric("Avaliações ≤ 2 a mais por atraso", f"+{num(causal['nota_ruim']['theta'] * 100, 0)} p.p.")
    c4.metric("PR-AUC no teste (LightGBM)", num(meta["metricas"]["teste_lgbm"]["PR-AUC"], 3),
              help="Aleatório = 0,037; regressão logística = 0,078")

    st.markdown(f"""
**O problema.** O prazo estimado no checkout é uma promessa. A operação só descobre o atraso
quando ele já aconteceu e a avaliação negativa já foi registrada.

**O que descobrimos.**
- Atrasar derruba a nota em **{num(abs(r['theta']))} ponto**, mesmo comparando pedidos de perfil
  igual — quase toda a diferença bruta ({num(r['ingenua'])}) é efeito do atraso em si.
- O efeito cresce com os dias de atraso e estabiliza depois de uma semana.
- Distância, prazo curto e destino no Norte/Nordeste aumentam o risco de atraso.
- Prever o atraso **no clique de compra** é difícil: os modelos ficam ~2× acima do acaso, e a
  ação preventiva só compensa em períodos de muito atraso ou quando o atraso é caro.
""")
    st.altair_chart(
        alt.Chart(base.groupby("mes_compra")["atrasou"].mean().reset_index())
        .mark_line(color=AZUL, point=True).encode(
            alt.X("mes_compra:T", title=None, axis=alt.Axis(format="%b/%y")),
            alt.Y("atrasou:Q", title="taxa de atraso", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("mes_compra:T", format="%b/%Y", title="mês"),
                     alt.Tooltip("atrasou:Q", format=".1%", title="taxa")],
        ).properties(title="Taxa de atraso por mês de compra", height=280),
        use_container_width=True,
    )


def exploracao():
    st.title("Exploração — o que está por trás do atraso")
    meses = sorted(base["mes_compra"].unique())
    c1, c2 = st.columns([2, 1])
    inicio, fim = c1.select_slider("Período de compra", options=meses, value=(meses[0], meses[-1]),
                                   format_func=lambda m: pd.Timestamp(m).strftime("%b/%Y"))
    regioes = c2.multiselect("Região do cliente", sorted(base["regiao_cliente"].dropna().unique()),
                             placeholder="todas")
    df = base[base["mes_compra"].between(inicio, fim)]
    if regioes:
        df = df[df["regiao_cliente"].isin(regioes)]
    media = df["atrasou"].mean()
    st.caption(f"{len(df):,} pedidos".replace(",", ".") + f" · taxa de atraso {pct(media)} (linha tracejada)")

    col1, col2 = st.columns(2)
    col1.altair_chart(barras_taxa(taxa_por(df, "uf_cliente"), "uf_cliente",
                                  "H-02 · Taxa por UF do cliente", horizontal=True, media=media),
                      use_container_width=True)
    dist = df.assign(faixa=pd.cut(df["distancia_km"], [0, 100, 300, 600, 1000, 2000, np.inf], right=False,
                                  labels=["<100", "100–300", "300–600", "600–1000", "1000–2000", "≥2000"]))
    col2.altair_chart(barras_taxa(taxa_por(dist, "faixa"), "faixa", "H-01 · Taxa por distância (km)",
                                  ordem=list(dist["faixa"].cat.categories), media=media),
                      use_container_width=True)
    prazo = df.assign(faixa=pd.cut(df["prazo_estimado_dias"], [0, 10, 15, 20, 25, 30, 40, np.inf], right=False,
                                   labels=["<10", "10–15", "15–20", "20–25", "25–30", "30–40", "≥40"]))
    col2.altair_chart(barras_taxa(taxa_por(prazo, "faixa"), "faixa", "H-03 · Taxa por prazo estimado (dias)",
                                  ordem=list(prazo["faixa"].cat.categories), media=media),
                      use_container_width=True)

    notas = (df.dropna(subset=["review_score"])
             .assign(situacao=lambda d: np.where(d["atrasou"] == 1, "atrasado", "no prazo"))
             .groupby("situacao")["review_score"].value_counts(normalize=True)
             .rename("fracao").reset_index())
    st.altair_chart(
        alt.Chart(notas).mark_bar(cornerRadiusEnd=4).encode(
            alt.X("review_score:O", title="nota"),
            alt.Y("fracao:Q", title="% das avaliações", axis=alt.Axis(format="%")),
            alt.XOffset("situacao:N", sort=["no prazo", "atrasado"]),
            alt.Color("situacao:N", sort=["no prazo", "atrasado"], title=None,
                      scale=alt.Scale(domain=["no prazo", "atrasado"], range=[AZUL, LARANJA])),
            tooltip=[alt.Tooltip("situacao:N", title="pedido"), alt.Tooltip("review_score:O", title="nota"),
                     alt.Tooltip("fracao:Q", format=".1%", title="% das avaliações")],
        ).properties(title="Distribuição da nota: atrasados × no prazo", height=300),
        use_container_width=True,
    )


def modelo():
    st.title("Modelo preditivo — dá para prever o atraso no clique?")
    m = meta["metricas"]
    tabela = pd.DataFrame({
        "Modelo": ["Aleatório (= taxa de atraso)", "Regressão logística (baseline)", "LightGBM"],
        "PR-AUC validação": [m["validacao_lgbm"]["taxa (PR-AUC aleatória)"], m["validacao_logit"]["PR-AUC"],
                             m["validacao_lgbm"]["PR-AUC"]],
        "PR-AUC teste": [m["teste_lgbm"]["taxa (PR-AUC aleatória)"], m["teste_logit"]["PR-AUC"],
                         m["teste_lgbm"]["PR-AUC"]],
        "ROC-AUC teste": [0.5, m["teste_logit"]["ROC-AUC"], m["teste_lgbm"]["ROC-AUC"]],
    })
    st.dataframe(tabela.style.format(precision=3, decimal=","), hide_index=True, use_container_width=True)
    c = meta["cortes"]
    st.markdown(f"""
Split temporal: treino até {c['treino_fim']}, validação até {c['validacao_fim']}, teste até
{c['teste_fim']} (as semanas seguintes ficaram fora — os atrasados ainda não tinham sido entregues).
**O LightGBM não superou o baseline no teste**; a diferença está dentro do ruído (IC 95% bootstrap
−0,017 a +0,004). Os dois ficam ~2× acima do acaso: com informação só do clique de compra, boa parte
do atraso depende do que acontece depois.
""")

    teste = previsoes[previsoes["conjunto"] == "teste"]
    curvas = []
    for col, nome in [("p_logit", "Regressão logística"), ("p_lgbm", "LightGBM")]:
        p, r, _ = precision_recall_curve(teste["atrasou"], teste[col])
        curvas.append(pd.DataFrame({"recall": r, "precisão": p, "modelo": nome}).iloc[::20])
    st.altair_chart(
        alt.Chart(pd.concat(curvas)).mark_line().encode(
            alt.X("recall:Q", axis=alt.Axis(format="%")), alt.Y("precisão:Q", axis=alt.Axis(format="%")),
            alt.Color("modelo:N", title=None, scale=alt.Scale(range=[AZUL, CINZA])),
            tooltip=["modelo", alt.Tooltip("recall:Q", format=".0%"), alt.Tooltip("precisão:Q", format=".1%")],
        ).properties(title="Curva precisão-recall no teste", height=320),
        use_container_width=True,
    )

    st.subheader("O que move o risco (SHAP)")
    c1, c2 = st.columns(2)
    c1.image(str(FIG / "04_shap_beeswarm.png"))
    c2.markdown("""
- **Prazo estimado:** quanto maior o prazo prometido, menor o risco (H-03).
- **UF do cliente:** Nordeste e Norte puxam o risco para cima (H-02).
- **Distância:** mais distância, mais risco (H-01).
- **Mês da compra:** capta os picos de nov/2017 e fev–mar/2018 — é o atributo mais forte e o mais
  frágil fora da amostra.
""")
    with st.expander("Três pedidos de alto risco explicados"):
        for i in range(1, 4):
            st.image(str(FIG / f"04_shap_pedido_{i}.png"))


def simulador():
    st.title("Simulador de risco de atraso")
    st.caption("Informe um pedido como ele é no clique de compra. O risco vem do LightGBM; as barras mostram "
               "quanto cada atributo empurra o risco para cima ou para baixo (SHAP, em log-odds).")
    cats = meta["categorias"]
    ufs_cli = [u for u in cats["uf_cliente"] if u != "desconhecida"]
    ufs_vend = [u for u in cats["uf_vendedor"] if u != "desconhecida"]

    c1, c2, c3 = st.columns(3)
    uf_cliente = c1.selectbox("UF do cliente", ufs_cli, index=ufs_cli.index("BA"))
    uf_vendedor = c1.selectbox("UF do vendedor", ufs_vend, index=ufs_vend.index("SP"))
    par = base[(base["uf_cliente"] == uf_cliente) & (base["uf_vendedor"] == uf_vendedor)]["distancia_km"]
    dist_padrao = float(par.median()) if par.notna().any() else float(base["distancia_km"].median())
    distancia = c1.number_input("Distância cliente–vendedor (km)", 0.0, 4000.0, float(round(dist_padrao)), step=50.0,
                                help="Começa na mediana do par de UFs")
    prazo = c2.slider("Prazo estimado (dias)", 3, 60, 24)
    prazo_envio = c2.slider("Prazo de postagem do vendedor (dias)", 1, 20, 6)
    mes = c2.selectbox("Mês da compra", list(range(1, 13)), index=5,
                       format_func=lambda m: pd.Timestamp(2018, m, 1).strftime("%B"))
    preco = c3.number_input("Preço total (R$)", 1.0, 5000.0, 86.5)
    frete = c3.number_input("Frete total (R$)", 0.0, 500.0, 17.2)
    categorias = sorted(c for c in cats["categoria_principal"] if c != "desconhecida")
    categoria = c3.selectbox("Categoria", categorias, index=categorias.index("health_beauty"))

    with st.expander("Outros atributos"):
        o1, o2, o3 = st.columns(3)
        n_itens = o1.number_input("Itens", 1, 20, 1)
        peso = o1.number_input("Peso total (g)", 0, 50000, 750)
        volume = o2.number_input("Volume total (cm³)", 100, 500000, 7240)
        pagamento = o2.selectbox("Pagamento", [c for c in cats["forma_pagamento"] if c != "desconhecida"])
        parcelas = o3.number_input("Parcelas", 0, 24, 2)
        hora = o3.slider("Hora da compra", 0, 23, 15)

    pedido = {
        "n_itens": n_itens, "n_produtos": n_itens, "n_vendedores": 1,
        "preco_total": preco, "frete_total": frete, "frete_relativo": frete / preco,
        "peso_total_g": peso, "volume_total_cm3": volume, "n_fotos_principal": 2,
        "parcelas": parcelas, "distancia_km": distancia, "mesmo_estado": int(uf_cliente == uf_vendedor),
        "prazo_estimado_dias": prazo, "prazo_envio_dias": prazo_envio,
        "mes": mes, "dia_semana": 2, "hora": hora,
        "categoria_principal": categoria, "forma_pagamento": pagamento,
        "uf_cliente": uf_cliente, "uf_vendedor": uf_vendedor,
    }
    x = pd.DataFrame([pedido])[meta["atributos"]]
    for col in meta["categoricos"]:
        x[col] = pd.Categorical(x[col], categories=cats[col])
    for col in x.columns.difference(meta["categoricos"]):
        x[col] = x[col].astype(float)

    lgbm = carregar_modelo()
    risco = lgbm.predict_proba(x)[0, 1]
    contrib = pd.Series(lgbm.predict(x, pred_contrib=True)[0, :-1], index=x.columns)

    limiar = carregar_json(MODELS / "limiar.json")["limiar"]
    percentil = min((previsoes.loc[previsoes["conjunto"] == "teste", "p_lgbm"] < risco).mean(), 0.99)
    m1, m2, m3 = st.columns(3)
    m1.metric("Risco de atraso", pct(risco))
    m2.metric("Comparado aos pedidos do teste", f"maior que {pct(percentil, 0)}")
    m3.metric("Ação preventiva?", "sim" if risco >= limiar else "não",
              help=f"Limiar do cenário central de custo: {pct(limiar)}")

    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:10]).reset_index()
    top.columns = ["atributo", "contribuicao"]
    top["sentido"] = np.where(top["contribuicao"] > 0, "aumenta o risco", "reduz o risco")
    st.altair_chart(
        alt.Chart(top).mark_bar(cornerRadiusEnd=4).encode(
            alt.Y("atributo:N", sort=None, title=None),
            alt.X("contribuicao:Q", title="contribuição para o risco (log-odds)"),
            alt.Color("sentido:N", title=None,
                      scale=alt.Scale(domain=["aumenta o risco", "reduz o risco"], range=[LARANJA, AZUL])),
            tooltip=["atributo", alt.Tooltip("contribuicao:Q", format="+.3f", title="contribuição")],
        ).properties(title="Por que esse risco", height=340),
        use_container_width=True,
    )


def decisao():
    st.title("Decisão por custo — quando vale agir?")
    st.caption("Agir = contato proativo com o cliente nos pedidos com risco acima do limiar. O limiar é escolhido "
               "na validação (mar–mai/2018) e aplicado no teste (jun–ago/2018).")
    c1, c2, c3 = st.columns(3)
    custo_acao = c1.slider("Custo da ação preventiva (R$)", 0.5, 20.0, 2.0, 0.5)
    custo_atraso = c2.slider("Custo de um atraso não tratado (R$)", 5.0, 150.0, 40.0, 5.0)
    efetividade = c3.slider("Fração do custo evitada pela ação", 0.05, 1.0, 0.5, 0.05)

    limiares = np.round(np.arange(0.01, 0.61, 0.005), 3)

    def custos(df):
        y = df["atrasou"].to_numpy().astype(bool)
        agir = df["p_lgbm"].to_numpy()[None, :] >= limiares[:, None]
        tp = (agir & y).sum(1)
        return agir.sum(1) * custo_acao + tp * custo_atraso * (1 - efetividade) + (y.sum() - tp) * custo_atraso, y

    val = previsoes[previsoes["conjunto"] == "validacao"]
    teste = previsoes[previsoes["conjunto"] == "teste"]
    custos_val, y_val = custos(val)
    custos_teste, y_teste = custos(teste)
    i = custos_val.argmin()
    sem_val, sem_teste = y_val.sum() * custo_atraso, y_teste.sum() * custo_atraso
    econ_val, econ_teste = 1 - custos_val[i] / sem_val, 1 - custos_teste[i] / sem_teste
    equilibrio = custo_acao / (efetividade * custo_atraso)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Limiar escolhido (validação)", pct(limiares[i], 1))
    m2.metric("Risco de equilíbrio", pct(equilibrio),
              help="Agir num pedido só compensa se risco × efetividade × custo do atraso > custo da ação")
    m3.metric("Economia na validação", pct(econ_val))
    m4.metric("Economia no teste", pct(econ_teste), delta="compensa" if econ_teste > 0 else "não compensa",
              delta_color="normal" if econ_teste > 0 else "inverse")

    curva = pd.concat([
        pd.DataFrame({"limiar": limiares, "economia": 1 - custos_val / sem_val, "conjunto": "validação"}),
        pd.DataFrame({"limiar": limiares, "economia": 1 - custos_teste / sem_teste, "conjunto": "teste"}),
    ])
    linhas = alt.Chart(curva).mark_line().encode(
        alt.X("limiar:Q", title="limiar de risco", axis=alt.Axis(format="%")),
        alt.Y("economia:Q", title="economia contra não agir", axis=alt.Axis(format="%")),
        alt.Color("conjunto:N", title=None, sort=["validação", "teste"],
                  scale=alt.Scale(domain=["validação", "teste"], range=[CINZA, AZUL])),
        tooltip=["conjunto", alt.Tooltip("limiar:Q", format=".1%"), alt.Tooltip("economia:Q", format=".1%")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=CINZA).encode(y="y:Q")
    escolhido = alt.Chart(pd.DataFrame({"x": [limiares[i]]})).mark_rule(color=LARANJA, strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart((linhas + zero + escolhido).properties(
        title="Economia por limiar (tracejado = limiar escolhido na validação)", height=340),
        use_container_width=True)
    st.markdown(f"""
A validação tem **{pct(y_val.mean())}** de atrasos e o teste só **{pct(y_teste.mean())}**. Com poucos
atrasos, a precisão entre os pedidos sinalizados cai perto do risco de equilíbrio e a ação passa a
custar mais do que evita. Aumente o custo do atraso para ver a partir de onde o modelo compensa no teste.
""")


def causal_pagina():
    st.title("Efeito causal — quanto o atraso, sozinho, derruba a nota?")
    r, rb = causal["nota"], causal["nota_ruim"]
    c1, c2 = st.columns(2)
    c1.metric("Efeito na nota (1–5)", num(r["theta"]),
              delta=f"diferença ingênua {num(r['ingenua'])}", delta_color="off")
    c2.metric("Efeito em P(nota ≤ 2)", f"+{num(rb['theta'] * 100, 1)} p.p.",
              delta=f"diferença ingênua +{num(rb['ingenua'] * 100, 1)} p.p.", delta_color="off")
    st.markdown(f"""
**Double Machine Learning** (modelo parcialmente linear, cross-fitting em 5 partições): prevê a nota e o
atraso a partir dos atributos pré-compra e regride um resíduo no outro. Assim se compara pedidos de mesmo
perfil (região, produto, frete, prazo, período). IC 95% do efeito na nota: {num(r['ic_inf'])} a
{num(r['ic_sup'])}. Só {num(abs(r['ingenua'] - r['theta']))} ponto da diferença bruta vinha do perfil
dos pedidos que atrasam.
""")

    dose = pd.DataFrame([{"faixa": f, **v} for f, v in causal["dose_resposta_dml"].items()])
    pontos = alt.Chart(dose).mark_point(filled=True, size=90, color=LARANJA).encode(
        alt.X("faixa:N", sort=None, title="dias de atraso"),
        alt.Y("theta:Q", title="efeito na nota (contra no prazo)", scale=alt.Scale(domain=[-3, 0])),
        tooltip=[alt.Tooltip("faixa:N", title="atraso"), alt.Tooltip("theta:Q", format="+.2f", title="efeito"),
                 alt.Tooltip("ic_inf:Q", format="+.2f", title="IC inf."),
                 alt.Tooltip("ic_sup:Q", format="+.2f", title="IC sup.")],
    )
    barras_erro = alt.Chart(dose).mark_rule(color=LARANJA).encode(
        alt.X("faixa:N", sort=None), alt.Y("ic_inf:Q"), alt.Y2("ic_sup:Q"))
    st.altair_chart((barras_erro + pontos).properties(
        title="H-05 · Efeito causal por duração do atraso (IC 95%)", height=320), use_container_width=True)

    st.subheader("Voz do cliente")
    st.markdown("Termos que um classificador TF-IDF associa a comentários de pedidos atrasados × no prazo "
                "(40 mil comentários, ROC-AUC 0,90). Os atrasados reclamam do **prazo**; defeito, troca e "
                "devolução aparecem do lado dos pedidos **no prazo**.")
    st.image(str(FIG / "07_termos.png"))
    st.info("**Limitação.** O efeito supõe que não há confundidores não observados. A qualidade do vendedor "
            "afeta atraso e nota ao mesmo tempo e não está no modelo, então o efeito pode estar superestimado.")


def impacto():
    st.title("Impacto — quanto vale reduzir atrasos?")
    reducao = st.slider("Redução da taxa de atraso", 0, 100, 50, 5, format="%d%%") / 100
    r, rb = causal["nota"], causal["nota_ruim"]
    evitados = TAXA_GLOBAL * reducao
    c1, c2, c3 = st.columns(3)
    c1.metric("Nova taxa de atraso", pct(TAXA_GLOBAL * (1 - reducao), 2))
    c2.metric("Avaliações ≤ 2 evitadas a cada 10 mil pedidos", f"{evitados * rb['theta'] * 10_000:.0f}",
              help=f"IC 95%: {evitados * rb['ic_inf'] * 10_000:.0f} a {evitados * rb['ic_sup'] * 10_000:.0f}")
    c3.metric("Ganho na nota média", f"+{num(evitados * -r['theta'], 3)}")
    st.markdown(f"""
**Premissas.** Taxa de atraso atual = {pct(TAXA_GLOBAL, 2)} (observada). Cada atraso evitado recupera o
efeito causal médio (DML). Como o efeito cresce com os dias de atraso, atacar primeiro os atrasos longos
renderia mais do que esta conta linear mostra.
""")


pg = st.navigation([
    st.Page(resumo, title="Resumo", icon=":material/summarize:", default=True),
    st.Page(exploracao, title="Exploração", icon=":material/query_stats:"),
    st.Page(modelo, title="Modelo preditivo", icon=":material/model_training:"),
    st.Page(simulador, title="Simulador de risco", icon=":material/tune:"),
    st.Page(decisao, title="Decisão por custo", icon=":material/payments:"),
    st.Page(causal_pagina, title="Efeito causal", icon=":material/science:", url_path="causal"),
    st.Page(impacto, title="Impacto", icon=":material/trending_up:"),
])
pg.run()
