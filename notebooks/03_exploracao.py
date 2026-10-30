#%%
# Setup - Etapa 3: análise exploratória (RA-03)
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figuras"
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.estilo import AZUL, CINZA, LARANJA, aplicar_estilo

aplicar_estilo()
base = pd.read_parquet(DATA_DIR / "base_analitica.parquet")
TAXA_GLOBAL = base["atrasou"].mean()
print(f"{len(base)} pedidos | taxa de atraso global: {TAXA_GLOBAL:.2%}")


def taxa_por(coluna, minimo=1):
    """Taxa de atraso e volume por grupo, descartando grupos com menos de `minimo` pedidos."""
    t = base.groupby(coluna, observed=True)["atrasou"].agg(taxa="mean", pedidos="size")
    return t[t["pedidos"] >= minimo]


def barras_taxa(t, titulo, arquivo, horizontal=False):
    fig, ax = plt.subplots(figsize=(8, 5 if horizontal else 4))
    if horizontal:
        ax.barh(t.index.astype(str), t["taxa"], color=AZUL, height=0.7)
        ax.axvline(TAXA_GLOBAL, color=CINZA, ls="--", lw=1, label="média global")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    else:
        ax.bar(t.index.astype(str), t["taxa"], color=AZUL, width=0.7)
        ax.axhline(TAXA_GLOBAL, color=CINZA, ls="--", lw=1, label="média global")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(titulo)
    ax.legend()
    fig.savefig(FIG_DIR / arquivo)
    plt.show()

#%%
# Qualidade de dados: ausentes por coluna e decisão
ausentes = base.isna().sum()
print(ausentes[ausentes > 0].to_string())
# Decisões:
# - distancia_km (CEP sem geolocalização): mantido como NaN; LightGBM trata nativamente,
#   a logística imputa pela mediana.
# - categoria_principal / n_fotos_principal (produto sem cadastro): categoria "desconhecida"
#   na modelagem; fotos imputadas.
# - peso/volume: poucos casos, imputação pela mediana.
# - review_score: pedidos sem avaliação ficam fora só da análise causal.
# - comentários: ausência é o normal (cliente não escreveu); só a Etapa 7 usa.

#%%
# Valores extremos
print(base[["preco_total", "frete_total", "frete_relativo", "peso_total_g",
            "distancia_km", "prazo_estimado_dias", "prazo_envio_dias", "dias_atraso"]]
      .describe(percentiles=[.01, .5, .99]).T.round(1).to_string())
# Caudas longas (preço, prazo de envio de até ~1000 dias) mas plausíveis; não removemos:
# árvores são robustas a elas e a logística recebe as variáveis padronizadas.

#%%
# Taxa de atraso por mês de compra
mensal = base.groupby(base["order_purchase_timestamp"].dt.to_period("M"))["atrasou"].mean()
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(mensal.index.to_timestamp(), mensal.values, color=AZUL, marker="o", ms=4)
ax.axhline(TAXA_GLOBAL, color=CINZA, ls="--", lw=1, label="média global")
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_title("Taxa de atraso por mês de compra")
ax.legend()
fig.savefig(FIG_DIR / "03_taxa_mensal.png")
plt.show()
print(mensal.map("{:.1%}".format).to_string())
# A taxa NÃO é estável: picos em nov/2017 (Black Friday) e fev–mar/2018. Isso exige
# validação temporal (RA-04) e explica por que validação e teste terão taxas diferentes.

#%%
# H-02: taxa de atraso por UF do cliente e por região
por_uf = taxa_por("uf_cliente").sort_values("taxa")
barras_taxa(por_uf, "Taxa de atraso por UF do cliente", "03_taxa_uf_cliente.png", horizontal=True)
por_regiao = taxa_por("regiao_cliente").sort_values("taxa", ascending=False)
print(por_regiao.assign(taxa=por_regiao["taxa"].map("{:.1%}".format)).to_string())

#%%
# Taxa de atraso por UF do vendedor (só UFs com pelo menos 200 pedidos)
por_uf_vend = taxa_por("uf_vendedor", minimo=200).sort_values("taxa")
barras_taxa(por_uf_vend, "Taxa de atraso por UF do vendedor (≥ 200 pedidos)",
            "03_taxa_uf_vendedor.png", horizontal=True)

#%%
# H-01: taxa de atraso por faixa de distância
FAIXAS_DIST = [0, 100, 300, 600, 1000, 2000, np.inf]
base["faixa_distancia"] = pd.cut(base["distancia_km"], FAIXAS_DIST, right=False,
                                 labels=["<100", "100–300", "300–600", "600–1000", "1000–2000", "≥2000"])
por_dist = taxa_por("faixa_distancia")
barras_taxa(por_dist, "Taxa de atraso por distância cliente–vendedor (km)", "03_taxa_distancia.png")
print(por_dist.to_string())

#%%
# H-03: taxa de atraso por faixa de prazo estimado
FAIXAS_PRAZO = [0, 10, 15, 20, 25, 30, 40, np.inf]
base["faixa_prazo"] = pd.cut(base["prazo_estimado_dias"], FAIXAS_PRAZO, right=False,
                             labels=["<10", "10–15", "15–20", "20–25", "25–30", "30–40", "≥40"])
por_prazo = taxa_por("faixa_prazo")
barras_taxa(por_prazo, "Taxa de atraso por prazo estimado (dias)", "03_taxa_prazo.png")
print(por_prazo.to_string())

#%%
# Distribuição das notas: atrasados x no prazo
notas = (base.dropna(subset=["review_score"])
         .groupby("atrasou")["review_score"].value_counts(normalize=True)
         .unstack(0).rename(columns={0: "no prazo", 1: "atrasado"}))
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(notas))
ax.bar(x - 0.2, notas["no prazo"], width=0.38, color=AZUL, label="no prazo")
ax.bar(x + 0.2, notas["atrasado"], width=0.38, color=LARANJA, label="atrasado")
ax.set_xticks(x, notas.index.astype(int))
ax.set_xlabel("nota")
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.set_title("Distribuição da nota de avaliação")
ax.legend()
fig.savefig(FIG_DIR / "03_notas.png")
plt.show()

media_nota = base.groupby("atrasou")["review_score"].mean()
print(notas.round(3).to_string())
print(f"\nnota média no prazo: {media_nota[0]:.2f} | atrasado: {media_nota[1]:.2f}")

#%%
# Conclusões preliminares (fechadas com SHAP na Etapa 4)
print(f"""
H-01 (distância): taxa vai de {por_dist['taxa'].iloc[0]:.1%} (<100 km) a {por_dist['taxa'].max():.1%} na faixa mais alta
H-02 (Norte/Nordeste): {por_regiao['taxa'].to_dict()}
H-03 (prazo curto): taxa na faixa <10 dias = {por_prazo['taxa'].iloc[0]:.1%}; ≥40 dias = {por_prazo['taxa'].iloc[-1]:.1%}
Nota: atrasados {media_nota[1]:.2f} contra {media_nota[0]:.2f} no prazo — queda grande o suficiente para a análise causal.
""")
