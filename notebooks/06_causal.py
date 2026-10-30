#%%
# Setup - Etapa 6: inferência causal (RA-07)
import json
import os
import random
import sys
from pathlib import Path

# O joblib tenta contar núcleos físicos via `wmic` (ausente no Windows 11) e imprime um
# traceback inofensivo; informar o limite de núcleos evita a consulta.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max((os.cpu_count() or 2) // 2, 1)))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figuras"
sys.path.insert(0, str(ROOT))

from src.estilo import AZUL, CINZA, LARANJA, aplicar_estilo
from src.features import preparar_x

aplicar_estilo()
base = pd.read_parquet(DATA_DIR / "base_analitica.parquet")

#%%
# Amostra causal: pedidos com avaliação (deduplicada na Etapa 2: a respondida por último)
causal = base.dropna(subset=["review_score"]).reset_index(drop=True)
Y_nota = causal["review_score"].to_numpy()
Y_ruim = (causal["review_score"] <= 2).astype(int).to_numpy()
T = causal["atrasou"].to_numpy()

# Confundidores: os mesmos atributos pré-compra do modelo preditivo (RA-02) mais a
# tendência temporal (meses desde jan/2017), porque atraso e nota variam com o período.
X = preparar_x(causal)
X["meses_desde_inicio"] = (causal["order_purchase_timestamp"].dt.to_period("M")
                           - pd.Period("2017-01", "M")).apply(lambda d: d.n)
print(f"{len(causal)} pedidos | tratados (atrasados): {T.mean():.2%}")

#%%
# Referência: diferença ingênua entre médias
def diferenca_ingenua(y, t):
    dif = y[t == 1].mean() - y[t == 0].mean()
    ep = np.sqrt(y[t == 1].var(ddof=1) / (t == 1).sum() + y[t == 0].var(ddof=1) / (t == 0).sum())
    return dif, ep


for nome, y in [("nota (1–5)", Y_nota), ("P(nota ≤ 2)", Y_ruim)]:
    dif, ep = diferenca_ingenua(y, T)
    print(f"ingênua, {nome}: {dif:+.3f} (IC 95% {dif - 1.96 * ep:+.3f} a {dif + 1.96 * ep:+.3f})")

#%%
# Double Machine Learning - modelo parcialmente linear (Chernozhukov et al., 2018)
#   Y = θ·T + g(X) + ε      T = m(X) + ν
# 1. prever Y a partir de X  -> resíduo Ỹ = Y - l̂(X)
# 2. prever T a partir de X  -> resíduo T̃ = T - m̂(X)
# 3. regredir Ỹ em T̃         -> θ̂ = Σ Ỹ·T̃ / Σ T̃²
# Cross-fitting: as previsões de cada partição vêm de modelos treinados nas outras K-1,
# o que evita o viés de sobreajuste dos modelos de nuisance.
PARAMS = {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
          "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
          "random_state": SEED, "verbose": -1}


def residuos(y, t, x, k=5):
    y_res, t_res = np.zeros(len(y)), np.zeros(len(t))
    for treino, aval in KFold(k, shuffle=True, random_state=SEED).split(x):
        l_hat = lgb.LGBMRegressor(**PARAMS).fit(x.iloc[treino], y[treino])
        m_hat = lgb.LGBMClassifier(**PARAMS).fit(x.iloc[treino], t[treino])
        y_res[aval] = y[aval] - l_hat.predict(x.iloc[aval])
        t_res[aval] = t[aval] - m_hat.predict_proba(x.iloc[aval])[:, 1]
    return y_res, t_res


def dml(y, t, x, k=5):
    """Efeito θ, erro padrão (pela função de influência) e IC 95%."""
    y_res, t_res = residuos(y, t, x, k)
    theta = (y_res * t_res).sum() / (t_res ** 2).sum()
    psi = (y_res - theta * t_res) * t_res
    ep = np.sqrt((psi ** 2).mean() / (t_res ** 2).mean() ** 2 / len(y))
    return {"theta": theta, "ep": ep, "ic_inf": theta - 1.96 * ep, "ic_sup": theta + 1.96 * ep,
            "p_valor": 2 * stats.norm.sf(abs(theta / ep)), "n": len(y)}


resultados = {}
for chave, nome, y in [("nota", "nota (1–5)", Y_nota), ("nota_ruim", "P(nota ≤ 2)", Y_ruim)]:
    r = dml(y, T, X)
    dif, ep = diferenca_ingenua(y, T)
    r.update({"ingenua": dif, "ingenua_ep": ep})
    resultados[chave] = r
    print(f"DML, {nome}: θ = {r['theta']:+.3f} (IC 95% {r['ic_inf']:+.3f} a {r['ic_sup']:+.3f}) "
          f"| ingênua {dif:+.3f}")

#%%
# Dose-resposta (H-05): nota média por faixa de dias em relação à data estimada
FAIXAS = [-np.inf, -15, -8, -1, 0, 3, 7, 14, np.inf]
ROTULOS = ["≤ -15", "-14 a -8", "-7 a -1", "no dia", "1–3", "4–7", "8–14", "≥ 15"]
causal["faixa_dias"] = pd.cut(causal["dias_atraso"], FAIXAS, labels=ROTULOS, right=True)
dose = causal.groupby("faixa_dias", observed=True).agg(
    pedidos=("review_score", "size"), nota_media=("review_score", "mean"),
    ep=("review_score", "sem"), pct_nota_ruim=("review_score", lambda s: (s <= 2).mean()),
)
print(dose.round(3).to_string())

fig, ax = plt.subplots(figsize=(9, 4))
cores = [AZUL if r in ROTULOS[:4] else LARANJA for r in dose.index]
ax.bar(dose.index.astype(str), dose["nota_media"], color=cores, width=0.7,
       yerr=1.96 * dose["ep"], ecolor=CINZA, capsize=3)
ax.set_ylim(1, 5)
ax.set_xlabel("dias de diferença em relação à data estimada (positivo = atraso)")
ax.set_ylabel("nota média")
ax.set_title("Nota média por dias de atraso (azul = no prazo, laranja = atrasado)")
fig.savefig(FIG_DIR / "06_dose_resposta.png")
plt.show()

#%%
# Dose-resposta ajustada: DML separado para cada faixa de atraso contra os pedidos no prazo
no_prazo = causal["atrasou"] == 0
dose_dml = {}
for faixa in ["1–3", "4–7", "8–14", "≥ 15"]:
    amostra = no_prazo | (causal["faixa_dias"] == faixa)
    t = (causal.loc[amostra, "faixa_dias"] == faixa).astype(int).to_numpy()
    r = dml(Y_nota[amostra.to_numpy()], t, X[amostra.to_numpy()].reset_index(drop=True))
    dose_dml[faixa] = r
    print(f"atraso {faixa:>5} dias: θ = {r['theta']:+.3f} (IC 95% {r['ic_inf']:+.3f} a {r['ic_sup']:+.3f})")
resultados["dose_resposta_dml"] = dose_dml

#%%
# Interpretação
r = resultados["nota"]
print(f"""
H-04: o atraso reduz a nota em {abs(r['theta']):.2f} ponto (IC 95% {abs(r['ic_sup']):.2f} a {abs(r['ic_inf']):.2f}),
      mesmo controlando o perfil do pedido. A diferença ingênua é {r['ingenua']:+.2f}: o ajuste
      {'reduz' if abs(r['theta']) < abs(r['ingenua']) else 'aumenta'} o efeito em {abs(abs(r['ingenua']) - abs(r['theta'])):.2f} ponto,
      parte da diferença bruta vinha do perfil dos pedidos que atrasam.
H-05: o efeito cresce com a duração do atraso — {dose_dml['1–3']['theta']:+.2f} (1–3 dias), {dose_dml['4–7']['theta']:+.2f} (4–7),
      {dose_dml['8–14']['theta']:+.2f} (8–14) — e estabiliza depois de uma semana ({dose_dml['≥ 15']['theta']:+.2f} em ≥ 15 dias):
      a nota já está perto do piso.
""")
# Limitações:
# - Premissa central: não há confundidores não observados. A qualidade do vendedor
#   (embalagem, produto conforme anúncio) afeta atraso e nota ao mesmo tempo e não está
#   em X; se vendedores ruins atrasam mais, θ ainda superestima o efeito do atraso.
# - A nota só existe para quem avaliou; se a propensão a avaliar depende do atraso, há
#   viés de seleção.
# - Modelo parcialmente linear: θ é um efeito médio; a heterogeneidade aparece só na
#   dose-resposta por faixa.

#%%
json.dump(resultados, open(REPORTS_DIR / "resultados_causais.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=float)
print("salvo: reports/resultados_causais.json")
