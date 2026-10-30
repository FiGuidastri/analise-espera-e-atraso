#%%
# Setup - Etapa 4: modelagem preditiva (RA-04, RA-06)
import json
import os
import random
import sys
from pathlib import Path

# O joblib tenta contar núcleos físicos via `wmic` (ausente no Windows 11) e imprime um
# traceback inofensivo; informar o limite de núcleos evita a consulta.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max((os.cpu_count() or 2) // 2, 1)))

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figuras"
MODELS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.estilo import AZUL, CINZA, LARANJA, aplicar_estilo
from src.features import ATRIBUTOS, ATRIBUTOS_CATEGORICOS, ATRIBUTOS_NUMERICOS, REGIAO_POR_UF, preparar_x

aplicar_estilo()
base = pd.read_parquet(DATA_DIR / "base_analitica.parquet")

#%%
# Divisão temporal - RA-04 (sem embaralhar)
# Treino: jan/2017–fev/2018 | Validação: mar–mai/2018 | Teste: jun/2018–12/ago/2018
# Compras a partir de 13/08/2018 ficam fora: na data de corte da base, os pedidos dessas
# semanas que atrasaram ainda não tinham sido entregues e não aparecem como "delivered",
# derrubando a taxa de atraso (3,6% → 2,2% → 0% nas últimas semanas, contra ~9% antes).
TREINO_FIM = pd.Timestamp("2018-03-01")
VALID_FIM = pd.Timestamp("2018-06-01")
TESTE_FIM = pd.Timestamp("2018-08-13")

compra = base["order_purchase_timestamp"]
base["conjunto"] = np.select(
    [compra < TREINO_FIM, compra < VALID_FIM, compra < TESTE_FIM],
    ["treino", "validacao", "teste"], default="excluido",
)
X = preparar_x(base)
y = base["atrasou"]
idx = {c: base.index[base["conjunto"] == c] for c in ["treino", "validacao", "teste"]}

print(base.groupby("conjunto", sort=False)["atrasou"].agg(pedidos="size", taxa_atraso="mean")
      .assign(taxa_atraso=lambda d: d["taxa_atraso"].map("{:.2%}".format)).to_string())


def metricas(y_true, p):
    return {"PR-AUC": average_precision_score(y_true, p), "ROC-AUC": roc_auc_score(y_true, p)}


# Conjuntos de atributos testados. `mes` é pré-compra, mas com só 14 meses de treino ele
# tende a decorar episódios (Black Friday 2017, pico de fev–mar/2018) em vez de sazonalidade
# que se repete. A escolha entre os dois conjuntos é feita na validação, como a dos hiperparâmetros.
CONJUNTOS = {
    "com_mes": ATRIBUTOS,
    "sem_mes": [a for a in ATRIBUTOS if a != "mes"],
}

#%%
# Baseline: regressão logística com imputação, padronização e one-hot
# C e conjunto de atributos escolhidos na validação.
def nova_logit(atributos, C):
    preproc = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
         [a for a in ATRIBUTOS_NUMERICOS if a in atributos]),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), ATRIBUTOS_CATEGORICOS),
    ])
    return make_pipeline(preproc, LogisticRegression(C=C, max_iter=3000))


melhor = None
for nome_conj, atributos in CONJUNTOS.items():
    for C in [0.01, 0.1, 1.0]:
        modelo = nova_logit(atributos, C).fit(X.loc[idx["treino"], atributos], y[idx["treino"]])
        pr = average_precision_score(
            y[idx["validacao"]], modelo.predict_proba(X.loc[idx["validacao"], atributos])[:, 1])
        print(f"{nome_conj}, C={C}: PR-AUC validação = {pr:.4f}")
        if melhor is None or pr > melhor[1]:
            melhor = (modelo, pr, {"conjunto": nome_conj, "C": C})

logit, _, escolha_logit = melhor
atrib_logit = CONJUNTOS[escolha_logit["conjunto"]]
print("escolhido:", escolha_logit)

#%%
# Modelo principal: LightGBM com parada antecipada na validação
# Pequena grade de complexidade x conjunto de atributos, escolhida pela PR-AUC de validação.
melhor = None
for nome_conj, atributos in CONJUNTOS.items():
    for nl in [7, 15, 31]:
        for mcs in [200, 500]:
            params = {"num_leaves": nl, "min_child_samples": mcs}
            modelo = lgb.LGBMClassifier(
                n_estimators=3000, learning_rate=0.03, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, random_state=SEED, verbose=-1, **params,
            )
            modelo.fit(
                X.loc[idx["treino"], atributos], y[idx["treino"]],
                eval_X=X.loc[idx["validacao"], atributos], eval_y=y[idx["validacao"]],
                eval_metric="average_precision",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            pr = average_precision_score(
                y[idx["validacao"]], modelo.predict_proba(X.loc[idx["validacao"], atributos])[:, 1])
            print(f"{nome_conj} {params} -> {modelo.best_iteration_} árvores, PR-AUC validação = {pr:.4f}")
            if melhor is None or pr > melhor[1]:
                melhor = (modelo, pr, {"conjunto": nome_conj, **params})

lgbm, _, melhores_params = melhor
atrib_lgbm = CONJUNTOS[melhores_params["conjunto"]]
print("escolhido:", melhores_params, "| árvores:", lgbm.best_iteration_)

#%%
# Comparação na validação e, uma única vez, no teste - critério de aceite do RA-04
pred = {}
for c in ["validacao", "teste"]:
    pred[c] = {
        "logit": logit.predict_proba(X.loc[idx[c], atrib_logit])[:, 1],
        "lgbm": lgbm.predict_proba(X.loc[idx[c], atrib_lgbm])[:, 1],
    }
tabela = pd.DataFrame({
    (c, m): {**metricas(y[idx[c]], pred[c][m]), "taxa (PR-AUC aleatória)": y[idx[c]].mean()}
    for c in ["validacao", "teste"] for m in ["logit", "lgbm"]
}).round(4)
print(tabela.to_string())

#%%
# A diferença de PR-AUC no teste é real ou ruído? Bootstrap de 1000 reamostragens do teste.
# (Só mede incerteza; nada aqui volta para a escolha do modelo.)
rng = np.random.default_rng(SEED)
y_teste = y[idx["teste"]].to_numpy()
difs = []
for _ in range(1000):
    i = rng.integers(0, len(y_teste), len(y_teste))
    difs.append(average_precision_score(y_teste[i], pred["teste"]["lgbm"][i])
                - average_precision_score(y_teste[i], pred["teste"]["logit"][i]))
ic_dif = np.percentile(difs, [2.5, 97.5])
dif = tabela.loc["PR-AUC", ("teste", "lgbm")] - tabela.loc["PR-AUC", ("teste", "logit")]
print(f"PR-AUC teste, LightGBM - logística: {dif:+.4f} (IC 95% bootstrap {ic_dif[0]:+.4f} a {ic_dif[1]:+.4f})")
print("critério do RA-04 (LightGBM > baseline no teste):", "atendido" if dif > 0 else "NÃO atendido")
# Leitura: os dois modelos ficam ~2x acima do aleatório no teste. Com atributos só de
# pré-compra, a maior parte do atraso depende do que acontece depois (transportadora,
# vendedor, greve de caminhoneiros em mai/2018), e o ganho do boosting sobre a logística
# fica dentro do ruído. Ver "próximos passos" no README.

#%%
# Curvas precisão-recall no teste
fig, ax = plt.subplots(figsize=(7, 5))
for m, cor, nome in [("logit", CINZA, "Regressão logística"), ("lgbm", AZUL, "LightGBM")]:
    p, r, _ = precision_recall_curve(y[idx["teste"]], pred["teste"][m])
    ax.plot(r, p, color=cor, label=f"{nome} (PR-AUC {tabela.loc['PR-AUC', ('teste', m)]:.3f})")
ax.axhline(y[idx["teste"]].mean(), color=LARANJA, ls="--", lw=1, label="aleatório")
ax.set_xlabel("recall"); ax.set_ylabel("precisão")
ax.set_title("Curva precisão-recall — teste")
ax.legend()
fig.savefig(FIG_DIR / "04_curva_pr.png")
plt.show()

#%%
# SHAP - importância global (RA-06), numa amostra do teste
amostra = X.loc[idx["teste"], atrib_lgbm].sample(5000, random_state=SEED)
explainer = shap.TreeExplainer(lgbm)
sv = explainer(amostra)

plt.close("all")  # o SHAP desenha na figura atual
shap.plots.bar(sv, max_display=15, show=False)
plt.title("Importância média |SHAP| — teste")
plt.savefig(FIG_DIR / "04_shap_importancia.png")
plt.show()

plt.close("all")  # o SHAP desenha na figura atual
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("Efeito dos atributos no risco de atraso (SHAP)")
plt.savefig(FIG_DIR / "04_shap_beeswarm.png")
plt.show()

importancia = pd.Series(np.abs(sv.values).mean(0), index=amostra.columns).sort_values(ascending=False)
print(importancia.round(4).to_string())

#%%
# Explicação local de três pedidos de alto risco do teste
p_teste = pd.Series(pred["teste"]["lgbm"], index=idx["teste"])
top3 = p_teste.nlargest(3).index
sv_top = explainer(X.loc[top3, atrib_lgbm])
for i, pedido in enumerate(top3):
    print(f"\npedido {base.loc[pedido, 'order_id']} | risco {p_teste[pedido]:.1%} | "
          f"atrasou de fato: {bool(y[pedido])}")
    print(X.loc[pedido, ["uf_cliente", "uf_vendedor", "distancia_km", "prazo_estimado_dias",
                         "frete_relativo", "mes"]].to_dict())
    plt.close("all")
    shap.plots.waterfall(sv_top[i], max_display=10, show=False)
    plt.title(f"Pedido de alto risco #{i + 1} ({p_teste[pedido]:.0%})")
    plt.savefig(FIG_DIR / f"04_shap_pedido_{i + 1}.png")
    plt.show()

#%%
# Fechamento das hipóteses H-01 a H-03 com SHAP
def direcao(col):
    """Correlação de Spearman entre o valor do atributo e sua contribuição SHAP."""
    v = amostra[col]
    ok = v.notna()
    return pd.Series(v[ok]).corr(pd.Series(sv[:, col].values[ok], index=v[ok].index), method="spearman")


regiao_shap = (pd.Series(sv[:, "uf_cliente"].values, index=amostra.index)
               .groupby(amostra["uf_cliente"].astype(str).map(REGIAO_POR_UF)).mean().sort_values())
print(f"H-01 distância: posição {importancia.index.get_loc('distancia_km') + 1} no ranking, "
      f"correlação valor×SHAP = {direcao('distancia_km'):+.2f}")
print(f"H-03 prazo estimado: posição {importancia.index.get_loc('prazo_estimado_dias') + 1}, "
      f"correlação valor×SHAP = {direcao('prazo_estimado_dias'):+.2f}")
print("H-02 SHAP médio de uf_cliente por região (log-odds):")
print(regiao_shap.round(3).to_string())

#%%
# Salva modelo, atributos e previsões para as próximas etapas
joblib.dump(lgbm, MODELS_DIR / "modelo_lgbm.joblib")
json.dump({
    "atributos": atrib_lgbm, "categoricos": ATRIBUTOS_CATEGORICOS,
    "categorias": {c: list(X[c].cat.categories) for c in ATRIBUTOS_CATEGORICOS},
    "parametros": melhores_params, "arvores": lgbm.best_iteration_,
    "cortes": {"treino_fim": str(TREINO_FIM.date()), "validacao_fim": str(VALID_FIM.date()),
               "teste_fim": str(TESTE_FIM.date())},
    "metricas": {f"{c}_{m}": tabela[(c, m)].to_dict() for c, m in tabela.columns},
}, open(MODELS_DIR / "atributos.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

previsoes = pd.concat([
    pd.DataFrame({"conjunto": c, "atrasou": y[idx[c]],
                  "p_logit": pred[c]["logit"], "p_lgbm": pred[c]["lgbm"]})
    for c in ["validacao", "teste"]
])
previsoes.to_parquet(ROOT / "reports" / "previsoes.parquet", index=False)
print("salvos: models/modelo_lgbm.joblib, models/atributos.json, reports/previsoes.parquet")
