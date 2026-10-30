#%%
# Setup - Etapa 5: regra de decisão por custo (RA-05)
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figuras"
sys.path.insert(0, str(ROOT))

from src.estilo import AZUL, CINZA, aplicar_estilo

aplicar_estilo()
previsoes = pd.read_parquet(DATA_DIR / "previsoes.parquet")
val = previsoes[previsoes["conjunto"] == "validacao"]
teste = previsoes[previsoes["conjunto"] == "teste"]

#%%
# Premissas (R$ por pedido) - declaradas explicitamente
# custo_acao: contato proativo com o cliente (mensagem + tempo de atendimento)
# custo_atraso: atraso não tratado (contato reativo no SAC, cupom de compensação,
#               efeito da avaliação negativa na conversão)
# efetividade: fração do custo do atraso evitada quando o cliente é avisado antes
CENARIOS = {
    "central":              {"custo_acao": 2.0, "custo_atraso": 40.0, "efetividade": 0.5},
    "ação mais cara":       {"custo_acao": 5.0, "custo_atraso": 40.0, "efetividade": 0.5},
    "atraso mais barato":   {"custo_acao": 2.0, "custo_atraso": 20.0, "efetividade": 0.5},
    "ação pouco efetiva":   {"custo_acao": 2.0, "custo_atraso": 40.0, "efetividade": 0.2},
    "atraso mais caro":     {"custo_acao": 2.0, "custo_atraso": 80.0, "efetividade": 0.5},
}


def custo_total(y, p, limiar, custo_acao, custo_atraso, efetividade):
    """Custo de agir em todo pedido com risco >= limiar.

    Pedido sinalizado: paga a ação e, se atrasar, só a parte não evitada do atraso.
    Pedido não sinalizado: se atrasar, paga o atraso inteiro.
    """
    agir = p >= limiar
    return (agir.sum() * custo_acao
            + (y & agir).sum() * custo_atraso * (1 - efetividade)
            + (y & ~agir).sum() * custo_atraso)


LIMIARES = np.round(np.arange(0.01, 0.61, 0.005), 3)

#%%
# Varredura de limiares na validação e aplicação única no teste
linhas = []
curvas = {}
for nome, premissas in CENARIOS.items():
    yv, pv = val["atrasou"].to_numpy().astype(bool), val["p_lgbm"].to_numpy()
    custos_val = np.array([custo_total(yv, pv, t, **premissas) for t in LIMIARES])
    curvas[nome] = custos_val
    limiar = LIMIARES[custos_val.argmin()]

    yt, pt = teste["atrasou"].to_numpy().astype(bool), teste["p_lgbm"].to_numpy()
    com_modelo = custo_total(yt, pt, limiar, **premissas)
    sem_modelo = custo_total(yt, pt, np.inf, **premissas)       # ninguém é sinalizado
    agir_todos = custo_total(yt, pt, -np.inf, **premissas)       # todos são sinalizados
    agir = pt >= limiar
    linhas.append({
        "cenário": nome, **premissas, "limiar": limiar,
        "economia_validação_%": 1 - custos_val.min() / custo_total(yv, pv, np.inf, **premissas),
        "sinalizados_%": agir.mean(),
        "precisão": (yt & agir).sum() / max(agir.sum(), 1),
        "recall": (yt & agir).sum() / yt.sum(),
        "custo_sem_modelo": sem_modelo, "custo_agir_todos": agir_todos, "custo_com_modelo": com_modelo,
        "economia_%": 1 - com_modelo / sem_modelo,
    })

resultado = pd.DataFrame(linhas).set_index("cenário")
print(resultado.round(3).to_string())

#%%
# Curva de custo na validação - cenário central
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(LIMIARES, curvas["central"] / 1000, color=AZUL)
t_central = resultado.loc["central", "limiar"]
ax.axvline(t_central, color=CINZA, ls="--", lw=1, label=f"limiar escolhido = {t_central:.3f}")
ax.set_xlabel("limiar de risco"); ax.set_ylabel("custo total (R$ mil)")
ax.set_title("Custo total por limiar — validação, cenário central")
ax.legend()
fig.savefig(FIG_DIR / "05_custo_limiar.png")
plt.show()

#%%
# Quando o modelo deixa de compensar?
# Agir só compensa num pedido com risco p se p * efetividade * custo_atraso > custo_acao,
# ou seja, p > custo_acao / (efetividade * custo_atraso). Se esse ponto de equilíbrio
# passa do risco máximo que o modelo atribui, ninguém é sinalizado e a economia é zero.
for nome, linha in resultado.iterrows():
    equilibrio = linha["custo_acao"] / (linha["efetividade"] * linha["custo_atraso"])
    print(f"{nome:<20} risco de equilíbrio = {equilibrio:.1%} | economia no teste = {linha['economia_%']:.1%}")
print(f"risco máximo previsto no teste: {teste['p_lgbm'].max():.1%}")
print(f"taxa de atraso: validação {val['atrasou'].mean():.1%} | teste {teste['atrasou'].mean():.1%}")
# Leitura: na validação (mar–mai/2018, 10% de atraso) a regra economiza; no teste
# (jun–ago/2018, 3,7%) há tão poucos atrasos que a precisão entre os sinalizados fica
# perto do risco de equilíbrio e a ação preventiva custa mais do que evita. O valor do
# modelo depende do regime de atraso: compensa em períodos críticos ou quando o atraso
# é caro (cenário "atraso mais caro"). Um limiar fixo também sofre com a mudança da taxa
# base entre períodos; recalibrar o risco por período é um próximo passo.

#%%
# Salva o limiar do cenário central
json.dump({"limiar": float(t_central), "premissas": CENARIOS["central"],
           "economia_teste": float(resultado.loc["central", "economia_%"])},
          open(MODELS_DIR / "limiar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
resultado.to_csv(ROOT / "reports" / "decisao_custo.csv")
print("salvos: models/limiar.json, reports/decisao_custo.csv")
