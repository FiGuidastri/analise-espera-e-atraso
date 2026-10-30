#%%
# Setup - Etapa 8: impacto de negócio (RA-09)
import json
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

base = pd.read_parquet(DATA_DIR / "base_analitica.parquet")
causal = json.load(open(REPORTS_DIR / "resultados_causais.json", encoding="utf-8"))

#%%
# Premissas
# - Taxa de atraso atual = taxa observada na base (jan/2017–ago/2018).
# - Cada atraso evitado vira um pedido no prazo e recupera o efeito causal médio do
#   atraso (θ do DML, Etapa 6) — tanto na nota quanto na chance de nota ≤ 2.
# - Efeito linear: evitar atrasos de qualquer duração vale o efeito médio. Como o efeito
#   cresce com os dias (H-05), atacar primeiro os atrasos longos renderia mais.
# - Intervalos herdam só a incerteza do θ (IC 95%), não a da taxa de atraso.
TAXA_ATRASO = base["atrasou"].mean()
nota, ruim = causal["nota"], causal["nota_ruim"]
POR_PEDIDOS = 10_000
print(f"taxa de atraso atual: {TAXA_ATRASO:.2%} | θ nota = {nota['theta']:+.3f} | θ P(nota ≤ 2) = {ruim['theta']:+.3f}")

#%%
# Tabela de cenários: reduções de 10%, 25% e 50% na taxa de atraso
linhas = []
for reducao in [0.10, 0.25, 0.50]:
    evitados = TAXA_ATRASO * reducao  # fração dos pedidos que deixa de atrasar
    linhas.append({
        "redução da taxa de atraso": f"{reducao:.0%}",
        "nova taxa de atraso": f"{TAXA_ATRASO * (1 - reducao):.2%}",
        "atrasos evitados / 10 mil pedidos": round(evitados * POR_PEDIDOS),
        "ganho na nota média": f"+{evitados * -nota['theta']:.3f} "
                               f"({evitados * -nota['ic_sup']:.3f} a {evitados * -nota['ic_inf']:.3f})",
        "avaliações ≤ 2 evitadas / 10 mil pedidos": f"{evitados * ruim['theta'] * POR_PEDIDOS:.0f} "
                                                    f"({evitados * ruim['ic_inf'] * POR_PEDIDOS:.0f} a "
                                                    f"{evitados * ruim['ic_sup'] * POR_PEDIDOS:.0f})",
    })
cenarios = pd.DataFrame(linhas)
print(cenarios.to_string(index=False))
cenarios.to_csv(REPORTS_DIR / "impacto_cenarios.csv", index=False)

#%%
# Em uma frase, para um gestor
meio = cenarios.iloc[2]
print(f"\nCortar os atrasos pela metade evita cerca de {meio['avaliações ≤ 2 evitadas / 10 mil pedidos'].split()[0]} "
      f"avaliações negativas a cada 10 mil pedidos e sobe a nota média em "
      f"{meio['ganho na nota média'].split()[0]} ponto.")
