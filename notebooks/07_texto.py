#%%
# Setup - Etapa 7: análise textual dos comentários (RA-08)
import os
import re
import sys
from pathlib import Path

# O joblib tenta contar núcleos físicos via `wmic` (ausente no Windows 11) e imprime um
# traceback inofensivo; informar o limite de núcleos evita a consulta.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max((os.cpu_count() or 2) // 2, 1)))

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from unidecode import unidecode

SEED = 42

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figuras"
sys.path.insert(0, str(ROOT))

from src.estilo import AZUL, LARANJA, aplicar_estilo

aplicar_estilo()
base = pd.read_parquet(DATA_DIR / "base_analitica.parquet")

#%%
# Avaliações com comentário preenchido (título + mensagem)
texto = (base["review_comment_title"].fillna("") + " " + base["review_comment_message"].fillna("")).str.strip()
comentarios = base.assign(texto=texto)[texto != ""]
print(f"{len(comentarios)} comentários ({len(comentarios) / len(base):.0%} dos pedidos) | "
      f"atrasados entre eles: {comentarios['atrasou'].mean():.1%}")

#%%
# Pré-processamento: minúsculas, sem acentos, só letras, sem stopwords em português.
# "nao", "nem" e "sem" ficam: em avaliação, negação muda o sentido ("nao recebi").
nltk.download("stopwords", quiet=True)
MANTER = {"nao", "nem", "sem"}
STOPWORDS = {unidecode(w) for w in nltk.corpus.stopwords.words("portuguese")} - MANTER


def limpar(s):
    s = re.sub(r"[^a-z ]", " ", unidecode(s.lower()))
    return " ".join(w for w in s.split() if w not in STOPWORDS and len(w) > 1)


comentarios["texto_limpo"] = comentarios["texto"].map(limpar)
print(comentarios[["texto", "texto_limpo"]].head(5).to_string())

#%%
# TF-IDF (unigramas e bigramas, frequência mínima de 10 documentos) + regressão logística
treino, aval = train_test_split(comentarios, test_size=0.25, stratify=comentarios["atrasou"],
                                random_state=SEED)
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=10)
X_treino = tfidf.fit_transform(treino["texto_limpo"])
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
clf.fit(X_treino, treino["atrasou"])

auc = roc_auc_score(aval["atrasou"], clf.predict_proba(tfidf.transform(aval["texto_limpo"]))[:, 1])
print(f"vocabulário: {len(tfidf.vocabulary_)} termos | ROC-AUC (25% separados): {auc:.3f}")

#%%
# Termos com maiores (atraso) e menores (no prazo) coeficientes
coefs = pd.Series(clf.coef_[0], index=tfidf.get_feature_names_out()).sort_values()
top_atraso = coefs.tail(20)[::-1]
top_prazo = coefs.head(20)
print("ASSOCIADOS A ATRASO:\n", top_atraso.round(2).to_string())
print("\nASSOCIADOS A NO PRAZO:\n", top_prazo.round(2).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 6))
for ax, termos, cor, titulo in [(axes[0], top_atraso[:15][::-1], LARANJA, "Termos de pedidos atrasados"),
                                (axes[1], top_prazo[:15].abs()[::-1], AZUL, "Termos de pedidos no prazo")]:
    ax.barh(termos.index, termos.values, color=cor, height=0.7)
    ax.set_title(titulo)
    ax.set_xlabel("|coeficiente|")
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "07_termos.png")
plt.show()

#%%
# Os termos de atraso falam de prazo/entrega ou de outros problemas?
PRAZO = r"prazo|entreg|receb|cheg|demor|atras|ainda|aguard|espera|data|correio|dias|hoje|mes|passou|cumpr"
de_prazo = top_atraso.index.str.contains(PRAZO)
print(f"{de_prazo.sum()} dos 20 termos mais associados a atraso falam de prazo/entrega:")
print(list(top_atraso.index[de_prazo]))
print("outros:", list(top_atraso.index[~de_prazo]))
# Conexão com o RA-07: os comentários de pedidos atrasados reclamam do prazo ("nao
# chegou", "nao recebi", "atraso", "correios", "nao cumpriu"). Já "defeito", "quebrado",
# "trocar", "devolucao" e "faltou" aparecem do lado dos pedidos NO PRAZO: reclamação de
# qualidade é outro tipo de insatisfação, que não anda junto com o atraso. Isso reforça
# que a queda de ~1,9 ponto estimada pelo DML vem do atraso em si.
