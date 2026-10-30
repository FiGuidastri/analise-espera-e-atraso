# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Visão geral

Projeto de portfólio de Ciência de Dados (em português) sobre o dataset Olist (Brazilian E-Commerce, Kaggle `olistbr/brazilian-ecommerce`). Duas perguntas centrais:

1. **Preditiva** — prever, no momento da compra, a probabilidade de um pedido atrasar.
2. **Causal** — estimar quanto o atraso, isoladamente, reduz a nota de avaliação (Double Machine Learning).

Documentos de referência (leia antes de implementar uma etapa):
- `README.md` — README final com resultados de todas as etapas (o roteiro original por etapa foi substituído; está no histórico do Git).
- `DOCs/PRD.md` — requisitos formais: hipóteses H-01..H-05, requisitos de análise RA-01..RA-09 e não funcionais RNF-01..RNF-05. Cite esses IDs no código/markdown dos notebooks.

## Estado atual e ambiente

- Etapas 0–9 implementadas; os 8 notebooks rodam em ordem (`cd notebooks; for n in 0*.py; do MPLBACKEND=Agg python "$n" || break; done`, ~6 min).
- **Notebooks são scripts `.py` no formato percent** (`#%%` delimita células). Não criar nem converter para `.ipynb`.
- **Dados:** baixados com `kagglehub` (não o pacote `kaggle`/`kaggle.json`), dataset `olistbr/brazilian-ecommerce/versions/2`, numa célula `#%%` no início de `01_carga_e_alvo.py`, gravados direto em `data/` via `output_dir` (não no cache do kagglehub). `data/` não deve ir para o Git.
- Não há build, lint nem testes. `requirements.txt` tem as versões fixadas (acrescentar libs conforme as etapas exigirem).

## Arquitetura

- Notebooks `.py` numerados por etapa: `01_carga_e_alvo`, `02_atributos`, `03_exploracao`, `04_modelagem`, `05_decisao_custo`, `06_causal`, `07_texto`, `08_impacto`.
- `src/features.py` — carga, alvo, agregações, `construir_base`, `preparar_x` e as listas `ATRIBUTOS*` (RNF-03). `src/estilo.py` — estilo único dos gráficos.
- Fluxo de arquivos: 02 grava `data/base_analitica.parquet` (lida por 03–08) e `reports/base_app.parquet` (versão enxuta, sem ids e comentários); 04 grava `reports/previsoes.parquet` (lida por 05 e pelo app); 06 grava `reports/resultados_causais.json` (lido por 08 e pelo app). Figuras em `reports/figuras/`.
- `app/relatorio.py` — relatório Streamlit (`streamlit run app/relatorio.py`); lê só arquivos versionados de `reports/` e `models/`, nunca `data/`, para funcionar no Streamlit Community Cloud.
- `models/` — `modelo_lgbm.joblib` (versionado, ~0,3 MB, usado pelo app), `atributos.json` e `limiar.json`.
- Split temporal fixo no 04: treino < 2018-03-01, validação < 2018-06-01, teste < 2018-08-13 (depois disso, excluído por viés de fim de base).
- Base analítica: **uma linha por pedido**, compras a partir de jan/2017, apenas pedidos `delivered` com data de entrega preenchida.

## Regras de domínio que não podem ser violadas

- **Alvo (RA-01):** atraso = data de entrega real > data estimada, comparando **apenas a data** (a estimada vem à meia-noite; comparar timestamps marcaria como atrasado pedido entregue no próprio dia). Variável contínua: dias de diferença (positivo = atraso).
- **Sem vazamento (RNF-02):** atributos só com informação disponível no clique de compra. Proibidos como atributos: `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date` e qualquer dado de avaliação. Os mesmos atributos pré-compra são os confundidores do modelo causal.
- **Validação temporal (RA-04):** split cronológico treino/validação/teste, sem embaralhar; excluir do teste as últimas semanas da base (só contêm pedidos já entregues, o que reduz artificialmente a taxa de atraso). Hiperparâmetros e limiar de custo escolhidos só na validação; teste usado uma única vez.
- **Métrica principal:** PR-AUC (apoio: ROC-AUC, precisão/recall no limiar). Baseline = regressão logística; modelo principal = LightGBM com early stopping; interpretabilidade com SHAP.
- **Pedidos com vários itens:** documentar regra para vendedor/categoria principal (ex.: item de maior preço) e o critério de deduplicação de avaliações por pedido.
- **Reprodutibilidade (RNF-01):** semente global fixa e dependências com versão fixada em `requirements.txt` (sem célula de print de versões nos notebooks).
