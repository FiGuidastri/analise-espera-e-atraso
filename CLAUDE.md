# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Visão geral

Projeto de portfólio de Ciência de Dados (em português) sobre o dataset Olist (Brazilian E-Commerce, Kaggle `olistbr/brazilian-ecommerce`). Duas perguntas centrais:

1. **Preditiva** — prever, no momento da compra, a probabilidade de um pedido atrasar.
2. **Causal** — estimar quanto o atraso, isoladamente, reduz a nota de avaliação (Double Machine Learning).

Documentos de referência (leia antes de implementar uma etapa):
- `README.md` — roteiro de execução por etapa (0 a 9), com checklists, perguntas de checagem e cuidados. Ao final do projeto, as seções de roteiro devem ser substituídas pelos resultados (modelo de README final no próprio arquivo).
- `DOCs/PRD.md` — requisitos formais: hipóteses H-01..H-05, requisitos de análise RA-01..RA-09 e não funcionais RNF-01..RNF-05. Cite esses IDs no código/markdown dos notebooks.

## Estado atual e ambiente

- Projeto em fase inicial: só existe `notebooks/01_carga_e_alvo.py` (vazio).
- **Notebooks são scripts `.py` no formato percent** (`#%%` delimita células). Não criar nem converter para `.ipynb`.
- **Dados:** baixados via API do Kaggle (dataset `olistbr/brazilian-ecommerce`) numa célula `#%%` dedicada no início de `01_carga_e_alvo.py`, salvos em `data/`. `data/` não deve ir para o Git.
- Não há build, lint nem testes. Ainda não existem `requirements.txt`, `src/` nem `models/` — serão criados nas etapas 2 e 9.

## Arquitetura planejada

- Notebooks `.py` numerados por etapa: `01_carga_e_alvo`, `02_atributos`, `03_exploracao`, `04_modelagem`, `05_decisao_custo`, `06_causal`, `07_texto`, `08_impacto`.
- `src/features.py` — funções de agregação/atributos reutilizadas entre notebooks (RNF-03).
- `models/` — modelo serializado, lista de atributos e limiar escolhido.
- Base analítica: **uma linha por pedido**, compras a partir de jan/2017, apenas pedidos `delivered` com data de entrega preenchida.

## Regras de domínio que não podem ser violadas

- **Alvo (RA-01):** atraso = data de entrega real > data estimada, comparando **apenas a data** (a estimada vem à meia-noite; comparar timestamps marcaria como atrasado pedido entregue no próprio dia). Variável contínua: dias de diferença (positivo = atraso).
- **Sem vazamento (RNF-02):** atributos só com informação disponível no clique de compra. Proibidos como atributos: `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date` e qualquer dado de avaliação. Os mesmos atributos pré-compra são os confundidores do modelo causal.
- **Validação temporal (RA-04):** split cronológico treino/validação/teste, sem embaralhar; excluir do teste as últimas semanas da base (só contêm pedidos já entregues, o que reduz artificialmente a taxa de atraso). Hiperparâmetros e limiar de custo escolhidos só na validação; teste usado uma única vez.
- **Métrica principal:** PR-AUC (apoio: ROC-AUC, precisão/recall no limiar). Baseline = regressão logística; modelo principal = LightGBM com early stopping; interpretabilidade com SHAP.
- **Pedidos com vários itens:** documentar regra para vendedor/categoria principal (ex.: item de maior preço) e o critério de deduplicação de avaliações por pedido.
- **Reprodutibilidade (RNF-01):** semente global fixa e versões das bibliotecas registradas.
