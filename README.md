# Atraso de Entrega e Satisfação do Cliente — Olist

Projeto de Ciência de Dados e Machine Learning que **prevê, no momento da compra, a probabilidade de um pedido atrasar** e **mede o efeito causal do atraso sobre a nota de avaliação**, com base no dataset público de e-commerce da Olist. Os requisitos completos de cada análise estão em [PRD.md](DOCs/PRD.md).

> **Como usar este README:** ele é o roteiro de execução. Cada etapa traz o objetivo, as tarefas, as perguntas que você deve conseguir responder ao final e o entregável. Marque as tarefas conforme avança e, ao terminar o projeto, substitua as seções de roteiro pelos resultados.

## Estrutura sugerida do repositório

```
analise-espera-e-atraso/
├── README.md
├── DOCs/
│   └── PRD.md
├── requirements.txt
├── data/                  # fora do Git (.gitignore)
├── notebooks/             # scripts .py com células #%% (formato percent)
│   ├── 01_carga_e_alvo.py
│   ├── 02_atributos.py
│   ├── 03_exploracao.py
│   ├── 04_modelagem.py
│   ├── 05_decisao_custo.py
│   ├── 06_causal.py
│   ├── 07_texto.py
│   └── 08_impacto.py
├── src/
│   └── features.py        # funções reutilizáveis entre notebooks
└── models/
```

## Visão geral das etapas

| Etapa | Notebook | Requisito no PRD | Entregável |
|---|---|---|---|
| 0. Preparação | 01 | RNF-01, RNF-05 | Dados baixados via API do Kaggle e ambiente reproduzível |
| 1. Carga e alvo | 01 | RA-01 | Variável de atraso e taxa global |
| 2. Atributos | 02 | RA-02 | Base com uma linha por pedido e dicionário de atributos |
| 3. Exploração | 03 | RA-03 | Gráficos e conclusões sobre H-01 a H-03 |
| 4. Modelagem | 04 | RA-04, RA-06 | Baseline, modelo principal e SHAP |
| 5. Decisão por custo | 05 | RA-05 | Limiar ótimo e economia estimada |
| 6. Inferência causal | 06 | RA-07 | Efeito do atraso na nota com intervalo de confiança |
| 7. Texto | 07 | RA-08 | Termos associados a atraso |
| 8. Impacto | 08 | RA-09 | Tabela de cenários de negócio |
| 9. Publicação | — | RNF-04 | README final e repositório no GitHub |

---

## Etapa 0 — Preparação do ambiente

**Objetivo.** Deixar os dados acessíveis de forma reproduzível, baixando-os pela API do Kaggle.

- [x] Criar em `notebooks/01_carga_e_alvo.py` uma célula `#%%` que baixa o dataset `olistbr/brazilian-ecommerce` (versão 2, fixa) com `kagglehub`, gravando direto em `data/` (`output_dir`) e pulando o download se os CSVs já existirem
- [x] Confirmar que os nove CSVs estão presentes (célula com `assert`)
- [x] Definir uma semente aleatória global e fixar as versões das bibliotecas em `requirements.txt`

**Pergunta de checagem.** Se outra pessoa clonar o repositório, ela consegue rodar o notebook apenas instalando o `requirements.txt`?

---

## Etapa 1 — Carga dos dados e definição do alvo

**Objetivo.** Entender as tabelas e construir a variável de atraso (RA-01).

- [ ] Carregar as nove tabelas e verificar o formato (linhas e colunas) de cada uma
- [ ] Desenhar, à mão ou numa ferramenta, o diagrama de relacionamento entre as tabelas e identificar as chaves (`order_id`, `customer_id`, `seller_id`, `product_id`, prefixo de CEP)
- [ ] Converter as colunas de data para o tipo datetime
- [ ] Ver a distribuição de `order_status` e decidir o filtro
- [ ] Criar a variável binária de atraso comparando apenas as datas (sem horário) de entrega real e estimada
- [ ] Criar a variável contínua de dias de diferença
- [ ] Calcular a taxa de atraso global

**Perguntas de checagem.** Qual é a taxa de atraso? Por que a comparação deve ignorar o horário? Quantos pedidos foram descartados pelo filtro e por quê?

**Cuidado.** A data estimada vem sem horário (meia-noite). Comparar os timestamps diretamente marcaria como atrasado um pedido entregue no próprio dia estimado.

---

## Etapa 2 — Engenharia de atributos

**Objetivo.** Construir a base analítica com uma linha por pedido, usando apenas informação disponível no momento da compra (RA-02).

- [ ] Para cada coluna candidata, perguntar: *essa informação existe no instante em que o cliente clica em comprar?* Descartar o que não existir
- [ ] Agregar os itens por pedido: quantidade de itens, de produtos e de vendedores, preço total, frete total, peso e volume totais
- [ ] Definir uma regra para o vendedor e a categoria principais (por exemplo, o item de maior preço) e documentá-la
- [ ] Agregar os pagamentos por pedido: forma de pagamento predominante e número de parcelas
- [ ] Tratar avaliações duplicadas por pedido (manter uma por pedido, com critério documentado)
- [ ] Calcular latitude e longitude médias por prefixo de CEP e, a partir delas, a distância cliente–vendedor (fórmula de Haversine)
- [ ] Criar os atributos derivados: prazo estimado em dias, prazo limite de envio em dias, frete relativo ao preço, mesmo estado (sim ou não), mês, dia da semana e hora da compra
- [ ] Filtrar as compras a partir de janeiro de 2017 (o volume de 2016 é muito baixo)
- [ ] Escrever o dicionário de atributos: nome, descrição, tabela de origem e por que não vaza informação
- [ ] Mover as funções de agregação para `src/features.py`

**Perguntas de checagem.** Quantas linhas tem a base final? Quantos pedidos ficaram sem distância calculada? Algum atributo depende de algo que acontece depois da compra?

**Cuidado.** `order_approved_at`, `order_delivered_carrier_date` e qualquer informação da avaliação são posteriores à compra e **não podem** ser atributos do modelo preditivo.

---

## Etapa 3 — Análise exploratória

**Objetivo.** Validar as hipóteses H-01 a H-03 e encontrar problemas de qualidade antes de modelar (RA-03).

- [ ] Ausentes por coluna e decisão de tratamento para cada uma
- [ ] Taxa de atraso por mês de compra (há picos? em quais meses?)
- [ ] Taxa de atraso por UF do cliente e por UF do vendedor
- [ ] Taxa de atraso por faixa de distância
- [ ] Taxa de atraso por faixa de prazo estimado
- [ ] Distribuição das notas para pedidos atrasados e no prazo
- [ ] Registrar uma conclusão preliminar para cada hipótese

**Perguntas de checagem.** A taxa de atraso é estável no tempo? O que isso implica para a forma de validar o modelo? A queda da nota com o atraso é grande o suficiente para justificar a análise causal?

---

## Etapa 4 — Modelagem preditiva

**Objetivo.** Construir um baseline e um modelo principal, avaliados de forma temporal (RA-04 e RA-06).

- [ ] Definir as datas de corte de treino, validação e teste em ordem cronológica
- [ ] Excluir do teste as últimas semanas da base e justificar (pedidos recentes só aparecem se já foram entregues, o que reduz artificialmente a taxa de atraso)
- [ ] Verificar a taxa de atraso em cada conjunto
- [ ] Treinar o baseline: regressão logística com imputação, padronização e one-hot
- [ ] Treinar o modelo principal: LightGBM com parada antecipada na validação
- [ ] Comparar os dois modelos por PR-AUC e ROC-AUC na validação e no teste
- [ ] Gerar a importância global dos atributos com SHAP
- [ ] Explicar individualmente três pedidos de alto risco
- [ ] Fechar as hipóteses H-01 a H-03 com base no SHAP

**Perguntas de checagem.** Por que PR-AUC e não acurácia? Quanto o modelo principal ganha sobre o baseline? Os atributos mais importantes fazem sentido para o negócio?

**Cuidado.** Nunca escolha hiperparâmetros ou limiares olhando o conjunto de teste. O teste é usado uma única vez, no final.

---

## Etapa 5 — Regra de decisão por custo

**Objetivo.** Transformar probabilidade em decisão: a partir de qual risco vale agir? (RA-05)

- [ ] Definir as premissas: custo de uma ação preventiva (contato proativo, upgrade de frete) e custo de um atraso não tratado (atendimento, cupom, avaliação negativa)
- [ ] Escrever a função de custo total para um dado limiar
- [ ] Varrer os limiares no conjunto de validação e escolher o de menor custo
- [ ] Aplicar o limiar no teste e comparar com o cenário sem modelo
- [ ] Repetir para pelo menos três combinações de premissas (análise de sensibilidade)

**Perguntas de checagem.** Qual é a economia percentual? Com que premissas de custo o modelo deixa de compensar?

---

## Etapa 6 — Inferência causal

**Objetivo.** Estimar quanto o atraso, isoladamente, reduz a nota (RA-07).

- [ ] Calcular a diferença ingênua de nota média entre atrasados e não atrasados
- [ ] Entender por que essa diferença pode estar enviesada: pedidos que atrasam têm perfil diferente (região, produto, frete)
- [ ] Estudar o Double Machine Learning no modelo parcialmente linear (referência: Chernozhukov et al., 2018)
- [ ] Implementar os três passos: prever a nota a partir dos confundidores, prever o atraso a partir dos confundidores e regredir o resíduo da nota no resíduo do atraso, com *cross-fitting* em K partições
- [ ] Calcular o erro padrão e o intervalo de confiança de 95%
- [ ] Repetir para o desfecho binário (nota menor ou igual a 2)
- [ ] Construir a análise dose-resposta: nota média por faixa de dias de diferença
- [ ] Comparar o efeito causal com a diferença ingênua e interpretar
- [ ] Redigir a seção de limitações (confundidores não observados, como a qualidade do vendedor)

**Perguntas de checagem.** O efeito causal é maior ou menor que a diferença ingênua? Por quê? O efeito cresce com os dias de atraso (H-05)?

**Alternativa.** Se preferir usar uma biblioteca, `DoubleML` e `EconML` implementam o mesmo método. Implementar manualmente primeiro ajuda a entender e a explicar em entrevistas.

---

## Etapa 7 — Análise textual dos comentários

**Objetivo.** Ouvir a voz do cliente para validar as conclusões causais (RA-08).

- [ ] Filtrar as avaliações com comentário preenchido e contar quantas são
- [ ] Pré-processar: minúsculas, remoção de acentos e de stopwords em português
- [ ] Vetorizar com TF-IDF (unigramas e bigramas, frequência mínima)
- [ ] Treinar uma regressão logística para separar comentários de pedidos atrasados e no prazo
- [ ] Listar os termos com os maiores e os menores coeficientes
- [ ] Interpretar: os termos confirmam que a insatisfação está ligada ao prazo?

**Pergunta de checagem.** Os termos associados a atraso falam de prazo e entrega ou de outros problemas, como produto errado ou defeito?

---

## Etapa 8 — Impacto de negócio

**Objetivo.** Responder quanto vale reduzir atrasos (RA-09).

- [ ] Usar o efeito causal e a taxa de atraso observada
- [ ] Calcular, para reduções de 10%, 25% e 50% na taxa de atraso, o ganho esperado na nota média e as avaliações negativas evitadas a cada 10 mil pedidos
- [ ] Montar a tabela de cenários e declarar as premissas

**Pergunta de checagem.** Um gestor não técnico entenderia o resultado em uma frase?

---

## Etapa 9 — Publicação no portfólio

**Objetivo.** Transformar o trabalho em algo que um recrutador entenda em dois minutos (RNF-04).

- [ ] Salvar o modelo, a lista de atributos e o limiar em `models/`
- [ ] Gerar o `requirements.txt`
- [ ] Reescrever este README substituindo o roteiro pelas seções abaixo
- [ ] Publicar no GitHub com `data/` no `.gitignore`
- [ ] Opcional: criar uma aplicação em Streamlit que recebe os dados de um pedido e retorna o risco de atraso

### Modelo do README final

| Seção | Conteúdo |
|---|---|
| Problema | Duas ou três frases sobre o problema de negócio |
| Dados | Fonte, período e volume |
| Abordagem | Diagrama ou lista curta das etapas |
| Resultados | PR-AUC contra o baseline, economia estimada, efeito causal com IC e tabela de impacto |
| Principais fatores de atraso | Gráfico SHAP e interpretação |
| Limitações | Premissas causais, premissas de custo e período dos dados |
| Como reproduzir | Instalar o `requirements.txt` e executar os notebooks `.py` em ordem (o dataset é público; o `kagglehub` baixa sem credencial) |
| Próximos passos | Melhorias planejadas |

---

## Próximos passos (após a primeira versão)

| Melhoria | Por que importa |
|---|---|
| Taxa histórica de atraso por vendedor, calculada só com entregas concluídas antes de cada compra | Costuma ser o atributo mais forte e exige controle rigoroso de vazamento |
| Calibração das probabilidades | Permite comunicar o risco em percentual confiável |
| Análise de sensibilidade causal | Mostra a robustez do efeito a confundidores não observados |
| API em FastAPI | Demonstra capacidade de levar o modelo para produção |

## Dados

Brazilian E-Commerce Public Dataset by Olist, disponível em [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).