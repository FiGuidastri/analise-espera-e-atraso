# PRD — Previsão de Atraso de Entrega e Impacto na Satisfação do Cliente

| Campo | Valor |
|---|---|
| Projeto | Atraso de entrega e satisfação — e-commerce Olist |
| Tipo | Projeto de portfólio em Ciência de Dados e Machine Learning |
| Autor | Filipe Guidastri |
| Status | Em definição |
| Versão | 0.1 |

## 1. Contexto

O comércio eletrônico brasileiro opera com **prazos de entrega longos e heterogêneos**, influenciados pela distância entre vendedor e cliente, pela malha logística regional e pelo perfil do produto. Em marketplaces, o prazo estimado informado no momento da compra funciona como uma promessa ao cliente. Quando essa promessa é quebrada, o efeito aparece na avaliação do pedido, na reputação do vendedor e no custo de atendimento.

O dataset público da Olist reúne cerca de 100 mil pedidos reais anonimizados, realizados entre 2016 e 2018, com informações de pedidos, itens, pagamentos, produtos, clientes, vendedores, geolocalização e avaliações. Ele permite estudar o problema de ponta a ponta: da previsão do atraso até a quantificação do seu impacto.

## 2. Problema

> **Problema de negócio:** a operação só descobre que um pedido atrasou depois que o atraso já ocorreu, quando o cliente já está insatisfeito e a avaliação negativa já foi registrada.

Duas perguntas orientam o projeto. A primeira é **preditiva**: é possível identificar, no momento da compra, quais pedidos têm maior probabilidade de atrasar? A segunda é **causal**: quanto o atraso, isoladamente, reduz a nota de avaliação, descontado o perfil dos pedidos que tendem a atrasar?

## 3. Objetivos

| ID | Objetivo | Tipo |
|---|---|---|
| OBJ-01 | Estimar a probabilidade de atraso de cada pedido no momento da compra | Preditivo |
| OBJ-02 | Definir uma regra de decisão sobre quando agir preventivamente, baseada em custo | Decisão |
| OBJ-03 | Estimar o efeito causal do atraso sobre a nota de avaliação | Causal |
| OBJ-04 | Identificar, nos comentários dos clientes, os temas associados a atraso | Exploratório (texto) |
| OBJ-05 | Traduzir os resultados em impacto de negócio mensurável | Negócio |

## 4. Stakeholders e uso pretendido

| Stakeholder | Necessidade | Como usa o resultado |
|---|---|---|
| Operações logísticas | Antecipar pedidos problemáticos | Prioriza pedidos de alto risco para ação preventiva |
| Atendimento ao cliente | Reduzir contatos reativos | Faz contato proativo antes do atraso |
| Gestão de sellers | Entender o que causa atraso | Usa os fatores explicativos para orientar vendedores |
| Diretoria | Justificar investimento em logística | Usa o impacto estimado na satisfação |

## 5. Hipóteses

| ID | Hipótese | Como será verificada |
|---|---|---|
| H-01 | A distância entre vendedor e cliente aumenta a probabilidade de atraso | Análise exploratória e importância de atributos |
| H-02 | Pedidos para as regiões Norte e Nordeste atrasam mais | Taxa de atraso por UF |
| H-03 | Prazos estimados mais curtos estão associados a mais atraso | Análise exploratória e SHAP |
| H-04 | O atraso reduz a nota mesmo após controlar o perfil do pedido | Estimativa causal |
| H-05 | O efeito do atraso sobre a nota cresce com a duração do atraso | Análise dose-resposta |

## 6. Escopo

**Dentro do escopo:** pedidos com status entregue e data de entrega registrada, compras a partir de janeiro de 2017, atributos disponíveis no momento da compra, modelagem preditiva supervisionada, estimativa causal com dados observacionais e análise textual dos comentários das avaliações.

**Fora do escopo:** pedidos cancelados ou não entregues, previsão do número exato de dias de atraso, otimização de rotas logísticas, sistemas de recomendação e coleta de dados externos pagos.

## 7. Dados

| Tabela | Conteúdo principal | Uso no projeto |
|---|---|---|
| orders | Datas de compra, entrega e estimativa; status | Alvo e atributos temporais |
| order_items | Itens, preço, frete, vendedor, data limite de envio | Atributos do pedido |
| order_payments | Forma de pagamento e parcelas | Atributos do pedido |
| order_reviews | Nota e comentário | Desfecho causal e texto |
| products | Categoria, peso, dimensões, fotos | Atributos do produto |
| customers | CEP e UF do cliente | Geografia |
| sellers | CEP e UF do vendedor | Geografia |
| geolocation | Latitude e longitude por prefixo de CEP | Distância |
| product_category_name_translation | Tradução das categorias | Legibilidade |

## 8. Requisitos das análises

### RA-01 — Definição do alvo

| Item | Requisito |
|---|---|
| Definição | Atraso = data de entrega ao cliente posterior à data estimada de entrega, comparando apenas a data, sem o horário |
| Variável derivada | Dias de diferença entre a entrega real e a estimada (positivo = atraso) |
| Filtro | Apenas pedidos entregues e com data de entrega preenchida |
| Critério de aceite | Taxa de atraso global calculada e documentada |

### RA-02 — Base analítica

| Item | Requisito |
|---|---|
| Granularidade | Uma linha por pedido |
| Restrição temporal | Somente informação conhecida no momento da compra (proibido usar data de aprovação, data de postagem e data de entrega como atributo) |
| Atributos mínimos | Quantidade de itens e vendedores, preço e frete totais, frete relativo ao preço, peso e volume, categoria principal, forma de pagamento e parcelas, UF do cliente e do vendedor, distância cliente–vendedor, prazo estimado em dias, prazo limite de envio, mês, dia da semana e hora da compra |
| Regra para pedidos com vários itens | Documentar a regra de agregação e o critério para definir vendedor e categoria principais |
| Critério de aceite | Dicionário de atributos com nome, descrição, origem e justificativa de que não há vazamento |

### RA-03 — Análise exploratória

| Item | Requisito |
|---|---|
| Perguntas obrigatórias | Evolução mensal da taxa de atraso; taxa de atraso por UF do cliente; distribuição da nota para pedidos no prazo e atrasados; relação entre distância e atraso; relação entre prazo estimado e atraso |
| Qualidade de dados | Ausentes por coluna, duplicidades de avaliação por pedido, CEPs sem geolocalização, valores extremos |
| Critério de aceite | Cada hipótese H-01 a H-03 recebe uma conclusão preliminar, com gráfico |

### RA-04 — Modelo preditivo

| Item | Requisito |
|---|---|
| Validação | Divisão temporal em treino, validação e teste, sem embaralhamento |
| Viés de fim de base | Excluir do teste as últimas semanas da base, em que só aparecem pedidos já entregues |
| Baseline | Modelo simples e interpretável (por exemplo, regressão logística) |
| Modelo principal | Gradient boosting (por exemplo, LightGBM) |
| Métrica principal | PR-AUC |
| Métricas de apoio | ROC-AUC e precisão e recall no limiar escolhido |
| Critério de aceite | Modelo principal supera o baseline em PR-AUC no conjunto de teste |

### RA-05 — Regra de decisão por custo

| Item | Requisito |
|---|---|
| Premissas | Custo de uma ação preventiva e custo de um atraso não tratado, em reais, declarados explicitamente |
| Escolha do limiar | Feita no conjunto de validação, minimizando o custo total |
| Comparação | Custo com o modelo contra o cenário sem modelo, no conjunto de teste |
| Sensibilidade | Recalcular o resultado para pelo menos três combinações de premissas de custo |
| Critério de aceite | Economia percentual reportada com as premissas usadas |

### RA-06 — Interpretabilidade

| Item | Requisito |
|---|---|
| Global | Ranking de importância dos atributos com SHAP |
| Local | Explicação de pelo menos três pedidos individuais de alto risco |
| Critério de aceite | Conclusão final sobre as hipóteses H-01 a H-03 |

### RA-07 — Efeito causal do atraso na avaliação

| Item | Requisito |
|---|---|
| Tratamento | Atraso (binário) |
| Desfechos | Nota média de 1 a 5 e probabilidade de nota menor ou igual a 2 |
| Confundidores | Somente atributos definidos antes da compra (RA-02) |
| Referência | Diferença ingênua entre médias de atrasados e não atrasados |
| Método | Double Machine Learning (modelo parcialmente linear com *cross-fitting*), ou alternativa justificada |
| Dose-resposta | Nota média por faixa de dias de diferença em relação à data estimada |
| Limitações | Premissa de ausência de confundidores não observados declarada explicitamente |
| Critério de aceite | Efeito com intervalo de confiança de 95%, comparado à diferença ingênua; H-04 e H-05 respondidas |

### RA-08 — Análise textual dos comentários

| Item | Requisito |
|---|---|
| Base | Avaliações com comentário preenchido |
| Pré-processamento | Minúsculas, remoção de acentos e de stopwords em português |
| Análise | Termos e bigramas mais associados a pedidos atrasados e a pedidos no prazo |
| Critério de aceite | Lista de termos interpretada e conectada às conclusões do RA-07 |

### RA-09 — Impacto de negócio

| Item | Requisito |
|---|---|
| Cenários | Redução de 10%, 25% e 50% na taxa de atraso |
| Saídas | Ganho esperado na nota média e avaliações negativas evitadas a cada 10 mil pedidos |
| Critério de aceite | Tabela de cenários com premissas declaradas |

## 9. Requisitos não funcionais

| ID | Requisito |
|---|---|
| RNF-01 | Reprodutibilidade: sementes aleatórias fixas e dependências listadas |
| RNF-02 | Ausência de vazamento: nenhum atributo posterior à compra |
| RNF-03 | Organização: notebooks numerados por etapa e funções reutilizáveis em módulo separado |
| RNF-04 | Documentação: README com problema, decisões, resultados e limitações |
| RNF-05 | Execução: dados obtidos via API do Kaggle e notebooks em scripts `.py` com células `#%%`, executáveis em ordem sem passos manuais além da credencial do Kaggle |

## 10. Métricas de sucesso do projeto

| Dimensão | Métrica | Meta |
|---|---|---|
| Preditiva | PR-AUC no teste | Superior ao baseline |
| Decisão | Economia de custo no teste | Positiva sob as premissas centrais |
| Causal | Intervalo de confiança do efeito | Não cruza zero |
| Comunicação | README | Explicado em até 2 minutos de leitura |

## 11. Riscos e mitigação

| Risco | Impacto | Mitigação |
|---|---|---|
| Vazamento temporal nos atributos | Métricas infladas | Checklist de atributos (RA-02) e divisão temporal |
| Mudança de padrão ao longo do tempo | Desempenho pior no teste | Validação temporal e análise mensal |
| Confundimento não observado | Efeito causal enviesado | Declarar a limitação e fazer análise de sensibilidade |
| Premissas de custo arbitrárias | Conclusão de negócio frágil | Análise de sensibilidade das premissas (RA-05) |
| Geolocalização ausente para alguns CEPs | Perda de informação | Quantificar e tratar os valores ausentes |

## 12. Entregáveis

| Entregável | Descrição |
|---|---|
| Notebooks por etapa | Exploração, atributos, modelagem, causal, texto e impacto |
| Modelo serializado | Modelo treinado, lista de atributos e limiar |
| Relatório de resultados | Seção de resultados no README |
| Aplicação (opcional) | Interface em Streamlit para estimar o risco de um pedido |