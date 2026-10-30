#%%
# Setup - Etapa 2: engenharia de atributos (RA-02)
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.features import ATRIBUTOS, carregar_tabelas, construir_base

#%%
# Base analítica: uma linha por pedido
# Regras (documentadas em src/features.py):
# - item principal = item de maior preço do pedido -> vendedor, categoria e fotos
# - pagamento: forma de maior valor pago e número máximo de parcelas
# - avaliação: uma por pedido, a respondida por último (só desfecho, nunca atributo)
# - distância: Haversine entre as coordenadas médias do prefixo de CEP do cliente
#   e do vendedor principal
tabelas = carregar_tabelas(DATA_DIR)
base = construir_base(tabelas)

print("linhas:", len(base), "| order_id único:", base["order_id"].is_unique)
print("pedidos sem distância calculada:", base["distancia_km"].isna().sum())
print("pedidos sem avaliação:", base["review_score"].isna().sum())

#%%
# Checagem de vazamento - RNF-02
PROIBIDAS = {
    "order_approved_at", "order_delivered_carrier_date", "order_delivered_customer_date",
    "review_score", "review_comment_title", "review_comment_message", "dias_atraso", "atrasou",
}
assert not PROIBIDAS & set(ATRIBUTOS), "atributo posterior à compra na lista de atributos"

#%%
# Dicionário de atributos - critério de aceite do RA-02
DICIONARIO = pd.DataFrame([
    ("n_itens", "Quantidade de itens do pedido", "order_items", "Itens do carrinho no clique"),
    ("n_produtos", "Quantidade de produtos distintos", "order_items", "Itens do carrinho no clique"),
    ("n_vendedores", "Quantidade de vendedores distintos", "order_items", "Itens do carrinho no clique"),
    ("preco_total", "Soma dos preços dos itens (R$)", "order_items", "Preço exibido no checkout"),
    ("frete_total", "Soma do frete dos itens (R$)", "order_items", "Frete cotado no checkout"),
    ("frete_relativo", "frete_total / preco_total", "order_items", "Derivado de valores do checkout"),
    ("peso_total_g", "Soma do peso cadastrado dos produtos", "products", "Cadastro do produto"),
    ("volume_total_cm3", "Soma do volume (C x A x L) dos produtos", "products", "Cadastro do produto"),
    ("n_fotos_principal", "Fotos do anúncio do item principal", "products", "Cadastro do produto"),
    ("categoria_principal", "Categoria (inglês) do item de maior preço", "products + translation", "Cadastro do produto"),
    ("forma_pagamento", "Forma de pagamento de maior valor", "order_payments", "Escolhida no checkout"),
    ("parcelas", "Número máximo de parcelas", "order_payments", "Escolhido no checkout"),
    ("uf_cliente", "UF do cliente", "customers", "Endereço de entrega informado na compra"),
    ("uf_vendedor", "UF do vendedor principal", "sellers", "Cadastro do vendedor"),
    ("mesmo_estado", "1 se cliente e vendedor na mesma UF", "customers + sellers", "Derivado de cadastros"),
    ("distancia_km", "Distância Haversine cliente–vendedor", "geolocation", "Derivado dos CEPs cadastrados"),
    ("prazo_estimado_dias", "Dias entre a compra e a data estimada de entrega", "orders", "Prazo prometido no checkout"),
    ("prazo_envio_dias", "Dias entre a compra e o limite de postagem do vendedor", "order_items", "Prazo de manuseio definido na venda (*)"),
    ("mes", "Mês da compra", "orders", "Momento da compra"),
    ("dia_semana", "Dia da semana da compra (0 = segunda)", "orders", "Momento da compra"),
    ("hora", "Hora da compra", "orders", "Momento da compra"),
], columns=["atributo", "descricao", "origem", "por_que_nao_vaza"])

assert set(DICIONARIO["atributo"]) == set(ATRIBUTOS)
DICIONARIO.to_csv(REPORTS_DIR / "dicionario_atributos.csv", index=False)
print(DICIONARIO.to_string(index=False))
# (*) shipping_limit_date é calculado pela Olist a partir do prazo de manuseio do
# vendedor, conhecido na venda; é a única ressalva da lista.

#%%
# Salva a base para os próximos notebooks
base.to_parquet(DATA_DIR / "base_analitica.parquet", index=False)
print("salvo:", DATA_DIR / "base_analitica.parquet")

# Versão enxuta para o app Streamlit (vai para o Git): sem identificadores nem texto
# dos comentários, que o app não usa.
base.drop(columns=["order_id", "review_comment_title", "review_comment_message"]).to_parquet(
    REPORTS_DIR / "base_app.parquet", index=False)
print("salvo:", REPORTS_DIR / "base_app.parquet")
