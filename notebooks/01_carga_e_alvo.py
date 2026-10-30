#%%
# Setup - RNF-01 (reprodutibilidade)
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#Funciona tanto rodando na raiz do projeto quanto dentro de notebooks/
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

from src.features import carregar_tabelas, construir_alvo, filtrar_pedidos

#%%
# Download dos dados via kagglehub, direto em data/ (sem passar pelo cache) - RNF-05
import kagglehub

DATASET = "olistbr/brazilian-ecommerce/versions/2" # versao fixa

if not list(DATA_DIR.glob("*.csv")):
    kagglehub.dataset_download(DATASET, output_dir=str(DATA_DIR))

print(sorted(p.name for p in DATA_DIR.glob("*.csv")))

#%%
# Checagem dos arquivos
ARQUIVOS_ESPERADOS = {
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
}
faltando = ARQUIVOS_ESPERADOS - {p.name for p in DATA_DIR.glob("*.csv")}
assert not faltando, f"CSVs faltando em {DATA_DIR}: {faltando}"

#%%
# Etapa 1 - Carga das tabelas (RA-01)
tabelas = carregar_tabelas(DATA_DIR)

for nome, df in tabelas.items():
    print(f"{nome:<35} {df.shape}")

#%%
# Chaves entre tabelas
# orders.order_id -> order_items / order_payments / order_reviews (1:N)
# orders.customer_id -> customers (1:1; a pessoa real é customer_unique_id)
# order_items.seller_id -> sellers ; order_items.product_id -> products
# customers/sellers.*_zip_code_prefix -> geolocation.geolocation_zip_code_prefix (1:N)
orders = tabelas["orders"]
items = tabelas["order_items"]
reviews = tabelas["order_reviews"]

print("order_id único em orders:", orders["order_id"].is_unique)
print("customer_id único em orders:", orders["customer_id"].is_unique)
print("pedidos com mais de 1 item:", (items.groupby("order_id").size() > 1).sum())
print("pedidos com mais de 1 avaliação:", (reviews.groupby("order_id").size() > 1).sum())
print("prefixos de CEP na geolocalização:", tabelas["geolocation"]["geolocation_zip_code_prefix"].nunique())

#%%
# Distribuição de order_status
print(orders["order_status"].value_counts())
print("\nperíodo das compras:",
      orders["order_purchase_timestamp"].min(), "a",
      orders["order_purchase_timestamp"].max())

#%%
# Filtro da base - RA-01: entregues, com data de entrega, compras a partir de 2017
base, funil = filtrar_pedidos(orders)

print(f"total:                     {funil['total']}")
print(f"- não delivered:           {funil['total'] - funil['delivered']}")
print(f"- delivered sem entrega:   {funil['delivered'] - funil['com_data_entrega']}")
print(f"- compras antes de 2017:   {funil['com_data_entrega'] - funil['a_partir_2017']}")
print(f"= base final:              {funil['a_partir_2017']}")

#%%
# Alvo - RA-01: compara só a data (a estimada vem à meia-noite)
base = construir_alvo(base)

print(f"taxa de atraso global: {base['atrasou'].mean():.2%}")
print(base["dias_atraso"].describe())

falsos = ((base["order_delivered_customer_date"] > base["order_estimated_delivery_date"])
          & (base["atrasou"] == 0)).sum()
print("falsos atrasos se comparássemos timestamps:", falsos)
