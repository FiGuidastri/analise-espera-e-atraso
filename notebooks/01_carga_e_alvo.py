#%%
# Setup - RNF-01 (reprodutibilidade)
from pathlib import Path
import random

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#Funciona tanto rodando na raiz do projeto quanto dentro de notebooks/
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

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
