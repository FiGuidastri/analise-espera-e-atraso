"""Funções de carga, alvo e atributos reutilizadas entre os notebooks (RNF-03).

Regra geral (RNF-02): todo atributo devolvido por `construir_base` existe no
instante do clique de compra. Colunas posteriores à compra (aprovação, postagem,
entrega, avaliação) só aparecem como alvo ou desfecho, nunca como atributo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_INICIO = pd.Timestamp("2017-01-01")

COLUNAS_DATA_ORDERS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

REGIAO_POR_UF = {
    "AC": "Norte", "AM": "Norte", "AP": "Norte", "PA": "Norte", "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste", "PB": "Nordeste",
    "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MS": "Centro-Oeste", "MT": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# Atributos do modelo preditivo e confundidores do modelo causal (RA-02, RA-07)
ATRIBUTOS_NUMERICOS = [
    "n_itens", "n_produtos", "n_vendedores",
    "preco_total", "frete_total", "frete_relativo",
    "peso_total_g", "volume_total_cm3", "n_fotos_principal",
    "parcelas", "distancia_km", "mesmo_estado",
    "prazo_estimado_dias", "prazo_envio_dias",
    "mes", "dia_semana", "hora",
]
ATRIBUTOS_CATEGORICOS = [
    "categoria_principal", "forma_pagamento", "uf_cliente", "uf_vendedor",
]
ATRIBUTOS = ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS


def preparar_x(base: pd.DataFrame) -> pd.DataFrame:
    """Matriz de atributos com categóricas como `category` (ausente = 'desconhecida').

    As categorias são as presentes em `base`; prepare a base inteira de uma vez
    para que treino, validação e teste compartilhem os mesmos códigos.
    """
    x = base[ATRIBUTOS].copy()
    for col in ATRIBUTOS_CATEGORICOS:
        x[col] = x[col].fillna("desconhecida").astype("category")
    return x


def carregar_tabelas(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Lê os nove CSVs com nomes curtos (orders, order_items, ...) e datas convertidas."""
    tabelas = {
        p.stem.removeprefix("olist_").removesuffix("_dataset"): pd.read_csv(p)
        for p in sorted(Path(data_dir).glob("*.csv"))
    }
    tabelas["orders"] = pd.read_csv(
        Path(data_dir) / "olist_orders_dataset.csv", parse_dates=COLUNAS_DATA_ORDERS
    )
    tabelas["order_items"]["shipping_limit_date"] = pd.to_datetime(
        tabelas["order_items"]["shipping_limit_date"]
    )
    for col in ["review_creation_date", "review_answer_timestamp"]:
        tabelas["order_reviews"][col] = pd.to_datetime(tabelas["order_reviews"][col])
    return tabelas


def filtrar_pedidos(orders: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Pedidos entregues, com data de entrega e comprados a partir de DATA_INICIO (RA-01).

    Devolve também o funil de descarte de cada filtro.
    """
    funil = {"total": len(orders)}
    base = orders[orders["order_status"] == "delivered"]
    funil["delivered"] = len(base)
    base = base[base["order_delivered_customer_date"].notna()]
    funil["com_data_entrega"] = len(base)
    base = base[base["order_purchase_timestamp"] >= DATA_INICIO].copy()
    funil["a_partir_2017"] = len(base)
    assert len(base) > 0, "base vazia após os filtros"
    return base, funil


def construir_alvo(pedidos: pd.DataFrame) -> pd.DataFrame:
    """Cria `dias_atraso` (positivo = atraso) e `atrasou` comparando só a data (RA-01).

    A data estimada vem à meia-noite: comparar timestamps marcaria como atrasado
    um pedido entregue no próprio dia estimado.
    """
    entrega = pedidos["order_delivered_customer_date"].dt.normalize()
    estimada = pedidos["order_estimated_delivery_date"].dt.normalize()
    pedidos = pedidos.copy()
    pedidos["dias_atraso"] = (entrega - estimada).dt.days
    pedidos["atrasou"] = (pedidos["dias_atraso"] > 0).astype(int)
    return pedidos


def agregar_itens(items: pd.DataFrame, products: pd.DataFrame, traducao: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por pedido com totais dos itens e vendedor/categoria principais.

    Regra do item principal: o item de maior preço do pedido (empate: o de menor
    `order_item_id`). Dele vêm o vendedor, a categoria e o número de fotos.
    """
    prod = products.merge(traducao, on="product_category_name", how="left")
    prod["categoria"] = prod["product_category_name_english"].fillna(prod["product_category_name"])
    prod["volume_cm3"] = prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"]
    itens = items.merge(
        prod[["product_id", "categoria", "product_weight_g", "volume_cm3", "product_photos_qty"]],
        on="product_id", how="left",
    )

    totais = itens.groupby("order_id").agg(
        n_itens=("order_item_id", "size"),
        n_produtos=("product_id", "nunique"),
        n_vendedores=("seller_id", "nunique"),
        preco_total=("price", "sum"),
        frete_total=("freight_value", "sum"),
        peso_total_g=("product_weight_g", lambda s: s.sum(min_count=1)),
        volume_total_cm3=("volume_cm3", lambda s: s.sum(min_count=1)),
        limite_envio=("shipping_limit_date", "min"),
    )

    principal = (
        itens.sort_values(["order_id", "price", "order_item_id"], ascending=[True, False, True])
        .drop_duplicates("order_id")
        .set_index("order_id")[["seller_id", "categoria", "product_photos_qty"]]
        .rename(columns={
            "seller_id": "vendedor_principal",
            "categoria": "categoria_principal",
            "product_photos_qty": "n_fotos_principal",
        })
    )
    return totais.join(principal).reset_index()


def agregar_pagamentos(payments: pd.DataFrame) -> pd.DataFrame:
    """Forma de pagamento predominante (maior valor pago) e número máximo de parcelas."""
    predominante = (
        payments.sort_values(["order_id", "payment_value"], ascending=[True, False])
        .drop_duplicates("order_id")
        .set_index("order_id")["payment_type"]
        .rename("forma_pagamento")
    )
    parcelas = payments.groupby("order_id")["payment_installments"].max().rename("parcelas")
    return pd.concat([predominante, parcelas], axis=1).reset_index()


def deduplicar_avaliacoes(reviews: pd.DataFrame) -> pd.DataFrame:
    """Uma avaliação por pedido: a respondida por último (`review_answer_timestamp`).

    É a versão final da opinião do cliente. Usada só como desfecho (RA-07/RA-08),
    nunca como atributo.
    """
    return (
        reviews.sort_values(["order_id", "review_answer_timestamp"])
        .drop_duplicates("order_id", keep="last")
        [["order_id", "review_score", "review_comment_title", "review_comment_message"]]
    )


def coordenadas_por_cep(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Latitude e longitude médias por prefixo de CEP, descartando pontos fora do Brasil."""
    geo = geolocation[
        geolocation["geolocation_lat"].between(-34, 6)
        & geolocation["geolocation_lng"].between(-74, -34)
    ]
    return (
        geo.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Distância em km sobre a esfera terrestre."""
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def construir_base(tabelas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Base analítica: uma linha por pedido, com alvo, atributos pré-compra e desfecho (RA-02)."""
    pedidos, _ = filtrar_pedidos(tabelas["orders"])
    pedidos = construir_alvo(pedidos)

    base = (
        pedidos
        .merge(agregar_itens(tabelas["order_items"], tabelas["products"],
                             tabelas["product_category_name_translation"]), on="order_id", how="inner")
        .merge(agregar_pagamentos(tabelas["order_payments"]), on="order_id", how="left")
        .merge(tabelas["customers"][["customer_id", "customer_zip_code_prefix", "customer_state"]],
               on="customer_id", how="left")
        .merge(tabelas["sellers"][["seller_id", "seller_zip_code_prefix", "seller_state"]],
               left_on="vendedor_principal", right_on="seller_id", how="left")
        .merge(deduplicar_avaliacoes(tabelas["order_reviews"]), on="order_id", how="left")
    )

    coords = coordenadas_por_cep(tabelas["geolocation"])
    cli = coords.reindex(base["customer_zip_code_prefix"]).to_numpy()
    vend = coords.reindex(base["seller_zip_code_prefix"]).to_numpy()
    base["distancia_km"] = haversine_km(cli[:, 0], cli[:, 1], vend[:, 0], vend[:, 1])

    compra = base["order_purchase_timestamp"]
    base["uf_cliente"] = base["customer_state"]
    base["uf_vendedor"] = base["seller_state"]
    base["regiao_cliente"] = base["uf_cliente"].map(REGIAO_POR_UF)
    base["mesmo_estado"] = (base["uf_cliente"] == base["uf_vendedor"]).astype(int)
    base["frete_relativo"] = base["frete_total"] / base["preco_total"]
    base["prazo_estimado_dias"] = (base["order_estimated_delivery_date"].dt.normalize() - compra.dt.normalize()).dt.days
    base["prazo_envio_dias"] = (base["limite_envio"] - compra).dt.total_seconds() / 86400
    base["mes"] = compra.dt.month
    base["dia_semana"] = compra.dt.dayofweek
    base["hora"] = compra.dt.hour

    colunas = (
        ["order_id", "order_purchase_timestamp", "regiao_cliente"]
        + ATRIBUTOS
        + ["dias_atraso", "atrasou", "review_score", "review_comment_title", "review_comment_message"]
    )
    return base[colunas].sort_values("order_purchase_timestamp").reset_index(drop=True)
