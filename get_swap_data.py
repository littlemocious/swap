import pandas as pd
import numpy as np
from configparser import RawConfigParser
from ds_utils.db.snowflake import SnowflakeProvider
#import pickle


def fetch_data(secrets):

    snowflake = SnowflakeProvider(
        **{ x: secrets.get("snowflake", x) for x in ["user", "password", "account", "database", "warehouse"]})

    swap_data = snowflake.sql2df("""
        with order_tb as (
        select distinct 
            order_seq.order_id, 
            order_seq.group_id, 
            order_seq.booking_id, 
            order_seq.style
        from rtrbi.v_ot_orderdata_seq order_seq
        join analytics.products prod on order_seq.style = prod.style
        where order_seq.style is not null and order_seq.booking_id > 0 and 
            order_seq.rental_begin_date > '2015-01-01' and prod.type = 'D' and prod.detailed_formality is not null
        ),
        replace_tb as (
        select distinct
            --annot.annotation_target as replace_group, 
            --annot.annotation_target_id as replace_id, 
            ord.order_id as order_id,
            --ord.group_id as group_id,
            ord.style as replace_style
        from etl.annotation annot
        join order_tb ord on ord.booking_id = annot.annotation_target_id
        join etl.annotation_detail detail on detail.annotation_id = annot.id
        where annot.annotation_group = 'REPLACEMENT' and annot.annotation_target = 'BOOKING' and order_id is not null 
            and (detail.detail_value = 'PO_CUSTOMER_SELECTION' or detail.detail_value = 'PO_CUSTOMER_SELF_SERVICE')
        ),
        origin_tb as (
        select distinct
            --annot.annotation_target as origin_group,
            --annot.annotation_target_id as origin_id,
            ord.order_id as order_id,
            --ord.group_id as group_id,
            ord.style as origin_style
        from etl.annotation annot
        join order_tb ord on ord.booking_id = annot.annotation_target_id
        where annot.annotation_group = 'PO' and annot.annotation_target = 'BOOKING' and order_id is not null 
        )
        select origin_tb.origin_style, 
        replace_tb.replace_style
        from replace_tb
        join origin_tb on replace_tb.order_id = origin_tb.order_id
        where replace_tb.replace_style <> origin_tb.origin_style
    """)

    formality = snowflake.sql2df("""
        select distinct  
            prod.style,
            prod.detailed_formality as formality
        from analytics.products prod
        where prod.style is not null and prod.type = 'D' and prod.detailed_formality is not null
            """)
    formality['formality'] = formality['formality'].apply(int)
    
    shape = snowflake.sql2df("""
    select distinct  
        prod.style,
        prod.high_level_shape as shape
    from analytics.products prod
    where prod.style is not null and prod.type = 'D' and prod.high_level_shape is not null
        """)

    return(formality, shape, swap_data)
