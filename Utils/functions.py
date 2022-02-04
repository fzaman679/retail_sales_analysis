import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import calendar

from sympy import false


def get_clean_orders_df(df):
    df[['order_date', 'delivery_date' ]] = df[['order_date', 'delivery_date' ]].apply(pd.to_datetime)
    df['Delivery_time'] = df['delivery_date'] - df['order_date']
    df['Year'] = df['order_date'].dt.year
    df['Month'] = df['order_date'].dt.month
    df['Day'] = df['order_date'].dt.day
    return df

def get_clean_customers(df):
    df.drop('country', axis=1, inplace=True)
    return df

def get_top_10_products_by_rev(df, df_products):
    revenue_per_product = df.groupby('product_id')['total_price'].sum().to_frame().reset_index()
    revenue_per_product = revenue_per_product.sort_values(by='total_price', ascending=False).head(20)
    revenue_per_product

    filter_products = df_products["product_ID"].isin(revenue_per_product.loc[:, 'product_id']) # selecting all rows from revenue product df under product_id column
    filtered_products_top_rev = df_products[filter_products]
    
    filtered_products_top_rev_copy = filtered_products_top_rev.copy()
    filtered_products_top_rev_copy = filtered_products_top_rev_copy.sort_values(by='product_ID', ascending=False)

    revenue_per_product_copy = revenue_per_product.copy()
    revenue_per_product_copy = revenue_per_product_copy.rename(columns={'product_id':'product_ID'})
    revenue_per_product_copy = revenue_per_product_copy.sort_values(by='product_ID', ascending=False)

    filtered_products_top_rev_FINAL = pd.merge(left=revenue_per_product_copy, right=filtered_products_top_rev_copy, how='inner', on='product_ID' )
    
    filtered_products_top_rev_FINAL['product_name_and_size'] = filtered_products_top_rev_FINAL['product_name'] + ', ' + filtered_products_top_rev_FINAL['size']

    filtered_products_top_rev_FINAL_grouped = filtered_products_top_rev_FINAL.groupby('product_name_and_size')['total_price'].sum().to_frame()
    filtered_products_top_rev_FINAL_grouped = filtered_products_top_rev_FINAL_grouped.reset_index()
    return filtered_products_top_rev_FINAL_grouped


def graph_top_10_products_by_rev(df):

    product = df['product_name_and_size']
    tot_price = df['total_price']

    graph_top_10_products_by_rev1 = plt.figure(figsize=(10,10))
    graph_top_10_products_by_rev1 = plt.barh(product, tot_price)
    graph_top_10_products_by_rev1 = plt.ylabel('Product Name', fontsize=16)
    graph_top_10_products_by_rev1 = plt.xlabel('Revenue', fontsize=16)
    graph_top_10_products_by_rev1 = plt.xticks(fontsize=14)
    graph_top_10_products_by_rev1 = plt.yticks(fontsize=14)
    graph_top_10_products_by_rev1 = plt.title('Total Revenue Per Product', fontsize=18)

    return graph_top_10_products_by_rev1

def get_20_best_customers(df_sales, df_orders, df_customers):

    top_price_by_order = df_sales.groupby('order_id')['total_price'].sum()
    top_price_by_order = top_price_by_order.to_frame()
    top_price_by_order.reset_index(inplace=True)
    
    top_price_by_order = top_price_by_order.sort_values(by='total_price', ascending=False)
    
    filter_orders = df_orders["order_id"].isin(top_price_by_order.loc[:, 'order_id'])
    filtered_order_top_rev = df_orders[filter_orders]
    filtered_order_top_rev = filtered_order_top_rev[['order_id', 'customer_id']]

    filtered_order_top_rev_updated = pd.merge(left=top_price_by_order, right=filtered_order_top_rev, how='inner', on='order_id' )
    filtered_order_top_rev_updated

    filtered_order_top_rev_updated = filtered_order_top_rev_updated.groupby('customer_id')['total_price'].sum().to_frame()
    filtered_order_top_rev_updated = filtered_order_top_rev_updated.reset_index()

    filtered_customers = df_customers['customer_id'].isin(filtered_order_top_rev_updated.loc[:, 'customer_id'])
    
    filtered_customers_top_rev = df_customers[filtered_customers]
    filtered_customers_top_rev = filtered_customers_top_rev[['customer_id','customer_name', 'age', 'home_address', 'zip_code', 'city', 'state']]

    filtered_order_top_rev_updated_copy = filtered_order_top_rev_updated.copy()
    filtered_order_top_rev_updated_copy = filtered_order_top_rev_updated_copy.sort_values(by='customer_id', ascending=False)
    filtered_customers_top_rev_copy = filtered_customers_top_rev.copy()
    filtered_customers_top_rev_copy = filtered_customers_top_rev_copy.sort_values(by='customer_id', ascending=False)
    filtered_order_top_rev_updated_FINAL = pd.merge(left=filtered_order_top_rev_updated_copy, right= filtered_customers_top_rev_copy, how='inner', on='customer_id')
    filtered_order_top_rev_updated_FINAL = filtered_order_top_rev_updated_FINAL.sort_values(by='total_price', ascending=False)
    
    TOP_20_BEST_customers_updated_FINAL = filtered_order_top_rev_updated_FINAL.head(20)
    return TOP_20_BEST_customers_updated_FINAL


def graph_top_20_customers(df):

    customer_id = df['customer_id']
    total_price = df['total_price']

    graph_top_20_cus = plt.figure(figsize=(14,6), dpi=100)
    graph_top_20_cus = plt.xticks(fontsize=14)
    graph_top_20_cus = plt.yticks(fontsize=14 )
    graph_top_20_cus = sns.barplot(x = customer_id, y = total_price, color='#1f77b4')
    graph_top_20_cus = plt.xlabel('customer_id', fontsize=16)
    graph_top_20_cus = plt.ylabel('total Price', fontsize=16)
    graph_top_20_cus = plt.title('Top 20 Customers', fontsize=18)
    return graph_top_20_cus


def get_delivery_time_for_best_customers(df_orders, top_20_customers):
    filtered_orders_updated_FOR_TOP_20 = df_orders['customer_id'].isin(top_20_customers['customer_id'])
    filtered_order_top_rev_LATEST_DATAFRAME = df_orders[filtered_orders_updated_FOR_TOP_20]
    grouped_filtered_order_top_rev_LATEST_DATAFRAME = filtered_order_top_rev_LATEST_DATAFRAME.sort_values(by='Delivery_time', ascending=False)
    grouped_filtered_order_top_rev_LATEST_DATAFRAME['Delivery_time'] = grouped_filtered_order_top_rev_LATEST_DATAFRAME['Delivery_time'].dt.days.astype('int64')

    grouped_filtered_order_top_rev_LATEST_DATAFRAME_agg= grouped_filtered_order_top_rev_LATEST_DATAFRAME.groupby('customer_id').agg({'Delivery_time': ['mean', 'min', 'max']}).round(0)
    return grouped_filtered_order_top_rev_LATEST_DATAFRAME_agg

def get_n_of_orders_by_month(df):
    orders_by_month = df.groupby('Month')['payment'].count().to_frame().reset_index()
    orders_by_month = orders_by_month.rename(columns={'payment':'n of orders'})
    orders_by_month = orders_by_month.sort_values(by='Month', ascending=True)
    orders_by_month['Month'] = orders_by_month['Month'].apply(lambda x: calendar.month_abbr[x])
    return orders_by_month

def graph_n_orders_per_month(df):
    n_sales_by_month_graph = plt.figure(figsize=(10,6), dpi=100)
    n_sales_by_month_graph = plt.xticks(fontsize=14)
    n_sales_by_month_graph = plt.yticks(fontsize=14)
    n_sales_by_month_graph = sns.barplot(x = df["Month"], y = df['n of orders'], color='#1f77b4')
    n_sales_by_month_graph = plt.xlabel('Month', fontsize=16)
    n_sales_by_month_graph = plt.ylabel('Number of Orders', fontsize=16)
    n_sales_by_month_graph = plt.title('Number of Orders per Month', fontsize=18)
    return n_sales_by_month_graph