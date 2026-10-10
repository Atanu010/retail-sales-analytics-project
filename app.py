"""Interactive Streamlit dashboard for the retail sales project."""

from pathlib import Path

import pandas as pd
import streamlit as st

from retail_sales.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "retail_sales_clean.csv"

st.set_page_config(page_title="Retail Sales Intelligence", layout="wide")


@st.cache_data(show_spinner="Preparing retail data and model artifacts...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists():
        run_pipeline(ROOT)
    data = pd.read_csv(DATA_PATH, parse_dates=["InvoiceDate"])
    forecast = pd.read_csv(
        ROOT / "data" / "processed" / "sales_forecast.csv", parse_dates=["date"]
    )
    metrics = pd.read_csv(ROOT / "data" / "processed" / "model_metrics.csv")
    return data, forecast, metrics


st.title("Retail Sales Intelligence")
st.caption("UCI Online Retail | Transaction performance and daily sales forecast")
try:
    transactions, forecast, model_metrics = load_data()
except Exception as error:
    st.error(f"Could not prepare dashboard data: {error}")
    st.stop()

with st.sidebar:
    st.header("Filters")
    countries = sorted(transactions["Country"].dropna().unique())
    selected_countries = st.multiselect("Country", countries, default=countries)
    products = sorted(transactions["Description"].dropna().unique())
    selected_products = st.multiselect("Product", products)
    min_date = transactions["InvoiceDate"].min().date()
    max_date = transactions["InvoiceDate"].max().date()
    selected_dates = st.date_input(
        "Invoice date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

filtered = transactions[transactions["Country"].isin(selected_countries)]
if selected_products:
    filtered = filtered[filtered["Description"].isin(selected_products)]
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = filtered[filtered["InvoiceDate"].dt.date.between(start_date, end_date)]

if filtered.empty:
    st.warning("No transactions match the selected filters.")
    st.stop()

sales = filtered["Sales"].sum()
orders = filtered["InvoiceNo"].nunique()
average_order = filtered.groupby("InvoiceNo")["Sales"].sum().mean()
best_country = filtered.groupby("Country")["Sales"].sum().idxmax()

st.subheader("Executive Summary")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Net sales", f"£{sales:,.0f}")
kpi2.metric("Valid orders", f"{orders:,}")
kpi3.metric("Average order value", f"£{average_order:,.2f}")
kpi4.metric("Top country", best_country)

left, right = st.columns(2)
with left:
    st.subheader("Sales Trend")
    daily = filtered.set_index("InvoiceDate")["Sales"].resample("W").sum()
    st.line_chart(daily, y_label="Sales (£)")
with right:
    st.subheader("Country Performance")
    country_sales = filtered.groupby("Country")["Sales"].sum().nlargest(10)
    st.bar_chart(country_sales, horizontal=True, x_label="Sales (£)")

left, right = st.columns(2)
with left:
    st.subheader("Product Performance")
    product_sales = filtered.groupby("Description")["Sales"].sum().nlargest(10)
    st.bar_chart(product_sales, horizontal=True, x_label="Sales (£)")
with right:
    st.subheader("Daily Forecast")
    st.line_chart(forecast.set_index("date")["sales"], y_label="Forecast sales (£)")

st.subheader("Model Comparison")
st.dataframe(model_metrics, hide_index=True, use_container_width=True)