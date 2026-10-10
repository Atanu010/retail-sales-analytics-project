"""Generate reusable project visualizations from cleaned transactions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import seaborn as sns

from retail_sales.analytics import daily_sales

sns.set_theme(style="whitegrid", palette="crest")


def create_visualizations(
    frame: pd.DataFrame,
    forecast: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save trend, product, region, correlation, forecast, and KPI charts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    daily = daily_sales(frame)

    figure, axis = plt.subplots(figsize=(12, 5))
    daily.resample("W").sum().plot(ax=axis, color="#126e75", linewidth=2)
    axis.set(title="Weekly Retail Sales", xlabel="Week", ylabel="Sales")
    outputs.append(_save(figure, output_dir / "sales_trends.png"))

    top_products = (
        frame.groupby("Description")["Sales"].sum().nlargest(10).sort_values()
    )
    figure, axis = plt.subplots(figsize=(10, 6))
    top_products.plot.barh(ax=axis, color="#d47a45")
    axis.set(title="Top 10 Products by Revenue", xlabel="Sales", ylabel="Product")
    outputs.append(_save(figure, output_dir / "product_performance.png"))

    top_countries = frame.groupby("Country")["Sales"].sum().nlargest(10).sort_values()
    figure, axis = plt.subplots(figsize=(10, 6))
    top_countries.plot.barh(ax=axis, color="#548c69")
    axis.set(title="Top 10 Countries by Revenue", xlabel="Sales", ylabel="Country")
    outputs.append(_save(figure, output_dir / "regional_performance.png"))

    numeric = frame[["Quantity", "UnitPrice", "Sales"]].corr()
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.heatmap(numeric, annot=True, cmap="vlag", center=0, fmt=".2f", ax=axis)
    axis.set_title("Transaction Metric Correlations")
    outputs.append(_save(figure, output_dir / "correlation_heatmap.png"))

    figure, axis = plt.subplots(figsize=(12, 5))
    daily.tail(90).plot(ax=axis, label="Actual", color="#126e75")
    forecast["sales"].plot(ax=axis, label="Forecast", color="#d47a45")
    axis.set(title="Daily Sales and 30-Day Forecast", xlabel="Date", ylabel="Sales")
    axis.legend()
    outputs.append(_save(figure, output_dir / "sales_forecast.png"))

    total_sales = frame["Sales"].sum()
    order_count = frame["InvoiceNo"].nunique()
    average_order = frame.groupby("InvoiceNo")["Sales"].sum().mean()
    country_count = frame["Country"].nunique()
    figure, axes = plt.subplots(2, 2, figsize=(11, 6))
    kpis = [
        ("Net Sales", f"£{total_sales:,.0f}"),
        ("Valid Orders", f"{order_count:,}"),
        ("Average Order Value", f"£{average_order:,.2f}"),
        ("Countries", f"{country_count:,}"),
    ]
    for axis, (label, value) in zip(axes.flat, kpis):
        axis.axis("off")
        axis.text(
            0.5, 0.62, value, ha="center", va="center", fontsize=22, weight="bold"
        )
        axis.text(0.5, 0.28, label, ha="center", va="center", fontsize=12)
    figure.suptitle("Retail Sales KPI Dashboard", fontsize=16, weight="bold")
    outputs.append(_save(figure, output_dir / "kpi_dashboard.png"))
    return outputs


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path