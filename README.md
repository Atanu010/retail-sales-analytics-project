# Retail Sales Analytics and Forecasting

End-to-end analysis of the UCI Online Retail transactions: validated cleaning,
exploratory analysis, time-aware daily sales forecasting, saved visualizations,
automated tests, and an interactive Streamlit dashboard.

## Dataset and provenance

This project uses the **Online Retail** dataset published by the UCI Machine
Learning Repository. It contains 541,909 invoice-line transactions from a
UK-based online retailer between 1 December 2010 and 9 December 2011.

- Dataset page: <https://archive.ics.uci.edu/dataset/352/online+retail>
- Download: <https://archive.ics.uci.edu/static/public/352/online+retail.zip>
- Dataset DOI: <https://doi.org/10.24432/C5BW33>
- License listed by UCI: CC BY 4.0

The workbook is downloaded by the pipeline into `data/raw/`; source data is not
committed. The analysis uses `Country` as its regional dimension and calculates
line sales as `Quantity * UnitPrice` in the source currency (GBP).

## Quick start

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m retail_sales.pipeline
```

The pipeline downloads and extracts the source workbook, validates and cleans
transactions, trains and compares forecasting models, and writes all derived
artifacts. Outputs are placed in `data/processed/`, `models/`, and
`visualizations/`.

Run the notebook from the project root using Jupyter, or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/retail_sales_analysis.ipynb
```

Launch the interactive dashboard after the pipeline completes:

```bash
streamlit run app.py
```

The dashboard also runs the pipeline automatically if processed data is absent.
Use the country, product, and invoice-date filters to update the executive KPIs
and performance charts. The forecast is a global daily-sales forecast, not a
filtered country- or product-specific forecast.

## Cleaning and validation

- Required source fields are checked before cleaning.
- Rows without invoice number, stock code, date, or country are dropped; missing
	product descriptions are labeled `Unknown product`.
- Missing customer IDs are retained as anonymous transactions because customer
	identity is not needed for sales aggregation or daily forecasting; an
	identifier is not imputed.
- Exact duplicate invoice lines are removed.
- Cancelled invoices (invoice numbers beginning with `C`) and rows with
	non-positive quantity or unit price are excluded from positive-sales
	analysis. These are commonly returns or non-sales adjustments in this source.
- Potential transaction-sales outliers are flagged using the 1.5-IQR upper
	fence and retained, since unusually large genuine orders should not be
	silently deleted.
- Cleaned records are checked for required fields, valid dates, and strictly
	positive quantities, unit prices, and sales before modeling, and are checked
	for exact duplicates and missing transaction dimensions.

The pipeline prints a cleaning summary and persists cleaned data for repeatable
analysis. The source workbook can be removed to force a fresh download.

## Analysis and modeling

The notebook includes descriptive statistics, monthly and weekday trends,
country and product revenue, and a transaction-metric correlation heatmap.
Daily sales are modeled using calendar variables, one- and seven-day lags, and
a trailing seven-day mean. Linear Regression and Random Forest are compared on
the final chronological 20% of complete daily observations using MAE, RMSE, and
R-squared. The lowest-MAE model is refit on all available observations, saved
with its metrics, and used for a recursive 30-day forecast.

This is a short historical series with a single retail cycle; the forecast is a
baseline, not a causal or seasonal planning guarantee. Holdout metrics should
be interpreted in that context.

## Project structure

```text
app.py
notebooks/retail_sales_analysis.ipynb
src/retail_sales/
	analytics.py       # daily series, features, model comparison, forecast
	data.py            # source download, cleaning, and validation
	pipeline.py        # reproducible end-to-end workflow
	visualize.py       # chart generation
tests/               # automated unit tests
project_report.md    # findings and limitations
```

## Tests and style

```bash
pytest
ruff check .
```

The saved model is a joblib artifact and should only be loaded from a trusted
source. Regenerate it from the documented source with the pipeline when needed.