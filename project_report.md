# Retail Sales Analytics Project Report

## Executive summary

This project analyzes invoice-line transactions from the UCI Online Retail
dataset and builds a reproducible daily sales forecasting baseline. It removes
exact duplicate lines, cancellations, and non-positive sales rows; retains and
reports potential sales outliers; compares Linear Regression and Random Forest
on a chronological holdout; and exposes sales, product, country, and forecast
views through notebook visuals and a Streamlit dashboard.

## Data source and scope

The Online Retail dataset is provided by the UCI Machine Learning Repository:
<https://archive.ics.uci.edu/dataset/352/online+retail> (DOI:
<https://doi.org/10.24432/C5BW33>). It contains 541,909 transaction lines for a
UK-based online retailer from 1 December 2010 through 9 December 2011. Source
values are in GBP. `Country` is used as the region field; it describes customer
country, not a finer store or sales territory.

## Data quality treatment

The pipeline checks the expected schema and parses invoice dates and numeric
quantity/price fields. Rows without invoice number, stock code, invoice date,
or country are excluded. Missing descriptions are retained as `Unknown
product`. Exact duplicate rows are removed. Invoice numbers prefixed with `C`
and rows with quantity or unit price at or below zero are excluded from the
positive-sales model. Line revenue is quantity times unit price. A 1.5-IQR
upper fence flags high transaction sales, which are retained rather than
automatically deleted. Cleaned data must have complete dates and positive,
non-missing sales.

## Analysis

The notebook produces descriptive statistics, daily/monthly/weekday trends,
top products, top countries, and correlations among quantity, unit price, and
line sales. Six generated images are written to `visualizations/`: the weekly
trend, product ranking, country ranking, correlation heatmap, forecast chart,
and KPI summary. The dashboard reports net sales, unique valid invoices,
average invoice value, and the top country under the selected filters.

## Forecasting approach

The target is total transaction sales per calendar day, with days without
transactions filled with zero. Features are calendar fields, one-day and
seven-day sales lags, and the mean of the preceding seven days. The last 20% of
feature-complete dates form a chronological holdout. Linear Regression and a
Random Forest with fixed random state are compared by MAE, RMSE, and R-squared;
the lowest-MAE estimator is then refit on all historical observations. Its
joblib artifact and recursive 30-day forecast are saved by the pipeline.

### Measured results

The executed pipeline read 541,909 source rows and retained 524,878 valid
positive-sales rows. It removed 5,268 exact duplicates and 11,763 deduplicated
rows with cancellations or non-positive quantity/price; no required transaction
identifiers were missing. It flagged 42,624 high transaction-sales observations
above the 1.5-IQR fence, retaining them for analysis. Missing customer IDs were
not imputed because they are not required for this analysis.

On the final chronological holdout, Random Forest achieved MAE 13,387.85, RMSE
23,573.84, and R-squared 0.411. Linear Regression achieved MAE 15,218.57, RMSE
23,906.56, and R-squared 0.394. Random Forest was selected by the lowest-MAE
criterion and refit on all available daily observations. The authoritative
full-precision metrics are written to `data/processed/model_metrics.csv` when
the pipeline runs.

## Limitations and recommendations

- The data covers roughly one year, so annual seasonality and long-run drift
  cannot be validated reliably.
- A calendar-day holdout evaluates future dates but may reflect a different
  seasonal regime than the training period; additional rolling-origin
  backtesting is recommended before operational use.
- Returns and cancellations are excluded rather than modeled as negative
  revenue, so the target is positive-sales revenue and not net accounting
  revenue.
- Country is a broad customer geography; the product ranking uses descriptions
  that may have inconsistent naming.
- The global forecast is not filtered by country or product. A production
  planning model should test segment-level forecasts, compare seasonal-naive
  baselines, and quantify prediction intervals.

## Reproducibility

Install with `python -m pip install -e '.[dev]'`, run `pytest`, then run
`python -m retail_sales.pipeline`. The same pipeline powers the notebook and
can bootstrap the dashboard when generated data is not present.