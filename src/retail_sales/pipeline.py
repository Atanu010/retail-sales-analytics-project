"""Reproducible end-to-end data preparation and modeling workflow."""

from pathlib import Path

from retail_sales.analytics import compare_models, daily_sales, forecast_next_days
from retail_sales.data import (
    clean_sales_data,
    download_dataset,
    load_raw_data,
    validate_sales_data,
)
from retail_sales.visualize import create_visualizations

ROOT = Path(__file__).resolve().parents[2]


def run_pipeline(root: str | Path = ROOT) -> dict[str, object]:
    """Download source data and write cleaned data, model, metrics, and charts."""
    root = Path(root)
    workbook = download_dataset(root / "data" / "raw")
    raw = load_raw_data(workbook)
    cleaned, cleaning_report = clean_sales_data(raw)
    validate_sales_data(cleaned)

    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(processed_dir / "retail_sales_clean.csv", index=False)

    daily = daily_sales(cleaned)
    metrics, predictions, artifact = compare_models(
        daily, root / "models" / "sales_forecast.joblib"
    )
    metrics.to_csv(processed_dir / "model_metrics.csv", index=False)
    predictions.to_csv(processed_dir / "holdout_predictions.csv", index_label="date")
    future = forecast_next_days(artifact["model"], daily)
    future.to_csv(processed_dir / "sales_forecast.csv", index_label="date")
    charts = create_visualizations(cleaned, future, root / "visualizations")

    return {
        "raw_rows": len(raw),
        "clean_rows": len(cleaned),
        "cleaning_report": cleaning_report,
        "metrics": metrics,
        "best_model": artifact["model_name"],
        "charts": charts,
    }


if __name__ == "__main__":
    results = run_pipeline()
    print(f"Rows: {results['raw_rows']:,} raw, {results['clean_rows']:,} clean")
    print(f"Selected model: {results['best_model']}")
    print(results["metrics"].to_string(index=False))
    print(f"Cleaning report: {results['cleaning_report']}")