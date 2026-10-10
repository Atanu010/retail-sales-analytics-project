import numpy as np
import pandas as pd
import pytest

from retail_sales.analytics import compare_models, daily_sales, forecast_next_days


def test_daily_sales_includes_zero_sales_days():
    transactions = pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(["2011-01-01", "2011-01-03"]),
            "Sales": [10.0, 12.0],
        }
    )

    result = daily_sales(transactions)

    assert result.tolist() == [10.0, 0.0, 12.0]


def test_model_comparison_and_forecast_are_reproducible(tmp_path):
    dates = pd.date_range("2011-01-01", periods=120, freq="D")
    series = pd.Series(100 + 20 * np.sin(np.arange(120) * 2 * np.pi / 7), index=dates)
    model_path = tmp_path / "model.joblib"

    metrics, predictions, artifact = compare_models(series, model_path)
    forecast = forecast_next_days(artifact["model"], series, horizon=14)

    assert model_path.exists()
    assert set(metrics["model"]) == {"Linear Regression", "Random Forest"}
    assert len(predictions) > 0
    assert len(forecast) == 14
    assert (forecast["sales"] >= 0).all()


def test_model_comparison_rejects_short_series(tmp_path):
    series = pd.Series([10.0] * 20, index=pd.date_range("2011-01-01", periods=20))

    with pytest.raises(ValueError, match="30"):
        compare_models(series, tmp_path / "model.joblib")


def test_model_comparison_accepts_unsorted_series(tmp_path):
    dates = pd.date_range("2011-01-01", periods=60, freq="D")
    series = pd.Series(
        100 + 20 * np.sin(np.arange(60) * 2 * np.pi / 7), index=dates[::-1]
    )

    metrics, predictions, artifact = compare_models(series, tmp_path / "model.joblib")
    forecast = forecast_next_days(artifact["model"], series, horizon=7)

    assert set(metrics["model"]) == {"Linear Regression", "Random Forest"}
    assert len(predictions) > 0
    assert len(forecast) == 7


def test_forecast_rejects_empty_history():
    empty = pd.Series(dtype=float)

    class DummyModel:
        def predict(self, row):
            return [0.0]

    with pytest.raises(ValueError, match="at least one historical observation"):
        forecast_next_days(DummyModel(), empty)