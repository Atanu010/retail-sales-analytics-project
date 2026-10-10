"""Time-series feature engineering and sales forecasting."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "day_of_year",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
]


def daily_sales(frame: pd.DataFrame) -> pd.Series:
    """Aggregate transaction revenue into a complete daily time series."""
    required = {"InvoiceDate", "Sales"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Sales data is missing required columns: {sorted(missing)}")

    cleaned = frame.copy()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned = cleaned.dropna(subset=["InvoiceDate", "Sales"]).sort_values("InvoiceDate")

    daily = (
        cleaned.set_index("InvoiceDate")["Sales"]
        .resample("D")
        .sum()
        .asfreq("D", fill_value=0)
    )
    daily.name = "Sales"
    return daily


def make_features(series: pd.Series) -> pd.DataFrame:
    """Build calendar and strictly historical sales features."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("Series must contain at least one observation.")

    series = series.copy().sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Series index must be datetime-like.") from exc

    features = pd.DataFrame(index=series.index)
    features["day_of_week"] = series.index.dayofweek
    features["day_of_month"] = series.index.day
    features["month"] = series.index.month
    features["year"] = series.index.year
    features["day_of_year"] = series.index.dayofyear
    features["lag_1"] = series.shift(1)
    features["lag_7"] = series.shift(7)
    features["rolling_mean_7"] = series.shift(1).rolling(7).mean()
    return features


def compare_models(
    series: pd.Series, model_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compare models on a chronological holdout and save the final model."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("Series must contain at least one observation.")

    series = series.copy().sort_index()
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except (TypeError, ValueError) as exc:
            raise ValueError("Series index must be datetime-like.") from exc

    features = make_features(series)
    supervised = features.join(series.rename("target")).dropna()
    if len(supervised) < 30:
        raise ValueError("At least 30 complete daily observations are required.")

    split_at = int(len(supervised) * 0.8)
    x_train = supervised[FEATURES].iloc[:split_at]
    y_train = supervised["target"].iloc[:split_at]
    x_test = supervised[FEATURES].iloc[split_at:]
    y_test = supervised["target"].iloc[split_at:]
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    records = []
    for name, estimator in candidates.items():
        estimator.fit(x_train, y_train)
        prediction = estimator.predict(x_test)
        records.append(
            {
                "model": name,
                "mae": mean_absolute_error(y_test, prediction),
                "rmse": np.sqrt(mean_squared_error(y_test, prediction)),
                "r2": r2_score(y_test, prediction),
            }
        )

    metrics = pd.DataFrame(records).sort_values("mae").reset_index(drop=True)
    best_name = str(metrics.iloc[0]["model"])
    final_model = candidates[best_name].fit(supervised[FEATURES], supervised["target"])
    artifact = {
        "model": final_model,
        "model_name": best_name,
        "features": FEATURES,
        "holdout_metrics": metrics.to_dict(orient="records"),
        "trained_through": str(series.index.max().date()),
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    predictions = pd.DataFrame(
        {"actual": y_test, "predicted": final_model.predict(x_test)}
    )
    return metrics, predictions, artifact


def forecast_next_days(
    model: object, history: pd.Series, horizon: int = 30
) -> pd.DataFrame:
    """Recursively forecast future daily revenue using only available history."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one day.")

    if not isinstance(history, pd.Series):
        raise TypeError("history must be a pandas Series")
    if history.empty:
        raise ValueError("Forecast requires at least one historical observation.")

    values = history.copy().sort_index().astype(float)
    if not isinstance(values.index, pd.DatetimeIndex):
        try:
            values.index = pd.to_datetime(values.index)
        except (TypeError, ValueError) as exc:
            raise ValueError("History index must be datetime-like.") from exc

    forecast = []
    for _ in range(horizon):
        next_date = values.index[-1] + pd.Timedelta(days=1)
        recent = values.iloc[-7:]
        row = pd.DataFrame(
            [[
                next_date.dayofweek,
                next_date.day,
                next_date.month,
                next_date.year,
                next_date.dayofyear,
                values.iloc[-1],
                values.iloc[-7] if len(values) >= 7 else values.mean(),
                recent.mean(),
            ]],
            columns=FEATURES,
            index=[next_date],
        )
        prediction = max(0.0, float(model.predict(row)[0]))
        forecast.append({"date": next_date, "sales": prediction})
        values.loc[next_date] = prediction
    return pd.DataFrame(forecast).set_index("date")