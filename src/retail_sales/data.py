"""Loading, cleaning, and validating the UCI Online Retail dataset."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
REQUIRED_COLUMNS = {
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "Country",
}


def download_dataset(destination: str | Path) -> Path:
    """Download and extract the original UCI workbook, returning its path."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "online-retail.zip"
    workbook_path = destination / "Online Retail.xlsx"
    if not workbook_path.exists():
        urlretrieve(DATA_URL, archive_path)
        with ZipFile(archive_path) as archive:
            workbook_name = next(
                name for name in archive.namelist() if name.lower().endswith(".xlsx")
            )
            archive.extract(workbook_name, destination)
            extracted_path = destination / workbook_name
            if extracted_path != workbook_path:
                extracted_path.rename(workbook_path)
    return workbook_path


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the first worksheet and validate its expected source schema."""
    frame = pd.read_excel(path, engine="openpyxl")
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return frame


def clean_sales_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean transactions while retaining legitimate high-value observations.

    Cancelled invoices and non-positive quantities or prices are excluded from
    sales analysis. Missing descriptions are labeled, while rows without the
    dimensions needed to identify and date a transaction are removed.
    """
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = frame.copy().replace(r"^\s*$", pd.NA, regex=True)
    original_count = len(cleaned)
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")
    cleaned["Description"] = cleaned["Description"].replace(r"^\s*$", pd.NA, regex=True)
    cleaned["Description"] = cleaned["Description"].fillna("Unknown product")

    valid_identifiers = cleaned[
        ["InvoiceNo", "StockCode", "InvoiceDate", "Country"]
    ].notna().all(axis=1)
    identified_rows = cleaned.loc[valid_identifiers]
    rows_before_deduplication = len(identified_rows)
    cleaned = identified_rows.drop_duplicates()
    after_duplicates = len(cleaned)
    non_cancelled = ~cleaned["InvoiceNo"].astype(str).str.startswith("C")
    positive_sales = (cleaned["Quantity"] > 0) & (cleaned["UnitPrice"] > 0)
    cleaned = cleaned.loc[non_cancelled & positive_sales].copy()
    cleaned["Sales"] = cleaned["Quantity"] * cleaned["UnitPrice"]

    if cleaned.empty:
        raise ValueError("No valid positive sales transactions remain after cleaning.")

    sales_q1 = cleaned["Sales"].quantile(0.25)
    sales_q3 = cleaned["Sales"].quantile(0.75)
    upper_fence = sales_q3 + 1.5 * (sales_q3 - sales_q1)
    outlier_count = int((cleaned["Sales"] > upper_fence).sum())

    report = {
        "input_rows": original_count,
        "rows_missing_required_identifiers": original_count - rows_before_deduplication,
        "duplicate_rows_removed": rows_before_deduplication - after_duplicates,
        "rows_removed_as_invalid_or_returns": after_duplicates - len(cleaned),
        "valid_sales_rows": len(cleaned),
        "sales_outliers_flagged": outlier_count,
    }
    return cleaned.reset_index(drop=True), report


def validate_sales_data(frame: pd.DataFrame) -> None:
    """Raise an informative error when cleaned data violates its contract."""
    expected = REQUIRED_COLUMNS | {"Sales"}
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"Cleaned data is missing columns: {sorted(missing)}")

    cleaned = frame.copy().replace(r"^\s*$", pd.NA, regex=True)
    if cleaned.empty:
        raise ValueError("Cleaned data must contain at least one transaction.")
    identifiers = ["InvoiceNo", "StockCode", "InvoiceDate", "Country"]
    if cleaned[identifiers].isna().any().any():
        raise ValueError("Transaction identifiers and dimensions must not be missing.")
    if cleaned.duplicated().any():
        raise ValueError("Cleaned data must not contain exact duplicate rows.")
    if cleaned["InvoiceDate"].isna().any():
        raise ValueError("InvoiceDate must not contain missing values.")
    if (cleaned["Quantity"] <= 0).any() or (cleaned["UnitPrice"] <= 0).any():
        raise ValueError("Quantity and UnitPrice must be positive.")
    if cleaned["Sales"].isna().any() or (cleaned["Sales"] <= 0).any():
        raise ValueError("Sales must contain only positive, non-missing values.")
