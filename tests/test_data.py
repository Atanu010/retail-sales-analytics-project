import pandas as pd
import pytest

from retail_sales.data import clean_sales_data, validate_sales_data


@pytest.fixture
def transactions():
    return pd.DataFrame(
        {
            "InvoiceNo": ["100", "100", "C101", "102", "103"],
            "StockCode": ["A", "A", "B", "C", "D"],
            "Description": ["Mug", "Mug", "Plate", None, "Lamp"],
            "Quantity": [2, 2, 1, 3, -1],
            "InvoiceDate": ["2011-01-01"] * 5,
            "UnitPrice": [5.0, 5.0, 10.0, 2.0, 4.0],
            "Country": ["France"] * 5,
        }
    )


def test_cleaning_removes_duplicates_returns_and_nonpositive_sales(transactions):
    cleaned, report = clean_sales_data(transactions)

    assert len(cleaned) == 2
    assert cleaned["Sales"].tolist() == [10.0, 6.0]
    assert report["duplicate_rows_removed"] == 1
    assert report["valid_sales_rows"] == 2
    assert cleaned["Description"].eq("Unknown product").any()
    validate_sales_data(cleaned)


def test_cleaning_requires_source_columns(transactions):
    with pytest.raises(ValueError, match="required columns"):
        clean_sales_data(transactions.drop(columns="Country"))


def test_validation_rejects_nonpositive_sales(transactions):
    cleaned, _ = clean_sales_data(transactions)
    cleaned.loc[0, "Sales"] = 0

    with pytest.raises(ValueError, match="positive"):
        validate_sales_data(cleaned)


def test_validation_rejects_duplicate_transactions(transactions):
    cleaned, _ = clean_sales_data(transactions)
    duplicated = pd.concat([cleaned, cleaned.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        validate_sales_data(duplicated)


def test_cleaning_treats_blank_strings_as_missing():
    data = pd.DataFrame(
        {
            "InvoiceNo": ["100", "  ", "C101", "102"],
            "StockCode": ["A", "B", "C", "D"],
            "Description": ["Mug", "", "Plate", "Lamp"],
            "Quantity": [2, 1, 1, 3],
            "InvoiceDate": [
                "2011-01-01",
                "2011-01-02",
                "2011-01-03",
                "2011-01-04",
            ],
            "UnitPrice": [5.0, 4.0, 10.0, 2.0],
            "Country": ["France", "France", "France", "France"],
        }
    )

    cleaned, report = clean_sales_data(data)

    assert len(cleaned) == 2
    assert cleaned["Description"].tolist() == ["Mug", "Lamp"]
    assert report["rows_missing_required_identifiers"] == 1
    validate_sales_data(cleaned)