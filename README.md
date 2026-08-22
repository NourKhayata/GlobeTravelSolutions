# Travel Insurance Production Analytics & Forecasting - Streamlit App

This Streamlit app replaces the Tableau EDA dashboards and adds a forecasting phase.

## Pages

1. Production Overview
2. Sales Performance
3. Geography & Product Analysis
4. Data Quality & Outlier Review
5. Forecasting Preview
6. Financial Performance — quarterly revenue, cost structure, profitability, a what-if profit simulator, and recommendations for increasing profit and sales

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

You can either:

1. Upload the Excel file through the app sidebar, or
2. Place the dataset here:

```text
data/Production report 2023- 2026 (Nour).xlsx
```

The app automatically unions all sheets whose names start with `Raw Data`.

For the Financial Performance page, place the quarterly financial summary beside the app (or in `data/`), or upload it on the page itself:

```text
Company_Financial_Summary_2023_2026.xlsx
```

It must contain a `Quarter` column (e.g. `Q1 2023`), a `Total Revenue` column, and the cost columns (payroll, benefits, cost of sales, administrative, other operating).

## Privacy note

The app removes personal fields from display, such as passport number, beneficiary full name, date of birth, email, phone number, and emergency contact fields.
