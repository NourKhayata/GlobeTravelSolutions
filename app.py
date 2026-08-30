import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

APP_TITLE = "Travel Insurance Production Analytics & Forecasting"
DEFAULT_DATA_PATHS = [
    Path("data/Production report 2023- 2026 (Nour).xlsx"),
    Path("Production report 2023- 2026 (Nour).xlsx"),
]

FINANCIAL_DATA_PATHS = [
    Path("data/Company_Financial_Summary_2023_2026.xlsx"),
    Path("Company_Financial_Summary_2023_2026.xlsx"),
]

REVENUE_COL = "Total Revenue"
QUARTER_COL = "Quarter"

# Long workbook headers shortened for axes and legends.
COST_LABELS = {
    "Total Payroll / Salary Expense": "Payroll",
    "Employee Benefits & Related Payroll Costs": "Benefits",
    "Cost of Sales (Insurance Related Costs)": "Cost of Sales",
    "Administrative Expenses": "Administrative",
    "Other Operating Expenses": "Other Operating",
}

# Categorical slots 1-5 of the validated palette, plus fixed status colors.
FINANCIAL_THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "text": "#0b0b0b",
        "muted": "#898781",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"],
        "template": "plotly_white",
    },
    "dark": {
        "surface": "#1a1a19",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "text": "#ffffff",
        "muted": "#898781",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"],
        "template": "plotly_dark",
    },
}
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

PERSONAL_FIELDS = [
    "Passport Number",
    "Beneficiary Full Name",
    "Beneficiary Date of Birth",
    "Email",
    "Phone Number",
    "Emergency Phone Number",
    "Emergency Contact Name",
]

AGE_BRACKET_COL = "Client Age Bracket"
AGE_BRACKET_ORDER = ["0-17", "18-25", "26-35", "36-45", "46-55", "56-65", "66-75", "76+"]
AGE_BRACKET_BINS = [-1, 17, 25, 35, 45, 55, 65, 75, 150]

st.set_page_config(page_title=APP_TITLE, layout="wide")


def clean_date(series: pd.Series) -> pd.Series:
    """Convert Excel serial dates or date strings to pandas datetime."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.to_datetime(series, errors="coerce")

    serial_mask = numeric.between(30000, 60000)
    parsed_serial = pd.to_datetime(
        numeric.where(serial_mask), unit="D", origin="1899-12-30", errors="coerce"
    )
    return parsed_serial.combine_first(parsed)


def parse_rates_text(text: str) -> dict:
    out = {}
    if not text:
        return out
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            continue
        currency, rate = part.split(":", 1)
        try:
            out[currency.strip()] = float(rate.strip())
        except ValueError:
            continue
    return out


def compute_usd_columns(df: pd.DataFrame, rates: dict) -> pd.DataFrame:
    df = df.copy()
    if "Currency" not in df.columns or not rates:
        return df

    def map_rate(value):
        try:
            return rates.get(str(value).strip(), np.nan)
        except Exception:
            return np.nan

    rate_series = df["Currency"].apply(map_rate)
    if "Gross" in df.columns:
        df["Gross_USD"] = pd.to_numeric(df["Gross"], errors="coerce") * rate_series
    if "Premium" in df.columns:
        df["Premium_USD"] = pd.to_numeric(df["Premium"], errors="coerce") * rate_series
    return df


@st.cache_data(show_spinner=True)
def load_excel(uploaded_file_bytes: Optional[bytes], fallback_path: Optional[str]) -> pd.DataFrame:
    if uploaded_file_bytes is not None:
        excel_obj = io.BytesIO(uploaded_file_bytes)
    elif fallback_path:
        excel_obj = fallback_path
    else:
        raise FileNotFoundError("No Excel file was uploaded and no local file was found.")

    xls = pd.ExcelFile(excel_obj)
    frames = []
    for sheet in xls.sheet_names:
        if sheet.lower().startswith("raw data"):
            temp = pd.read_excel(excel_obj, sheet_name=sheet)
            temp["Source Sheet"] = sheet
            frames.append(temp)

    if not frames:
        raise ValueError("No sheets starting with 'Raw Data' were found.")

    df = pd.concat(frames, ignore_index=True)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

    for col in ["Issue Date", "Creation", "Modification Date", "Inception Date", "Expiry Date", "Beneficiary Date of Birth"]:
        if col in df.columns:
            df[col] = clean_date(df[col])

    numeric_cols = [
        "Number Of Members",
        "Gross",
        "Net To Pay",
        "Net",
        "Premium",
        "Premium Before Tax",
        "Discount Amount",
        "Additional Fees",
        "Mark Up",
        "Ceded Premium",
        "Duration",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Issue Date" in df.columns:
        df["Issue Month"] = df["Issue Date"].dt.to_period("M").dt.to_timestamp()
        df["Issue Year"] = df["Issue Date"].dt.year

    if "Beneficiary Date of Birth" in df.columns and "Issue Date" in df.columns:
        age_at_issue = (df["Issue Date"] - df["Beneficiary Date of Birth"]).dt.days / 365.25
        df[AGE_BRACKET_COL] = pd.cut(
            age_at_issue, bins=AGE_BRACKET_BINS, labels=AGE_BRACKET_ORDER
        ).astype(str).replace({"nan": np.nan})

    if "Reference Number" in df.columns:
        df["Policy Count"] = 1

    return df


def find_local_data_file() -> Optional[str]:
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            return str(path)
    return None


def multiselect_filter(label: str, df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        return df
    values = sorted([x for x in df[col].dropna().unique().tolist() if str(x).strip() != ""])
    selected = st.sidebar.multiselect(label, values, default=[])
    if selected:
        return df[df[col].isin(selected)]
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    if "Issue Date" in df.columns and df["Issue Date"].notna().any():
        min_date = df["Issue Date"].min().date()
        max_date = df["Issue Date"].max().date()
        date_range = st.sidebar.date_input(
            "Issue Date Range",
            (min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = date_range
            df = df[(df["Issue Date"].dt.date >= start) & (df["Issue Date"].dt.date <= end)]

    for label, col in [
        ("Currency", "Currency"),
        ("Status", "Status"),
        ("Type", "Type"),
        ("Insurance Company", "InsuranceCompany"),
        ("Country of Residence", "Country of Residence"),
        ("Plan", "Plan"),
    ]:
        df = multiselect_filter(label, df, col)

    return df


def kpi_cards(df: pd.DataFrame, use_usd: bool = False):
    total_policies = int(df["Policy Count"].sum()) if "Policy Count" in df.columns else len(df)
    total_members = df["Number Of Members"].sum() if "Number Of Members" in df.columns else np.nan
    gross_col = "Gross_USD" if use_usd and "Gross_USD" in df.columns else "Gross"
    premium_col = "Premium_USD" if use_usd and "Premium_USD" in df.columns else "Premium"
    total_gross = df[gross_col].sum() if gross_col in df.columns else np.nan
    total_premium = df[premium_col].sum() if premium_col in df.columns else np.nan

    cancelled = 0
    if "Status" in df.columns:
        cancelled = df["Status"].astype(str).str.lower().eq("cancelled").sum()
    cancellation_rate = cancelled / total_policies if total_policies else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Policies", f"{total_policies:,.0f}")
    c2.metric("Total Members", f"{total_members:,.0f}" if pd.notna(total_members) else "N/A")

    label_gross = "Total Gross (USD)" if use_usd and gross_col == "Gross_USD" else "Total Gross"
    c3.metric(label_gross, f"${total_gross:,.2f}" if pd.notna(total_gross) else "N/A")

    label_premium = "Total Premium (USD)" if use_usd and premium_col == "Premium_USD" else "Total Premium"
    c4.metric(label_premium, f"${total_premium:,.2f}" if pd.notna(total_premium) else "N/A")
    c5.metric("Cancellation Rate", f"{cancellation_rate:.2%}")


def bar_count(df: pd.DataFrame, col: str, title: str, top_n: int = 15, category_order: Optional[list] = None):
    if col not in df.columns:
        st.info(f"Column not found: {col}")
        return
    out = df[col].fillna("(Missing)").value_counts(dropna=False).reset_index()
    out.columns = [col, "Policies"]

    if category_order:
        order = [c for c in category_order if c in out[col].values] + ["(Missing)"]
        out[col] = pd.Categorical(out[col], categories=order, ordered=True)
        out = out.sort_values(col)
    else:
        out = out.head(top_n)

    fig = px.bar(out, x="Policies", y=col, orientation="h", title=title, text_auto=",.0f")
    if category_order:
        fig.update_layout(yaxis={"categoryorder": "array", "categoryarray": order[::-1]})
    else:
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def sum_bar(
    df: pd.DataFrame,
    group_col: str,
    measure: str,
    title: str,
    top_n: int = 15,
    use_usd: bool = False,
):
    measure_col = measure
    if use_usd and f"{measure}_USD" in df.columns:
        measure_col = f"{measure}_USD"

    if group_col not in df.columns or measure_col not in df.columns:
        st.info(f"Required columns not found: {group_col}, {measure_col}")
        return

    out = (
        df.groupby(group_col, dropna=False)[measure_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig = px.bar(out, x=measure_col, y=group_col, orientation="h", title=title, text_auto=",.0f")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


def overview_page(df: pd.DataFrame, use_usd: bool = False):
    st.subheader("01 Executive / Production Overview")
    st.caption("This page replaces the first Tableau dashboard and is used for the first EDA step.")
    kpi_cards(df, use_usd=use_usd)

    if "Issue Month" in df.columns:
        gross_col = "Gross_USD" if use_usd and "Gross_USD" in df.columns else "Gross"
        premium_col = "Premium_USD" if use_usd and "Premium_USD" in df.columns else "Premium"
        agg_map = {
            "Policies": ("Policy Count", "sum"),
            "Members": ("Number Of Members", "sum"),
        }
        if gross_col in df.columns:
            agg_map["Gross"] = (gross_col, "sum")
        if premium_col in df.columns:
            agg_map["Premium"] = (premium_col, "sum")

        monthly = df.groupby("Issue Month", as_index=False).agg(**agg_map)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(monthly, x="Issue Month", y="Policies", markers=True, title="Monthly Production Trend")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "Gross" in monthly.columns:
                title = "Gross by Month"
                if use_usd and gross_col == "Gross_USD":
                    title = "Gross by Month (USD)"
                fig = px.line(monthly, x="Issue Month", y="Gross", markers=True, title=title)
                st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        bar_count(df, "Status", "Status Distribution", top_n=10)
    with c2:
        bar_count(df, "Type", "Type Distribution", top_n=10)
    with c3:
        bar_count(df, "Currency", "Currency Distribution", top_n=10)


def sales_page(df: pd.DataFrame, use_usd: bool = False):
    st.subheader("02 Sales & Production Performance")
    c1, c2 = st.columns(2)
    with c1:
        sum_bar(df, "InsuranceCompany", "Gross", "Top Insurance Companies by Gross", use_usd=use_usd)
        sum_bar(df, "POS Name", "Gross", "Top POS by Gross", use_usd=use_usd)
    with c2:
        sum_bar(df, "Distributor Name", "Gross", "Top Distributors by Gross", use_usd=use_usd)
        sum_bar(df, "Account Executive Name", "Gross", "Top Account Executives by Gross", use_usd=use_usd)


def geography_product_page(df: pd.DataFrame, use_usd: bool = False):
    st.subheader("03 Geography, Destination & Product Analysis")
    c1, c2 = st.columns(2)
    with c1:
        bar_count(df, "Country of Residence", "Top Countries of Residence", top_n=20)
        bar_count(df, "Coverage Zone", "Coverage Zone Distribution", top_n=15)
    with c2:
        bar_count(df, "Destination", "Top Destinations", top_n=20)
        bar_count(df, "Plan", "Top Plans", top_n=20)

    bar_count(
        df, AGE_BRACKET_COL, "Client Age Bracket Distribution",
        category_order=AGE_BRACKET_ORDER,
    )


def data_quality_page(df: pd.DataFrame, use_usd: bool = False):
    st.subheader("04 Data Quality & Outlier Review")
    rows = []
    for col in ["Issue Date", "Reference Number", "Status", "Gross", "Premium", "Currency", "Destination", "Plan"]:
        if col in df.columns:
            rows.append({"Check": f"Missing {col}", "Count": int(df[col].isna().sum())})
    if "Gross" in df.columns:
        rows.append({"Check": "Negative Gross", "Count": int((df["Gross"] < 0).sum())})
        rows.append({"Check": "Zero Gross", "Count": int((df["Gross"] == 0).sum())})
    if "Reference Number" in df.columns:
        rows.append({"Check": "Duplicate Reference Numbers", "Count": int(df["Reference Number"].duplicated().sum())})
    dq = pd.DataFrame(rows)
    st.dataframe(dq, use_container_width=True, hide_index=True)
    if "Gross" in df.columns:
        st.markdown("#### Sample records with negative or zero Gross")
        gross_col = "Gross_USD" if use_usd and "Gross_USD" in df.columns else "Gross"
        premium_col = "Premium_USD" if use_usd and "Premium_USD" in df.columns else "Premium"
        sample_cols = [
            c for c in ["Issue Date", "Reference Number", "Status", "Type", "Currency", gross_col, premium_col, "Plan"]
            if c in df.columns
        ]
        display_df = df[(df["Gross"] <= 0)][sample_cols].head(100).copy()
        if gross_col in display_df.columns:
            display_df[gross_col] = display_df[gross_col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )
        if premium_col in display_df.columns:
            display_df[premium_col] = display_df[premium_col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    if "Number Of Members" in df.columns:
        st.markdown("#### Highest Number of Members")
        gross_col = "Gross_USD" if use_usd and "Gross_USD" in df.columns else "Gross"
        sample_cols = [
            c for c in ["Issue Date", "Reference Number", "Type", "Number Of Members", gross_col, "Currency", "Plan"]
            if c in df.columns
        ]
        display_df = df.sort_values("Number Of Members", ascending=False)[sample_cols].head(50).copy()
        if "Number Of Members" in display_df.columns:
            display_df["Number Of Members"] = display_df["Number Of Members"].apply(
                lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
            )
        if gross_col in display_df.columns:
            display_df[gross_col] = display_df[gross_col].apply(
                lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A"
            )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

def baseline_forecast(series: pd.Series, horizon: int) -> pd.Series:
    series = series.asfreq("MS").fillna(0)
    train = series.copy()
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    if STATSMODELS_AVAILABLE and len(series) >= 24:
        try:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="estimated",
            )
            forecast_values = model.fit(optimized=True).forecast(horizon)
        except Exception:
            forecast_values = pd.Series([train.tail(3).mean()] * horizon, index=future_index)
    else:
        forecast_values = pd.Series([train.tail(3).mean()] * horizon, index=future_index)
    return pd.Series(forecast_values.values, index=future_index)


def arima_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for ARIMA forecasting")
    from statsmodels.tsa.arima.model import ARIMA

    series = series.asfreq("MS").fillna(method="ffill").fillna(0)
    model = ARIMA(series, order=(1, 1, 1))
    fit = model.fit()
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series(fit.forecast(horizon), index=future_index)


def sarima_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for SARIMA forecasting")
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = series.asfreq("MS").fillna(method="ffill").fillna(0)
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series(fit.forecast(horizon), index=future_index)


def stl_arima_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for STL + ARIMA forecasting")
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.forecasting.stl import STLForecast

    series = series.asfreq("MS").fillna(method="ffill").fillna(0).astype(float)
    if len(series) < 24:
        raise ValueError("STL + ARIMA needs at least two full years of monthly data")
    stlf = STLForecast(
        series,
        ARIMA,
        model_kwargs={"order": (1, 1, 1)},
        period=12,
        robust=True,
    )
    fit = stlf.fit()
    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series(fit.forecast(horizon).values, index=future_index)


def ml_regression_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for machine-learning forecasting")
    from sklearn.linear_model import Ridge

    series = series.asfreq("MS").fillna(method="ffill").fillna(0).astype(float)
    n_lags = min(12, len(series) - 1)
    if n_lags < 3:
        raise ValueError("Not enough data for ML regression forecasting")

    X, y = [], []
    values = series.values
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags:i])
        y.append(values[i])
    model = Ridge(alpha=1.0)
    model.fit(np.array(X), np.array(y))

    history = list(values[-n_lags:])
    preds = []
    for _ in range(horizon):
        pred = model.predict(np.array(history[-n_lags:]).reshape(1, -1))[0]
        preds.append(pred)
        history.append(pred)

    future_index = pd.date_range(series.index.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    return pd.Series(preds, index=future_index)


def prophet_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet is not installed")
    df = series.asfreq("MS").fillna(method="ffill").fillna(0).reset_index()
    df.columns = ["ds", "y"]
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq="MS")
    forecast = model.predict(future)
    return pd.Series(forecast["yhat"].iloc[-horizon:].values, index=future["ds"].iloc[-horizon:])


def evaluate_forecast(true: pd.Series, pred: pd.Series) -> dict:
    true = true.astype(float)
    pred = pred.astype(float)
    common = true.index.intersection(pred.index)
    true = true.loc[common]
    pred = pred.loc[common]
    error = true - pred
    rmse = np.sqrt(np.mean(error ** 2))
    denom = np.where(np.abs(true) < 1e-6, 1.0, np.abs(true))
    mape = np.mean(np.abs(error / denom)) * 100
    return {"RMSE": rmse, "MAPE": mape}


def forecasting_page(df: pd.DataFrame, use_usd: bool = False):
    st.subheader("05 Forecasting Preview")
    st.caption("This page is the next phase after EDA. It prepares monthly data and tests a first forecasting baseline.")
    targets = ["Policy Count", "Gross", "Premium", "Number Of Members"]
    available_targets = [
        t for t in targets if t in df.columns or (use_usd and f"{t}_USD" in df.columns)
    ]
    target = st.selectbox("Forecast target", available_targets)
    horizon = st.slider("Forecast horizon in months", 1, 12, 6)
    target_col = target
    if use_usd and f"{target}_USD" in df.columns:
        target_col = f"{target}_USD"
    if "Issue Month" not in df.columns:
        st.warning("Issue Month is required for forecasting.")
        return
    monthly = df.groupby("Issue Month")[target_col].sum().sort_index()
    if monthly.empty or len(monthly) < 6:
        st.warning("Not enough monthly observations for forecasting.")
        return

    available_methods = {
        "Baseline": baseline_forecast,
        "ARIMA": arima_forecast,
        "SARIMA": sarima_forecast,
        "STL + ARIMA": stl_arima_forecast,
        "ML Regression": ml_regression_forecast,
    }
    if PROPHET_AVAILABLE:
        available_methods["Prophet"] = prophet_forecast

    method_descriptions = {
        "Baseline": "A simple baseline using moving average or Holt-Winters seasonal smoothing if enough data exists.",
        "ARIMA": "A classical ARIMA model that captures autoregression, differencing, and moving averages.",
        "SARIMA": "A seasonal ARIMA model for monthly data that captures yearly seasonal patterns.",
        "STL + ARIMA": "Decomposes the series with STL (trend, seasonal, remainder), fits ARIMA on the deseasonalized series, then adds the extrapolated seasonal component back in.",
        "ML Regression": "A lag-based regression model using Ridge to forecast future values from recent history.",
    }
    if PROPHET_AVAILABLE:
        method_descriptions["Prophet"] = "A time-series model from Prophet that captures weekly/yearly seasonality and trend."

    default_selection = ["Baseline"]
    if STATSMODELS_AVAILABLE:
        default_selection.extend([m for m in ["ARIMA", "SARIMA", "STL + ARIMA"] if m in available_methods])
    if SKLEARN_AVAILABLE:
        default_selection.append("ML Regression")

    selected_methods = st.multiselect(
        "Forecast methods to compare",
        list(available_methods.keys()),
        default=[m for m in default_selection if m in available_methods],
    )

    with st.expander("Method descriptions", expanded=False):
        for method_name in selected_methods:
            st.markdown(f"**{method_name}**: {method_descriptions.get(method_name, '')}")

    validation_months = st.slider(
        "Validation window (months)",
        min_value=1,
        max_value=min(12, max(1, len(monthly) - 3)),
        value=min(3, max(1, len(monthly) - 3)),
    )
    auto_select = st.checkbox("Select best model by validation error", value=True)

    trainer = monthly.iloc[:-validation_months]
    validation = monthly.iloc[-validation_months:]

    results = []
    errors = []
    validation_scores = []

    for method_name in selected_methods:
        method = available_methods[method_name]
        try:
            valid_forecast = method(trainer, validation_months)
            score = evaluate_forecast(validation, valid_forecast)
            validation_scores.append(
                {
                    "Method": method_name,
                    "RMSE": score["RMSE"],
                    "MAPE": score["MAPE"],
                }
            )
        except Exception as exc:
            errors.append(f"Validation {method_name}: {exc}")
            continue

    best_method_name = None
    if auto_select and validation_scores:
        best_method_name = min(validation_scores, key=lambda row: row["MAPE"])["Method"]
        st.success(f"Best model on validation set: {best_method_name}")

    if validation_scores:
        st.markdown("#### Validation scores")
        st.dataframe(pd.DataFrame(validation_scores).sort_values("MAPE"), use_container_width=True, hide_index=True)

    if best_method_name:
        methods_to_run = [best_method_name]
    else:
        methods_to_run = selected_methods

    for method_name in methods_to_run:
        method = available_methods[method_name]
        try:
            forecast_series = method(monthly, horizon)
            forecast_df = pd.DataFrame({"Month": forecast_series.index, "Value": forecast_series.values, "Method": method_name})
            results.append(forecast_df)
        except Exception as exc:
            errors.append(f"{method_name}: {exc}")

    actual = pd.DataFrame({"Month": monthly.index, "Value": monthly.values, "Method": "Actual"})
    display_df = pd.concat([actual] + results, ignore_index=True)

    title = f"{target} Forecast"
    if use_usd and target_col.endswith("_USD"):
        title = f"{target} Forecast (USD)"
    fig = px.line(display_df, x="Month", y="Value", color="Method", markers=True, title=title)
    st.plotly_chart(fig, use_container_width=True)

    if errors:
        st.warning("Some methods could not be computed:")
        for err in errors:
            st.write(err)

    if results:
        combined = pd.concat(results, ignore_index=True)
        st.dataframe(combined.sort_values(["Method", "Month"]).tail(horizon * len(results)), use_container_width=True, hide_index=True)
    else:
        st.info("No forecast methods could be compared.")

def parse_quarter_label(value) -> pd.Timestamp:
    """Turn a label such as 'Q1 2023' into the timestamp of the quarter start."""
    text = str(value).strip().upper()
    match = re.match(r"Q([1-4])\D*(\d{4})", text)
    if match is None:
        match = re.match(r"(\d{4})\D*Q([1-4])", text)
        if match is None:
            return pd.NaT
        year, quarter = int(match.group(1)), int(match.group(2))
    else:
        quarter, year = int(match.group(1)), int(match.group(2))
    return pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)


@st.cache_data(show_spinner=True)
def load_financials(uploaded_file_bytes: Optional[bytes], fallback_path: Optional[str]) -> pd.DataFrame:
    if uploaded_file_bytes is not None:
        excel_obj = io.BytesIO(uploaded_file_bytes)
    elif fallback_path:
        excel_obj = fallback_path
    else:
        raise FileNotFoundError("No financial summary file was uploaded and no local file was found.")

    df = pd.read_excel(excel_obj, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    if QUARTER_COL not in df.columns or REVENUE_COL not in df.columns:
        raise ValueError(f"The financial file must contain '{QUARTER_COL}' and '{REVENUE_COL}' columns.")

    df = df.rename(columns=COST_LABELS)
    cost_cols = [c for c in df.columns if c not in (QUARTER_COL, REVENUE_COL)]
    for col in cost_cols + [REVENUE_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Quarter Start"] = df[QUARTER_COL].apply(parse_quarter_label)
    df = df.dropna(subset=["Quarter Start"]).sort_values("Quarter Start").reset_index(drop=True)

    df["Total Costs"] = df[cost_cols].sum(axis=1)
    df["Net Profit"] = df[REVENUE_COL] - df["Total Costs"]
    df["Net Margin %"] = np.where(df[REVENUE_COL] > 0, df["Net Profit"] / df[REVENUE_COL] * 100, np.nan)
    df["Year"] = df["Quarter Start"].dt.year
    df.attrs["cost_cols"] = cost_cols
    return df


def find_local_financial_file() -> Optional[str]:
    for path in FINANCIAL_DATA_PATHS:
        if path.exists():
            return str(path)
    return None


def financial_theme() -> dict:
    """Streamlit's active theme, so chart chrome matches the surface it renders on."""
    base = None
    try:
        context_theme = getattr(st.context, "theme", None)
        base = getattr(context_theme, "type", None)
    except Exception:
        base = None
    if base is None:
        try:
            base = st.get_option("theme.base")
        except Exception:
            base = None
    return FINANCIAL_THEMES["dark" if str(base).lower() == "dark" else "light"]


def quarter_order(fin: pd.DataFrame) -> list:
    """Plotly orders categorical axes by trace appearance, which splits colour-grouped bars."""
    return fin.sort_values("Quarter Start")[QUARTER_COL].tolist()


def style_fig(fig: go.Figure, theme: dict, height: int = 400) -> go.Figure:
    """Recessive hairline grid, thin marks, generous padding."""
    fig.update_layout(
        template=theme["template"],
        height=height,
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font=dict(color=theme["text"], size=13),
        # Title sits at the top of the margin band; the legend clears it underneath.
        title=dict(font=dict(size=15), x=0, xanchor="left", y=0.97, yanchor="top"),
        margin=dict(l=10, r=10, t=104, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor=theme["axis"], tickfont=dict(color=theme["muted"]), title_text="")
    fig.update_yaxes(
        gridcolor=theme["grid"],
        griddash="solid",
        zerolinecolor=theme["axis"],
        linecolor=theme["axis"],
        tickfont=dict(color=theme["muted"]),
    )
    return fig


def financial_kpis(fin: pd.DataFrame):
    latest = fin.iloc[-1]
    ttm = fin.tail(4)
    prior_ttm = fin.iloc[-8:-4] if len(fin) >= 8 else None

    revenue_ttm = ttm[REVENUE_COL].sum()
    profit_ttm = ttm["Net Profit"].sum()
    margin_ttm = profit_ttm / revenue_ttm * 100 if revenue_ttm else np.nan

    revenue_delta = margin_delta = None
    if prior_ttm is not None and prior_ttm[REVENUE_COL].sum():
        prior_revenue = prior_ttm[REVENUE_COL].sum()
        prior_margin = prior_ttm["Net Profit"].sum() / prior_revenue * 100
        revenue_delta = f"{(revenue_ttm / prior_revenue - 1) * 100:+.1f}% vs prior 4Q"
        margin_delta = f"{margin_ttm - prior_margin:+.1f} pts vs prior 4Q"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue (last 4Q)", f"${revenue_ttm:,.0f}", revenue_delta)
    c2.metric("Costs (last 4Q)", f"${ttm['Total Costs'].sum():,.0f}")
    c3.metric("Net Profit (last 4Q)", f"${profit_ttm:,.0f}")
    c4.metric("Net Margin (last 4Q)", f"{margin_ttm:.1f}%", margin_delta)
    c5.metric(f"Latest Quarter ({latest[QUARTER_COL]})", f"${latest['Net Profit']:,.0f}", f"{latest['Net Margin %']:.1f}% margin")


def financial_trend_charts(fin: pd.DataFrame, theme: dict):
    c1, c2 = st.columns(2)
    order = {QUARTER_COL: quarter_order(fin)}

    with c1:
        trend = fin.melt(
            id_vars=QUARTER_COL,
            value_vars=[REVENUE_COL, "Total Costs"],
            var_name="Measure",
            value_name="Amount",
        )
        fig = px.line(
            trend,
            x=QUARTER_COL,
            y="Amount",
            color="Measure",
            markers=True,
            title="Revenue vs Total Costs by Quarter",
            category_orders=order,
            color_discrete_map={REVENUE_COL: theme["series"][0], "Total Costs": theme["series"][1]},
        )
        fig.update_traces(line_width=2, marker_size=8)
        # theme=None keeps Streamlit from overriding the palette and barmode set above.
        st.plotly_chart(style_fig(fig, theme), use_container_width=True, theme=None)

    with c2:
        profit = fin.copy()
        profit["Result"] = np.where(profit["Net Profit"] >= 0, "Profit", "Loss")
        fig = px.bar(
            profit,
            x=QUARTER_COL,
            y="Net Profit",
            color="Result",
            title="Net Profit by Quarter (loss quarters flagged)",
            category_orders={**order, "Result": ["Profit", "Loss"]},
            color_discrete_map={"Profit": STATUS_GOOD, "Loss": STATUS_CRITICAL},
        )
        fig.update_traces(marker_line_color=theme["surface"], marker_line_width=1)
        # theme=None keeps Streamlit from overriding the palette and barmode set above.
        st.plotly_chart(style_fig(fig, theme), use_container_width=True, theme=None)


def cost_structure_charts(fin: pd.DataFrame, theme: dict, cost_cols: list):
    c1, c2 = st.columns(2)

    with c1:
        stacked = fin.melt(id_vars=QUARTER_COL, value_vars=cost_cols, var_name="Cost Category", value_name="Amount")
        fig = px.bar(
            stacked,
            x=QUARTER_COL,
            y="Amount",
            color="Cost Category",
            title="Cost Structure by Quarter",
            category_orders={QUARTER_COL: quarter_order(fin), "Cost Category": cost_cols},
            color_discrete_sequence=theme["series"],
        )
        fig.update_traces(marker_line_color=theme["surface"], marker_line_width=1)
        fig.update_layout(barmode="stack")
        # theme=None keeps Streamlit from overriding the palette and barmode set above.
        st.plotly_chart(style_fig(fig, theme), use_container_width=True, theme=None)

    with c2:
        share = fin[[QUARTER_COL]].copy()
        for col in cost_cols:
            share[col] = fin[col] / fin[REVENUE_COL] * 100
        share = share.melt(id_vars=QUARTER_COL, value_vars=cost_cols, var_name="Cost Category", value_name="Share")
        fig = px.line(
            share,
            x=QUARTER_COL,
            y="Share",
            color="Cost Category",
            markers=True,
            title="Each Cost as % of Revenue (cost discipline)",
            category_orders={QUARTER_COL: quarter_order(fin), "Cost Category": cost_cols},
            color_discrete_sequence=theme["series"],
        )
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_yaxes(ticksuffix="%")
        # theme=None keeps Streamlit from overriding the palette and barmode set above.
        st.plotly_chart(style_fig(fig, theme), use_container_width=True, theme=None)


def margin_and_waterfall(fin: pd.DataFrame, theme: dict, cost_cols: list):
    c1, c2 = st.columns(2)

    with c1:
        fig = px.line(
            fin,
            x=QUARTER_COL,
            y="Net Margin %",
            markers=True,
            title="Net Margin % by Quarter",
            color_discrete_sequence=[theme["series"][0]],
        )
        fig.update_traces(line_width=2, marker_size=8)
        fig.add_hline(y=0, line_color=theme["axis"], line_width=1)
        fig.update_yaxes(ticksuffix="%")
        # theme=None keeps Streamlit from overriding the palette and barmode set above.
        st.plotly_chart(style_fig(fig, theme), use_container_width=True, theme=None)

    with c2:
        latest = fin.iloc[-1]
        labels = [REVENUE_COL] + cost_cols + ["Net Profit"]
        # Revenue is 'relative' from zero, not 'absolute', so it takes the increasing
        # colour instead of sharing the green/red reserved for the closing total.
        values = [latest[REVENUE_COL]] + [-latest[c] for c in cost_cols] + [0]
        fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["relative"] * (len(cost_cols) + 1) + ["total"],
                x=labels,
                y=values,
                connector=dict(line=dict(color=theme["axis"], width=1)),
                increasing=dict(marker=dict(color=theme["series"][0])),
                decreasing=dict(marker=dict(color=theme["series"][1])),
                totals=dict(marker=dict(color=STATUS_GOOD if latest["Net Profit"] >= 0 else STATUS_CRITICAL)),
                text=[f"${abs(v):,.0f}" if v else f"${latest['Net Profit']:,.0f}" for v in values],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig.update_layout(title=f"Revenue to Profit Bridge — {latest[QUARTER_COL]}", showlegend=False)
        fig = style_fig(fig, theme)
        # Headroom so the outside label on the revenue bar is not clipped.
        fig.update_yaxes(range=[0, latest[REVENUE_COL] * 1.18])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, use_container_width=True, theme=None)


def profit_levers(fin: pd.DataFrame, cost_cols: list) -> pd.DataFrame:
    """Quantify what one point of improvement on each lever is worth per quarter."""
    ttm = fin.tail(4)
    revenue = ttm[REVENUE_COL].sum() / len(ttm)
    rows = []
    for col in cost_cols:
        share = ttm[col].sum() / ttm[REVENUE_COL].sum() * 100
        rows.append(
            {
                "Lever": col,
                "Avg per Quarter": ttm[col].mean(),
                "% of Revenue": share,
                "Profit if cut 10%": ttm[col].mean() * 0.10,
            }
        )
    rows.append(
        {
            "Lever": "Revenue growth (+10%)",
            "Avg per Quarter": revenue,
            "% of Revenue": 100.0,
            "Profit if cut 10%": revenue * 0.10 * (1 - ttm["Cost of Sales"].sum() / ttm[REVENUE_COL].sum())
            if "Cost of Sales" in ttm.columns
            else np.nan,
        }
    )
    return pd.DataFrame(rows).sort_values("Profit if cut 10%", ascending=False)


def what_if_simulator(fin: pd.DataFrame, theme: dict, cost_cols: list):
    st.markdown("#### What-if simulator — model a path back to higher profit")
    st.caption(
        "Baseline carries the last 4 quarters forward unchanged: revenue keeps its recent growth rate, "
        "Cost of Sales stays at today's share of revenue, payroll and overhead keep their recent growth. "
        "The sliders start on that baseline, so whatever you move is the profit you gain or give up."
    )

    def avg_growth(series: pd.Series) -> float:
        values = series.dropna().astype(float)
        if len(values) < 2 or (values <= 0).any():
            return 0.0
        return float((values.iloc[-1] / values.iloc[0]) ** (1 / (len(values) - 1)) - 1)

    def clamp(value: float, low: float, high: float) -> float:
        return float(min(max(value, low), high))

    def snap(value: float, low: float, high: float) -> float:
        """Slider defaults land on a 0.5 step so the baseline is reproducible from the sliders."""
        return round(clamp(value, low, high) * 2) / 2

    ttm = fin.tail(4)
    last = fin.iloc[-1]
    fixed_cols = [c for c in cost_cols if c != "Cost of Sales"]

    if "Cost of Sales" in ttm.columns and ttm[REVENUE_COL].sum():
        cos_default = snap(ttm["Cost of Sales"].sum() / ttm[REVENUE_COL].sum() * 100, 30.0, 55.0)
    else:
        cos_default = 45.0
    revenue_default = snap(avg_growth(ttm[REVENUE_COL]) * 100, -10.0, 20.0)
    opex_default = snap(
        float(np.mean([avg_growth(ttm[c]) for c in fixed_cols])) * 100 if fixed_cols else 0.0, -5.0, 10.0
    )

    horizon = st.slider("Quarters to project", 2, 8, 4, key="fin_horizon")

    c1, c2, c3 = st.columns(3)
    revenue_growth = c1.slider("Revenue growth per quarter (%)", -10.0, 20.0, revenue_default, 0.5, key="fin_rev")
    cos_target = c2.slider("Cost of Sales as % of revenue", 30.0, 55.0, cos_default, 0.5, key="fin_cos")
    opex_growth = c3.slider("Payroll & overhead growth per quarter (%)", -5.0, 10.0, opex_default, 0.5, key="fin_opex")

    def project(rev_growth: float, cos_pct: float, fixed_growth: float) -> list:
        """Baseline and scenario share one model, so untouched sliders reproduce the baseline exactly."""
        revenue = last[REVENUE_COL]
        fixed = {col: last[col] for col in fixed_cols}
        profits = []
        for _ in range(horizon):
            revenue *= 1 + rev_growth / 100
            fixed = {col: value * (1 + fixed_growth / 100) for col, value in fixed.items()}
            profits.append(revenue - revenue * cos_pct / 100 - sum(fixed.values()))
        return profits

    baseline_profit = project(revenue_default, cos_default, opex_default)
    scenario_profit = project(revenue_growth, cos_target, opex_growth)

    quarters = []
    period = last["Quarter Start"]
    for _ in range(horizon):
        period = period + pd.offsets.QuarterBegin(1, startingMonth=1)
        quarters.append(f"Q{period.quarter} {period.year}")

    projection = pd.DataFrame(
        {
            QUARTER_COL: quarters * 2,
            "Net Profit": baseline_profit + scenario_profit,
            "Plan": ["Baseline (current trajectory)"] * horizon + ["Scenario (your levers)"] * horizon,
        }
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        fig = px.line(
            projection,
            x=QUARTER_COL,
            y="Net Profit",
            color="Plan",
            markers=True,
            title="Projected Net Profit — baseline vs scenario",
            color_discrete_map={
                "Baseline (current trajectory)": theme["series"][0],
                "Scenario (your levers)": theme["series"][1],
            },
        )
        fig.update_traces(line_width=2, marker_size=8)
        fig.add_hline(y=0, line_color=theme["axis"], line_width=1)
        # theme=None keeps Streamlit from overriding the palette and barmode set above.
        st.plotly_chart(style_fig(fig, theme), use_container_width=True, theme=None)
    with c2:
        gain = sum(scenario_profit) - sum(baseline_profit)
        st.metric("Scenario profit", f"${sum(scenario_profit):,.0f}")
        st.metric("Baseline profit", f"${sum(baseline_profit):,.0f}")
        st.metric("Difference", f"${gain:,.0f}", f"{'above' if gain >= 0 else 'below'} baseline")

    with st.expander("Projection table"):
        table = projection.copy()
        table["Net Profit"] = table["Net Profit"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(table, use_container_width=True, hide_index=True)


def growth_recommendations(fin: pd.DataFrame, cost_cols: list):
    st.markdown("### How to increase profit and sales")

    ttm = fin.tail(4)
    revenue_ttm = ttm[REVENUE_COL].sum()
    payroll_cols = [c for c in ("Payroll", "Benefits") if c in fin.columns]
    fixed_cols = [c for c in cost_cols if c != "Cost of Sales"]

    worst = fin.loc[fin["Net Margin %"].idxmin()]
    best = fin.loc[fin["Net Margin %"].idxmax()]
    cos_share = ttm["Cost of Sales"].sum() / revenue_ttm * 100 if "Cost of Sales" in ttm.columns else np.nan
    contribution_margin = (100 - cos_share) / 100 if pd.notna(cos_share) else np.nan
    fixed_per_quarter = ttm[fixed_cols].sum(axis=1).mean()
    breakeven = fixed_per_quarter / contribution_margin if contribution_margin else np.nan
    cos_point_value = revenue_ttm / 4 / 100

    worst_payroll_share = (
        fin.loc[fin["Net Margin %"].idxmin(), payroll_cols].sum()
        / worst[REVENUE_COL]
        * 100
        if payroll_cols
        else np.nan
    )
    best_payroll_share = (
        fin.loc[fin["Net Margin %"].idxmax(), payroll_cols].sum() / best[REVENUE_COL] * 100 if payroll_cols else np.nan
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Contribution margin", f"{contribution_margin * 100:.1f}%", "revenue left after Cost of Sales")
    c2.metric("Break-even revenue / quarter", f"${breakeven:,.0f}", "at today's fixed cost base")
    c3.metric("Value of 1pt lower Cost of Sales", f"${cos_point_value:,.0f}", "profit per quarter")

    st.markdown(
        "Ordered by how soon and how cheaply each can be attempted, not by size of "
        "impact: a distributor without a dedicated data or analytics team has to "
        "sequence its effort, not just diagnose it."
    )
    st.markdown(
        f"""
**1. Now, at no cost (owner: sales / account management).**
If a single scheme or partner drives a large, irregular spike in production (see Sales Performance and
Production Overview for which one), ask to shift its renewal into a quiet quarter instead of the peak. Part of
what looks like seasonality is often a renewal calendar the company can ask to change; this is a conversation,
not a program, and it reduces the concentration risk of one renewal moving unexpectedly.

**2. Now, at no cost (owner: whoever processes cancellations).**
Add a save-the-sale step, such as a call or a small offer, before a cancellation is finalized, and enforce
complete point-of-sale data capture at that same step. Track the current cancellation rate on the Production
Overview page; every point recovered there is acquisition effort that already happened.

**3. This quarter (owner: sales / marketing).**
Redirect marketing spend and account-executive push toward the weak quarters using plans already in the
catalogue, not new products: launching a new plan needs underwriting sign-off from the partner insurance
companies and is not something the distributor controls unilaterally.

**4. This quarter (owner: sales).**
Concentrate incentive spend on the top distributors, POS and account executives already producing the highest
Gross (see Sales Performance), plus a specific push on family and group bundling to raise members per policy.
Uses the sales channel the company already has rather than building a new one.

**5. Next (owner: finance / management, structural).**
Prioritize commission and ceded-premium renegotiation with the single highest-volume insurance company first
(see Sales Performance for which one), before attempting the rest. Cost of Sales is the largest single line at
{cos_share:.1f}% of revenue, and every point shaved is about ${cos_point_value:,.0f} of profit per quarter with
no extra sales needed, but a renegotiation ranked by volume is realistic where a blanket renegotiation across
every partner at once is not.

**6. Ongoing (owner: finance / management, structural).**
Add seasonal or contract staffing for peak quarters instead of restructuring core compensation, and use this
simulator itself as a standing quarterly check: the worst quarter ({worst[QUARTER_COL]},
{worst['Net Margin %']:.1f}% margin) was not a cost explosion, it was revenue falling to
${worst[REVENUE_COL]:,.0f} while payroll and benefits kept climbing to {worst_payroll_share:.0f}% of revenue,
against {best_payroll_share:.0f}% in the best quarter ({best[QUARTER_COL]}). If that share starts climbing back
toward {worst_payroll_share:.0f}%, treat it as an early warning rather than waiting for the quarter to close at
a loss.
"""
    )

    st.markdown("#### What each lever is worth per quarter")
    levers = profit_levers(fin, cost_cols)
    display = levers.copy()
    display["Avg per Quarter"] = display["Avg per Quarter"].apply(lambda x: f"${x:,.0f}")
    display["% of Revenue"] = display["% of Revenue"].apply(lambda x: f"{x:.1f}%")
    display["Profit if cut 10%"] = display["Profit if cut 10%"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
    )
    display = display.rename(columns={"Profit if cut 10%": "Profit impact of a 10% move"})
    st.dataframe(display, use_container_width=True, hide_index=True)


def financial_page():
    st.subheader("06 Company Financial Performance")
    st.caption(
        "Quarterly revenue, cost structure and profitability from the company financial summary, "
        "with levers for improving profit and sales."
    )

    fallback_path = find_local_financial_file()
    uploaded = st.file_uploader("Upload the company financial summary Excel file", type=["xlsx"], key="fin_upload")
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    if uploaded is None and fallback_path is None:
        st.warning("Upload `Company_Financial_Summary_2023_2026.xlsx` or place it beside the app.")
        return

    try:
        fin = load_financials(uploaded_bytes, fallback_path)
    except Exception as exc:
        st.error(f"Could not read the financial summary: {exc}")
        return

    cost_cols = fin.attrs.get("cost_cols", [])
    if not cost_cols:
        st.warning("No cost columns were found in the financial summary.")
        return

    theme = financial_theme()

    years = sorted(fin["Year"].unique().tolist())
    selected_years = st.multiselect("Years", years, default=years, key="fin_years")
    if selected_years:
        fin = fin[fin["Year"].isin(selected_years)].reset_index(drop=True)
    if fin.empty:
        st.warning("No quarters match the selected years.")
        return

    financial_kpis(fin)
    st.markdown("---")
    financial_trend_charts(fin, theme)
    cost_structure_charts(fin, theme, cost_cols)
    margin_and_waterfall(fin, theme, cost_cols)

    with st.expander("Table view — quarterly financials"):
        table = fin[[QUARTER_COL] + cost_cols + [REVENUE_COL, "Total Costs", "Net Profit", "Net Margin %"]].copy()
        for col in cost_cols + [REVENUE_COL, "Total Costs", "Net Profit"]:
            table[col] = table[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
        table["Net Margin %"] = table["Net Margin %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("---")
    what_if_simulator(fin, theme, cost_cols)
    st.markdown("---")
    growth_recommendations(fin, cost_cols)


def main():
    st.title(APP_TITLE)
    st.markdown("This app shows EDA dashboards and adds a forecasting phase for future production/sales.")
    fallback_path = find_local_data_file()
    uploaded = st.sidebar.file_uploader("Upload production Excel file", type=["xlsx"])
    uploaded_bytes = uploaded.getvalue() if uploaded is not None else None
    if uploaded is None and fallback_path is None:
        st.warning("Upload the Excel dataset or place it inside a local `data/` folder.")
        st.stop()
    with st.spinner("Loading and cleaning data..."):
        df = load_excel(uploaded_bytes, fallback_path)
    safe_df = df.drop(columns=[c for c in PERSONAL_FIELDS if c in df.columns], errors="ignore")
    filtered = apply_filters(safe_df)
    st.sidebar.markdown("---")
    st.sidebar.write(f"Rows after filters: **{len(filtered):,}**")
    st.sidebar.markdown("---")
    use_usd = st.sidebar.checkbox("Show amounts in USD (convert)", value=False)
    rates_file = st.sidebar.file_uploader("Upload currency rates CSV (Currency,Rate)", type=["csv"])
    rates_text = st.sidebar.text_area(
        "Or paste currency:rate pairs (comma separated)", value="USD:1, LBP:0.00066, EUR:1.08", height=80
    )
    rates = {}
    if rates_file is not None:
        try:
            rates_df = pd.read_csv(rates_file)
            if "Currency" in rates_df.columns and "Rate" in rates_df.columns:
                rates = {
                    str(row["Currency"]).strip(): float(row["Rate"])
                    for _, row in rates_df.iterrows()
                }
        except Exception:
            st.sidebar.warning("Could not read rates CSV; falling back to manual input.")
    if not rates:
        rates = parse_rates_text(rates_text)
    df_for_display = compute_usd_columns(filtered, rates) if use_usd else filtered
    page = st.sidebar.radio(
        "Pages",
        [
            "01 Production Overview",
            "02 Sales Performance",
            "03 Geography & Product",
            "04 Data Quality",
            "05 Forecasting Preview",
            "06 Financial Performance",
        ],
    )
    if page == "01 Production Overview":
        overview_page(df_for_display, use_usd=use_usd)
    elif page == "02 Sales Performance":
        sales_page(df_for_display, use_usd=use_usd)
    elif page == "03 Geography & Product":
        geography_product_page(df_for_display, use_usd=use_usd)
    elif page == "04 Data Quality":
        data_quality_page(df_for_display, use_usd=use_usd)
    elif page == "05 Forecasting Preview":
        forecasting_page(df_for_display, use_usd=use_usd)
    elif page == "06 Financial Performance":
        financial_page()

    if page != "06 Financial Performance":
        with st.expander("View cleaned filtered data sample"):
            st.dataframe(df_for_display.head(200), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
