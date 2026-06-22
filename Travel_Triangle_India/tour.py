"""
Kerala Tour Packages — Streamlit Dashboard
===========================================
This dashboard is built by reusing the data-cleaning and analysis logic from
the `Kerala_Tour.ipynb` notebook (city splitting, numeric extraction from
Duration/Discount/Ratings, price-per-day, outlier detection, destination and
correlation analysis). Static matplotlib/seaborn charts from the notebook
have been re-created as interactive Plotly charts.

Run with:
    pip install -r requirements.txt
    streamlit run tour.py
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG — must be the first Streamlit call
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kerala Tour Packages Dashboard",
    page_icon="🌴",
    layout="wide",
)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
# The notebook scrapes data live from the web rather than saving a CSV, but
# this app expects the resulting dataset to sit next to it as a file. We try
# a few likely names first, then fall back to "any csv/xlsx in this folder".
CANDIDATE_FILENAMES = [
    "kerala_tour_packages.csv",
    "Kerala_Tour.csv",
    "kerala_tour.csv",
    "Kerala_Tour_Packages.csv",
    "data.csv",
]
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_dataset_path():
    """Look for a known filename first, then any csv/xlsx file in the folder."""
    for name in CANDIDATE_FILENAMES:
        path = os.path.join(APP_DIR, name)
        if os.path.exists(path):
            return path
    # Fallback: grab the first csv/xlsx sitting alongside the app
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        matches = glob.glob(os.path.join(APP_DIR, ext))
        if matches:
            return matches[0]
    return None


# ----------------------------------------------------------------------------
# DATA LOADING + CLEANING (mirrors the notebook's preprocessing steps)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and cleaning dataset...")
def load_data(path: str):
    """Load the raw file and apply the same cleaning steps as the notebook:
    - split the 'Cities' string into 1D/2D/3D city columns
    - fill missing Old Prices with the column mean
    - extract numeric Days/Nights from Duration
    - extract numeric Discount % and Ratings
    - compute Price per Day
    - flag price outliers using the IQR method
    All steps are wrapped defensively so the app still works if some
    expected columns are missing from the dataset.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df.columns = [c.strip() for c in df.columns]

    # --- Split the Cities string into 1D / 2D / 3D city columns (notebook cell 8) ---
    if "Cities" in df.columns:
        def split_cities(city_string):
            if not isinstance(city_string, str):
                return pd.Series(["", "", ""])
            import re
            matches = re.findall(r"([^→]+?)\s*\((\d+)D\)", city_string)
            one_day, two_day, three_day = [], [], []
            for city, day in matches:
                city = city.strip()
                if day == "1":
                    one_day.append(city)
                elif day == "2":
                    two_day.append(city)
                elif day == "3":
                    three_day.append(city)
            return pd.Series([", ".join(one_day), ", ".join(two_day), ", ".join(three_day)])

        df[["1D Cities", "2D Cities", "3D Cities"]] = df["Cities"].apply(split_cities)

    # --- Fill missing Old Prices with the mean (notebook cell 13) ---
    if "Old Prices" in df.columns:
        df["Old Prices"] = pd.to_numeric(df["Old Prices"], errors="coerce")
        df["Old Prices"] = df["Old Prices"].fillna(df["Old Prices"].mean())

    if "New Prices" in df.columns:
        df["New Prices"] = pd.to_numeric(df["New Prices"], errors="coerce")

    # --- Extract numeric Days / Nights from Duration (notebook cell 24) ---
    if "Duration" in df.columns:
        df["Days"] = df["Duration"].astype(str).str.extract(r"(\d+)\s*Days?").astype(float)
        df["Nights"] = df["Duration"].astype(str).str.extract(r"(\d+)\s*Nights?").astype(float)

    # --- Extract numeric Discount % (notebook cell 24) ---
    if "Discount" in df.columns:
        df["Discount_num"] = df["Discount"].astype(str).str.extract(r"(\d+)").astype(float)

    # --- Extract numeric Ratings (notebook cell 24) ---
    if "Ratings" in df.columns:
        df["Ratings_num"] = df["Ratings"].astype(str).str.extract(r"(\d+)").astype(float)

    # --- Price per day (notebook cell 32) ---
    if "New Prices" in df.columns and "Days" in df.columns:
        df["Price_per_day"] = df["New Prices"] / df["Days"].replace(0, np.nan)

    # --- Outlier flag using IQR method (notebook cell 39) ---
    if "New Prices" in df.columns and df["New Prices"].notna().any():
        q1 = df["New Prices"].quantile(0.25)
        q3 = df["New Prices"].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df["Price_Outlier"] = (df["New Prices"] < lower) | (df["New Prices"] > upper)

    # --- Try to detect any date-like columns for filtering later ---
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def get_column_groups(df: pd.DataFrame):
    """Classify columns so charts/filters can adapt automatically to the dataset."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    date_cols = df.select_dtypes(include="datetime64[ns]").columns.tolist()
    categorical_cols = [
        c for c in df.select_dtypes(include="object").columns
        if df[c].nunique(dropna=True) <= 50  # skip free-text/high-cardinality columns
    ]
    return numeric_cols, categorical_cols, date_cols


# ----------------------------------------------------------------------------
# LOAD DATA (with graceful error handling)
# ----------------------------------------------------------------------------
dataset_path = _find_dataset_path()

if dataset_path is None:
    st.title("🌴 Kerala Tour Packages Dashboard")
    st.error(
        "No dataset file (.csv / .xlsx) was found in the app's folder. "
        "Please export the scraped DataFrame from `Kerala_Tour.ipynb` "
        "(e.g. `df.to_csv('kerala_tour_packages.csv', index=False)`) "
        "and place it in the same directory as this app, or upload one below."
    )
    uploaded = st.file_uploader("Upload a dataset file", type=["csv", "xlsx", "xls"])
    if uploaded is None:
        st.stop()
    temp_path = os.path.join(APP_DIR, "_uploaded_dataset.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    dataset_path = temp_path

try:
    df = load_data(dataset_path)
except Exception as e:
    st.error(f"Failed to load/process the dataset: {e}")
    st.stop()

if df.empty:
    st.warning("The dataset loaded successfully but contains no rows.")
    st.stop()

numeric_cols, categorical_cols, date_cols = get_column_groups(df)

# ----------------------------------------------------------------------------
# SIDEBAR — dataset overview, filters, navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🌴 Kerala Tour")
st.sidebar.caption(f"Source file: `{os.path.basename(dataset_path)}`")

# --- Dataset overview ---
with st.sidebar.expander("📋 Dataset Overview", expanded=True):
    st.write(f"**Rows:** {df.shape[0]:,}")
    st.write(f"**Columns:** {df.shape[1]:,}")
    missing_total = int(df.isnull().sum().sum())
    st.write(f"**Missing values:** {missing_total:,}")
    st.write(f"**Numeric columns:** {len(numeric_cols)}")
    st.write(f"**Categorical columns:** {len(categorical_cols)}")

# --- Filters ---
st.sidebar.subheader("🔍 Filters")
filtered_df = df.copy()

# Categorical filters — one multiselect per categorical column (capped to a
# handful of the most useful columns so the sidebar doesn't get too long)
filter_candidates = [c for c in categorical_cols if c not in ("Cities", "1D Cities", "2D Cities", "3D Cities")]
for col in filter_candidates[:6]:
    options = sorted(df[col].dropna().unique().tolist())
    if 1 < len(options) <= 50:
        selected = st.sidebar.multiselect(f"{col}", options, default=[])
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

# Date filters (if any date columns exist in the dataset)
for col in date_cols:
    valid_dates = df[col].dropna()
    if valid_dates.empty:
        continue
    min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
    date_range = st.sidebar.date_input(f"{col} range", value=(min_d, max_d))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start, end = date_range
        filtered_df = filtered_df[
            (filtered_df[col].dt.date >= start) & (filtered_df[col].dt.date <= end)
        ]

# Numeric range filter for price, if present, since it's the most common ask
if "New Prices" in numeric_cols:
    p_min, p_max = float(df["New Prices"].min()), float(df["New Prices"].max())
    if p_min < p_max:
        price_range = st.sidebar.slider(
            "Price range (₹)", min_value=p_min, max_value=p_max, value=(p_min, p_max)
        )
        filtered_df = filtered_df[
            (filtered_df["New Prices"] >= price_range[0])
            & (filtered_df["New Prices"] <= price_range[1])
        ]

if st.sidebar.button("Reset filters"):
    st.rerun()

# --- Navigation ---
st.sidebar.subheader("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Data Explorer", "Visualizations", "Destination Insights", "Top & Bottom Packages", "Correlation"],
)

st.sidebar.divider()
st.sidebar.caption(f"Showing **{len(filtered_df):,}** of **{len(df):,}** rows after filters.")

# ----------------------------------------------------------------------------
# MAIN AREA
# ----------------------------------------------------------------------------
st.title("🌴 Kerala Tour Packages Dashboard")
st.write(
    "Explore tour packages for Kerala scraped from TravelTriangle — pricing, "
    "duration, ratings, discounts, and destination patterns. Use the sidebar "
    "to filter the data and navigate between sections."
)

# ============================== OVERVIEW PAGE ===============================
if page == "Overview":
    st.subheader("Key Metrics")

    # Build metric cards only for columns that actually exist, so the layout
    # adapts automatically to whatever dataset is loaded.
    metrics = []
    metrics.append(("Total Packages", f"{len(filtered_df):,}"))
    if "New Prices" in filtered_df.columns:
        metrics.append(("Avg. Current Price", f"₹{filtered_df['New Prices'].mean():,.0f}"))
    if "Old Prices" in filtered_df.columns:
        metrics.append(("Avg. Original Price", f"₹{filtered_df['Old Prices'].mean():,.0f}"))
    if "Discount_num" in filtered_df.columns:
        metrics.append(("Avg. Discount", f"{filtered_df['Discount_num'].mean():.1f}%"))
    if "Ratings_num" in filtered_df.columns:
        metrics.append(("Avg. Rating", f"{filtered_df['Ratings_num'].mean():.1f} ★"))
    if "Days" in filtered_df.columns:
        metrics.append(("Avg. Duration", f"{filtered_df['Days'].mean():.1f} days"))

    cols = st.columns(min(len(metrics), 6) or 1)
    for c, (label, value) in zip(cols, metrics):
        c.metric(label, value)

    st.divider()
    st.subheader("Data Preview")
    st.dataframe(filtered_df.head(20), use_container_width=True)

    st.subheader("Summary Statistics")
    if numeric_cols:
        st.dataframe(filtered_df[numeric_cols].describe().T, use_container_width=True)
    else:
        st.info("No numeric columns available for summary statistics.")

# ============================ DATA EXPLORER PAGE ============================
elif page == "Data Explorer":
    st.subheader("Full Data Preview")
    st.dataframe(filtered_df, use_container_width=True)

    st.subheader("Missing Values by Column")
    missing = filtered_df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in the current filtered data. 🎉")
    else:
        fig = px.bar(
            missing.sort_values(ascending=False),
            labels={"index": "Column", "value": "Missing Count"},
            title="Missing Values per Column",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Statistics")
    if numeric_cols:
        st.dataframe(filtered_df[numeric_cols].describe().T, use_container_width=True)

# ============================= VISUALIZATIONS PAGE ===========================
elif page == "Visualizations":
    st.subheader("Distributions (from the notebook's univariate analysis)")

    col1, col2 = st.columns(2)

    # Price distributions (notebook cells 16-17)
    with col1:
        if "Old Prices" in filtered_df.columns:
            fig = px.histogram(
                filtered_df, x="Old Prices", nbins=36, title="Previous Price Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "New Prices" in filtered_df.columns:
            fig = px.histogram(
                filtered_df, x="New Prices", nbins=36, title="Current Price Distribution",
                color_discrete_sequence=["indianred"],
            )
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # Duration distribution (notebook cells 18-19)
    with col3:
        if "Duration" in filtered_df.columns:
            dur_counts = filtered_df["Duration"].value_counts().reset_index()
            dur_counts.columns = ["Duration", "Count"]
            fig = px.bar(dur_counts, x="Duration", y="Count", title="Duration Distribution")
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if "Duration" in filtered_df.columns:
            fig = px.pie(dur_counts, names="Duration", values="Count", title="Duration Share")
            st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    # Ratings pie (notebook cell 20)
    with col5:
        if "Ratings" in filtered_df.columns:
            rat_counts = filtered_df["Ratings"].value_counts().reset_index()
            rat_counts.columns = ["Ratings", "Count"]
            fig = px.pie(rat_counts, names="Ratings", values="Count", title="Ratings Distribution")
            st.plotly_chart(fig, use_container_width=True)

    # Discount distribution (notebook cell 21)
    with col6:
        if "Discount" in filtered_df.columns:
            disc_counts = filtered_df["Discount"].value_counts().reset_index()
            disc_counts.columns = ["Discount", "Count"]
            fig = px.bar(disc_counts, x="Discount", y="Count", title="Discount Distribution")
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Generic Explorer (auto-adapts to any column in this dataset)")
    g1, g2 = st.columns(2)
    with g1:
        if numeric_cols:
            num_choice = st.selectbox("Numeric column", numeric_cols, key="num_choice")
            fig = px.histogram(filtered_df, x=num_choice, nbins=30, title=f"{num_choice} Distribution")
            st.plotly_chart(fig, use_container_width=True)
    with g2:
        if categorical_cols:
            cat_choice = st.selectbox("Categorical column", categorical_cols, key="cat_choice")
            counts = filtered_df[cat_choice].value_counts().reset_index()
            counts.columns = [cat_choice, "Count"]
            fig = px.bar(counts, x=cat_choice, y="Count", title=f"{cat_choice} Counts")
            st.plotly_chart(fig, use_container_width=True)

# =========================== DESTINATION INSIGHTS PAGE =======================
elif page == "Destination Insights":
    st.subheader("Most Frequently Visited Destinations")

    if all(c in filtered_df.columns for c in ["1D Cities", "2D Cities", "3D Cities"]):
        all_cities = pd.concat([
            filtered_df["1D Cities"].astype(str).str.split(", "),
            filtered_df["2D Cities"].astype(str).str.split(", "),
            filtered_df["3D Cities"].astype(str).str.split(", "),
        ])
        city_list = [c.strip() for sub in all_cities.dropna() for c in sub if c.strip() not in ("", "nan")]
        if city_list:
            city_counts = pd.Series(city_list).value_counts().head(15).reset_index()
            city_counts.columns = ["City", "Packages"]
            fig = px.bar(
                city_counts, x="City", y="Packages",
                title="Top 15 Most Frequently Visited Destinations", color_discrete_sequence=["teal"],
            )
            st.plotly_chart(fig, use_container_width=True)

            # Average price per top destination (notebook cell 29)
            if "Cities" in filtered_df.columns and "New Prices" in filtered_df.columns:
                top_cities = city_counts["City"].head(10)
                price_by_city = {}
                for city in top_cities:
                    mask = filtered_df["Cities"].astype(str).str.contains(city, regex=False, na=False)
                    if mask.any():
                        price_by_city[city] = filtered_df.loc[mask, "New Prices"].mean()
                if price_by_city:
                    price_series = pd.Series(price_by_city).sort_values(ascending=False).reset_index()
                    price_series.columns = ["City", "Avg Price"]
                    fig = px.bar(
                        price_series, x="City", y="Avg Price",
                        title="Average Package Price by Top Destination", color_discrete_sequence=["coral"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No destination/city data available in the current filter selection.")
    else:
        st.info("This dataset has no 'Cities' breakdown columns to analyze.")

# ========================= TOP & BOTTOM PACKAGES PAGE =========================
elif page == "Top & Bottom Packages":
    st.subheader("Cheapest, Most Expensive & Best-Value Packages")

    display_cols = [c for c in ["Package Name", "Duration", "New Prices", "Ratings", "Discount"] if c in filtered_df.columns]

    if "New Prices" in filtered_df.columns and display_cols:
        st.markdown("**Top 10 Cheapest Packages**")
        st.dataframe(filtered_df.nsmallest(10, "New Prices")[display_cols], use_container_width=True)

        st.markdown("**Top 10 Most Expensive Packages**")
        st.dataframe(filtered_df.nlargest(10, "New Prices")[display_cols], use_container_width=True)

        if "Ratings_num" in filtered_df.columns and "Price_per_day" in filtered_df.columns:
            st.markdown("**Top 10 Best Value (lowest price/day, rating 4+)**")
            value_cols = [c for c in ["Package Name", "Duration", "New Prices", "Price_per_day", "Ratings"] if c in filtered_df.columns]
            best_value = filtered_df[filtered_df["Ratings_num"] >= 4].nsmallest(10, "Price_per_day")[value_cols]
            st.dataframe(best_value, use_container_width=True)
    else:
        st.info("Price data isn't available to rank packages.")

    st.divider()
    st.subheader("Price Outliers (IQR method)")
    if "Price_Outlier" in filtered_df.columns:
        outliers = filtered_df[filtered_df["Price_Outlier"]]
        st.write(f"**{len(outliers)}** of **{len(filtered_df)}** packages flagged as price outliers.")
        outlier_cols = [c for c in ["Package Name", "New Prices", "Duration"] if c in filtered_df.columns]
        if not outliers.empty and outlier_cols:
            st.dataframe(
                outliers[outlier_cols].sort_values("New Prices", ascending=False).head(15),
                use_container_width=True,
            )
    else:
        st.info("Outlier flag not available for this dataset.")

# =============================== CORRELATION PAGE ============================
elif page == "Correlation":
    st.subheader("Relationships Between Numeric Features")

    # Price per day, discount vs price, rating vs price (notebook cells 32-34)
    c1, c2 = st.columns(2)
    with c1:
        if "Price_per_day" in filtered_df.columns:
            fig = px.histogram(
                filtered_df, x="Price_per_day", nbins=36, title="Price per Day Distribution",
                color_discrete_sequence=["purple"],
            )
            st.plotly_chart(fig, use_container_width=True)
        if "New Prices" in filtered_df.columns and "Discount_num" in filtered_df.columns:
            fig = px.scatter(
                filtered_df, x="New Prices", y="Discount_num", opacity=0.4,
                title="Discount % vs Current Price",
            )
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "Ratings_num" in filtered_df.columns and "New Prices" in filtered_df.columns:
            fig = px.scatter(
                filtered_df, x="Ratings_num", y="New Prices", opacity=0.4,
                title="Rating vs Current Price", color_discrete_sequence=["darkgreen"],
            )
            st.plotly_chart(fig, use_container_width=True)
        if "Package Type" in filtered_df.columns and "New Prices" in filtered_df.columns:
            type_summary = (
                filtered_df.groupby("Package Type")["New Prices"].mean()
                .sort_values(ascending=False).reset_index()
            )
            fig = px.bar(
                type_summary, x="Package Type", y="New Prices",
                title="Average Price by Package Type",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Correlation Heatmap")
    corr_candidates = [c for c in ["New Prices", "Old Prices", "Days", "Discount_num", "Ratings_num", "Price_per_day"] if c in filtered_df.columns]
    if len(corr_candidates) >= 2:
        corr = filtered_df[corr_candidates].corr()
        fig = px.imshow(
            corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
            title="Correlation Matrix (Cleaned Numeric Features)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough numeric columns available to compute a correlation matrix.")

st.divider()
st.caption("Dashboard built from `Kerala_Tour.ipynb` — data originally scraped from TravelTriangle.")
