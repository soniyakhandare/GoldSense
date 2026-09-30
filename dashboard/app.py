import joblib
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GoldSense | Gold Analytics",
    page_icon="G",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background-color: #0E1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #151922;
    }

    /* Main title */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 15px;
        margin-bottom: 30px;
    }

    /* KPI cards */
    .metric-card {
        background-color: #171B24;
        border: 1px solid #292F3B;
        border-radius: 12px;
        padding: 20px;
        min-height: 120px;
    }

    .metric-label {
        color: #9CA3AF;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
    }

    .metric-change {
        color: #22C55E;
        font-size: 13px;
        margin-top: 5px;
    }

    /* Section headings */
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 12px;
        margin-top: 40px;
        padding: 20px;
    }

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# REAL GOLD DATA
# --------------------------------------------------

# Connect to GoldSense database
conn = sqlite3.connect("database/goldsense.db")

# Load real gold data
df = pd.read_sql_query(
    "SELECT * FROM gold_prices",
    conn
)

conn.close()

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

linear_model = joblib.load("models/linear_regression.joblib")
random_forest_model = joblib.load("models/random_forest.joblib")
gradient_boosting_model = joblib.load("models/gradient_boosting.joblib")
arima_model = joblib.load("models/arima_1_0_1.joblib")
sarima_model = joblib.load("models/sarima_1_0_1_1_0_1_30.joblib")


arima_model = joblib.load("models/arima_1_0_1.joblib")


# ----------------------------
# ----------------------

# SIDEBAR
# --------------------------------------------------

st.sidebar.markdown(
    "<h1 style='font-size:28px;'>GoldSense</h1>",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "<p style='color:#9CA3AF;'>Gold Price Intelligence</p>",
    unsafe_allow_html=True
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "EDA & Trends",
        "Market Analytics",
        "Anomaly Detection",
        "ML Prediction",
        "Forecasting",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown(
    "<p style='color:#6B7280;font-size:12px;'>"
    "GoldSense Analytics System<br>"
    "MSc Big Data Analytics"
    "</p>",
    unsafe_allow_html=True
)


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

if page == "Dashboard":

    # Header
    st.markdown(
        "<div class='main-title'>Gold Price Dashboard</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>"
        "Market overview, price trends and analytical insights"
        "</div>",
        unsafe_allow_html=True
    )

    # Date Filter
    st.markdown("### Filter by Date")

    date_range = st.date_input(
    "Select Date Range",
    value=(df["Date"].min().date(), df["Date"].max().date())
)

    filtered_df = df[
    (df["Date"].dt.date >= date_range[0]) &
    (df["Date"].dt.date <= date_range[1])
]
    # --------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------

    latest_price = filtered_df["Price"].iloc[-1]
    average_price = filtered_df["Price"].mean()
    highest_price = filtered_df["Price"].max()
    lowest_price = filtered_df["Price"].min()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Latest Price</div>
                <div class="metric-value">₹{latest_price:,.0f}</div>
                <div class="metric-change">Current market value</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Average Price</div>
                <div class="metric-value">₹{average_price:,.0f}</div>
                <div class="metric-change">Average over period</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Highest Price</div>
                <div class="metric-value">₹{highest_price:,.0f}</div>
                <div class="metric-change">Period high</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Lowest Price</div>
                <div class="metric-value">₹{lowest_price:,.0f}</div>
                <div class="metric-change">Period low</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------
    # PRICE CHART
    # --------------------------------------------------

    st.markdown(
        "<div class='section-title'>Gold Price Trend</div>",
        unsafe_allow_html=True
    )

    indicators = st.multiselect(
        "Select Indicators",
        ["Gold Price", "30-Day Moving Average"],
        default=["Gold Price", "30-Day Moving Average"]
)

    fig = go.Figure()

    if "Gold Price" in indicators:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["Price"],
                mode="lines",
                name="Gold Price",
                line=dict(width=2)
            )
        )
    
    if "30-Day Moving Average" in indicators:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["Date"],
                y=filtered_df["MA_30"],
                mode="lines",
                name="30-Day Moving Average",
                line=dict(width=2, dash="dash")
            )
        )

    fig.update_layout(
        height=430,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            y=1.02,
            x=0
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

   # --------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------

    st.markdown(
        "<div class='section-title'>Dataset Summary</div>",
        unsafe_allow_html=True
    )

    summary = pd.DataFrame({
        "Metric": [
            "Records",
            "Start Date",
            "End Date",
            "Average Price",
            "Highest Price",
            "Lowest Price"
        ],
        "Value": [
            len(filtered_df),
            filtered_df["Date"].min().strftime("%d %b %Y"),
            filtered_df["Date"].max().strftime("%d %b %Y"),
            f"₹{average_price:,.0f}",
            f"₹{highest_price:,.0f}",
            f"₹{lowest_price:,.0f}"
        ]
    })

    summary["Value"] = summary["Value"].astype(str)

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )
        # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.markdown(
        """
        <div class="footer">
            GoldSense • Gold Price Analytics & Forecasting System
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# OTHER PAGES
# --------------------------------------------------

elif page == "EDA & Trends":

    # Load Person 1's cleaned dataset for EDA
    cleaned_df = pd.read_csv("data/cleaned_gold.csv")
    cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"])
    cleaned_df = cleaned_df.sort_values("Date").reset_index(drop=True)

    st.markdown(
        "<div class='main-title'>Exploratory Data Analysis</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>"
        "Understanding historical gold-price behaviour and market patterns"
        "</div>",
        unsafe_allow_html=True
    )

    # -------------------------------
    # EDA SUMMARY
    # -------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(cleaned_df):,}")

    with col2:
        st.metric("Average Price", f"₹{cleaned_df['Price'].mean():,.0f}")

    with col3:
        st.metric("Highest Price", f"₹{cleaned_df['Price'].max():,.0f}")

    with col4:
        st.metric("Lowest Price", f"₹{cleaned_df['Price'].min():,.0f}")

    # -------------------------------
    # PRICE TREND
    # -------------------------------

    st.markdown(
        "<div class='section-title'>Historical Gold Price</div>",
        unsafe_allow_html=True
    )

    price_fig = go.Figure()

    price_fig.add_trace(
        go.Scatter(
            x=cleaned_df["Date"],
            y=cleaned_df["Price"],
            mode="lines",
            name="Gold Price",
            line=dict(width=2)
        )
    )

    price_fig.update_layout(
        height=420,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Gold Price",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(price_fig, use_container_width=True)

    # -------------------------------
    # MONTHLY AVERAGE + DISTRIBUTION
    # -------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div class='section-title'>Monthly Average Price</div>",
            unsafe_allow_html=True
        )

        monthly = (
            cleaned_df.set_index("Date")["Price"]
            .resample("ME")
            .mean()
            .reset_index()
        )

        monthly_fig = go.Figure()
        monthly_fig.add_trace(
            go.Bar(
                x=monthly["Date"],
                y=monthly["Price"],
                name="Monthly Average"
            )
        )

        monthly_fig.update_layout(
            height=350,
            template="plotly_dark",
            xaxis_title="Month",
            yaxis_title="Average Price",
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(monthly_fig, use_container_width=True)

    with col2:
        st.markdown(
            "<div class='section-title'>Price Distribution</div>",
            unsafe_allow_html=True
        )

        distribution_fig = go.Figure()
        distribution_fig.add_trace(
            go.Histogram(
                x=cleaned_df["Price"],
                nbinsx=30,
                name="Price Distribution"
            )
        )

        distribution_fig.update_layout(
            height=350,
            template="plotly_dark",
            xaxis_title="Gold Price",
            yaxis_title="Frequency",
            margin=dict(l=10, r=10, t=20, b=10)
        )

        st.plotly_chart(distribution_fig, use_container_width=True)

    # -------------------------------
    # DAILY RETURNS
    # -------------------------------

    st.markdown(
        "<div class='section-title'>Daily Returns</div>",
        unsafe_allow_html=True
    )

    returns_fig = go.Figure()
    returns_fig.add_trace(
        go.Scatter(
            x=cleaned_df["Date"],
            y=cleaned_df["Daily_Return"],
            mode="lines",
            name="Daily Return",
            line=dict(width=1.5)
        )
    )

    returns_fig.update_layout(
        height=350,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Daily Return (%)",
        hovermode="x unified",
        margin=dict(l=10, r=10, t=20, b=10)
    )

    st.plotly_chart(returns_fig, use_container_width=True)

    # -------------------------------
    # DATASET PREVIEW
    # -------------------------------

    st.markdown(
        "<div class='section-title'>Cleaned Dataset Preview</div>",
        unsafe_allow_html=True
    )

    st.dataframe(
        cleaned_df.head(10),
        use_container_width=True,
        hide_index=True
    )


elif page == "Market Analytics":

    # Load Person 2's feature-engineered dataset
    feature_df = pd.read_csv("data/feature_engineered_gold.csv")
    feature_df["Date"] = pd.to_datetime(feature_df["Date"])
    feature_df = feature_df.sort_values("Date").reset_index(drop=True)

    st.markdown(
        "<div class='main-title'>Market Analytics</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='subtitle'>Analyzing returns, moving averages, volatility and momentum</div>",
        unsafe_allow_html=True
    )

    # Latest values for KPI cards
    latest = feature_df.iloc[-1]
    avg_return = feature_df["Daily_Return"].mean()
    latest_ma7 = latest["MA_7"]
    latest_ma30 = latest["MA_30"]
    latest_volatility = latest["Volatility_30"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Average Daily Return", f"{avg_return:.2f}%")

    with col2:
        st.metric(
            "Current MA 7",
            f"₹{latest_ma7:,.0f}" if pd.notna(latest_ma7) else "N/A"
        )

    with col3:
        st.metric(
            "Current MA 30",
            f"₹{latest_ma30:,.0f}" if pd.notna(latest_ma30) else "N/A"
        )

    with col4:
        st.metric(
            "30-Day Volatility",
            f"{latest_volatility:.2f}" if pd.notna(latest_volatility) else "N/A"
        )

    # Price and moving averages
    st.markdown(
        "<div class='section-title'>Price vs Moving Averages</div>",
        unsafe_allow_html=True
    )

    fig_ma = go.Figure()
    fig_ma.add_trace(
        go.Scatter(
            x=feature_df["Date"],
            y=feature_df["Price"],
            name="Gold Price",
            mode="lines"
        )
    )
    fig_ma.add_trace(
        go.Scatter(
            x=feature_df["Date"],
            y=feature_df["MA_7"],
            name="7-Day MA",
            mode="lines"
        )
    )
    fig_ma.add_trace(
        go.Scatter(
            x=feature_df["Date"],
            y=feature_df["MA_30"],
            name="30-Day MA",
            mode="lines"
        )
    )
    fig_ma.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        legend_title="Metric"
    )
    st.plotly_chart(fig_ma, use_container_width=True)

    # Daily returns
    st.markdown(
        "<div class='section-title'>Daily Returns</div>",
        unsafe_allow_html=True
    )

    returns_df = feature_df.dropna(subset=["Daily_Return"])
    fig_returns = go.Figure()
    fig_returns.add_trace(
        go.Scatter(
            x=returns_df["Date"],
            y=returns_df["Daily_Return"],
            name="Daily Return",
            mode="lines"
        )
    )
    fig_returns.add_hline(y=0, line_dash="dash")
    fig_returns.update_layout(
        template="plotly_dark",
        height=350,
        xaxis_title="Date",
        yaxis_title="Daily Return (%)"
    )
    st.plotly_chart(fig_returns, use_container_width=True)

    # Volatility and momentum
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            "<div class='section-title'>30-Day Volatility</div>",
            unsafe_allow_html=True
        )
        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=feature_df["Date"],
                y=feature_df["Volatility_30"],
                name="Volatility",
                mode="lines"
            )
        )
        fig_vol.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Date",
            yaxis_title="Volatility"
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        st.markdown(
            "<div class='section-title'>7-Day Momentum</div>",
            unsafe_allow_html=True
        )
        fig_momentum = go.Figure()
        fig_momentum.add_trace(
            go.Scatter(
                x=feature_df["Date"],
                y=feature_df["Momentum_7"],
                name="Momentum",
                mode="lines"
            )
        )
        fig_momentum.add_hline(y=0, line_dash="dash")
        fig_momentum.update_layout(
            template="plotly_dark",
            height=350,
            xaxis_title="Date",
            yaxis_title="Momentum"
        )
        st.plotly_chart(fig_momentum, use_container_width=True)

    st.markdown(
        "<div class='section-title'>Feature-Engineered Dataset Preview</div>",
        unsafe_allow_html=True
    )
    st.dataframe(
        feature_df.head(10),
        use_container_width=True,
        hide_index=True
    )
    
elif page == "Anomaly Detection":

    st.markdown("<div class='main-title'>Anomaly Detection</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Identifying unusual gold-price movements</div>",
        unsafe_allow_html=True
    )

    # Load feature-engineered data
    anomaly_df = pd.read_csv("data/feature_engineered_gold.csv")
    anomaly_df["Date"] = pd.to_datetime(anomaly_df["Date"])
    anomaly_df = anomaly_df.sort_values("Date")

    # Calculate anomaly statistics
    total_records = len(anomaly_df)
    anomalies = anomaly_df["Is_Anomaly"].sum()
    normal_records = total_records - anomalies
    anomaly_percentage = (anomalies / total_records) * 100

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{total_records:,}")

    with col2:
        st.metric("Anomalies Detected", f"{anomalies:,}")

    with col3:
        st.metric("Normal Records", f"{normal_records:,}")

    with col4:
        st.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

    # Price and Anomaly Chart
    st.markdown("### Gold Price with Anomalies")

    fig = go.Figure()

    # Gold price line
    fig.add_trace(
        go.Scatter(
            x=anomaly_df["Date"],
            y=anomaly_df["Price"],
            mode="lines",
            name="Gold Price"
        )
    )

    # Anomaly points
    anomaly_points = anomaly_df[anomaly_df["Is_Anomaly"] == True]

    fig.add_trace(
        go.Scatter(
            x=anomaly_points["Date"],
            y=anomaly_points["Price"],
            mode="markers",
            name="Anomaly",
            marker=dict(
                size=8,
                symbol="circle"
            )
        )
    )

    fig.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Gold Price (₹)",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Anomaly Details
    st.markdown("### Anomaly Details")

    anomaly_details = anomaly_df[
        anomaly_df["Is_Anomaly"] == True
    ][
        [
            "Date",
            "Price",
            "Daily_Return",
            "Volatility_30",
            "Momentum_7"
        ]
    ].copy()

    anomaly_details = anomaly_details.sort_values(
        "Date",
        ascending=False
    )

    st.dataframe(
        anomaly_details,
        use_container_width=True,
        hide_index=True
    )

elif page == "ML Prediction":

    st.markdown(
        "<div class='main-title'>ML Prediction</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Machine learning based gold-price prediction</div>",
        unsafe_allow_html=True
    )

    st.markdown("### Model Prediction")

        # Load feature-engineered data for ML
    ml_df = pd.read_csv("data/feature_engineered_gold.csv")

    ml_df["Date"] = pd.to_datetime(ml_df["Date"])

    ml_df = ml_df.sort_values("Date").reset_index(drop=True)

    # Features used by Person 3's ML models
    lagged_features = [
        "Daily_Return",
        "MA_7",
        "MA_30",
        "Volatility_30",
        "Momentum_7",
        "Price_Lag_1",
        "Price_Lag_7",
        "Price_Lag_30",
        "Price_vs_MA30",
        "MA_90"
    ]

    non_lagged_features = [
        "Year",
        "Month"
    ]

    # Create ML dataset
    ml_data = ml_df[
        ["Date"] + lagged_features + non_lagged_features + ["Price"]
    ].copy()

    # Shift price-based features by one day
    for column in lagged_features:
        ml_data[column] = ml_data[column].shift(1)

    # Remove rows with missing values
    ml_data = ml_data.dropna().reset_index(drop=True)

    # Final feature set
    features = lagged_features + non_lagged_features

    X = ml_data[features]
    y = ml_data["Price"]



    # Make predictions using the trained models
    linear_pred = linear_model.predict(X)
    random_forest_pred = random_forest_model.predict(X)
    gradient_boosting_pred = gradient_boosting_model.predict(X)

    selected_model = st.selectbox(
    "Select Prediction Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

    st.markdown("### Model Predictions")

    model_predictions = {
    "Linear Regression": linear_pred[-1],
    "Random Forest": random_forest_pred[-1],
    "Gradient Boosting": gradient_boosting_pred[-1]
}

    st.metric(
        "Predicted Price",
        f"₹{model_predictions[selected_model]:,.2f}"
    )

    # Create a DataFrame to display the predictions
    pred_df = pd.DataFrame({
        "Date": ml_data["Date"],
        "Actual": y,
        "Linear Regression": linear_pred,
        "Random Forest": random_forest_pred,
        "Gradient Boosting": gradient_boosting_pred
    })

    st.markdown("### Model Predictions")

    st.dataframe(
        pred_df,
        use_container_width=True,
        hide_index=True
    )

elif page == "Forecasting":

    st.markdown(
        "<div class='main-title'>Time-Series Forecasting</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>ARIMA-based gold price forecasting</div>",
        unsafe_allow_html=True
    )

    forecast = arima_model.forecast(steps=30)
    sarima_forecast = sarima_model.forecast(steps=30)
    

    future_dates = pd.date_range(
        start=df["Date"].iloc[-1] + pd.Timedelta(days=1),
        periods=30
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecasted Price": forecast,
        "ARIMA Forecast": forecast,
        "SARIMA Forecast": sarima_forecast
    })

    st.markdown("### 30-Day Gold Price Forecast")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"].tail(100),
            y=df["Price"].tail(100),
            mode="lines",
            name="Historical Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecasted Price"],
            mode="lines",
            name="SARIMA Forecast"
        )
    )

    fig.update_layout(
        height=500,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Gold Price (₹)"
    )

    st.plotly_chart(fig, width="stretch")

    st.markdown("### Forecasted Values")

    st.dataframe(
        forecast_df,
        hide_index=True,
        width="stretch"
    )


elif page == "About Project":

    st.markdown(
        "<div class='main-title'>About GoldSense</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Gold Price Intelligence and Analytics System</div>",
        unsafe_allow_html=True
    )

    st.markdown("### Project Overview")

    st.write(
        """
        GoldSense is a gold-price analytics system designed to study
        historical gold-price behaviour, identify unusual market movements,
        and provide analytical insights through an interactive dashboard.
        """
    )

    st.markdown("### Project Workflow")

    st.write(
        """
        The project follows a complete data analytics workflow:

        **Raw Data → Data Cleaning → EDA → Feature Engineering
        → SQL → Anomaly Detection → ML Prediction → Forecasting
        → Streamlit Dashboard**
        """
    )

    st.markdown("### Technologies Used")

    st.write(
        """
        - Python
        - Pandas
        - NumPy
        - Plotly
        - SQLite
        - Streamlit
        - Machine Learning
        - Time-Series Forecasting
        """
    )

    st.markdown("### Project Components")

    st.write(
        """
        **Data Cleaning & EDA:**  
        Cleaning the dataset and analysing historical gold-price patterns.

        **Feature Engineering:**  
        Creating returns, moving averages, volatility, momentum and lag features.

        **SQL & Anomaly Detection:**  
        Storing analytical data in SQLite and identifying unusual price movements.

        **ML Prediction:**  
        Applying machine-learning models to predict gold prices.

        **Forecasting:**  
        Using time-series methods to forecast future gold prices.

        **Streamlit Dashboard:**  
        Combining the outputs into an interactive analytical application.
        """
    )

    st.markdown("### Team Project")

    st.write(
        """
        GoldSense is a collaborative academic project developed as part of
        the M.Sc. Big Data Analytics practical project.
        """
    )
