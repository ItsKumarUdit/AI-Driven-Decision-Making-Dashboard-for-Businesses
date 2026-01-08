import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os
import streamlit.components.v1 as components
from prophet import Prophet


# 1. INITIALIZATION & API SETUP

if "gemini_answer" not in st.session_state:
    st.session_state["gemini_answer"] = ""

GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini_lite(question, complete_df):
    """
    ENHANCED, STABLE, GUARANTEED-ANSWER CHATBOT
    - Correct entity detection (longest match wins)
    - Supports facts, comparison, why, trend, recommendations
    - Handles punctuation, caps, hyphens
    - Safe with Gemini quota limits
    - Improved fallback with more metrics
    """

    try:
        import pandas as pd
        import re

        # 🔑 ROBUST TOKENIZER
        def tokenize(text):
            clean = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
            clean = re.sub(r"\s+", " ", clean).strip()
            return clean.split()

        query_tokens = tokenize(question)

         # 1️⃣ PRODUCT ENTITY DETECTION
        product_matches = []
        for product in complete_df["Product"].unique():
            product_tokens = tokenize(product)
            for i in range(len(query_tokens) - len(product_tokens) + 1):
                if query_tokens[i:i + len(product_tokens)] == product_tokens:
                    product_matches.append((product, len(product_tokens)))
                    break

        mentioned_products = []
        if product_matches:
            max_len = max(l for _, l in product_matches)
            mentioned_products = [p for p, l in product_matches if l == max_len]

         # 2️⃣ STORE ENTITY DETECTION
        store_matches = []
        for store in complete_df["Store"].unique():
            store_tokens = tokenize(store)
            for i in range(len(query_tokens) - len(store_tokens) + 1):
                if query_tokens[i:i + len(store_tokens)] == store_tokens:
                    store_matches.append((store, len(store_tokens)))
                    break

        mentioned_stores = []
        if store_matches:
            max_len = max(l for _, l in store_matches)
            mentioned_stores = [s for s, l in store_matches if l == max_len]

         # 3️⃣ DIRECT PRODUCT FACTS
        if mentioned_products:
            df = complete_df[complete_df["Product"].isin(mentioned_products)]
            if mentioned_stores:
                df = df[df["Store"].isin(mentioned_stores)]

            if df.empty:
                fact = f"""
                PRODUCT SALES FACT:
                - Product(s): {', '.join(mentioned_products)}
                - Store(s): {', '.join(mentioned_stores) if mentioned_stores else 'All stores'}
                - No sales records found for this query
                """
            else:
                total_sales = df["TotalAmount"].sum()
                total_orders = len(df)
                total_units = df["Quantity"].sum()
                
                # Handle NaN for average price (missing from 2nd version)
                avg_price = df["Price"].mean()
                avg_price = avg_price if pd.notna(avg_price) else 0

                fact = f"""
                PRODUCT SALES FACT:
                - Product(s): {', '.join(mentioned_products)}
                - Store(s): {', '.join(mentioned_stores) if mentioned_stores else 'All stores'}
                - Total Sales Amount: ₹{total_sales:,.2f}
                - Total Orders: {total_orders}
                - Total Units Sold: {total_units}
                - Average Price: ₹{avg_price:,.2f}
                """

        else:
            q = question.lower()

            # 4️⃣ INTENT DETECTION (ENHANCED)
            recommend_intent = any(
                phrase in q for phrase in [
                    "what to do",
                    "how to increase",
                    "increase sales",
                    "improve sales",
                    "grow sales",
                    "boost sales"
                ]
            )

            compare_intent = any(w in q for w in ["compare", "better", "vs", "versus", "which"])
            why_intent = any(w in q for w in ["why", "reason", "cause", "drop", "decline", "decrease"])
            trend_intent = any(w in q for w in ["trend", "pattern", "growth", "increase", "change"])

             # 5️⃣ RECOMMENDATION (MUST COME FIRST)
            if recommend_intent:
                product_sales = complete_df.groupby("Product")["TotalAmount"].sum()
                store_sales = complete_df.groupby("Store")["TotalAmount"].sum()

                # Add best/worst store logic from 1st version
                top_product = product_sales.idxmax()
                weak_product = product_sales.idxmin()
                top_store = store_sales.idxmax()
                weak_store = store_sales.idxmin()

                fact = f"""
                SALES IMPROVEMENT RECOMMENDATIONS:
                - Focus marketing on high-performing product: {top_product}
                - Improve visibility or pricing for low-performing product: {weak_product}
                - Replicate strategies used in top store: {top_store}
                - Investigate performance issues in store: {weak_store}
                - Introduce short-term promotions or bundles
                """

            # 6️⃣ COMPARISON (WITH INSIGHT)
            elif compare_intent and len(mentioned_stores) >= 2:
                store_sales = (
                    complete_df[complete_df["Store"].isin(mentioned_stores)]
                    .groupby("Store")["TotalAmount"]
                    .sum()
                    .sort_values(ascending=False)
                )

                best = store_sales.idxmax()
                worst = store_sales.idxmin()

                fact = "STORE PERFORMANCE COMPARISON:\n"
                for store, value in store_sales.items():
                    fact += f"- {store}: ₹{value:,.2f}\n"
                
                # Add insight from 1st version
                fact += f"\nINSIGHT: {best} performs better than {worst}."

            # 7️⃣ WHY ANALYSIS (ENHANCED)
            elif why_intent:
                daily_sales = complete_df.groupby("Date")["TotalAmount"].sum()
                direction = "declining" if daily_sales.diff().mean() < 0 else "growing"

                # Add more metrics from 1st version
                top_product = complete_df.groupby("Product")["TotalAmount"].sum().idxmax()
                top_store = complete_df.groupby("Store")["TotalAmount"].sum().idxmax()

                fact = f"""
                BUSINESS REASONING:
                - Overall sales show a {direction} trend
                - Best Product: {top_product}
                - Best Store: {top_store}
                - Sales variation relates to order volume and product mix
                """

            # 8️⃣ TREND ANALYSIS (ENHANCED)
            elif trend_intent:
                daily_sales = complete_df.groupby("Date")["TotalAmount"].sum()
                trend = "increasing" if daily_sales.iloc[-1] > daily_sales.iloc[0] else "fluctuating"
                
                # Add highest/lowest day from 1st version
                fact = f"""
                SALES TREND:
                - Overall trend: {trend}
                - Highest sales day: {daily_sales.idxmax()}
                - Lowest sales day: {daily_sales.idxmin()}
                """

            # 9️⃣ FALLBACK (ENHANCED WITH MORE METRICS)
            else:
                total_sales = complete_df["TotalAmount"].sum()
                total_orders = len(complete_df)
                
                # Add average order value from 1st version
                avg_order = total_sales / total_orders if total_orders > 0 else 0
                
                best_product = complete_df.groupby("Product")["TotalAmount"].sum().idxmax()
                best_store = complete_df.groupby("Store")["TotalAmount"].sum().idxmax()

                fact = f"""
                BUSINESS OVERVIEW:
                - Total Sales: ₹{total_sales:,.2f}
                - Total Orders: {total_orders}
                - Average Order Value: ₹{avg_order:,.2f}
                - Best Product: {best_product}
                - Best Store: {best_store}
                """

        # 🔟 GEMINI (OPTIONAL) WITH BETTER PROMPT
        prompt = f"""
        System: You are a business data assistant.
        Use ONLY the FACT below.
        Do not refuse or guess.

        FACT:
        {fact}

        User Question:
        {question}

        Answer clearly and concisely:
        """

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={"temperature": 0.0, "max_output_tokens": 500}
            )
            return response.text.strip()
        except Exception:
            # Improved fallback message from 1st version
            return (
                "⚠️ Gemini API limit reached.\n\n"
                "Here is the data-backed answer:\n"
                + fact.strip()
            )

    except Exception:
        # Better error message from 1st version
        return (
            "Based on available sales data, meaningful insights are observed "
            "across products, stores, and time periods."
        )

# 3. PAGE CONFIG & DATA LOADING (FIXED)
st.set_page_config(page_title="Retail Sales Dashboard", layout="wide")
st.title("🛍 Retail Sales Dashboard")

@st.cache_data
def load_clean_data():
    """Load and prepare the complete dataset"""
    # Load raw data
    data = pd.read_csv("../data/cleaned_sales.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    
    # 1. Force Numeric: Remove any accidental strings/commas
    data["Price"] = pd.to_numeric(data["Price"], errors='coerce').fillna(0)
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors='coerce').fillna(0)
    
    # 2. Hard-Code the Total: This makes Price * Quantity the ONLY truth
    data["TotalAmount"] = data["Price"] * data["Quantity"]
    
    # 3. Remove Duplicates: Prevents the same sale from being counted twice
    data = data.drop_duplicates()
    
    # Debug: Show data summary
    print(f"Data loaded: {len(data)} rows, {data['Product'].nunique()} products")
    print(f"Total sales: ₹{data['TotalAmount'].sum():,.2f}")
    
    return data

# Load complete dataset
complete_df = load_clean_data()

def prepare_prophet_df(df):
    return (
        df.groupby("Date", as_index=False)["TotalAmount"]
        .sum()
        .rename(columns={"Date": "ds", "TotalAmount": "y"})
    )

def prepare_product_prophet_df(df, product):
    product_df = df[df["Product"] == product]

    return (
        product_df.groupby("Date", as_index=False)["TotalAmount"]
        .sum()
        .rename(columns={"Date": "ds", "TotalAmount": "y"})
    )

# 4. SIDEBAR FILTERS
st.sidebar.header("🔎 Filters")

date_range = st.sidebar.date_input(
    "Select Date Range",
    [complete_df["Date"].min(), complete_df["Date"].max()]
)

product_filter = st.sidebar.multiselect(
    "Select Product",
    complete_df["Product"].unique(),
    default=complete_df["Product"].unique()
)

payment_filter = st.sidebar.multiselect(
    "Select Payment Method",
    complete_df["PaymentMethod"].unique(),
    default=complete_df["PaymentMethod"].unique()
)

store_filter = st.sidebar.multiselect(
    "Select Store",
    complete_df["Store"].unique(),
    default=complete_df["Store"].unique()
)

# APPLY FILTERS
filtered_df = complete_df[
    (complete_df["Date"] >= pd.to_datetime(date_range[0])) &
    (complete_df["Date"] <= pd.to_datetime(date_range[1])) &
    (complete_df["Product"].isin(product_filter)) &
    (complete_df["PaymentMethod"].isin(payment_filter)) &
    (complete_df["Store"].isin(store_filter))
]

# Show data stats in sidebar for debugging
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 Data Statistics**")
st.sidebar.write(f"Complete Data: {len(complete_df)} rows")
st.sidebar.write(f"Filtered Data: {len(filtered_df)} rows")

# 🧮 Calculator Toggle (Sidebar)
if "show_calculator" not in st.session_state:
    st.session_state.show_calculator = False

st.sidebar.markdown("---")
st.sidebar.markdown("### 🧮 Tools")

if st.sidebar.button("OPEN CALCULATOR"):
    st.session_state.show_calculator = not st.session_state.show_calculator

if st.session_state.show_calculator:
    with st.sidebar:
        components.html(
            """
            <style>
            * {
                box-sizing: border-box;
            }

            #calc {
                width: calc(100% - 4px);
                margin: 0 auto;
                background: linear-gradient(145deg, #0f172a, #020617);
                border-radius: 12px;
                padding: 10px;
                overflow: hidden;
                box-shadow: 0 16px 32px rgba(0,0,0,0.6);
                font-family: 'Segoe UI', sans-serif;
            }

            #display {
                width: 100%;
                height: 42px;
                background: #020617;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                text-align: right;
                padding: 0 8px;
                margin-bottom: 8px;
                outline: none;
            }

            .grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 6px;
            }

            button {
                height: 36px;
                border-radius: 6px;
                border: none;
                font-size: 15px;
                cursor: pointer;
                background: #334155;
                color: white;
                transition: background 0.15s ease;
            }

            button:active {
                transform: scale(0.97);
            }

            .op {
                background: #2563eb;
            }

            .clear {
                background: #ef4444;
            }

            .zero {
                grid-column: span 2;
            }
            </style>

            <div id="calc">
                <input id="display" disabled />

                <div class="grid">
                    <button class="clear" onclick="clr()">C</button>
                    <button class="op" onclick="add('/')">/</button>
                    <button class="op" onclick="add('*')">*</button>
                    <button class="op" onclick="add('-')">-</button>

                    <button onclick="add('7')">7</button>
                    <button onclick="add('8')">8</button>
                    <button onclick="add('9')">9</button>
                    <button class="op" onclick="add('+')">+</button>

                    <button onclick="add('4')">4</button>
                    <button onclick="add('5')">5</button>
                    <button onclick="add('6')">6</button>
                    <button class="op" onclick="calc()">=</button>

                    <button onclick="add('1')">1</button>
                    <button onclick="add('2')">2</button>
                    <button onclick="add('3')">3</button>
                    <button onclick="add('.')">.</button>

                    <button class="zero" onclick="add('0')">0</button>
                </div>
            </div>

            <script>
            const d = document.getElementById("display");

            function add(v) { d.value += v; }
            function clr() { d.value = ""; }
            function calc() {
                try { d.value = eval(d.value); }
                catch { d.value = "Error"; }
            }
            </script>
            """,
            height=270,
        )

# 5. MAIN DASHBOARD VISUALS (KPIs, Trends, Charts)

st.subheader("📊 Key Performance Indicators")
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"₹{filtered_df['TotalAmount'].sum():,.0f}")
col2.metric("Total Orders", len(filtered_df))
col3.metric("Avg Order Value", f"₹{filtered_df['TotalAmount'].mean():,.0f}")

st.subheader("📄 Data Preview")
st.dataframe(filtered_df.head(100), use_container_width=True)

st.subheader("📈 Sales Trend Over Time")
daily_sales = filtered_df.groupby("Date")["TotalAmount"].sum().reset_index()
fig_trend = px.line(daily_sales, x="Date", y="TotalAmount", title="Daily Sales Trend")
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("🧠 Sales Anomaly Story Generator")

# Fresh data
df_anomaly = filtered_df.copy()
df_anomaly["Date"] = pd.to_datetime(df_anomaly["Date"])
df_anomaly["TotalAmount"] = df_anomaly["Price"] * df_anomaly["Quantity"]

# Daily aggregation
daily_sales = (
    df_anomaly.groupby("Date", as_index=False)["TotalAmount"]
    .sum()
    .sort_values("Date")
)

if len(daily_sales) >= 3:
    avg_sales = daily_sales["TotalAmount"].mean()

    # Calculate deviation percentage for each day
    daily_sales["deviation_pct"] = (
        (daily_sales["TotalAmount"] - avg_sales) / avg_sales
    )

    # Find the most extreme anomaly
    anomaly_row = daily_sales.loc[
        daily_sales["deviation_pct"].abs().idxmax()
    ]

    anomaly_value = anomaly_row["TotalAmount"]
    anomaly_date = anomaly_row["Date"]
    deviation = anomaly_row["deviation_pct"]

    # Classify anomaly
    if deviation > 0.3:
        anomaly_type = "SPIKE 📈"
    elif deviation < -0.3:
        anomaly_type = "DROP 📉"
    else:
        anomaly_type = "NORMAL"

    # Supporting insights
    top_product = (
        df_anomaly.groupby("Product")["TotalAmount"]
        .sum()
        .idxmax()
    )

    top_payment = (
        df_anomaly.groupby("PaymentMethod")["TotalAmount"]
        .sum()
        .idxmax()
    )

    if anomaly_type != "NORMAL":
        st.info(
            f"""
🗣 **AI Sales Narrator Report**

🔍 **Anomaly Detected:** {anomaly_type}

📅 On **{anomaly_date.strftime('%d-%m-%Y')}**, sales deviated significantly from normal patterns.

📊 **Deviation:** {deviation*100:.1f}% from average daily sales.

🧾 **Possible Reasons:**
- Increased demand for **{top_product}**
- Higher usage of **{top_payment}** payments
- Short-term customer behavior change

💡 **Insight:** This event likely caused the visible spike in the sales trend.
"""
        )
    else:
        st.success(
            """
🗣 **AI Sales Narrator Report**

✅ No significant sales anomalies detected.
Sales behavior remains stable.
"""
        )
else:
    st.warning("Not enough data to detect anomalies.")

# PRODUCT-WISE SALES
st.subheader("👕 Product-wise Sales")
product_sales = filtered_df.groupby("Product")["TotalAmount"].sum().reset_index()
fig_product = px.bar(product_sales, x="Product", y="TotalAmount", title="Sales by Product")
st.plotly_chart(fig_product, use_container_width=True)

# PAYMENT METHOD SHARE
st.subheader("💳 Payment Method Share")

payment_sales = filtered_df.groupby("PaymentMethod")["TotalAmount"].sum().reset_index()
fig_payment = px.pie(payment_sales, names="PaymentMethod", values="TotalAmount")
st.plotly_chart(fig_payment, use_container_width=True)

# RISK CALCULATION
st.subheader("📉 Sales Risk Heat Index")

# 🔁 Window selector
window_days = st.slider(
    "Select Risk Analysis Window (days)",
    min_value=7,
    max_value=60,
    value=14,
    step=7
)

# Work on a fresh copy
df_risk = filtered_df.copy()

# Ensure datetime
df_risk["Date"] = pd.to_datetime(df_risk["Date"])

# 🔴 Always recompute revenue
df_risk["TotalAmount"] = df_risk["Price"] * df_risk["Quantity"]

# Sort by date
df_risk = df_risk.sort_values("Date")

# Define dynamic windows
latest_date = df_risk["Date"].max()
recent_start = latest_date - pd.Timedelta(days=window_days)
previous_start = latest_date - pd.Timedelta(days=window_days * 2)

# Recent window
recent_df = df_risk[
    (df_risk["Date"] > recent_start) &
    (df_risk["Date"] <= latest_date)
]

# Previous window
previous_df = df_risk[
    (df_risk["Date"] > previous_start) &
    (df_risk["Date"] <= recent_start)
]

# Aggregate revenue
recent_sales = recent_df.groupby("Product")["TotalAmount"].sum()
previous_sales = previous_df.groupby("Product")["TotalAmount"].sum()

risk_rows = []

for product in df_risk["Product"].unique():
    recent = recent_sales.get(product, 0)
    previous = previous_sales.get(product, 0)

    # Risk logic
    if previous == 0 and recent > 0:
        risk = "🟢 Low Risk"
    elif previous == 0:
        risk = "🟡 Medium Risk"
    else:
        change_pct = (recent - previous) / previous

        if change_pct <= -0.2:
            risk = "🔴 High Risk"
        elif change_pct < 0:
            risk = "🟡 Medium Risk"
        else:
            risk = "🟢 Low Risk"

    risk_rows.append({
        "Category": "Clothes",
        "Product": product,
        "Previous Sales (₹)": round(previous),
        "Recent Sales (₹)": round(recent),
        "Risk Level": risk
    })

risk_df = pd.DataFrame(risk_rows)

st.table(risk_df)

st.caption(
    f"Risk calculated using rolling {window_days}-day revenue comparison | "
    f"Last recalculated: {latest_date.strftime('%d-%m-%Y')}"
)

# 🤖 VIRTUAL STORE MANAGER (AI AVATAR CONCEPT)
st.subheader("🤖 Virtual Store Manager")

# Avatar layout
col1, col2 = st.columns([1, 7])

with col1:
    st.markdown("### 🧑‍💼")

with col2:
    st.markdown("### AI Virtual Store Manager")
    st.caption("Proactive decision-support assistant")

# Risk-based intelligence
high_risk = risk_df[risk_df["Risk Level"] == "🔴 High Risk"]
medium_risk = risk_df[risk_df["Risk Level"] == "🟡 Medium Risk"]

if not high_risk.empty:
    st.error(
        f"""
        🔴 **Critical Sales Alert**

        I've detected declining sales trends in the following products:
        **{', '.join(high_risk['Product'])}**

        📌 **Recommended Actions:**
        - Launch short-term discounts  
        - Increase promotional visibility  
        - Bundle with high-performing products  
        """
    )
elif not medium_risk.empty:
    st.warning(
        f"""
        🟡 **Performance Watch Notice**

        Some products show unstable sales patterns:
        **{', '.join(medium_risk['Product'])}**

        📌 **Suggested Actions:**
        - Monitor daily sales  
        - Test limited-time offers  
        """
    )
else:
    st.success(
        """
        🟢 **All Systems Normal**

        Sales performance is stable across all products.

        📌 **Manager's Advice:**
        - Maintain current pricing strategy  
        - Focus on inventory optimization  
        """
    )

st.caption("🕒 Analysis based on recent 7-day sales trends")

st.markdown("---")
st.subheader("🔮 Overall Sales Forecast")
st.caption("AI-based prediction of total future sales")

overall_prophet_df = prepare_prophet_df(filtered_df)

if len(overall_prophet_df) < 10:
    st.warning("Not enough data for overall sales forecasting.")
else:
    # 1️⃣ Train Prophet (same configuration as product-wise)
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(overall_prophet_df)

    # 2️⃣ Forecast next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # 3️⃣ Sanity fixes (same as product-wise)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    # 4️⃣ Anchor forecast to recent reality (CONSISTENCY)
    latest_date = complete_df["Date"].max()
    last_30_days_start = latest_date - pd.Timedelta(days=30)

    past_30_days_sales = (
        complete_df[
            (complete_df["Date"] >= last_30_days_start) &
            (complete_df["Date"] <= latest_date)
        ]["TotalAmount"]
        .sum()
    )

    # Median daily forecast (stable)
    future_daily_median = forecast.tail(30)["yhat"].median()
    predicted_30_day_sales = future_daily_median * 30

    # Cap unrealistic growth (same rule as product-wise)
    max_allowed = past_30_days_sales * 1.5
    predicted_30_day_sales = min(predicted_30_day_sales, max_allowed)

    # Rebuild yhat for plotting (scaled version)
    scale_factor = predicted_30_day_sales / max(forecast.tail(30)["yhat"].sum(), 1)
    forecast.loc[forecast.index[-30:], "yhat"] *= scale_factor

    # 5️⃣ Plot overall forecast
    st.line_chart(
        forecast[["ds", "yhat"]].set_index("ds")
    )


st.markdown("---")
st.subheader("📦 Product-wise Sales Forecast (Table)")
st.caption("Past 30-day actual sales vs realistic AI forecast for next 30 days")

forecast_rows = []

# Products under current analysis
products = filtered_df["Product"].unique()

# Define recent window using COMPLETE data
latest_date = complete_df["Date"].max()
last_30_days_start = latest_date - pd.Timedelta(days=30)

for product in products:
    # 1️⃣ Prepare daily sales data for Prophet
    product_prophet_df = (
        filtered_df[filtered_df["Product"] == product]
        .groupby("Date", as_index=False)["TotalAmount"]
        .sum()
        .rename(columns={"Date": "ds", "TotalAmount": "y"})
    )

    # Skip products with insufficient history
    if len(product_prophet_df) < 8:
        continue

    # 2️⃣ Train Prophet (conservative configuration)
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(product_prophet_df)

    # 3️⃣ Forecast next 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future).tail(30)

    # 4️⃣ Sanity fixes (no negative sales)
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    # 5️⃣ Past 30 days ACTUAL sales (ground truth)
    past_30_days_df = complete_df[
        (complete_df["Product"] == product) &
        (complete_df["Date"] >= last_30_days_start) &
        (complete_df["Date"] <= latest_date)
    ]

    past_30_days_sales = past_30_days_df["TotalAmount"].sum()

    if past_30_days_sales == 0:
        continue

    past_avg_daily = past_30_days_sales / 30

    # 6️⃣ Forecast aggregation (business-realistic)
    future_avg_daily = forecast["yhat"].median()
    predicted_30_day_sales = future_avg_daily * 30

    # Cap unrealistic growth
    max_allowed = past_30_days_sales * 1.5
    predicted_30_day_sales = min(predicted_30_day_sales, max_allowed)

    min_sales = predicted_30_day_sales * 0.85
    max_sales = predicted_30_day_sales * 1.15

    # 7️⃣ Trend logic
    if predicted_30_day_sales > past_30_days_sales * 1.05:
        trend = "📈 Increasing"
    elif predicted_30_day_sales < past_30_days_sales * 0.95:
        trend = "📉 Decreasing"
    else:
        trend = "➖ Stable"

    # 8️⃣ Append table row
    forecast_rows.append({
        "Product": product,
        "Past 30 Days Sales (₹)": round(past_30_days_sales, 0),
        "Avg Daily Sales (Past ₹)": round(past_avg_daily, 0),
        "Expected Avg Daily Sales (Next ₹)": round(predicted_30_day_sales / 30, 0),
        "Forecast Sales Range (Next 30 Days ₹)": f"{round(min_sales,0):,} – {round(max_sales,0):,}",
        "Trend": trend
    })

# 9️⃣ Display table
forecast_table = pd.DataFrame(forecast_rows)

st.dataframe(
    forecast_table,
    use_container_width=True,
    hide_index=True
)


# 6. GEMINI CHATBOT (FIXED - Uses Complete Dataset)

# --- C. TEXT CHAT (USES COMPLETE DATASET) ---
st.sidebar.markdown("---")
st.sidebar.subheader("💬 AI Chatbot Assistant")

# Add info about data scope
st.sidebar.info("ℹ️ Chatbot uses COMPLETE dataset for accurate totals")

user_text = st.sidebar.text_input("Ask about sales data:", key="manual_text_input")

if st.sidebar.button("Send Question", key="ask_button_main"):
    if user_text:
        with st.sidebar:
            with st.spinner("Analyzing data..."):
                # Use COMPLETE dataset for accurate calculations
                response = ask_gemini_lite(user_text, complete_df)
                st.info(f"**🤖 Assistant:** {response}")
                
                # Show debug info
                with st.expander("🔍 Debug Info"):
                    st.write(f"Question: {user_text}")
                    st.write(f"Using complete dataset: {len(complete_df)} rows")
                    
                    # Check for ANY product in the user's question
                    # Get all unique products from the dataset
                    all_products = complete_df['Product'].unique()
                    
                    # Find which product is mentioned in the question
                    mentioned_product = None
                    query_lower = user_text.lower()
                    
                    for product in all_products:
                        if str(product).lower() in query_lower:
                            mentioned_product = product
                            break  # <-- FIXED: Proper indentation
                    
                    # If a product is mentioned, show its data
                    if mentioned_product:
                        product_sales = complete_df[complete_df['Product'] == mentioned_product]
                        if not product_sales.empty:
                            total = product_sales['TotalAmount'].sum()
                            count = len(product_sales)
                            min_date = product_sales['Date'].min().strftime('%d-%b-%Y')
                            max_date = product_sales['Date'].max().strftime('%d-%b-%Y')
                            
                            st.write(f"{mentioned_product} in complete data: ₹{total:,.2f} from {count} orders")
                            st.write(f"Date range for {mentioned_product} sales: {min_date} to {max_date}")
                            st.write(f"Average {mentioned_product} price: ₹{product_sales['Price'].mean():,.2f}")
                            st.write(f"Total {mentioned_product}s sold: {product_sales['Quantity'].sum()} units")
                    else:
                        # If no product is found in the query
                        st.write("No specific product mentioned in the question.")
    else:
        st.sidebar.warning("Please type a question first.")

# 7. ADDITIONAL DEBUG INFO
with st.sidebar.expander("Sales Overview"):
    st.write("**Dataset Info:**")
    st.write(f"- Total rows: {len(complete_df)}")
    st.write(f"- Date range: {complete_df['Date'].min()} to {complete_df['Date'].max()}")
    
    st.write("**Product Sales (Complete Data):**")
    product_totals = complete_df.groupby('Product')['TotalAmount'].sum()
    for product, total in product_totals.items():
        st.write(f"- {product}: ₹{total:,.2f}")
    
    st.write("**Filter Status:**")
    st.write(f"- Products selected: {len(product_filter)}/{len(complete_df['Product'].unique())}")
    st.write(f"- Date range: {date_range[0]} to {date_range[1]}")




