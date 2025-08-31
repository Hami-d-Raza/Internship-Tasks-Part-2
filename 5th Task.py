# ====================================================
# Task 5: Interactive Business Dashboard in Streamlit
# ====================================================
# Objective:
# Develop an interactive dashboard for analyzing sales, profit,
# and segment-wise performance using Global Superstore dataset.
#
# Dataset: Global Superstore
# ====================================================
import os
print("Current Working Directory:", os.getcwd())
import streamlit as st
import pandas as pd
import plotly.express as px

# -------------------------
# Step 1: Load Dataset
# -------------------------
@st.cache_data
def load_data():
    # Use raw string or forward slash for Windows path
    df = pd.read_csv(r"DS_Tasks 2\Global_Superstore2.csv", encoding="ISO-8859-1")
    return df

df = load_data()

# -------------------------
# Step 2: Data Cleaning
# -------------------------
# Drop missing values in key columns
df = df.dropna(subset=["Sales", "Profit", "Customer Name", "Category", "Sub-Category", "Region"])

# Convert numeric columns
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")
df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")

# Convert Order Date if exists
if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)

# -------------------------
# Step 3: Sidebar Filters
# -------------------------
st.sidebar.header("🔎 Filters")

regions = df["Region"].unique().tolist()
categories = df["Category"].unique().tolist()
sub_categories = df["Sub-Category"].unique().tolist()

selected_region = st.sidebar.multiselect("Select Region(s):", regions, default=regions)
selected_category = st.sidebar.multiselect("Select Category:", categories, default=categories)
selected_sub_category = st.sidebar.multiselect("Select Sub-Category:", sub_categories, default=sub_categories)

# Apply filters
filtered_df = df[
    (df["Region"].isin(selected_region)) &
    (df["Category"].isin(selected_category)) &
    (df["Sub-Category"].isin(selected_sub_category))
]

# -------------------------
# Step 4: KPIs
# -------------------------
st.title("📊 Global Superstore Business Dashboard")

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()

col1, col2 = st.columns(2)
col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📈 Total Profit", f"${total_profit:,.0f}")

# -------------------------
# Step 5: Charts
# -------------------------

# Sales by Category
fig_category = px.bar(
    filtered_df.groupby("Category")["Sales"].sum().reset_index(),
    x="Category", y="Sales", title="Sales by Category", color="Category"
)
st.plotly_chart(fig_category, use_container_width=True)

# Profit by Region
fig_region = px.bar(
    filtered_df.groupby("Region")["Profit"].sum().reset_index(),
    x="Region", y="Profit", title="Profit by Region", color="Region"
)
st.plotly_chart(fig_region, use_container_width=True)

# Top 5 Customers by Sales
top_customers = filtered_df.groupby("Customer Name")["Sales"].sum().nlargest(5).reset_index()
fig_customers = px.bar(
    top_customers, x="Customer Name", y="Sales", 
    title="Top 5 Customers by Sales", color="Customer Name"
)
st.plotly_chart(fig_customers, use_container_width=True)

# Sales & Profit Trend over Time (if Order Date exists)
if "YearMonth" in filtered_df.columns:
    trend = filtered_df.groupby("YearMonth")[["Sales", "Profit"]].sum().reset_index()
    fig_trend = px.line(
        trend, x="YearMonth", y=["Sales", "Profit"], 
        title="Monthly Sales & Profit Trend"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------
# Step 6: Show Raw Data (Optional)
# -------------------------
with st.expander("📂 Show Raw Data"):
    st.dataframe(filtered_df)
