import pandas as pd

# Load cleaned data
df = pd.read_csv("data/cleaned_sales.csv")

# KPIs
total_sales = df['TotalAmount'].sum()
total_orders = len(df)
avg_order_value = total_sales / total_orders
top_products = df.groupby('Product')['TotalAmount'].sum().sort_values(ascending=False).head(5)

print("📊 Total Sales:", total_sales)
print("📦 Total Orders:", total_orders)
print("💰 Average Order Value:", avg_order_value)
print("🔥 Top Products:\n", top_products)
