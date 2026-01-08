import pandas as pd
from datetime import datetime

# Load raw data
df = pd.read_csv("data/raw_sales.csv")

# Convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Calculate total sale amount
df['TotalAmount'] = df['Quantity'] * df['Price']

# Save cleaned data
df.to_csv("data/cleaned_sales.csv", index=False)

print("✅ Data cleaned & saved!")
