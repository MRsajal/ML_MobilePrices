import pandas as pd
import numpy as np
import re

def clean_numeric_column(series):
    def convert_to_gb(value):
        if pd.isna(value):
            return np.nan
        
        value = str(value).upper().strip().replace(',', '')

        if 'GB' in value:
            return float(re.sub(r'[^0-9.]', '', value.replace('GB', '')))
        
        elif 'MB' in value:
            return float(re.sub(r'[^0-9.]', '', value.replace('MB', ''))) / 1024
        
        else:
            try:
                return float(re.sub(r'[^0-9.]', '', value))
            except ValueError:
                return np.nan

    return series.apply(convert_to_gb)


# --------------------------
# Load and Combine Datasets
# --------------------------

df1 = pd.read_csv('mobile_prices_filled.csv')
df2 = pd.read_csv('mobile_prices_filled2.csv')

print('Number of records in first dataset:', len(df1))
print('Number of records in second dataset:', len(df2))

df = pd.concat([df1, df2], ignore_index=True)
print('Total number of records after combining datasets:', len(df))

# Standardize column names
df.columns = [col.strip().replace(' ', '_') for col in df.columns]


# --------------------------
# Clean Numeric Columns
# --------------------------

for col in ['RAM', 'Storage', 'Used_Price', 'Original_Price']:
    if col in df.columns:
        df[col] = clean_numeric_column(df[col])

# Ensure we're predicting Used Price
df.dropna(subset=['Used_Price'], inplace=True)

# Only keep rows with Original Price available
df_model = df.dropna(subset=['Original_Price']).copy()


# --------------------------
# Feature Engineering
# --------------------------

# Extract brand (first word)
df_model['Brand'] = df_model['Mobile_Name'].str.split().str[0]
df_model['Brand'] = df_model['Brand'].astype('category').cat.codes

# Ratios and log features
df_model['Price_Ratio'] = df_model['Used_Price'] / df_model['Original_Price']
df_model['RAM_Storage_Ratio'] = df_model['RAM'] / df_model['Storage']
df_model['Log_Original_Price'] = np.log1p(df_model['Original_Price'])
df_model['Log_Used_Price'] = np.log1p(df_model['Used_Price'])

# Replace infinities once
df_model.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with missing essential numeric features
essential_cols = [
    'RAM', 'Storage', 'Original_Price', 'Used_Price',
    'Price_Ratio', 'RAM_Storage_Ratio',
    'Log_Original_Price', 'Log_Used_Price'
]

existing_cols = [c for c in essential_cols if c in df_model.columns]
df_model.dropna(subset=existing_cols, inplace=True)


# --------------------------
# Save Cleaned Data
# --------------------------

print("Remaining rows after cleaning:", len(df_model))
df_model.to_csv('Data.csv', index=False)
print('Processed data saved to Data.csv')
