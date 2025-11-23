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


df1=pd.read_csv('mobile_prices_filled.csv')
df2=pd.read_csv('mobile_prices_filled2.csv')
print('Number of records in first dataset:', len(df1))
print('Number of records in second dataset:', len(df2))

df = pd.concat([df1, df2], ignore_index=True)
print('Total number of records after combining datasets:', len(df))

# Standardize column names for easier access
df.columns = [col.strip().replace(' ', '_') for col in df.columns]

# 2. Clean numeric columns
for col in ['RAM', 'Storage', 'Used_Price', 'Original_Price']:
    if col in df.columns:
        df[col] = clean_numeric_column(df[col])

# Ensure the target variable is not missing
df.dropna(subset=['Used_Price'], inplace=True)

# 3. Handle missing 'Original Price' and select features
df_model = df.dropna(subset=['Original_Price']).copy()

# Convert brand names into features
df_model['Brand'] = df_model['Mobile_Name'].str.split().str[0]  # first word as brand
df_model['Brand'] = df_model['Brand'].astype('category').cat.codes

# Add ratio and log features
df_model['Price_Ratio'] = df_model['Used_Price'] / df_model['Original_Price']
df_model['RAM_Storage_Ratio'] = df_model['RAM'] / df_model['Storage']
df_model['Log_Original_Price'] = np.log1p(df_model['Original_Price'])
df_model['Log_Used_Price'] = np.log1p(df_model['Used_Price'])

# --- Fix infinities and extreme values ---
df_model.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN again after creating ratio/log features
df_model.dropna(subset=['RAM', 'Storage', 'Original_Price', 'Used_Price'], inplace=True)

# Ensure all values are finite and within float32 range
for col in df_model.select_dtypes(include=[np.number]).columns:
    df_model = df_model[np.isfinite(df_model[col])]

# Save the processed dataframe to CSV
df_model.to_csv('Data.csv', index=False)
print('Processed data saved to Data.csv')