import pandas as pd

# --- Load both CSV files ---
used_df = pd.read_csv("bikroy_mobile_prices2.csv")          # First file (used prices)
original_df = pd.read_csv("all_mobile_specs_scraped.csv")          # Second file (original prices)

# --- Clean and standardize column names ---
used_df.columns = used_df.columns.str.strip()
original_df.columns = original_df.columns.str.strip()

# --- Normalize mobile names to uppercase for better matching ---
used_df['Mobile Name'] = used_df['Mobile Name'].str.upper().str.strip()
original_df['Mobile Name'] = original_df['Mobile Name'].str.upper().str.strip()

# --- Rename 'price' column to 'Original Price' in the second file if needed ---
if 'price' in original_df.columns:
    original_df.rename(columns={'price': 'Original Price'}, inplace=True)

# --- Select relevant columns for merging ---
merge_columns = ['Mobile Name', 'Original Price', 'RAM', 'Storage']
available_cols = [col for col in merge_columns if col in original_df.columns]

# --- Merge to fill missing data ---
merged_df = used_df.merge(
    original_df[available_cols],
    on='Mobile Name',
    how='left',
    suffixes=('', '_new')
)

# --- Count missing values before filling ---
missing_before_price = used_df['Original Price'].isna().sum() if 'Original Price' in used_df.columns else 0
missing_before_ram = used_df['RAM'].isna().sum() if 'RAM' in used_df.columns else 0
missing_before_storage = used_df['Storage'].isna().sum() if 'Storage' in used_df.columns else 0

# --- Fill missing values only if empty in the first file ---
for col in ['Original Price', 'RAM', 'Storage']:
    if f"{col}_new" in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(merged_df[f"{col}_new"])

# --- Drop temporary columns ---
cols_to_drop = [f"{col}_new" for col in ['Original Price', 'RAM', 'Storage'] if f"{col}_new" in merged_df.columns]
merged_df.drop(columns=cols_to_drop, inplace=True)

# --- Count missing values after filling ---
missing_after_price = merged_df['Original Price'].isna().sum() if 'Original Price' in merged_df.columns else 0
missing_after_ram = merged_df['RAM'].isna().sum() if 'RAM' in merged_df.columns else 0
missing_after_storage = merged_df['Storage'].isna().sum() if 'Storage' in merged_df.columns else 0

# --- Calculate filled counts ---
filled_price = missing_before_price - missing_after_price
filled_ram = missing_before_ram - missing_after_ram
filled_storage = missing_before_storage - missing_after_storage

# --- Save the result ---
output_file = "mobile_prices_filled2.csv"
merged_df.to_csv(output_file, index=False)

# --- Print summary ---
print("✅ Merging completed successfully!")
print(f"📄 Output saved as: {output_file}")
print(f"➡️ Total rows: {len(merged_df)}")
print(f"💰 Original Price values filled: {filled_price}")
print(f"📱 RAM values filled: {filled_ram}")
print(f"💾 Storage values filled: {filled_storage}")
print(f"🚫 Still missing Original Price values: {missing_after_price}")
print(f"🚫 Still missing RAM values: {missing_after_ram}")
print(f"🚫 Still missing Storage values: {missing_after_storage}")
