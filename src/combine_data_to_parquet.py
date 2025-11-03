import pandas as pd
import matplotlib.pyplot as plt
import glob


excel_files = glob.glob("../data/*.xls")
print(excel_files)

# Check if every columns are identical
headers = []

for f in excel_files:
    try:
        # Try reading as Excel
        df = pd.read_excel(f, engine="xlrd", nrows=1, header=None)
        header = df.iloc[0].tolist()
    except Exception as e:
        # Read as CSV
        df = pd.read_csv(f, engine="python", on_bad_lines='skip', sep="\t", nrows=1, header=None)
        header = df.iloc[0].tolist()
    headers.append((f, header))

if headers:
    first_header = headers[0][1]
    all_identical = all(header == first_header for _, header in headers)

    for file, header in headers:
        print(f"{file}: {header}")

    if all_identical:
        print("All headers are identical")
    else:
        print("All headers are NOT identical")
else:
    print("Data has not been found")
    
    
# Combine all data into a single DataFrame    
df_list = []

for f in excel_files:
    try:
        #Try reading as Excel
        df = pd.read_excel(f, engine="xlrd")
        print(f"Read {f} as Excel")
    except Exception as e:
        #read as CSV
        df = pd.read_csv(f, engine="python", on_bad_lines='skip', sep="\t")
        print(f"Read {f} as CSV")
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)
print("DataFrames combined successfully")

    
# Reshape the DataFrame to tidy format
tidy_rows = []

cols = combined_df.columns.tolist()
for idx, row in combined_df.iterrows():
    for i in range(0, len(cols), 3):
        date_col = cols[i]
        meas_col = cols[i+1]
        status_col = cols[i+2]
        tidy_rows.append({
            "MeasureDate": row[date_col],
            "MeasurementName": meas_col,
            "Value": row[meas_col],
            "Status": row[status_col]
        })

tidy_df = pd.DataFrame(tidy_rows, columns=["MeasureDate", "MeasurementName", "Value", "Status"])
tidy_df.head()

# Save the tidy DataFrame to a Parquet file
try:
    tidy_df['Status'] = tidy_df['Status'].astype(str)
    tidy_df.to_parquet("../data/combined_data.parquet")
    print("Data saved to ../data/combined_data.parquet")
except Exception as e:
    print(f"Error saving data: {e}")
    