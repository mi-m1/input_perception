import os
import pandas as pd
import glob

# directory = 'meta-llama/*/*.csv'

# Find all 'file.csv' in subdirectories of 'results'
csv_files = glob.glob('Qwen/*/*.csv')
# csv_files = glob.glob('meta-llama/*/*.csv')


# csv_files = [
#     "meta-llama/Llama-3.2-1B-Instruct/cws_y0.5.csv",
#     "meta-llama/Llama-3.2-3B-Instruct/cws_y0.5.csv",
# ]

print(f'csv_files: {csv_files}')

for file_path in csv_files:

    df = pd.read_csv(file_path)
    print(f"{file_path}: {df.shape}")
    
    columns = df.columns.to_list()
    print(f"columns: {columns}")
    print(len(columns))