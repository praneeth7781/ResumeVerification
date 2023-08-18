import os
import pandas as pd


folder_path = "./0. combined"
output_file = "./0. combined/combined_data.csv"

# temp list to store all dataframes
data_frames = []


for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        # read each CSV file into a DataFrame
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # appending to temp list
        data_frames.append(df)

# Concatenate the DataFrames into a single DataFrame
combined_df = pd.concat(data_frames)
combined_df = combined_df.reset_index(drop=True)

# creating final file
combined_df.to_csv(output_file, index=False)
