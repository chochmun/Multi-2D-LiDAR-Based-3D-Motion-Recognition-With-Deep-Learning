import os
import pandas as pd
import json

# File paths
json_file_path = 'lidar_interface/poses.json'
csv_folder_path = 'csv_files/'
output_file_path = 'csv_files/merged_selection.csv'
expected_num_columns = 273  # 고정된 열의 수

# Load JSON file with class labels
with open(json_file_path, 'r') as f:
    class_labels = json.load(f)

# User selection (for example, user selects classes 0, 1, and 5)
user_selection = [0, 1, 5]  # Modify this based on the user input

# Convert the selected class indices to corresponding labels
selected_labels = [class_labels[str(idx)] for idx in user_selection]

# List of files in the CSV folder
csv_files = os.listdir(csv_folder_path)

# Initialize an empty list to store dataframes
dfs = []

# Iterate through the selected labels and match them with CSV file names
for label in selected_labels:
    for file in csv_files:
        if label in file:
            # Read the corresponding CSV file
            df = pd.read_csv(os.path.join(csv_folder_path, file))

            # Adjust the number of columns to 273
            if df.shape[1] < expected_num_columns:
                # If the dataframe has fewer columns, add NaN columns
                additional_columns = expected_num_columns - df.shape[1]
                for i in range(additional_columns):
                    df[f'Unnamed_column_{df.shape[1] + i}'] = pd.NA
            elif df.shape[1] > expected_num_columns:
                # If the dataframe has more columns, truncate the extra columns
                df = df.iloc[:, :expected_num_columns]

            # Append the adjusted dataframe to the list
            dfs.append(df)

# Concatenate all selected CSV files into one dataframe
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged dataframe to a new CSV file
merged_df.to_csv(output_file_path, index=False)

print(f"Merged CSV saved to {output_file_path}")
