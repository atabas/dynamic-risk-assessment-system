import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]


#############Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    current_dir = os.getcwd()
    input_path = f"{current_dir}/{input_folder_path}"

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    output_path = os.path.join(os.getcwd(), output_folder_path)

    files = os.listdir(input_path)
    d = [pd.read_csv(os.path.join(input_path, fname)) for fname in files]

    combined_df = pd.concat(d)
    combined_df.drop_duplicates(inplace=True)
    print(output_path)
    output_csv_path = os.path.join(output_path, "finaldata.csv")
    combined_df.to_csv(output_csv_path, index=False)

    output_txt_path = os.path.join(output_path, "ingestedfiles.txt")
    with open(output_txt_path, "w") as f:
        f.write(str(files))


if __name__ == "__main__":
    print("Running ingestion.py......")
    merge_multiple_dataframe()