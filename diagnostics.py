import pandas as pd
import numpy as np
import time
import os
import json
import pickle
import subprocess as sp
import sys

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

##################Function to get model predictions
def model_predictions(dataset):
    # read the deployed model and a test dataset, calculate predictions
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), "rb") as f:
        model = pickle.load(f)
    test_dataset_path = os.path.join(test_data_path, dataset)
    df = pd.read_csv(test_dataset_path)
    X_test = df.loc[
        :, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ]
    predictions = model.predict(X_test)
    return predictions  # return value should be a list containing all predictions


##################Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X = df.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]]

    summary_stats = ((col, [X[col].mean(), X[col].median(), X[col].std()]) for col in X)
    return list(
        summary_stats
    )  # return value should be a list containing all summary statistics


def dataframe_missing_values():
    # Calculate percentage of missing values
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    return df.isna().sum() / df.count().sum()


##################Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    start_time_training = time.time()
    process = sp.call(["python", "training.py"])
    end_time_training = time.time() - start_time_training

    start_time_ingestion = time.time()
    process = sp.call(["python", "ingestion.py"])
    end_time_ingestion = time.time() - start_time_ingestion
    return [
        end_time_training,
        end_time_ingestion,
    ]  # return a list of 2 timing values in seconds


##################Function to check dependencies
def outdated_packages_list():
    # get a list of outdated packages
    args = [sys.executable, "-m", "pip", "list", "--outdated"]
    results = sp.run(args, capture_output=True, check=True).stdout
    indented_results = ("\n" + results.decode()).replace("\n", "\n    ")
    return indented_results


if __name__ == "__main__":
    model_predictions("testdata.csv")
    dataframe_summary()
    dataframe_missing_values()
    execution_time()
    outdated_packages_list()
