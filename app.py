import json
import os
import pickle
import pandas as pd

from flask import Flask, request

from diagnostics import (dataframe_missing_values, dataframe_summary,
                         execution_time, model_predictions,
                         outdated_packages_list)
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
output_model_path = os.path.join(config["output_model_path"])

model_path = os.path.join(output_model_path, "trainedmodel.pkl")
with open(model_path, "rb") as f:
    prediction_model = pickle.load(f)


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # call the prediction function you created in Step 3
    dataset = request.args.get("dataset")
    df = pd.read_csv(dataset)
    preds = model_predictions(df)
    return str(preds)


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    # check the score of the deployed model
    return str(score_model(test_data_path, "testdata.csv"))


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def summarystats():
    # check means, medians, and modes for each column
    return str(
        dataframe_summary()
    )  # return a list of all calculated summary statistics


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    # check timing and percent NA values
    return str(
        {
            "time_in_seconds": execution_time(),
            "percent_missing_values": dataframe_missing_values(),
            "outdated_packages_list": outdated_packages_list(),
        }
    )  # add return value for all diagnostics


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
