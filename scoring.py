import json
import os
import pickle

import pandas as pd
from sklearn import metrics

#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
output_model_path = os.path.join(config["output_model_path"])


#################Function for model scoring
def score_model(test_data_path, test_data_csv_name):
    # this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file
    model_path = os.path.join(output_model_path, "trainedmodel.pkl")
    test_dataset_path = os.path.join(test_data_path, test_data_csv_name)
    output_scores_path = os.path.join(output_model_path, "latestscore.txt")

    df = pd.read_csv(test_dataset_path)
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    X_test = df.loc[
        :, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ]
    y_test = df["exited"]

    predictions = model.predict(X_test)
    f1_score = metrics.f1_score(predictions, y_test)
    with open(output_scores_path, "w") as f:
        f.write(str(f1_score))

    print("f1 score is: ", f1_score)
    return f1_score


if __name__ == "__main__":
    score_model(test_data_path, "testdata.csv")
