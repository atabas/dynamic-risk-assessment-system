import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from diagnostics import model_predictions

###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"])
output_model_path = os.path.join(config["output_model_path"])


##############Function for reporting
def score_model(test_data_path, test_data_csv_name):
    # calculate a confusion matrix using the test data and the deployed model
    # write the confusion matrix to the workspace
    print(type(test_data_path), type(test_data_csv_name))
    test_dataset_path = os.path.join(test_data_path, test_data_csv_name)
    df = pd.read_csv(test_dataset_path)

    predictions = model_predictions(test_data_csv_name)
    y_test = df["exited"]
    cfm = metrics.confusion_matrix(y_test, predictions)

    # Credit for confusion matrix code:
    # https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cfm.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cfm.flatten() / np.sum(cfm)
    ]
    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cfm, annot=labels, fmt="", cmap="Blues")

    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))


if __name__ == "__main__":
    score_model(test_data_path, "testdata.csv")
