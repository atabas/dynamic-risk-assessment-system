import json
import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

###################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
model_path = os.path.join(config["output_model_path"])


#################Function for training the model
def train_model():

    # use this logistic regression for training
    m = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="ovr",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X = df.loc[:, ["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    y = df["exited"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = m.fit(X_train, y_train)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # write the trained model to your workspace in a file called trainedmodel.pkl
    save_model_filename = os.path.join(model_path, "trainedmodel.pkl")
    pickle.dump(model, open(save_model_filename, "wb"))


if __name__ == "__main__":
    train_model()
