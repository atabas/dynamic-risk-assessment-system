import json
import os
import shutil

##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])
output_model_path = os.path.join(config["output_model_path"])

model_to_copy = os.path.join(output_model_path, "trainedmodel.pkl")
scores_to_copy = os.path.join(output_model_path, "latestscore.txt")
ingestedfiles_to_copy = os.path.join(dataset_csv_path, "ingestedfiles.txt")
copy_all_files = [model_to_copy, scores_to_copy, ingestedfiles_to_copy]

####################function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    if not os.path.exists(prod_deployment_path):
        os.makedirs(prod_deployment_path)
    for f in copy_all_files:
        shutil.copy(f, prod_deployment_path)


if __name__ == "__main__":
    store_model_into_pickle()
