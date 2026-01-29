"""
This scripts gets the average of all the possible models and returns a .csv
with the average data.
"""

import os
import ast
from pathlib import Path
import argparse

import pandas as pd
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        description="Mean Calculator for .csv training output"
    )

    parser.add_argument(
        '--data',
        required=True,
        help="The route to the .csv (folder, not file) that contains the metrics."
    )

    return parser.parse_args()

def sanitize_df(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: np.mean(ast.literal_eval(x))
                if isinstance(x, str) and x.startswith("[")
                else x
            )
    return df

def main():
    args = get_args()

    path = Path(args.data)
    models = {"yolov8n": [], "yolo11n": [], "yolo26n": []}

    # obtain and divide depending on the model trained
    for p in path.iterdir():
        if "yolov8n" in str(p):
            models["yolov8n"].append(p)
        elif "yolo11n" in str(p):
            models["yolo11n"].append(p)
        elif "yolo26n" in str(p):
            models["yolo26n"].append(p)
        else:
            raise Exception("Error, a folder does not follow the format expected")

    for model in models.keys():
        print(f"\n\n\t[main] :: Creating mean .csv for {model}\n")

        folders = models[model]
        # we will use a LIFO method for calculating the mean
        last = folders.pop()

        training = pd.read_csv(f"{last}/{last.name}_training_metrics.csv")
        testing = pd.read_csv(f"{last}/{last.name}_test_metrics.csv")

        testing = sanitize_df(testing)

        while len(folders):
            last = folders.pop()
            dummy = pd.read_csv(f"{last}/{last.name}_training_metrics.csv")
            dummy_test = pd.read_csv(f"{last}/{last.name}_test_metrics.csv")

            dummy_test = sanitize_df(dummy_test)
            
            training_mean = pd.concat([training, dummy]).groupby(level=0).mean()
            training = training_mean

            testing_mean = pd.concat([testing, dummy_test]).groupby(level=0).mean()
            testing = testing_mean

        training.to_csv(f"{model}_training_mean_results.csv", index=True)
        testing.to_csv(f"{model}_test_mean_results.csv", index=True)


if __name__ == "__main__":
    main()

