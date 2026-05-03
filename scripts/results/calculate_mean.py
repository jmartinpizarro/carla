"""
This scripts gets the average of all the possible models and returns a .csv
with the average data. Groups models by brand (yolov8, yolo11, yolo26) and
by training approach (tiled or not tiled).
"""

import ast
import re
from pathlib import Path
import argparse
import os

import pandas as pd
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        description='Mean Calculator for .csv training output'
    )

    parser.add_argument(
        '--data',
        required=True,
        help='The route to the folder that contains the metrics (results_nt or results_t).',
    )

    parser.add_argument(
        '--tiled',
        action='store_true',
        help='Flag to indicate if models were trained with tiling approach.',
    )

    parser.add_argument(
        '--output',
        default=None,
        help='Output directory for mean results (default: model_<yolov8|yolo11|yolo26>).',
    )

    return parser.parse_args()


def extract_model_brand(folder_name: str) -> str | None:
    """Extract model brand from folder name.

    Expected format: results_{model_name}_e{epochs}_b{batch}_s{seed}_{id}
    E.g.: results_yolov8n_e200_b16_s42_e3157 -> yolov8
    """
    # Match yolo followed by v8, v11, v26, 8, 11, or 26
    match = re.search(r'(yolo(?:v)?(?:8|11|26))', folder_name)
    if match:
        return match.group(1)
    return None


def sanitize_df(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: (
                    np.mean(ast.literal_eval(x))
                    if isinstance(x, str) and x.startswith('[')
                    else x
                )
            )
    return df


def main():
    args = get_args()

    path = Path(args.data)

    # Group models by brand
    models_by_brand = {'yolov8': [], 'yolo11': [], 'yolo26': []}

    # obtain and divide depending on the model brand
    for p in path.iterdir():
        if not p.is_dir():
            continue

        folder_name = p.name
        brand = extract_model_brand(folder_name)

        if brand in models_by_brand:
            models_by_brand[brand].append(p)
        else:
            print(f'[warning] Skipping folder: {folder_name} (unknown model brand)')

    # Determine tiling suffix for output
    tiling_suffix = '_tiled' if args.tiled else '_non_tiled'

    for brand, folders in models_by_brand.items():
        if not folders:
            print(f'[skip] No models found for {brand}')
            continue

        print(f'\n[main] :: Creating mean .csv for {brand}{tiling_suffix}\n')

        # Output directory
        output_dir = args.output or f'model_{brand}'
        os.makedirs(output_dir, exist_ok=True)

        # we will use a LIFO method for calculating the mean
        last = folders.pop()

        training_path = os.path.join(last, 'results.csv')
        testing_path = os.path.join(last, 'test_results.csv')

        if not os.path.exists(training_path) or not os.path.exists(testing_path):
            print(f'[skip] CSV files not found for {last.name}')
            continue

        training = pd.read_csv(training_path)
        testing = pd.read_csv(testing_path)

        testing = sanitize_df(testing)

        while len(folders):
            last = folders.pop()
            training_path = os.path.join(last, 'results.csv')
            testing_path = os.path.join(last, 'test_results.csv')

            if not os.path.exists(training_path) or not os.path.exists(testing_path):
                print(f'[skip] CSV files not found for {last.name}')
                continue

            dummy = pd.read_csv(training_path)
            dummy_test = pd.read_csv(testing_path)

            dummy_test = sanitize_df(dummy_test)

            training_mean = pd.concat([training, dummy]).groupby(level=0).mean()
            training = training_mean

            testing_mean = pd.concat([testing, dummy_test]).groupby(level=0).mean()
            testing = testing_mean

        # Save with tiling suffix
        training_output = os.path.join(
            output_dir, f'{brand}{tiling_suffix}_training_mean_results.csv'
        )
        testing_output = os.path.join(
            output_dir, f'{brand}{tiling_suffix}_test_mean_results.csv'
        )

        training.to_csv(training_output, index=True)
        testing.to_csv(testing_output, index=True)

        print(f'[done] Saved to {training_output} and {testing_output}')


if __name__ == '__main__':
    main()
