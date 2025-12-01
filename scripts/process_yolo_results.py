"""
This script contains the code for visualizing and computing accuracy metrics for the YOLO predictions (and its modifications)
"""

import os
import argparse

import pandas as pd
from ultralytics import YOLO

TESTING_EXPORT_DIR = 'model_testing_results'

def get_args():
    parser = argparse.ArgumentParser(
        description='YOLO and YOLO-mods Results Renderer'
    )

    parser.add_argument(
        "--data",
        default='data/yolo_v3/data.yaml',
        type=str,
        help="Route where the .yaml definition of the dataset for YOLO can be found."
    )

    parser.add_argument(
        '--results-folder',
        default='runs',
        type=str,
        help="Route where the output model and its results are generated. By default is 'runs/'",
    )

    return parser.parse_args()


def main():
    args = get_args()

    best_precision = 0.0
    best_precission_model = None
    best_map50 = 0.0
    best_map50_model = None

    models = os.listdir(args.results_folder)

    models_output = {}

    print('[process_yolo_results] :: Starting the script\n')

    for model in models:
        print(f'\n\n\t[process_yolo_results] :: Testing {model}\n\n')

        route = os.path.join(args.results_folder, model)
        models_output[model] = {}

        # if the model could not be trained, skip it
        if model == 'detect' or not os.listdir(os.path.join(route, 'weights')):
            continue

        df = pd.read_csv(os.path.join(route, 'results.csv'))
        # it will just contain the metrics/precision(B),metrics/mAP50(B),metrics/mAP50-95(B)
        metrics = df[
            [
                'epoch',
                'metrics/precision(B)',
                'metrics/mAP50(B)',
                'metrics/mAP50-95(B)',
            ]
        ]

        models_output[model]['training_metrics'] = metrics

        # now it is necessary to obtain the metrics with the test set
        YOLO_MODEL = YOLO(f'{route}/weights/best.pt')
        prediction_metrics = YOLO_MODEL.val(
            data=args.data,
            split='test',
            imgsz=640,
            half=True,
            device='cuda',
            save_json=False,
            verbose=False,
        )

        models_output[model]['test_metrics'] = {
            'mAP50': prediction_metrics.box.map50,
            'mAP50-95': prediction_metrics.box.map,
            'precision': prediction_metrics.box.p,
            'recall': prediction_metrics.box.r,
        }

        # without any more relevant analysis, its complicated to define which one is the
        # best. However, it is possible to save the best precission and mAP50
        if prediction_metrics.box.p >= best_precision:
            best_precision = prediction_metrics.box.p[0] # only one class
            best_precission_model = model

        if prediction_metrics.box.map50 >= best_map50:
            best_map50 = prediction_metrics.box.map50
            best_map50_model = model

        print(
            '[process_yolo_results] :: The script has ended\n',
            f'\t The model with the best precission was: {best_precission_model} with {best_precision}\n',
            f'\t The model with the best map50 was: {best_map50_model} with {best_map50}\n',
        )   

        os.makedirs(f"{TESTING_EXPORT_DIR}/{model}", exist_ok=True)

        # Training metrics → CSV
        training_csv_path = os.path.join(
            f"{TESTING_EXPORT_DIR}/{model}", f'{model}_training_metrics.csv'
        )
        models_output[model]['training_metrics'].to_csv(
            training_csv_path, index=False
        )

        # Test metrics → CSV
        test_metrics_df = pd.DataFrame([models_output[model]['test_metrics']])
        test_csv_path = os.path.join(
            f"{TESTING_EXPORT_DIR}/{model}", f'{model}_test_metrics.csv'
        )
        test_metrics_df.to_csv(test_csv_path, index=False)




if __name__ == '__main__':
    main()
