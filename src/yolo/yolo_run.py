"""
YoloRun stands for a single instance of YOLO being training. It is able to
train, test and calculatemetrics from the test dataset partition
"""

import os
import uuid
from typing import Dict

from src.yolo.utils.coverage_utils import (
    gt_coverage_percent,
    pred_coverage_percent,
)

import numpy as np
from ultralytics import YOLO


class YoloRun:
    def __init__(self, model, epochs, batch, seed, box, data, tiled):
        self.model: str = model
        self.epochs: int = epochs
        self.batch: int = batch
        self.seed: int = seed
        self.box: float = box
        self.tiled: bool = tiled
        self.data = data  # it must not include data.yaml
        self.metrics: Dict = None

    def train(
        self,
    ):
        exp_id = uuid.uuid4().hex[:5]

        model_path = self.model
        model_name = os.path.splitext(os.path.basename(model_path))[0]

        exp_name = f'{model_name}_e{self.epochs}_b{self.batch}_s{self.seed}_{exp_id}'

        print(
            f'\n[YoloRun] :: Starting training experiment: {exp_name} with config:\n',
            f'\t-Model: {self.model}\n',
            f'\t-Epochs: {self.epochs}\n',
            f'\t-Batch Size: {self.batch}\n',
            f'\t-Seed: {self.seed}\n',
            f'\t-Bbox Loss Size: {self.box}\n',
            f'\t-Tiled: {self.tiled}\n',
        )
        try:
            model = YOLO(f'{self.model}.pt')
        except Exception as e:
            print(
                f'[YoloRun] :: An error has ocurred when importing the model {exp_name}:\n{e}\n'
            )

        try:
            model.train(
                data=f'{self.data}/data.yaml',
                epochs=self.epochs,
                batch=self.batch,
                imgsz=640,
                optimizer='auto',
                seed=self.seed,
                box=self.box,
                project=f'results_{exp_name}',
                name=exp_name,
                cache='disk',
                plots=True,
            )
        except Exception as e:
            print(
                f'[YoloRun] :: An error has ocurred during the training of the model {exp_name}:\n{e}\n'
            )

        return f'results_{exp_name}'

    def test(self, model_route: str):
        """
        Test implementation for the YoloRun

        :param model_route (exp_name in training): str -> The folder where the output of the training was done
        """

        print(
            f'\n[YoloRun] :: Starting training experiment: {model_route} with config:\n',
            f'\t-Model: {self.model}\n',
            f'\t-Epochs: {self.epochs}\n',
            f'\t-Batch Size: {self.batch}\n',
            f'\t-Seed: {self.seed}\n',
            f'\t-Bbox Loss Size: {self.box}\n',
            f'\t-Tiled: {self.tiled}\n',
        )

        try:
            YOLO_VAL = YOLO(f'{model_route}/weights/best.pt')
        except Exception as e:
            print(
                f'[YoloRun] :: An error has ocurred when importing the model {model_route}:\n{e}\n'
            )

        try:
            prediction_metrics = YOLO_VAL.val(
                data=f'{self.data}/data.yaml',
                split='test',
                imgsz=640,
                half=True,
                device='cuda',
                save=True,
                name=f'test_{model_route}',
                save_conf=True,
                save_txt=False,
                save_json=False,
                verbose=False,
            )
        except Exception as e:
            print(
                f'[YoloRun] :: An error has ocurred during the training of the model {model_route}:\n{e}\n'
            )

        model_coverage = {'gt': [], 'pred': []}

        # for each image of the dataset, process it and add it to the dict
        for image in os.listdir(f'{self.data}/test/images'):
            image_route = f'{self.data}/test/images/{image}'
            label_route = f'{self.data}/test/labels/{image[:-4]}.txt'

            model_coverage['gt'].append(gt_coverage_percent(image_route, label_route))
            model_coverage['pred'].append(pred_coverage_percent(image_route, YOLO_VAL))

        mean_pred = np.mean(model_coverage['pred'])
        mean_gt = np.mean(model_coverage['gt'])
        abs_diff = abs(mean_pred - mean_gt)

        self.metrics = {
            'mAP50': prediction_metrics.box.map50,
            'mAP50-95': prediction_metrics.box.map,
            'precision': prediction_metrics.box.p,
            'recall': prediction_metrics.box.r,
            'coverage': abs_diff,
        }

        return
