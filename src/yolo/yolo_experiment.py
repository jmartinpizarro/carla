"""
Class that represents a single instance of a YoloExperiment, which is considered an abstraction
of an entire experiment: all possible models are trained with different seeds.
"""

from typing import Dict
import itertools
import yaml
import os
import pandas as pd

from src.yolo.yolo_run import YoloRun

TESTING_EXPORT_DIR = 'model_testing_results'


class YoloExperiment:
    def __init__(self, config: str):
        """
        :param config: str -> Route to the .yaml file that contains the training config
        """
        self.config: Dict = self.parse_config(config)

    def parse_config(self, config_route: str):
        """
        Reads the .yaml file and generates a configuration for the experiment

        :param config_route: str -> Route to the .yaml file that contains the training config
        """

        try:
            with open(config_route, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(
                f'[YoloExperiment :: parse_config()] :: An error has ocurred when importing the .yaml file:\n{e}\n'
            )
            return

        # asume user is not retard and it will not do some weird combinations
        if config['epochs'] <= 0:
            print(
                '[YoloExperiment :: parse_config()] :: The epochs parameter cannot be less or equal than 0'
            )
            return

        if config['min_seed'] >= config['max_seed']:
            print(
                '[YoloExperiment :: parse_config()] :: The min and max seed where reversed. min_seed must be lower than max_seed'
            )

        if config['results_folder'] == '':
            config['results_folder'] = 'results_yolo'

        return config

    def grid_search(self, config):
        """
        Generate all possible combinations for the training config.
        Seeds are handled sequentially: for each parameter combination,
        iterate through seeds from min_seed to max_seed.

        :param config: Dict -> the parsed config from the .yaml file
        """
        # Extract seed range
        min_seed = config['min_seed']
        max_seed = config['max_seed']

        # Exclude seed parameters from grid search
        grid_config = {
            k: v for k, v in config.items() if k not in ['min_seed', 'max_seed']
        }

        # Create grid search for all parameters except seeds
        keys = list(grid_config.keys())
        values = (
            grid_config[k] if isinstance(grid_config[k], list) else [grid_config[k]]
            for k in keys
        )

        # For each parameter combination, iterate through seeds sequentially
        for combo in itertools.product(*values):
            param_combo = dict(zip(keys, combo))
            # Generate sequential seeds for this parameter combination
            for seed in range(min_seed, max_seed + 1):
                result = param_combo.copy()
                result['seed'] = seed
                yield result

    def start_experiment(self, data: str):
        """
        Starts the experiment, creating a YoloRun object for each experiment.

        :param data: str -> the route where the data is located. It must not point to the .yaml file
        """

        runs = []

        for config in self.grid_search(self.config):
            run = YoloRun(
                config['model'],
                config['epochs'],
                config['batch'],
                config['seed'],
                config['box'],
                data,
                config['tiled'],
            )

            # train
            results_folder = run.train()
            # then test
            run.test(results_folder)
            # as metrics have been saved into the YoloRun object, it is possible to retrieve them after the training. If the program breaks (e.g. VRAM), training metrics are lost
            runs.append(run)

            os.makedirs(f'{TESTING_EXPORT_DIR}/{results_folder}', exist_ok=True)

            test_metrics_df = pd.DataFrame(run.metrics)
            test_csv_path = os.path.join(
                f'{TESTING_EXPORT_DIR}/{results_folder}',
                f'{results_folder}_test_metrics.csv',
            )
            test_metrics_df.to_csv(test_csv_path, index=False)
