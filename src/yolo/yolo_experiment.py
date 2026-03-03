"""
Class that represents a single instance of a YoloExperiment, which is considered an abstraction
of an entire experiment: all possible models are trained with different seeds.
"""

from typing import Dict


class YoloExperiment:
    def __init__(self, config: str):
        """
        :param config: str -> Route to the .yaml file that contains the training config
        """
        self.config: Dict = self.parse_config(config)

    def parse_config(self, config_route: str):
        pass

    def calculate_average_metrics(self):
        pass
