#!usr/bin/bash python3

"""
This file runs the code for extracting and generating the initial
labeled dataset (just a few images - up to 200-300) in order to
create a pipeline to do it automatically
"""

import os
import time
import argparse
from pathlib import Path


def main(fileName: str) -> int:
	print(fileName)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
		description='CARLA Initial Dataset Labeling Script: Use it if you need it to generate boxes for your data.'
	)
    
    parser.add_argument("--fileName", required=True, help="A .mp4 file that will be processed and segmented into different random images.")
    args = parser.parse_args()
    
    main(Path(args.fileName))
