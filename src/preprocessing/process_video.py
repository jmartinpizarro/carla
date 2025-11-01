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

from moviepy import *
from tqdm import tqdm

DATA_ORIGIN_ROUTE = 'data/'
SECONDS_OFFSET = 3  # every 5 seconds an action will be apply


def main(outputRoute: str, fileName: str, offset: str) -> int:
    # transform arguments into usable contents

    outputRoute = str(outputRoute)
    fileName = str(fileName)
    offset = int(str(offset))

    print(
        '\n\n[script.process_video]::Starting this scripts with config:\n\n',
        f'\t--outputRoute: {outputRoute}\n',
        f'\t--fileName: {fileName}\n',
        f'\t--offset: {offset}\n',
    )

    start = time.time()

    extract_frames(outputRoute, fileName, offset)

    end = time.time()

    print(
        f'\n[script.process_video]::Script has ended, taking a total of {(end - start):.2f} seconds\n'
    )

    return 0


def extract_frames(outputRoute: str, fileName: str, offset: int) -> None:
    """
    Extracts n frames until the video has finished.
    @param outputRoute: str -> output folder for the images
    @param fileName: str -> .mp4 video route
    @param offset: int -> an integer that defines since what second the frame generator should start
    """

    if not os.path.exists(outputRoute):
        os.makedirs(outputRoute)

    fileRoute = DATA_ORIGIN_ROUTE + fileName

    clip = VideoFileClip(fileRoute)
    # the user may want to remove some extra and useless frames
    clip = clip.subclipped(offset)

    c = 0
    with tqdm(
        total=int(clip.duration // SECONDS_OFFSET),
        desc='Extracting frames',
        unit='frame',
    ) as pbar:
        while c < clip.duration:
            imgPath = os.path.join(outputRoute, f'{fileName}_{c}.png')
            clip.save_frame(imgPath, c)

            c += SECONDS_OFFSET  # each five seconds get an image
            pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CARLA Video Processor, transforms an .mp4 into a number of images for labelling them.'
    )

    parser.add_argument(
        '--outputRoute',
        required=True,
        help='The output route folder for the frames to be saved.',
    )

    parser.add_argument(
        '--fileName',
        required=True,
        help='A .mp4 file that will be processed and segmented into different random images.',
    )
    parser.add_argument(
        '--offset',
        required=True,
        help='A number (in seconds), that will be used for shortcuting an .mp4 and obtaining images in that interval',
    )
    args = parser.parse_args()

    main(Path(args.outputRoute), Path(args.fileName), Path(args.offset))
