"""Basic inference script. It should be used for prototyping or quick testing"""

import argparse
import matplotlib

matplotlib.use('Agg')  # otherwise memory goes fucked
from src.yolo.yolo_model import YoloModel
from src.yolo.utils.unit_conversor import UnitConversor
import torch
import geopandas as gpd
from shapely.geometry import Point
import cv2
import os
import glob


def get_args():
    parser = argparse.ArgumentParser('PoC Inference')

    parser.add_argument('--model', required=True, help='The route to the .pt model')

    parser.add_argument(
        '--tiled',
        action='store_true',
        required=False,
        help='The model that uses the inference uses tiling mechanisms or not',
    )

    parser.add_argument(
        '--input-data',
        required=True,
        help='The route to the file (image or video) to which inference is going to be made',
    )

    parser.add_argument(
        '--log-files',
        required=False,
        help='Depuration or extra information. Creates a file and anotates there coverage, boxes location and more.',
    )

    return parser.parse_args()


def create_kde_video(
    kde_dir='kde_plots', output_video='kde_plots/kde_heatmap_video.mp4', fps=5
):
    """
    Creates a video from KDE PNG files in the kde_plots directory

    :param kde_dir: Directory containing the PNG files
    :param output_video: Output video filename
    :param fps: Frames per second for the output video
    """
    # Get all PNG files matching the pattern
    png_files = sorted(
        glob.glob(os.path.join(kde_dir, 'density_heatmap_kde_*.png')),
        key=lambda x: int(x.split('_')[-1].split('.')[0]),
    )

    if not png_files:
        print(f'[create_kde_video] :: No PNG files found in {kde_dir}')
        return

    print(f'[create_kde_video] :: Found {len(png_files)} PNG files')

    # Read first image to get dimensions
    first_frame = cv2.imread(png_files[0])
    height, width, _ = first_frame.shape

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Write all frames to video
    for i, png_file in enumerate(png_files):
        frame = cv2.imread(png_file)
        out.write(frame)
        if (i + 1) % 10 == 0:
            print(f'[create_kde_video] :: Processed {i + 1}/{len(png_files)} frames')

    out.release()
    print(f'[create_kde_video] :: Video saved to {output_video}')


def main():
    args = get_args()
    model = YoloModel(
        model=args.model,
        tiled=args.tiled,
        input_data=args.input_data,
        log_files=args.log_files,
    )
    # PART 1: THE INFERENCE
    r_boxes = model.inference()

    if not r_boxes:
        print('No detections found. KDE not possible to display')
        return

    print(r_boxes)

    # PART 2: THE KDE THING
    # as r_boxes saves all the predictions from every frame, it is possible
    # to access (if video) to all predictions and do, per frame, a KDE. This is
    # computationally fucking expensive, but thats how life works.
    for k in r_boxes.keys():
        print(f'[run_inference] :: Computing KDE for frame {k}\n')
        boxes = torch.tensor(r_boxes[k], dtype=torch.int16)
        drone_pos = (-33.253099, -54.504020)
        rel_alt = 7.845

        conversor = UnitConversor(
            rel_altitude=rel_alt, boxes=boxes, drone_pos=drone_pos, gb_yaw=29.5
        )

        # transform positions into lat, lon
        lats, lons = conversor.calc_rw_positions_boxes()

        # (no drone pos included) - could be useful in order to see the evolution
        # of the flight
        geometries = [Point(lon, lat) for lat, lon in zip(lats, lons)]
        gdf = gpd.GeoDataFrame(
            {'type': ['detection'] * len(geometries)},
            geometry=geometries,
            crs='EPSG:4326',
        ).to_crs(epsg=3857)

        model.generate_density_heatmap(gdf, bandwidths=[1.0, 2.0, 5.0], frame=k)

    # PART 3: CREATE VIDEO FROM KDE PLOTS
    print('\n[run_inference] :: Creating video from KDE plots...\n')
    create_kde_video()


if __name__ == '__main__':
    main()
