"""Basic inference script. It should be used for prototyping or quick testing"""

import argparse
from src.yolo.yolo_model import YoloModel
from src.yolo.utils.unit_conversor import UnitConversor
import torch
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import contextily as ctx


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


def main():
    args = get_args()
    model = YoloModel(
        model=args.model,
        tiled=args.tiled,
        input_data=args.input_data,
        log_files=args.log_files,
    )
    r_boxes = model.inference()

    boxes = torch.tensor(r_boxes, dtype=torch.int16)
    drone_pos = (-33.253099, -54.504020)
    rel_alt = 4.793

    conversor = UnitConversor(
        rel_altitude=rel_alt, boxes=boxes, drone_pos=drone_pos, gb_yaw=29.5
    )
    lats, lons = conversor.calc_rw_positions_boxes()

    # Crear GeoDataFrame con detecciones y dron juntos
    geometries = [Point(lon, lat) for lat, lon in zip(lats, lons)]
    geometries.append(Point(drone_pos[1], drone_pos[0]))

    types = ['detection'] * len(lats) + ['drone']

    gdf = gpd.GeoDataFrame(
        {'type': types}, geometry=geometries, crs='EPSG:4326'
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(12, 10))

    # change styles depending on the thing we are plotting
    gdf[gdf['type'] == 'detection'].plot(
        ax=ax, color='red', markersize=50, label='Detections', zorder=2
    )
    gdf[gdf['type'] == 'drone'].plot(
        ax=ax, color='blue', markersize=100, marker='^', label='Drone', zorder=3
    )

    # add satellite map - it does not work so fuck me
    ctx.add_basemap(ax, source=ctx.providers.NASAGIBS.BlueMarble)

    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
