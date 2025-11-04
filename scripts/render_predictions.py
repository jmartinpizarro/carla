"""
This script contains the code for rendering the rectangles of an image
based on T-Rex predictions (x_min, y_min, x_max, y_max) stored in a nested JSON.
"""

import cv2
import json
import argparse
import os

def draw_trex_predictions(image_path, json_path, output_path):
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Cargar predicciones
    with open(json_path, 'r') as f:
        predictions = json.load(f)

    # Buscar la entrada correspondiente a la imagen
    if image_name not in predictions:
        print(f"No predictions found for {image_name}")
        return

    objects = (
        predictions[image_name]
        .get("data", {})
        .get("result", {})
        .get("objects", [])
    )

    for obj in objects:
        bbox = obj.get("bbox", [])
        if len(bbox) == 4:
            x_min, y_min, x_max, y_max = map(int, bbox)
            score = obj.get("score", 0.0)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(
                image,
                f"{score:.2f}",
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, image)
    print(f"Rendered image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Render T-Rex predictions for a given image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--json", default="trex_predictions_v2.json", help="Path to the T-Rex predictions JSON file.")
    parser.add_argument("--output", default="rendered_trex.jpg", help="Path to save the rendered image.")
    args = parser.parse_args()

    draw_trex_predictions(args.image, args.json, args.output)

if __name__ == "__main__":
    main()
