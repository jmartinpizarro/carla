import argparse
import base64
import json
import time
import os

import requests

CREATE_TASK_API_URL = 'https://api.deepdataspace.com/v2/task/trex/detection'
QUERY_TASK_API_URL = (
    'https://api.deepdataspace.com/v2/task_status'  # + /{task_uuid}
)
# this image is used for giving T-Rex2 some information about the phenotype of the plant
HELPER_IMAGE = 'data/v1/train/YDRAY-DJI_20250515122715_0039_D_MP4_162_png.rf.41bcd46db0c603f36d6037f31d757ecd.jpg'


def encode_image_to_base64(path):
    """Encodes an image - don't care about the extension - into a base64 format. Used for calling the trex api"""

    with open(path, 'rb') as file:
        data = file.read()
    ext = path.split('.')[-1].lower()
    mime = 'jpeg' if ext in ['jpg', 'jpeg'] else 'png'
    return f'data:image/{mime};base64,{base64.b64encode(data).decode()}'


def get_args():
    parser = argparse.ArgumentParser(description='T-Rex2 API Inference')
    parser.add_argument(
        '--token', required=True, type=str, help='Token of the T-Rex2 API'
    )

    parser.add_argument(
        '--input-dir',
        required=True,
        type=str,
        help='Input directory to take images from',
    )

    parser.add_argument(
        '--output-dir', type=str, default='./output/', help='Output directory'
    )

    return parser.parse_args()

def main():
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    print(
        '[T-Rex 2 Inference] :: Running the script for the T-Rex 2 inference...\n',
        f'\t Data will be taken from {args.input_dir}\n',
        f'\t Predictions will be saved in {args.output_dir}\n',
    )

    start = time.time()

    images = [f for f in os.listdir(args.input_dir) if f.endswith('.jpg')]

    helper_filename = os.path.basename(HELPER_IMAGE)
    helper_image_route = os.path.join(
        args.input_dir, os.path.basename(HELPER_IMAGE)
    )

    predictions = {}

    # asume data will be always follow the COCO format
    annotations_file = 'data/v1/train/_annotations.coco.json'

    with open(annotations_file, 'r') as _f:
        labels = json.load(_f)

    for target_image in images:
        filename = os.path.join(args.input_dir, os.path.basename(target_image))

        if filename == HELPER_IMAGE:
            continue

        # next(filter()) returns the first element where the condition is true
        # filter(condition, iterableObject)
        helper_image_id = next(
            filter(
                lambda x: x['file_name'] == helper_filename, labels['images']
            )
        )['id']

        # take the box labels info from the helper image
        helper_boxes = list(
            filter(
                lambda x: x['image_id'] == helper_image_id,
                labels['annotations'],
            )
        )
        
        # trex only processes rect and the bbox of it. Other thing will suppose an error
        helper_boxes = [b for b in helper_boxes if 'bbox' in b]

        # create a copy bcs if i modify the original i fuck myself
        helper_boxes_copy = []
        for box in helper_boxes:
            box_copy = box.copy()  # copy dict
            box_copy['type'] = 'rect'
            box_copy['rect'] = box_copy['bbox']  # don't use pop, can cause error
            del box_copy['bbox']  # this is optional
            helper_boxes_copy.append(box_copy)

        parameters = {
            'model': 'T-Rex-2.0',
            'image': encode_image_to_base64(filename),
            'targets': ['bbox', 'embedding'],
            'prompt': {
                'type': 'visual_images',
                'visual_images': [
                    {
                        'image': encode_image_to_base64(helper_image_route),
                        'interactions': helper_boxes_copy,  # Usa la copia
                    }
                ],
            },
        }

        headers = {'Content-Type': 'application/json', 'Token': args.token}
        resp = requests.post(
            url=CREATE_TASK_API_URL,
            json=parameters,  # parameters
            headers=headers,
        )

        json_resp = resp.json()

        # Step 3: Extract task_uuid
        try:
            task_uuid = json_resp['data'][
                'task_uuid'
            ]  # could get a task_uuid error if the petition is not done

            # Step 4: Poll for task status
            while True:
                resp = requests.get(
                    f'https://api.deepdataspace.com/v2/task_status/{task_uuid}',
                    headers=headers,
                )
                json_resp = resp.json()
                if json_resp['data']['status'] not in ['waiting', 'running']:
                    break
                time.sleep(1)

            if json_resp['data']['status'] == 'failed':
                predictions[target_image] = '-1'
                print(
                    f'[T-Rex 2 Inference] :: An error has ocurred when processing image: {target_image}\n',
                    f'\t{json_resp}',
                )
            elif json_resp['data']['status'] == 'success':
                print(
                    f'[T-Rex 2 Inference] :: Prediction obtained for: {target_image}\n'
                )
                predictions[target_image] = json_resp
        except KeyError as e:
            print(
                f'[T-Rex 2 Inference] :: An error has ocurred when processing image: {target_image}\n',
                f'\t{e}',
            )

    with open(
        os.path.join(args.output_dir, 'trex_predictions.json'), 'w'
    ) as fp:
        json.dump(predictions, fp)

    end = time.time()

    print(
        '[T-Rex 2 Inference] :: All possible predictions has been obtained!\n'
    )
    print('\tThe program has lasted a total of: %s seconds\n' % (end - start))

    return 1


if __name__ == '__main__':
    main()