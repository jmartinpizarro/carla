"""
Contains the class YoloModel used for doing inference with a .pt model.
"""

from src.yolo.utils.tiling_utils import process_frame_with_grids

import cv2
from ultralytics import YOLO


class YoloModel:
    def __init__(
        self, model: str, tiled: bool, input_data: str, log_files: str
    ):
        """
        :param model: str -> Route to the model
        :param tiled: bool -> The model uses tiling or not
        :param input_data: str -> Route to the file where you want to do the
        inference
        :param log_files: str -> Route where the program is going to write
        logging and the predictions
        """
        self.model: str = model
        self.tiled: bool = tiled
        self.input_data: str = input_data
        self.log_files: str = log_files

    def inference(self, conf_threshold=0.4, iou=0.75):
        try:
            YOLO_MODEL = YOLO(self.model)
        except Exception:
            print(
                f'[YoloModel] :: An error has ocurred when importing the model {self.model}\n'
            )
            return

        is_video = self.input_data.lower().endswith('.mp4')

        if is_video:
            # create some stuff we will need for procesing those files
            cap = cv2.VideoCapture(self.input_data)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        else:
            cap = None
            frame = cv2.imread(self.input_data)
            frames = [frame]

        frame_count = 0

        if self.log_files is not None:
            try:
                log_file = open(self.log_files, 'w')
                log_file.write(f'model:{self.model}\n')
                log_file.write(f'conf:{conf_threshold}\n')
                log_file.write(f'iou:{iou}\n')
                log_file.write('\n')
            except Exception as e:
                print(
                    f'[YoloModel] :: An error has ocurred when opening the file for writing the predictions:\n{e}\n'
                )
                return

        while True:
            if is_video:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                if frame_count >= len(frames):
                    break
                frame = frames[frame_count].copy()

            if self.log_files is not None:
                try:
                    if is_video:
                        log_file.write(f'<{frame_count}>\n')

                    if self.tiled:
                        boxes, scores, classes = process_frame_with_grids(
                            frame, YOLO_MODEL, conf_threshold
                        )
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box)
                            log_file.write(f'{x1},{y1},{x2},{y2}\n')
                            cv2.rectangle(
                                frame, (x1, y1), (x2, y2), (0, 0, 255), 3
                            )
                    else:
                        results = YOLO_MODEL(
                            frame, conf=conf_threshold, iou=iou
                        )

                        for r in results:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                log_file.write(f'{x1},{y1},{x2},{y2}\n')
                                cv2.rectangle(
                                    frame, (x1, y1), (x2, y2), (0, 0, 255), 3
                                )
                except Exception as e:
                    print(
                        f'[YoloModel] :: An error has ocurred when writing the predictions:\n{e}\n'
                    )
                    if self.log_files is not None:
                        log_file.close()
                    return

            if is_video:
                out.write(frame)
                frame_count += 1
                if frame_count % 30 == 0:
                    print(
                        f'Processed {frame_count}/{total_frames} frames ({100 * frame_count / total_frames:.1f}%)'
                    )
                cv2.imshow('YOLO Video Prediction', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cv2.imwrite('output.jpg', frame)
                cv2.imshow('YOLO Prediction', frame)
                cv2.waitKey(0)
                frame_count += 1

        if self.log_files is not None:
            log_file.close()

        if is_video:
            out.release()
            cap.release()

        cv2.destroyAllWindows()

    def write_predictions(self):
        pass
