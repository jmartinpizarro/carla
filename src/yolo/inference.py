from ultralytics import YOLO
import cv2

MODEL_ROUTE = 'results_yolo_lowFlight_v7/yolov8l_e200_b16_s42_box7.5_1a939/weights/best.pt'
TO_PREDICT = 'data/lowFlightRGB_v7/test/images/DJI_20251205100553_0006_D_MP4_87_png.rf.f08b6669dcd4b0994bd1c495c0e4fbb7.jpg'

CONF_THRESHOLD = 0.31
CONF_IOU = 0.5

# Text config
font = cv2.FONT_HERSHEY_SIMPLEX
org = (100, 100)
fontScale = 3
color = (0, 0, 255)
thickness = 5

# Load model
model = YOLO(MODEL_ROUTE)

# Detect
results = model(TO_PREDICT, stream=False, conf=CONF_THRESHOLD, iou=CONF_IOU)
result = results[0]  # first (and only) result for images

# If it's an image → just render once
if not TO_PREDICT.lower().endswith('.mp4'):
    frame = result.plot(font_size=20.0, line_width=3)

    cv2.putText(
        frame,
        f"Cardilla's Number: {len(result.boxes)}",
        org,
        font,
        fontScale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    cv2.imwrite('output.png', frame)
    cv2.imshow('YOLO Prediction', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# If it's a video → process frame by frame
else:
    cap = cv2.VideoCapture(TO_PREDICT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    results = model(TO_PREDICT, stream=True, conf=CONF_THRESHOLD, iou=CONF_IOU)

    for result in results:
        frame = result.plot(font_size=20.0, line_width=3)
        cv2.putText(
            frame,
            f"Cardilla's Number: {len(result.boxes)}",
            org,
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )

        out.write(frame)
        cv2.imshow('YOLO Video Prediction', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
