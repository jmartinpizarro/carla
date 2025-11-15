from ultralytics import YOLO
import cv2

MODEL_ROUTE = 'runs/detect/train2_batch_8/weights/best.pt'
VIDEO_SOURCE = 'data/YDRAY-DJI_20250515124024_0043_D.MP4'

# load model
model = YOLO(MODEL_ROUTE)

# inference
results = model(VIDEO_SOURCE, stream=True)

# cv2 config
# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (100, 100)
# fontScale
fontScale = 3
# Blue color in BGR
color = (0, 0, 255)
# Line thickness of 2 px
thickness = 5

# render results per frame
for result in results:
    frame = result.plot(font_size=20.0, line_width=3)  # draw the output
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
    cv2.imshow(f'{VIDEO_SOURCE} YOLO Predictions', frame)

    # Sexit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
