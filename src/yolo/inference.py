from ultralytics import YOLO
import cv2

MODEL_ROUTE = 'runs/yolo11l_e100_b16_s42_box7.5_09775/weights/best.pt'
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

# for video saving
# open video to get properties
cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# define output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

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
    out.write(frame)
    cv2.imshow(f'{VIDEO_SOURCE} YOLO Predictions', frame)

    # Sexit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

out.release()
cap.release()
