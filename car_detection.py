import cv2
from yolov5 import YOLOv5
import time

# Load the pre-trained model (YOLOv5s)
model = YOLOv5("yolov5s.pt", device="cpu")  # Change "cpu" to "cuda" if using GPU

# Load the video
video_path = "videos/traffic.mp4"  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Check if the video is loaded successfully
if not cap.isOpened():
    print("Error: Could not load the video. Please check the path.")
else:
    print("Video loaded successfully.")

# Get the frame rate (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video frame rate: {fps}")

# Calculate the frame interval (every 3 seconds)
frame_interval = int(fps * 1)  # 3 seconds

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Capture a frame every 3 seconds
    if frame_count % frame_interval == 0:
        # Detect vehicles in the frame
        results = model.predict(frame)

        # Count the number of vehicles
        car_count = 0
        for detection in results.pred[0]:
            class_id = int(detection[5])  # Class ID
            if class_id == 2:  # Class ID 2 is for cars in COCO dataset
                car_count += 1

        # Display the number of detected vehicles
        print(f"Number of vehicles detected in frame {frame_count}: {car_count}")

        # Display the frame with detection results
        results.show()

        # Save the frame as an image (optional)
        output_image_path = f"output/frame_{frame_count}.jpg"  # Replace with your save path
        cv2.imwrite(output_image_path, frame)
        print(f"Frame {frame_count} saved as: {output_image_path}")

    frame_count += 1

    # Press 'q' to exit the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()