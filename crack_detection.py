from ultralytics import YOLO
import cv2

# Load your trained YOLO model (update path as per your model location)
model = YOLO("C:/Users/kanis/runs/detect/train/weights/best.pt")

# Open the default webcam
cap = cv2.VideoCapture(0)

# Check if webcam is accessible
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO inference on the frame
    results = model(frame)

    # Get image with boxes drawn
    result_frame = results[0].plot()

    # Draw "Crack Detected" label on each detection
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.putText(result_frame, "Crack Detected", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Crack Detection - Webcam", result_frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
