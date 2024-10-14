import cv2
from ultralytics import YOLO
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the image using OpenCV
image_path = '/Users/jamie/Desktop/untitled folder/16AB_2.jpg'
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not read the image.")
    exit()

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # You can also use 'yolov8s.pt' for a larger version

# Perform object detection
results = model(image_path)  # YOLOv8 works with image paths directly

# Make a copy of the original image for visualization
image_with_detections = image.copy()

# Iterate over the detections and draw bounding boxes
for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x_min, y_min, x_max, y_max)
    scores = result.boxes.conf.cpu().numpy()  # Confidence scores
    labels = result.boxes.cls.cpu().numpy()   # Class labels
    
    for i in range(len(boxes)):
        # Extract coordinates, label, and confidence
        xmin, ymin, xmax, ymax = boxes[i].astype(int)
        score = scores[i]
        class_id = int(labels[i])
        class_name = model.names[class_id]  # Get the class name

        # Print the detection in the terminal
        print(f"Detected {class_name} at [{xmin}, {ymin}, {xmax}, {ymax}] with confidence {score:.2f}")

        # Draw the bounding box
        cv2.rectangle(image_with_detections, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Display label and confidence score
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image_with_detections, label, (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('Detections', image_with_detections)
cv2.waitKey(0)
cv2.destroyAllWindows()
