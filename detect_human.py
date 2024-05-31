import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2

# Open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    exit("Could not open webcam")
    
# Loop through frames
while webcam.isOpened():

    # Read frame from webcam
    status, frame = webcam.read()

    if not status:
        break

    # Apply object detection
    bbox, label, conf = cv.detect_common_objects(frame)

    # Filter for only people
    person_bbox = []
    person_label = []
    person_conf = []

    for i in range(len(label)):
        if label[i] == "person":
            person_bbox.append(bbox[i])
            person_label.append(label[i])
            person_conf.append(conf[i])

    # Draw bounding box over detected people
    out = draw_bbox(frame, person_bbox, person_label, person_conf, write_conf=True)

    # Display output
    cv2.imshow("Real-time object detection", out)

    # Press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Release resources
webcam.release()
cv2.destroyAllWindows()
