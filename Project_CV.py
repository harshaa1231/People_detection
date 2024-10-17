import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Function to detect and classify objects, and count people
def detect_and_classify_objects(frame):
    results = model.predict(frame)  # Run detection on the current frame
    person_count = 0  # Initialize count for people

    # Loop through detected objects
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get box coordinates
        class_id = int(box.cls)  # Get the predicted class ID
        label = results[0].names[class_id]  # Get the class label from the names list

        # Draw bounding box and classify objects
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green box
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 255, 0), 2)  # Display label above the bounding box

        # Count people (or any other specific object you want to count)
        if label == 'person':
            person_count += 1  # Increment the people count

    # Display the total people count on the frame
    cv2.putText(frame, f"People Count: {person_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red text for people count
    
    return frame, person_count  # Return both the frame and people count

# Function to process the video, display results, and save to file
def process_video(input_path, output_video_path):
    cap = cv2.VideoCapture(input_path)  # Create VideoCapture object

    # Get video properties (width, height, frames per second)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Exit if no frame is read

        # Detect and classify objects, and get the people count
        output_frame, people_count = detect_and_classify_objects(frame)

        # Write the frame with bounding boxes, labels, and people count to the output video
        out.write(output_frame)

        # Display the processed frame with bounding boxes, labels, and people count
        cv2.imshow('Object Detection and People Counting', output_frame)

        # Press 'q' to quit the video display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects, and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Main function to run the program
if __name__ == "__main__":
    input_path = input("Enter the path to a video: ")  # e.g., '/path/to/video.mp4'
    output_path = input("Enter the path to save the output video: ")  # e.g., 'output.mp4'
    
    process_video(input_path, output_path)  # Call the function to process and save the video
